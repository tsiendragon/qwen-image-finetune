import glob
import importlib
import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from qflux.data.cache_manager import EmbeddingCacheManager
from qflux.data.config import DatasetInitArgs
from qflux.losses.edit_mask_loss import map_mask_to_latent
from qflux.utils.huggingface import is_huggingface_repo, load_editing_dataset
from qflux.utils.tools import hash_string_md5, pad_to_max_shape


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


_pat_end = re.compile(r"control_(\d+)\.(?:png|jpe?g|webp)$", re.IGNORECASE)


def is_control_image(path: str):
    """返回 (ok, d)。ok=True 表示以 control_{d}.png/jpg/jpeg/webp 结尾；d 为整数。"""
    name = Path(path).name
    m = _pat_end.search(name)
    return m is not None


def _first_existing(base_dir: str, stem: str, exts=IMG_EXTS) -> str | None:
    for ext in exts:
        p = os.path.join(base_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def get_number_of_controls(control_dir: str, stem: str) -> int:
    for ext in IMG_EXTS:
        control_paths = glob.glob(os.path.join(control_dir, f"{stem}_control_[0-99]*{ext}"))
        print("control_paths", control_paths, "stem", stem)
        if len(control_paths) > 0:
            return len(control_paths)
    return 0


def _collect_extra_controls(control_dir: str, stem: str, num_controls: int) -> list[str]:
    """匹配 stem_control_1.*, stem_control_2.*, …（使用 _control_N 格式），排除 *_mask."""
    out = []
    for i in range(1, num_controls + 1):
        for ext in IMG_EXTS:
            control_path = os.path.join(control_dir, f"{stem}_control_{i}{ext}")
            if os.path.exists(control_path):
                out.append(control_path)
                continue
    return out


def _find_mask(images_dir: str, control_dir: str, stem: str) -> str | None:
    cands = [
        os.path.join(images_dir, f"{stem}_mask.png"),
        os.path.join(control_dir, f"{stem}_mask.png"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


class ImageDataset(Dataset):
    """
    图像-控制图像-文本 三元组数据集。

    返回的样本在不同缓存模式下包含不同键：
    - cached=False: 包含 'image' (C,H,W), 'control' (C,H,W), 'prompt' (str), 'file_paths' (dict)，以及可选 'mask'
    - cached=True 且旧式缓存存在: 另含 'pixel_latent', 'control_latent',
      'prompt_embeds', 'prompt_embeds_mask', 'empty_prompt_embed',
      'empty_prompt_embeds_mask' 等张量
    - cached=True 且仅有新式缓存: 返回缓存子目录名对应的键及其张量

    使用示例:
    ```python
    from qflux.data.dataset import ImageDataset, collate_fn
    from torch.utils.data import DataLoader

    data_config = {
        'dataset_path': '/path/to/dataset',
        'image_size': (512, 512),
        'use_cache': True,
        'cache_dir': '/path/to/cache',
    }
    dataset = ImageDataset(data_config)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    ```
    """

    def __init__(self, data_config: DatasetInitArgs):
        """
        初始化数据集。
        Args:
            data_config: 包含数据集路径和处理配置的字典，常见键：
                - dataset_path: 数据集根目录路径（str 或 [str]）
                - image_size: 目标尺寸 [img_w, img_h] 或 int，默认 None（不缩放）
                - cache_dir: 缓存目录路径，提供则可用缓存
                - use_cache: 是否使用缓存，默认 True
                - selected_control_indexes: 选择的控制图像索引，默认 None

        期望的数据集目录结构（每个根路径下）:
            dataset_root/
              training_images/
                xxx.png
                xxx.txt          # 与图像同名的 caption 文本
              control_images/
                xxx.png
                xxx_1.png
                xxx_2.png
                xxx_mask.png     # 可选，若存在则会返回 'mask'

        示例:
        ```python
        cfg = {
            'dataset_path': ['/data/ds1', '/data/ds2'],
            'use_cache': True,
            'cache_dir': '/data/cache'
        }
        ds = ImageDataset(cfg)
        print(len(ds))
        sample = ds[0]
        ```
        """
        self.data_config = data_config
        dataset_path = data_config.dataset_path

        # 支持多个数据集路径
        if isinstance(dataset_path, (list, tuple)):
            self.dataset_paths = list(dataset_path)
        else:
            self.dataset_paths = [dataset_path]

        self.hf_datasets: dict[str, Any] = {}
        self.cache_dir = data_config.cache_dir
        self.use_cache = data_config.use_cache
        self.selected_control_indexes = data_config.selected_control_indexes

        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_manager = EmbeddingCacheManager(self.cache_dir)
            print(f"缓存已启用，缓存目录: {self.cache_dir}")
            print(f"use_cache: {self.use_cache}")
        else:
            self.cache_manager = None  # type: ignore
            print("缓存未启用")
        if self.cache_manager:
            self.cache_exists = self.cache_manager.exist(self.cache_dir)  # type: ignore
        else:
            self.cache_exists = False
        # loading datsets
        self._load_all_datasets()
        self.load_processor()

    def load_processor(self):
        """load processor"""
        from qflux.utils.tools import instantiate_class

        class_path = self.data_config.processor.class_path
        init_args = self.data_config.processor.init_args
        self.preprocessor = instantiate_class(class_path, init_args)

    def _load_all_datasets(self):
        """Load datasets from local directories or Hugging Face repositories."""
        self.all_samples = []  # Unified list of all samples
        # [file_info]
        # file_info:
        # dict['dataset_type', 'local_index', 'repo_id','global_index' ]

        for dataset_path in self.dataset_paths:
            split = None
            if isinstance(dataset_path, dict):
                repo_id = dataset_path["repo_id"]
                split = dataset_path["split"]
                dataset_path = repo_id
            if is_huggingface_repo(dataset_path):
                samples = self._load_huggingface_dataset(dataset_path, split=split)
            elif isinstance(dataset_path, str) and dataset_path.endswith(".csv"):
                samples = self._load_csv_dataset(dataset_path)
            else:
                samples = self._load_local_dataset(dataset_path)
                # [image_file, control_files, prompt_file, mask_file, dataset_type, local_index, global_index]
            if samples is None:
                logging.warning(f"No samples loaded from {dataset_path}")
                continue
            self.all_samples += samples

    def __len__(self):
        """Return total number of samples across all datasets."""
        return len(self.all_samples)

    def _load_huggingface_dataset(self, repo_id: str, split: str | None = None):
        """
        Load dataset from Hugging Face using lazy loading approach.

        No enumeration happens here - we just store the dataset object
        and metadata for later access.
        """
        # Load the dataset (this is fast, just creates the dataset object)
        dataset = load_editing_dataset(repo_id, split=split)
        # Store HF dataset reference
        dataset_info = {
            "type": "huggingface",
            "repo_id": repo_id,
            "dataset": dataset,
            "length": len(dataset),  # This is typically cached by HF datasets
            "start_idx": len(self.all_samples),  # Track where this dataset starts
        }
        logging.info(f"Loaded Hugging Face dataset: {repo_id}, {dataset_info}")

        self.hf_datasets[repo_id] = dataset_info

        # Add entries for each sample without iterating
        # We'll create lightweight placeholders
        samples = []
        for idx in range(dataset_info["length"]):
            sample_ref = {
                "dataset_type": "huggingface",
                "repo_id": repo_id,
                "local_index": idx,
                "global_index": dataset_info["start_idx"] + idx,
            }
            samples.append(sample_ref)
        return samples

    def _load_local_dataset(self, dataset_path: str) -> list[dict]:
        """Load dataset from local directory."""
        # under the path, find
        # 1. find the images_dir and control_dir
        image_dir, control_dir = self._find_directories(dataset_path)
        # 2. find the image_files and control_files, prompt file or mask file [optional]
        if image_dir is None or control_dir is None:
            raise ValueError(f"Could not find image or control directory in {dataset_path}")
        samples = self._scan_image_files(image_dir, control_dir)
        return samples

    def _load_csv_dataset(self, dataset_path: str) -> list[dict]:
        """Load dataset from CSV file.
        samples.append(
                {
                    "image": image_path,
                    "control": controls,
                    "caption": prompt_file,
                    "mask_file": mask_file,
                    "dataset_type": "local",
                    "local_index": n,
                    "global_index": start_idx + n,
                }
            )
        """

        df = pd.read_csv(dataset_path)
        start_idx = len(self.all_samples)
        # calculate contrl numbers
        columns = df.columns
        columns = [x for x in columns if "path_control" in x]
        control_keys = sorted(columns)
        samples = []
        for idx, row in df.iterrows():
            controls = [row[x] for x in control_keys]
            prompt = row["prompt"]
            data = {
                "image": row["path_target"],
                "control": controls,
                "caption": prompt,
                "dataset_type": "local_csv",
                "local_index": idx,
                "global_index": start_idx + idx,
            }
            if "path_mask" in row:
                mask_file = row["path_mask"]
                data["mask_file"] = mask_file
            samples.append(data)
        return samples

    def _find_directories(self, dataset_path: str) -> tuple[str | None, str | None]:
        """
        查找所有数据集根路径下的图像目录与控制图像目录。
        优先匹配:
            images: ['training_images', 'images', 'target_images']
            control: ['control_images', 'control', 'condition_images']
        若都未找到则回退到默认 'training_images' 与 'control_images'。
        Returns:
            images_dirs, control_dirs
        """

        image_possible_names = ["training_images", "images", "target_images", "target", "targets"]
        control_possible_names = ["control_images", "control", "condition_images", "controls"]

        # 查找图像目录
        images_dir = None
        for name in image_possible_names:
            path = os.path.join(dataset_path, name)
            if os.path.exists(path):
                print(f"Found images directory: {path}")
                images_dir = path
                break

        # 查找控制图像目录
        control_dir = None
        for name in control_possible_names:
            path = os.path.join(dataset_path, name)
            if os.path.exists(path):
                print(f"Found control directory: {path}")
                control_dir = path
                break
        return images_dir, control_dir

    def _scan_image_files(self, images_dir: str, control_dir: str) -> list[dict]:
        """
        扫描所有数据集中的成对样本。
        假定 target 图片一定存在
        样本需满足：
            - images_dir/xyz.(jpg|jpeg|png|bmp)
            - control_dir/xyz.(jpg|jpeg|png|bmp)
            - control_dir/xyz_control_1.(jpg|jpeg|png|bmp) [additional control images]
            - control_dir/xyz_control_2.(jpg|jpeg|png|bmp) [additional control images]
        prompt file could be in either images_dir or control_dir
            - images_dir/xyz.txt 作为 prompt
            - control_dir/xyz.txt 作为 prompt
        (mask could be either in control_dir or images_dir)
            - control_dir/xxx_mask.png  [additional mask images]
            - images_dir/xyz_mask.png [additional mask images]

        Returns:
            List[dict]: 'image' 'control: list[str]' 'caption' 'dataset_index' 'mask_file'

        """
        # first looking for prompt text

        # first search target images
        target_images = glob.glob(os.path.join(images_dir, "*.*"))
        target_images = [img for img in target_images if img.endswith(IMG_EXTS)]
        # exclude mask images
        target_images = [img for img in target_images if not img.endswith("_mask.png")]
        target_images = [img for img in target_images if not is_control_image(img)]
        # exclude control images

        samples = []

        start_idx = len(self.all_samples)

        # 用 stem 归并（同 stem 出现两边时优先 images_dir 文本）
        stems = [os.path.splitext(os.path.basename(p))[0] for p in target_images]

        # filter these stems that dont have correspoding images
        stems = [s for s in stems if _first_existing(images_dir, s) is not None]

        logging.info("found %d prompts", len(stems))

        # 2) 对每个 stem 按需拼装样本
        n = 0
        from tqdm import tqdm

        num_controls = get_number_of_controls(control_dir, stems[0])

        print("num_controls", num_controls)
        logging.info("found %d controls", num_controls)
        logging.info(f"found with stem {control_dir}/{stems[0]}")
        for stem in tqdm(stems, desc="matching prompts"):
            # source image（必须在 images_dir）
            image_path = _first_existing(images_dir, stem)
            if image_path is None:
                logging.info(f"skipping {stem} because no image found")
                continue

            # control image（必须在 control_dir）
            main_control = _first_existing(control_dir, stem)

            # 额外控制图
            if main_control is not None:
                extras = _collect_extra_controls(control_dir, stem, num_controls)
                controls = [main_control] + list(extras)
            else:
                controls = []

            img_txt = os.path.join(images_dir, f"{stem}.txt")
            ctl_txt = os.path.join(control_dir, f"{stem}.txt")
            if os.path.exists(img_txt):
                prompt_file = img_txt
            elif os.path.exists(ctl_txt):
                prompt_file = ctl_txt
            else:
                continue

            # mask（可选）
            mask_file = _find_mask(images_dir, control_dir, stem)

            samples.append(
                {
                    "image": image_path,
                    "control": controls,
                    "caption": prompt_file,
                    "mask_file": mask_file,
                    "dataset_type": "local",
                    "local_index": n,
                    "global_index": start_idx + n,
                }
            )
            n += 1
        logging.info(f" samples[0]: {samples[0]}")
        return samples

    def __repr__(self) -> str:
        msg = f"""ImageDataset(
            dataset_paths={self.dataset_paths},
            cache_dir={self.cache_dir},
            use_cache={self.use_cache},
        """
        return msg

    def get_file_hashes(self, data: dict[str, Any]) -> dict[str, Any]:
        file_hashes = {}
        main_hash = ""
        if "image" in data:
            file_hashes["image_hash"] = self.cache_manager.get_hash(data["image"])
            main_hash += file_hashes["image_hash"]
        if "control" in data:
            file_hashes["control_hash"] = self.cache_manager.get_hash(data["control"])
            main_hash += file_hashes["control_hash"]
        if "prompt" in data:
            file_hashes["prompt_hash"] = hash_string_md5(data["prompt"])
            main_hash += file_hashes["prompt_hash"]
        if "prompt" in data:
            file_hashes["empty_prompt_hash"] = hash_string_md5("empty")
        if "control" in data and "prompt" in data:
            file_hashes["control_prompt_hash"] = self.cache_manager.get_hash(data["control"], data["prompt"])
        if "control" in data and "prompt" in data:
            file_hashes["control_empty_prompt_hash"] = self.cache_manager.get_hash(data["control"], "empty")
        if "controls" in data:
            controls_sum_hash = file_hashes["control_hash"]
            for i in range(len(data["controls"])):
                file_hashes[f"control_{i + 1}_hash"] = self.cache_manager.get_hash(data["controls"][i])
                controls_sum_hash += file_hashes[f"control_{i + 1}_hash"]
            file_hashes["controls_sum_hash"] = controls_sum_hash
        elif "control" in data:
            file_hashes["controls_sum_hash"] = file_hashes["control_hash"]
        file_hashes["main_hash"] = main_hash
        return file_hashes

    def data_key_exist(self, data: dict[str, Any], key: str) -> bool:
        if key in data and data[key] is not None:
            return True
        return False

    def load_data(self, idx: int) -> dict[str, Any]:
        if idx >= self.__len__():
            raise IndexError(f"Index {idx} out of range for dataset of size {self.__len__()}")
        sample = self.all_samples[idx]
        data = {}
        if sample["dataset_type"] == "huggingface":
            local_index = sample["local_index"]
            repo_id = sample["repo_id"]
            data_item = self.hf_datasets[repo_id]["dataset"][local_index]
            if data_item["target_image"] is not None:
                image = data_item["target_image"].convert("RGB")
                data["image"] = image
            control = data_item["control_images"]
            if control is not None:
                data["control"] = control[0].convert("RGB")
                if len(control) > 1:
                    data["controls"] = control[1:]
                    data["controls"] = [img.convert("RGB") for img in data["controls"]]
                    if self.selected_control_indexes is not None:
                        data["controls"] = [data["controls"][i - 1] for i in self.selected_control_indexes]

            prompt = data_item["prompt"]
            data["prompt"] = prompt
            if data_item["control_mask"] is not None:
                data["mask"] = np.array(data_item["control_mask"].convert("L"))

            if self.cache_manager is not None:
                file_hashes = self.get_file_hashes(data)
                data["file_hashes"] = file_hashes
        else:  # loaded locally
            data_item = self.all_samples[idx]
            #  {
            #         "image": image_path,
            #         "control": controls,
            #         "caption": prompt_file,
            #         "mask_file": mask_file,
            #         "dataset_type": "local",
            #         "local_index": n,
            #         "global_index": start_idx + n,
            #     }
            # ) If, not exist return None
            # 读取提示文本
            if self.data_key_exist(data_item, "image"):
                data["image"] = data_item["image"]
            if self.data_key_exist(data_item, "control"):
                data["control"] = data_item["control"][0]
                if len(data_item["control"]) > 1:
                    data["controls"] = data_item["control"][1:]
                    if self.selected_control_indexes is not None:
                        data["controls"] = [data["controls"][i - 1] for i in self.selected_control_indexes]
            if self.data_key_exist(data_item, "mask_file"):
                data["mask"] = cv2.imread(data_item["mask_file"], 0)
            if self.data_key_exist(data_item, "caption") and data_item["dataset_type"] == "local":
                with open(data_item["caption"], encoding="utf-8") as f:
                    prompt = f.read().strip()
                data["prompt"] = prompt
            else:
                prompt = data_item["caption"]
                data["prompt"] = prompt
            if self.cache_manager is not None:
                file_hashes = self.get_file_hashes(data)
            data["file_hashes"] = file_hashes
        return data

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        获取索引样本。
        Args:
            idx: 样本索引
        Returns:
            dict: 根据缓存状态返回原始图像+文本或缓存嵌入的字典

        示例（非缓存模式）:
        ```python
        item = dataset[0]
        image = item['image']     # (C,H,W) numpy
        control = item['control'] # (C,H,W) numpy
        prompt = item['prompt']   # str
        img_shapes = item['img_shapes']  # [(C,H,W), ...] original image shapes
        ```
        """
        data = self.load_data(idx)

        data = self.preprocessor.preprocess(data)
        data["cached"] = False

        # Generate img_shapes with original dimensions (before VAE encoding)
        img_shapes = self._generate_img_shapes(data)
        data["img_shapes"] = img_shapes

        if self.use_cache and self.cache_exists:
            if random.random() < self.data_config.caption_dropout_rate:
                replace_empty_embeddings = True
            else:
                replace_empty_embeddings = False
            prompt_empty_drop_keys = self.data_config.prompt_empty_drop_keys
            data = self.cache_manager.load_cache(data, replace_empty_embeddings, prompt_empty_drop_keys)
            data["cached"] = True
        if "controls" in data:
            n_controls = len(data["controls"])
            for i in range(n_controls):
                data[f"control_{i + 1}"] = data["controls"][i]
            del data["controls"]
            data["n_controls"] = n_controls
        else:
            data["n_controls"] = 0
        return data

    def _generate_img_shapes(self, data: dict) -> list[tuple]:
        """Generate img_shapes list with original image dimensions

        This method collects the original dimensions (C, H, W) of all images
        in the sample, including target image, control, and additional controls.

        The dimensions are in original pixel space, before any VAE encoding
        or packing. The trainer will handle the conversion to latent space.

        Args:
            data: Dictionary containing preprocessed images

        Returns:
            List of tuples: [(C, H, W), ...] where:
                - First tuple: target image shape
                - Second tuple: control image shape
                - Remaining tuples: additional control images (control_1, control_2, ...)

        Example:
            >>> data = {
            ...     'image': np.array with shape (3, 512, 512),
            ...     'control': np.array with shape (3, 512, 512),
            ...     'controls': [np.array with shape (3, 640, 640)]
            ... }
            >>> shapes = self._generate_img_shapes(data)
            >>> shapes  # [(3, 512, 512), (3, 512, 512), (3, 640, 640)]
        """
        img_shapes = []

        # Target image shape
        if "image" in data:
            image = data["image"]
            if isinstance(image, torch.Tensor):
                C, H, W = image.shape
            elif isinstance(image, np.ndarray):
                if image.ndim == 3:
                    C, H, W = image.shape
                else:
                    raise ValueError(f"Expected 3D image array, got shape {image.shape}")
            else:
                raise TypeError(f"Unsupported image type: {type(image)}")
            img_shapes.append((C, H, W))

        # Control image shape
        if "control" in data:
            control = data["control"]
            if isinstance(control, torch.Tensor):
                C, H, W = control.shape
            elif isinstance(control, np.ndarray):
                if control.ndim == 3:
                    C, H, W = control.shape
                else:
                    raise ValueError(f"Expected 3D control array, got shape {control.shape}")
            else:
                raise TypeError(f"Unsupported control type: {type(control)}")
            img_shapes.append((C, H, W))

        # Additional control images
        if "controls" in data:
            for i, control_i in enumerate(data["controls"]):
                if isinstance(control_i, torch.Tensor):
                    C, H, W = control_i.shape
                elif isinstance(control_i, np.ndarray):
                    if control_i.ndim == 3:
                        C, H, W = control_i.shape
                    else:
                        raise ValueError(f"Expected 3D control_{i} array, got shape {control_i.shape}")
                else:
                    raise TypeError(f"Unsupported control_{i} type: {type(control_i)}")
                img_shapes.append((C, H, W))

        return img_shapes


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    自定义 collate 函数，自动对张量右端填充并对嵌套字典递归聚合。
    Args:
        batch: 由 __getitem__ 返回的样本字典组成的列表
    Returns:
        Dict[str, Any]: 已聚合并对齐后的批次字典

    示例:
    ```python
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    ```
    """
    # [{a:1,b:2, c:{d:1,g:2}},{a:3,b:4, c:{e:3,g:4}}] -> {a: [1,3], b: [2,4], c:{d: [1,3], e: [2,4], g: [2,4]}}
    # 分离cached和non-cached的数据
    keys = list(batch[0].keys())
    # flattten
    batch_dict = {key: [item[key] for item in batch] for key in keys}

    # Special handling for mask: convert each sample to latent space BEFORE padding
    # because each sample may have different dimensions
    edit_mask_list = None
    if "mask" in batch_dict:
        mask_list = batch_dict["mask"]  # List of masks, each may have different shape
        edit_mask_list = []
        for mask in mask_list:
            # Convert to tensor if needed
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            # Add batch dimension if needed: [H, W] -> [1, H, W]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # Convert this individual mask to latent space
            edit_mask = map_mask_to_latent(mask)  # [1, seq_len]
            edit_mask_list.append(edit_mask.squeeze(0))  # [seq_len]

    # if torch tensor, padding to maximal length
    for key in batch_dict:
        if isinstance(batch_dict[key][0], np.ndarray):
            batch_dict[key] = [torch.from_numpy(item) for item in batch_dict[key]]
        if isinstance(batch_dict[key][0], torch.Tensor):
            batch_dict[key] = pad_to_max_shape(batch_dict[key])
        elif isinstance(batch_dict[key][0], dict):
            batch_list = batch_dict[key]
            batch_dict[key] = collate_fn(batch_list)  # type: ignore[assignment]
            # [ {d:1,g:2}, {e:3,g:4}] -> {d: [1,3], g: [2,4]}
            # [{a:1,b:2, c:{d:1,g:2}},{a:3,b:4, c:{d:3,g:4}}] -> {a: [1,3], b: [2,4], c:{d: [1,3], g: [2,4]}}

    # Pad edit_mask_list and add to batch_dict
    if edit_mask_list is not None:
        # Pad edit masks to same length
        batch_dict["edit_mask"] = pad_to_max_shape(edit_mask_list)  # [B, max_seq_len]

    return batch_dict


def loader(
    class_path: str,
    init_args: dict,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    动态加载数据集类并创建 DataLoader。
    Args:
        class_path: 类的完整路径，如 'qlufx.data.dataset.ImageDataset'
        init_args: 用于初始化该类的参数字典（传给其 __init__）
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱
    Returns:
        torch.utils.data.DataLoader: 附加属性 'cache_manager'

    示例:
    ```python
    from qflux.data.dataset import loader
    dl = loader(
        class_path='qflux.data.dataset.ImageDataset',
        init_args={'dataset_path': '/data/ds', 'use_cache': False},
        batch_size=2,
        num_workers=2,
        shuffle=True,
    )
    for batch in dl:
        pass
    ```
    """
    # 解析类路径
    module_path, class_name = class_path.rsplit(".", 1)

    # 动态导入模块
    module = importlib.import_module(module_path)

    # 获取类对象
    dataset_class = getattr(module, class_name)

    # 使用init_args实例化类
    dataset = dataset_class(init_args)
    cache_manager = dataset.cache_manager
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    dataloader.cache_manager = cache_manager
    return dataloader


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level="INFO",
    )
    from qflux.data.config import load_config_from_yaml

    config_file = "configs/face_seg_flux_kontext_fp16_huggingface_dataset.yaml"
    config_file = "tests/test_configs/test_example_fluxkontext_fp16_character_composition.yaml"
    config = load_config_from_yaml(config_file)
    data_config = config.data
    dataloader = loader(
        data_config.class_path,
        data_config.init_args,
        data_config.batch_size,
        data_config.num_workers,
        data_config.shuffle,
    )
    for batch in dataloader:
        print("batch keys", batch.keys())

        for k, v in batch.items():
            print(k)
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, v)
        break
    print(batch["cached"])
    print(batch["file_hashes"])
    print(dataloader.cache_manager)
    # print('batch type', type(batch['image'][0]), batch['image'])
    print(batch["prompt"])
    print(batch["control_1"].shape)
    print(batch["img_shapes"])

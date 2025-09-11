import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional

import glob
import numpy as np
import cv2
import importlib
import logging
import hashlib
# 删除重复的 typing 导入，已在上方统一导入 Optional, Dict, List, Any
from src.data.cache_manager import EmbeddingCacheManager, check_cache_exists
from src.utils.hugginface import load_editing_dataset, is_huggingface_repo


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _first_existing(base_dir: str, stem: str, exts=IMG_EXTS) -> Optional[str]:
    for ext in exts:
        p = os.path.join(base_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _collect_extra_controls(control_dir: str, stem: str) -> List[str]:
    """匹配 stem_1.*, stem_2.*, …（只收集数值后缀），排除 *_mask."""
    out = []
    for ext in IMG_EXTS:
        for p in glob.glob(os.path.join(control_dir, f"{stem}_*{ext}")):
            bn = os.path.basename(p)
            name_no_ext = os.path.splitext(bn)[0]
            if name_no_ext.endswith("_mask"):
                continue
            suf = name_no_ext[len(stem) + 1:]  # 去掉 'stem_'
            if suf.isdigit():
                out.append(p)
    # 依据数字后缀排序，确保确定性
    out.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1]))
    return out


def _find_mask(images_dir: str, control_dir: str, stem: str) -> Optional[str]:
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
      'prompt_embed', 'prompt_embeds_mask', 'empty_prompt_embed',
      'empty_prompt_embeds_mask' 等张量
    - cached=True 且仅有新式缓存: 返回缓存子目录名对应的键及其张量

    使用示例:
    ```python
    from src.data.dataset import ImageDataset, collate_fn
    from torch.utils.data import DataLoader

    data_config = {
        'dataset_path': '/path/to/dataset',
        'image_size': (512, 512),
        'use_cache': True,
        'cache_dir': '/path/to/cache',
        'random_crop': False,
        'center_crop': True,
        'center_crop_ratio': 0.9,
    }
    dataset = ImageDataset(data_config)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    ```
    """
    def __init__(self, data_config):
        """
        初始化数据集。
        Args:
            data_config: 包含数据集路径和处理配置的字典，常见键：
                - dataset_path: 数据集根目录路径（str 或 [str]）
                - image_size: 目标尺寸 [img_w, img_h] 或 int，默认 None（不缩放）
                - cache_dir: 缓存目录路径，提供则可用缓存
                - use_cache: 是否使用缓存，默认 True
                - random_crop: 是否随机正方形裁剪，默认 False（需配合 crop_size）
                - crop_size: 随机裁剪后缩放到的边长，默认 None
                - crop_scale: 随机裁剪的缩放范围 [min, max]，默认 [0.8, 1.0]
                - center_crop: 是否启用按比例的中心裁剪，默认 False
                - center_crop_ratio: 中心裁剪相对最大可裁区域的缩放比例，默认 1.0

        期望的数据集目录结构（每个根路径下）:
            dataset_root/
              training_images/
                xxx.png
                xxx.txt          # 与图像同名的 caption 文本
              control_images/
                xxx.png
                xxx_mask.png     # 可选，若存在则会返回 'mask'

        示例:
        ```python
        cfg = {
            'dataset_path': ['/data/ds1', '/data/ds2'],
            'image_size': (512, 512),
            'use_cache': True,
            'cache_dir': '/data/cache'
        }
        ds = ImageDataset(cfg)
        print(len(ds))
        sample = ds[0]
        ```
        """
        self.data_config = data_config
        dataset_path = data_config.get('dataset_path', './dataset')

        # 支持多个数据集路径
        if isinstance(dataset_path, (list, tuple)):
            self.dataset_paths = list(dataset_path)
        else:
            self.dataset_paths = [dataset_path]

        self.hf_datasets = {}

        self.image_size = data_config.get('image_size', None)
        self.cache_dir = data_config.get('cache_dir', None)
        self.use_cache = data_config.get('use_cache', True)

        # 随机裁剪配置
        self.random_crop = data_config.get('random_crop', False)
        self.crop_size = data_config.get('crop_size', None)
        self.crop_scale = data_config.get('crop_scale', [0.8, 1.0])

        # center crop 配置
        self.center_crop = data_config.get('center_crop', False)
        self.center_crop_ratio = data_config.get('center_crop_ratio', 1.0)

        # 初始化缓存管理器
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_manager = EmbeddingCacheManager(self.cache_dir)
            print(f"缓存已启用，缓存目录: {self.cache_dir}")
        else:
            self.cache_manager = None
            print("缓存未启用")

        self.cache_exists = check_cache_exists(self.cache_dir) if self.cache_dir else False
        if self.cache_exists:
            cache_subfolders = glob.glob(self.cache_dir+"/*")
            self.cache_keys = [os.path.basename(cache_subfolder) for cache_subfolder in cache_subfolders]
        else:
            self.cache_keys = []
        # loading datsets
        self._load_all_datasets()

        # 图像预处理变换 - 将[img_w, img_h] 转换为transforms.Resize期望的 (height, width) 格式
        if isinstance(self.image_size, (tuple, list)):
            self.resize_size = (self.image_size[0], self.image_size[1])  # (width, height)
        else:
            self.resize_size = self.image_size

    def _load_all_datasets(self):
        """Load datasets from local directories or Hugging Face repositories."""
        self.all_samples = []  # Unified list of all samples
        # [file_info]
        # file_info:
        # dict['dataset_type', 'local_index', 'repo_id','global_index' ]

        for dataset_path in self.dataset_paths:
            split = None
            if isinstance(dataset_path, dict):
                repo_id = dataset_path['repo_id']
                split = dataset_path['split']
                dataset_path = repo_id
            if is_huggingface_repo(dataset_path):
                samples = self._load_huggingface_dataset(dataset_path, split=split)
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

    def _load_huggingface_dataset(self, repo_id: str, split: Optional[str] = None):
        """
        Load dataset from Hugging Face using lazy loading approach.

        No enumeration happens here - we just store the dataset object
        and metadata for later access.
        """
        # Load the dataset (this is fast, just creates the dataset object)
        dataset = load_editing_dataset(repo_id, split=self.data_config.get('split', split))
        # Store HF dataset reference
        dataset_info = {
            'type': 'huggingface',
            'repo_id': repo_id,
            'dataset': dataset,
            'length': len(dataset),  # This is typically cached by HF datasets
            'start_idx': len(self.all_samples)  # Track where this dataset starts
        }
        logging.info(f"Loaded Hugging Face dataset: {repo_id}, {dataset_info}")

        self.hf_datasets[repo_id] = dataset_info

        # Add entries for each sample without iterating
        # We'll create lightweight placeholders
        samples = []
        for idx in range(dataset_info['length']):
            sample_ref = {
                'dataset_type': 'huggingface',
                'repo_id': repo_id,
                'local_index': idx,
                'global_index': dataset_info['start_idx'] + idx
            }
            samples.append(sample_ref)
        return samples

    def _load_local_dataset(self, dataset_path: str) -> List[dict]:
        """Load dataset from local directory."""
        # under the path, find
        # 1. find the images_dir and control_dir
        image_dir, control_dir = self._find_directories(dataset_path)
        # 2. find the image_files and control_files, prompt file or mask file [optional]
        samples = self._scan_image_files(image_dir, control_dir)
        return samples

    def _find_directories(self, dataset_path: str) -> List[str]:
        """
        查找所有数据集根路径下的图像目录与控制图像目录。
        优先匹配:
            images: ['training_images', 'images', 'target_images']
            control: ['control_images', 'control', 'condition_images']
        若都未找到则回退到默认 'training_images' 与 'control_images'。
        Returns:
            images_dirs, control_dirs
        """

        image_possible_names = ['training_images', 'images', 'target_images']
        control_possible_names = ['control_images', 'control', 'condition_images']

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

    def _scan_image_files(self, images_dir, control_dir):
        """
        扫描所有数据集中的成对样本。
        样本需满足：
            - images_dir/xyz.(jpg|jpeg|png|bmp)
            - control_dir/xyz.(jpg|jpeg|png|bmp)
            - control_dir/xyz_1.(jpg|jpeg|png|bmp) [additional control images]
            - control_dir/xyz_2.(jpg|jpeg|png|bmp) [additional control images]
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
        prompt_files = glob.glob(os.path.join(images_dir, '*.txt'))
        prompt_files += glob.glob(os.path.join(control_dir, '*.txt'))
        samples = []

        start_idx = len(self.all_samples)

        # 用 stem 归并（同 stem 出现两边时优先 images_dir 文本）
        stem_to_prompt = {}
        for p in prompt_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            # 如果已存在 images_dir 的 prompt，就不被 control_dir 覆盖
            if stem in stem_to_prompt:
                # 维持优先级：images_dir 优先；否则若当前是 images_dir 则覆盖
                if os.path.dirname(stem_to_prompt[stem]) != images_dir and os.path.dirname(p) == images_dir:
                    stem_to_prompt[stem] = p
            else:
                stem_to_prompt[stem] = p

        # 2) 对每个 stem 按需拼装样本
        n = 0
        for stem, ptxt in sorted(stem_to_prompt.items()):
            # source image（必须在 images_dir）
            image_path = _first_existing(images_dir, stem)
            if not image_path:
                continue  # 无图跳过

            # control image（必须在 control_dir）
            main_control = _first_existing(control_dir, stem)
            if not main_control:
                continue  # 无控制图跳过

            # 额外控制图
            extras = _collect_extra_controls(control_dir, stem)
            controls = [main_control] + extras

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
        return samples

    def __repr__(self) -> str:
        msg = f"""ImageDataset(
            dataset_paths={self.dataset_paths},
            image_size={self.image_size},
            cache_dir={self.cache_dir},
            use_cache={self.use_cache},
            random_crop={self.random_crop},
            crop_size={self.crop_size},
            crop_scale={self.crop_scale},
            center_crop={self.center_crop},
            center_crop_ratio={self.center_crop_ratio})
        """
        return msg

    def get_random_crop_bbox(self, h, w):
        """
        生成随机正方形裁剪的边界框（需启用 random_crop）。
        Args:
            h: 图像高度
            w: 图像宽度
        Returns:
            tuple: (x, y, crop_w, crop_h) 裁剪区域；未启用时返回 None

        示例:
        ```python
        dataset.random_crop = True
        dataset.crop_size = 256
        bbox = dataset.get_random_crop_bbox(h=768, w=1024)
        ```
        """
        if not self.random_crop:
            return None

        # 计算随机缩放因子
        scale = random.uniform(self.crop_scale[0], self.crop_scale[1])

        # 选择较小的边作为基准，确保裁剪出正方形
        min_side = min(h, w)
        crop_size_final = int(min_side * scale)

        # 随机选择裁剪位置
        if w > crop_size_final:
            x = random.randint(0, w - crop_size_final)
        else:
            x = 0

        if h > crop_size_final:
            y = random.randint(0, h - crop_size_final)
        else:
            y = 0

        return (x, y, crop_size_final, crop_size_final)

    def get_center_crop_bbox(self, h, w):
        """
        生成中心裁剪的边界框，按目标宽高比取最大区域（需启用 center_crop）。
        Args:
            h: 图像高度
            w: 图像宽度
        Returns:
            tuple: (x, y, crop_w, crop_h) 裁剪区域；未启用时返回 None

        示例:
        ```python
        dataset.center_crop = True
        dataset.resize_size = (512, 512)
        bbox = dataset.get_center_crop_bbox(h=720, w=1280)
        ```
        """
        if not self.center_crop:
            return None

        # 确定目标宽高比
        if self.resize_size is not None:
            if isinstance(self.resize_size, (tuple, list)):
                target_w, target_h = self.resize_size
            else:
                target_w = target_h = self.resize_size
        else:
            # 如果没有指定 resize_size，默认使用正方形
            target_w = target_h = min(h, w)

        # 计算目标宽高比
        target_ratio = target_w / target_h

        # 根据图像尺寸和目标比例，计算能容纳的最大裁剪区域
        # 按照目标比例，计算两种可能的裁剪尺寸
        crop_w_by_height = int(h * target_ratio)  # 按高度限制计算宽度
        crop_h_by_width = int(w / target_ratio)   # 按宽度限制计算高度

        if crop_w_by_height <= w:
            # 高度是限制因素，使用完整高度
            crop_h = h
            crop_w = crop_w_by_height
        else:
            # 宽度是限制因素，使用完整宽度
            crop_w = w
            crop_h = crop_h_by_width

        # 应用 center_crop_ratio 进行缩放
        crop_w = int(crop_w * self.center_crop_ratio)
        crop_h = int(crop_h * self.center_crop_ratio)

        # 计算中心裁剪位置
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2

        return (x, y, crop_w, crop_h)

    def apply_crop_and_resize(self, img, bbox=None):
        """
        应用裁剪并按配置调整大小。
        Args:
            img: 输入图像，(H, W, C) 或 (H, W)
            bbox: 裁剪边界框 (x, y, w, h)，None 则不裁剪
        Returns:
            numpy.ndarray: 处理后的图像

        示例:
        ```python
        img = cv2.imread('/path/a.png')[:, :, ::-1]
        out = dataset.apply_crop_and_resize(img, bbox=(10, 20, 256, 256))
        ```
        """
        if bbox is not None:
            x, y, w, h = bbox
            img = img[y:y+h, x:x+w]

        # 调整到目标尺寸
        if self.random_crop and self.crop_size is not None:
            img = cv2.resize(img, (self.crop_size, self.crop_size))
        elif self.center_crop and self.resize_size is not None:
            # center crop 后需要 resize 到目标尺寸
            img = cv2.resize(img, self.resize_size)
        elif self.resize_size is not None:
            img = cv2.resize(img, self.resize_size)

        return img

    def preprocess(self, image_path, crop_bbox=None):
        """
        读取并预处理单个图像，返回 CHW 格式。
        Args:
            image_path: 图像路径或已加载的 numpy 数组
            crop_bbox: 可选裁剪边界框 (x, y, w, h)
        Returns:
            numpy.ndarray: (C, H, W)

        示例:
        ```python
        arr = dataset.preprocess('/path/a.png', crop_bbox=None)  # (3,H,W)
        ```
        """
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img = img[:, :, ::-1]  # BGR to RGB
        else:
            img = image_path

        # 应用裁剪和调整大小
        img = self.apply_crop_and_resize(img, crop_bbox)

        # 转换为CHW格式
        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1)  # ，C,H,W
        return img

    def __getitem__(self, idx):
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
        ```
        """
        if idx >= self.__len__():
            raise IndexError(f"Index {idx} out of range for dataset of size {self.__len__()}")
        sample = self.all_samples[idx]
        if sample['dataset_type'] == 'huggingface':
            local_index = sample['local_index']
            repo_id = sample['repo_id']
            data_item = self.hf_datasets[repo_id]['dataset'][local_index]
            image = np.array(data_item['target_image'].convert('RGB'))
            control = data_item['control_images']
            if control is not None:
                control = np.array(control[0].convert('RGB'))
            else:
                control = None
            prompt = data_item['prompt']
            if data_item['control_mask'] is not None:
                mask_numpy = np.array(data_item['control_mask'].convert('L'))
            else:
                mask_numpy = None

            if self.use_cache:
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
                repo_id_str = repo_id.replace('/', '_')
                image_hash = f"{repo_id_str}_{local_index}_{prompt_hash}"
                control_hash = f"{repo_id_str}_{local_index}_{prompt_hash}"
                prompt_hash = f"{repo_id_str}_{local_index}_{prompt_hash}"
                empty_prompt_hash = f"{repo_id_str}_{local_index}_{prompt_hash}"

        else:
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

            with open(data_item['caption'], 'r', encoding='utf-8') as f:
                prompt = f.read().strip()

            if self.use_cache:
                image_hash = self.cache_manager.get_file_hash_for_image(data_item['image'])
                control_hash = self.cache_manager.get_file_hash_for_image(data_item['control'][0])
                prompt_hash = self.cache_manager.get_file_hash_for_prompt(data_item['image'], prompt)
                empty_prompt_hash = self.cache_manager.get_file_hash_for_prompt(data_item['image'], "empty")

            image = data_item['image']
            control = data_item['control'][0]
            mask_file = data_item['mask_file']
            mask_numpy = None
            if mask_file is not None and os.path.exists(mask_file):
                mask_numpy = cv2.imread(mask_file, 0)

        # 如果启用裁剪，生成共同的裁剪边界框
        crop_bbox = None
        if (self.random_crop and self.crop_size is not None) or self.center_crop:
            # 读取第一个图像来获取尺寸（假设image和control尺寸相同）
            if isinstance(image, str):
                temp_img = cv2.imread(image)
            else:
                temp_img = image
            if temp_img is not None:
                h, w = temp_img.shape[:2]
                if self.center_crop:
                    crop_bbox = self.get_center_crop_bbox(h, w)
                elif self.random_crop and self.crop_size is not None:
                    crop_bbox = self.get_random_crop_bbox(h, w)

        image_numpy = self.preprocess(image, crop_bbox)

        # 加载控制图像（使用相同的裁剪边界框）
        control_numpy = self.preprocess(control, crop_bbox)

        has_mask = False
        if mask_numpy is not None:
            mask_numpy = self.preprocess(mask_numpy, crop_bbox)
            has_mask = True
        print('cache_exists', self.cache_exists)
        print('use_cache', self.use_cache)
        if self.cache_exists and self.use_cache:
            # 如果启用缓存，尝试加载缓存的嵌入
            # 检查缓存是否存在
            # TODO: original implementation
            cache_file = os.path.join(self.cache_dir, 'prompt_embeds_mask', f"{image_hash}.pt")
            old_style = os.path.exists(cache_file)

            if old_style:
                cached_data = {}
                for cache_type, file_hash in [
                    ('pixel_latent', image_hash),
                    ('control_latent', control_hash),
                    ('prompt_embed', prompt_hash),
                    ('prompt_embeds_mask', prompt_hash),
                    ('empty_prompt_embed', empty_prompt_hash),
                    ('empty_prompt_embeds_mask', empty_prompt_hash),
                ]:
                    cached_embedding = self.cache_manager.load_cache(cache_type, file_hash)
                    cached_data[cache_type] = cached_embedding
                if random.random() < self.data_config.get('cache_drop_rate', 0.0):
                    for key in self.data_config.get('prompt_empty_drop_keys', []):
                        empty_key = f'empty_{key}'
                        cached_data[key] = cached_data[empty_key]

                # 如果所有缓存都存在，返回缓存数据
                data = {
                    'cached': True,
                    'image': image_numpy,
                    'control': control_numpy,
                    'pixel_latent': cached_data['pixel_latent'],
                    'control_latent': cached_data['control_latent'],
                    'prompt_embed': cached_data['prompt_embed'],
                    'prompt_embeds_mask': cached_data['prompt_embeds_mask'],
                    'prompt': prompt,
                    'file_hashes': {
                        'image_hash': image_hash,
                        'control_hash': control_hash,
                        'prompt_hash': prompt_hash,
                        'empty_prompt_hash': empty_prompt_hash
                    }
                }
                if has_mask:
                    data['mask'] = (mask_numpy > 125).astype(np.float32)  # convet to 0 or 1
                self.check_none_output(data)
                return data
            else:
                data = {}
                for cache_type in self.cache_keys:
                    cache_path = os.path.join(self.cache_dir, cache_type, f"{prompt_hash}.pt")
                    loaded_data = torch.load(cache_path, map_location='cpu', weights_only=False)
                    # 确保加载的数据没有梯度信息
                    loaded_data = loaded_data.detach()
                    data[cache_type] = loaded_data
                data.update(
                    {
                        'cached': True,
                        'image': image_numpy,
                        'control': control_numpy,
                        'prompt': prompt,
                    }
                )
                if has_mask:
                    data['mask'] = (mask_numpy > 125).astype(np.float32)  # convet to 0 or 1
                self.check_none_output(data)
                return data
        else:

            # 如果没有缓存或缓存不完整，返回原始数据

            data = {
                'cached': False,
                'image': image_numpy,
                'control': control_numpy,
                'prompt': prompt,
            }
            if self.use_cache:
                data['file_hashes'] = {
                    'image_hash': image_hash,
                    'control_hash': control_hash,
                    'prompt_hash': prompt_hash,
                    'empty_prompt_hash': empty_prompt_hash
                }
            if has_mask:
                data['mask'] = (mask_numpy > 125).astype(np.float32)  # convet to 0 or 1
            self.check_none_output(data)
            return data

    def check_none_output(self, data: dict):
        """
        断言数据字典（含嵌套）中的值均非 None。
        Args:
            data: 返回的数据字典
        Raises:
            AssertionError: 若存在 None 值
        """
        for k, v in data.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    assert vv is not None, f"value is None for key {kk} in {k}"
            else:
                assert v is not None, f"value is None for key {k}"

    def get_cache_stats(self) -> Optional[Dict[str, int]]:
        """
        获取缓存统计信息。
        Returns:
            Optional[Dict[str, int]]: 缓存统计字典；未启用缓存返回 None

        示例:
        ```python
        stats = dataset.get_cache_stats()
        if stats:
            print(stats)
        ```
        """
        if self.cache_manager:
            return self.cache_manager.get_cache_stats()
        return None


def pad_to_max_shape(tensors, padding_value=0):
    """
    将一组不同形状但维度数一致的张量沿各维右侧填充到最大形状后堆叠。
    Args:
        tensors: List[Tensor]，形状可能不同
        padding_value: 填充值，默认 0
    Returns:
        Tensor: 形状为 (N, ...) 的批次张量

    示例:
    ```python
    x = [torch.ones(2,3), torch.zeros(4,1)]
    y = pad_to_max_shape(x, padding_value=0)  # -> (2,4,3)
    ```
    """
    # Find maximum shape
    max_shape = [max(sizes) for sizes in zip(*[t.shape for t in tensors])]
    padded = []
    for t in tensors:
        pad_sizes = []
        for i in range(len(max_shape) - 1, -1, -1):
            diff = max_shape[i] - t.shape[i]
            pad_sizes.extend([0, diff])  # (left, right) per dimension
        padded_tensor = F.pad(t, pad_sizes, value=padding_value)
        padded.append(padded_tensor)

    return torch.stack(padded, dim=0)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    # if torch tensor, padding to maximal length
    for key in batch_dict:
        if isinstance(batch_dict[key][0], np.ndarray):
            batch_dict[key] = [torch.from_numpy(item) for item in batch_dict[key]]
        if isinstance(batch_dict[key][0], torch.Tensor):
            batch_dict[key] = pad_to_max_shape(batch_dict[key])
        elif isinstance(batch_dict[key][0], dict):
            batch_list = batch_dict[key]
            batch_list = collate_fn(batch_list)
            batch_dict[key] = batch_list
            # [ {d:1,g:2}, {e:3,g:4}] -> {d: [1,3], g: [2,4]}
            # [{a:1,b:2, c:{d:1,g:2}},{a:3,b:4, c:{d:3,g:4}}] -> {a: [1,3], b: [2,4], c:{d: [1,3], g: [2,4]}}
    return batch_dict


def loader(
        class_path: str,
        init_args: dict,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True) -> DataLoader:
    """
    动态加载数据集类并创建 DataLoader。
    Args:
        class_path: 类的完整路径，如 'src.data.dataset.ImageDataset'
        init_args: 用于初始化该类的参数字典（传给其 __init__）
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱
    Returns:
        torch.utils.data.DataLoader: 附加属性 'cache_manager'

    示例:
    ```python
    from src.data.dataset import loader
    dl = loader(
        class_path='src.data.dataset.ImageDataset',
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
    module_path, class_name = class_path.rsplit('.', 1)

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
        collate_fn=collate_fn
    )
    setattr(dataloader, 'cache_manager', cache_manager)
    return dataloader


if __name__ == "__main__":
    from src.data.config import load_config_from_yaml
    config_file = 'configs/face_seg_flux_kontext_fp16_huggingface_dataset.yaml'
    config = load_config_from_yaml(config_file)
    data_config = config.data
    dataloader = loader(
        data_config.class_path,
        data_config.init_args,
        data_config.batch_size,
        data_config.num_workers,
        data_config.shuffle
    )
    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, v)

        break
    print(batch['cached'])
    print(batch['file_hashes'])
    print(dataloader.cache_manager)

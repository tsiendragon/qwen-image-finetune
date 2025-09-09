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
# 删除重复的 typing 导入，已在上方统一导入 Optional, Dict, List, Any
from src.data.cache_manager import EmbeddingCacheManager, check_cache_exists


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

        self.images_dirs, self.control_dirs = self._find_directories()

        # 扫描所有图像文件
        self.image_files = self._scan_image_files()
        # 图像预处理变换 - 将[img_w, img_h] 转换为transforms.Resize期望的 (height, width) 格式
        if isinstance(self.image_size, (tuple, list)):
            self.resize_size = (self.image_size[0], self.image_size[1])  # (width, height)
        else:
            self.resize_size = self.image_size

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

    def _find_directories(self):
        """
        查找所有数据集根路径下的图像目录与控制图像目录。
        优先匹配:
            images: ['training_images', 'images', 'target_images']
            control: ['control_images', 'control', 'condition_images']
        若都未找到则回退到默认 'training_images' 与 'control_images'。
        Returns:
            Tuple[List[str], List[str]]: images_dirs, control_dirs
        """
        images_dirs = []
        control_dirs = []

        image_possible_names = ['training_images', 'images', 'target_images']
        control_possible_names = ['control_images', 'control', 'condition_images']

        for dataset_path in self.dataset_paths:
            # 查找图像目录
            images_dir = None
            for name in image_possible_names:
                path = os.path.join(dataset_path, name)
                if os.path.exists(path):
                    print(f"Found images directory: {path}")
                    images_dir = path
                    break

            if images_dir is None:
                # 如果都没找到，使用默认路径
                images_dir = os.path.join(dataset_path, 'training_images')
                print(f"Using default images directory: {images_dir}")

            # 查找控制图像目录
            control_dir = None
            for name in control_possible_names:
                path = os.path.join(dataset_path, name)
                if os.path.exists(path):
                    print(f"Found control directory: {path}")
                    control_dir = path
                    break

            if control_dir is None:
                # 如果都没找到，使用默认路径
                control_dir = os.path.join(dataset_path, 'control_images')
                print(f"Using default control directory: {control_dir}")

            # 检查目录是否存在
            if not os.path.exists(images_dir):
                print(f"Warning: Images directory not found: {images_dir}")
                continue
            if not os.path.exists(control_dir):
                print(f"Warning: Control directory not found: {control_dir}")
                continue

            images_dirs.append(images_dir)
            control_dirs.append(control_dir)

        if not images_dirs:
            raise ValueError("No valid dataset directories found")

        return images_dirs, control_dirs

    def _scan_image_files(self):
        """
        扫描所有数据集中的成对样本。
        样本需满足：
            - images_dir/xxx.(jpg|jpeg|png|bmp)
            - control_dir/xxx.(jpg|jpeg|png|bmp)
            - images_dir/xxx.txt 作为 caption
            - 可选 control_dir/xxx_mask.png
        Returns:
            List[dict]: 'image' 'control' 'caption' 'dataset_index' 'mask_file'
        """
        all_valid_files = []

        # 遍历所有数据集目录
        for i, (images_dir, control_dir) in enumerate(zip(self.images_dirs, self.control_dirs)):
            print(f"Scanning dataset {i+1}: {images_dir}")
            image_files = []

            # 获取images目录下的所有图像文件
            image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for pattern in image_patterns:
                files = glob.glob(os.path.join(images_dir, pattern))
                image_files.extend(files)

            # 过滤掉没有对应control图像和caption的文件
            valid_files = []
            for image_file in image_files:
                # 获取文件名（不含扩展名）
                base_name = os.path.splitext(os.path.basename(image_file))[0]

                # 检查是否存在对应的control图像
                control_file = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    control_path = os.path.join(control_dir, base_name + ext)
                    if os.path.exists(control_path):
                        control_file = control_path
                        break

                mask_file = os.path.join(control_dir, base_name + '_mask.png')

                # 检查是否存在对应的caption文件
                caption_file = os.path.join(images_dir, base_name + '.txt')

                if control_file and os.path.exists(caption_file):
                    valid_files.append({
                        'image': image_file,
                        'control': control_file,
                        'caption': caption_file,
                        'dataset_index': i,  # 记录来自哪个数据集,
                        'mask_file': mask_file,
                    })
                else:
                    print(f"Warning: Skipping {image_file} - missing control image or caption file")

            print(f"Dataset {i+1} found {len(valid_files)} valid image pairs")
            all_valid_files.extend(valid_files)

        print(f"Total found {len(all_valid_files)} valid image pairs from {len(self.images_dirs)} datasets")
        return all_valid_files

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)

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
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_files)}")

        file_info = self.image_files[idx]

        # 读取提示文本
        with open(file_info['caption'], 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        # 如果启用裁剪，生成共同的裁剪边界框
        crop_bbox = None
        if (self.random_crop and self.crop_size is not None) or self.center_crop:
            # 读取第一个图像来获取尺寸（假设image和control尺寸相同）
            temp_img = cv2.imread(file_info['image'])
            if temp_img is not None:
                h, w = temp_img.shape[:2]
                if self.center_crop:
                    crop_bbox = self.get_center_crop_bbox(h, w)
                elif self.random_crop and self.crop_size is not None:
                    crop_bbox = self.get_random_crop_bbox(h, w)

        image_numpy = self.preprocess(file_info['image'], crop_bbox)

        # 加载控制图像（使用相同的裁剪边界框）
        control_numpy = self.preprocess(file_info['control'], crop_bbox)

        has_mask = False
        if os.path.exists(file_info['mask_file']):
            mask_numpy = self.preprocess(cv2.imread(file_info['mask_file'], 0), crop_bbox)
            has_mask = True

        if self.use_cache:
            image_hash = self.cache_manager.get_file_hash_for_image(file_info['image'])
            control_hash = self.cache_manager.get_file_hash_for_image(file_info['control'])
            prompt_hash = self.cache_manager.get_file_hash_for_prompt(file_info['image'], prompt)
            empty_prompt_hash = self.cache_manager.get_file_hash_for_prompt(file_info['image'], "empty")

        if self.cache_exists and self.use_cache:
            # 如果启用缓存，尝试加载缓存的嵌入
            # 检查缓存是否存在
            # TODO: original implementation
            cache_file = os.path.join(self.cache_dir, 'pixel_latent', f"{image_hash}.pt")
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
                'file_paths': file_info,
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
    config_file = 'configs/qwen_image_edit_config.yaml'
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

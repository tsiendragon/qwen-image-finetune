import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
import cv2
import importlib
from typing import Optional, Dict, List, Any
from src.data.cache_manager import EmbeddingCacheManager, check_cache_exists


class ImageDataset(Dataset):
    """
    Dataset:
        image: RGB numpy array [3, img_h, img_w]
        control: RGB numpy array [3, img_h, img_w]
        prompt: str
        file_hashes: dict
            - image_hash: str
            - control_hash: str
            - prompt_hash: str
            - empty_prompt_hash: str
    """
    def __init__(self, data_config):
        """
        初始化数据集
        Args:
            data_config: 包含数据集路径和配置的字典
                - dataset_path: 数据集根目录路径，可以是单个路径字符串或路径列表
                - image_size: 图像尺寸 [img_w, img_h]，默认为(512, 512)
                - cache_dir: 缓存目录路径，如果提供则启用缓存
                - use_cache: 是否使用缓存，默认为 True
                - random_crop: 是否启用随机裁剪，默认为 False
                - crop_size: 裁剪后的正方形尺寸，默认为 None
                - crop_scale: 裁剪缩放范围 [min, max]，默认为 [0.8, 1.0]
                - center_crop: 是否启用中心裁剪，默认为 False
                - center_crop_ratio: 中心裁剪比例，默认为 1.0

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

    def get_random_crop_bbox(self, h, w):
        """
        生成随机裁剪的边界框
        Args:
            h: 图像高度
            w: 图像宽度
        Returns:
            tuple: (x, y, crop_w, crop_h) 裁剪区域
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
        生成中心裁剪的边界框，保持目标比例且在图片内部能达到的最大尺寸
        Args:
            h: 图像高度
            w: 图像宽度
        Returns:
            tuple: (x, y, crop_w, crop_h) 裁剪区域
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
        应用裁剪和调整大小
        Args:
            img: 输入图像 (H, W, C)
            bbox: 裁剪边界框 (x, y, w, h)，如果为None则不裁剪
        Returns:
            处理后的图像
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
        预处理单个图像
        Args:
            image_path: 图像路径
            crop_bbox: 可选的裁剪边界框 (x, y, w, h)
        Returns:
            处理后的图像数组 (C, H, W)
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
        """查找所有数据集路径下的图像和控制图像目录"""
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
        """扫描所有数据集中的图像文件"""
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
        获取数据集中的一个样本
        Args:
            idx: 样本索引
        Returns:
            tuple: (image_tensor, prompt) 或 带缓存的 embedding 字典
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
        for k, v in data.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    assert vv is not None, f"value is None for key {kk} in {k}"
            else:
                assert v is not None, f"value is None for key {k}"

    def get_cache_stats(self) -> Optional[Dict[str, int]]:
        """
        获取缓存统计信息
        Returns:
            缓存统计字典，如果未启用缓存则返回 None
        """
        if self.cache_manager:
            return self.cache_manager.get_cache_stats()
        return None


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义collate函数，用于处理不同长度的prompt_embed和prompt_embeds_mask的padding

    Args:
        batch: 批次数据列表，每个元素是__getitem__返回的字典

    Returns:
        处理后的批次数据字典，其中prompt_embed和prompt_embeds_mask已进行padding
    """
    # 分离cached和non-cached的数据
    cached_items = [item for item in batch if item.get('cached', False)]
    non_cached_items = [item for item in batch if not item.get('cached', False)]

    # 如果batch中有cached数据，需要进行padding
    if cached_items:
        # 找到prompt_embed和prompt_embeds_mask的最大长度
        max_prompt_len = 0
        for item in cached_items:
            if item['prompt_embed'] is not None:
                prompt_len = item['prompt_embed'].shape[0]
                max_prompt_len = max(max_prompt_len, prompt_len)

        # 对所有cached items进行padding
        for item in cached_items:
            if item['prompt_embed'] is not None:
                prompt_embed = item['prompt_embed']
                prompt_embeds_mask = item['prompt_embeds_mask']

                # 转换为torch tensor（如果还不是）
                if not isinstance(prompt_embed, torch.Tensor):
                    prompt_embed = torch.tensor(prompt_embed)
                if not isinstance(prompt_embeds_mask, torch.Tensor):
                    prompt_embeds_mask = torch.tensor(prompt_embeds_mask)

                current_len = prompt_embed.shape[0]

                if current_len < max_prompt_len:
                    # 计算需要padding的长度
                    pad_len = max_prompt_len - current_len

                    # 对prompt_embed进行padding
                    if len(prompt_embed.shape) == 1:
                        # 1D tensor
                        pad_embed = torch.zeros(pad_len, dtype=prompt_embed.dtype)
                        item['prompt_embed'] = torch.cat([prompt_embed, pad_embed], dim=0)
                    else:
                        # 2D tensor [seq_len, hidden_dim]
                        pad_embed = torch.zeros(pad_len, prompt_embed.shape[1], dtype=prompt_embed.dtype)
                        item['prompt_embed'] = torch.cat([prompt_embed, pad_embed], dim=0)

                    # 对prompt_embeds_mask进行padding
                    if len(prompt_embeds_mask.shape) == 1:
                        # 1D tensor
                        pad_mask = torch.zeros(pad_len, dtype=prompt_embeds_mask.dtype)
                        item['prompt_embeds_mask'] = torch.cat([prompt_embeds_mask, pad_mask], dim=0)
                    else:
                        # 2D tensor
                        pad_mask = torch.zeros(pad_len, prompt_embeds_mask.shape[1], dtype=prompt_embeds_mask.dtype)
                        item['prompt_embeds_mask'] = torch.cat([prompt_embeds_mask, pad_mask], dim=0)

    # 重新组合批次数据
    all_items = cached_items + non_cached_items

    # 创建批次字典
    batch_dict = {}

    # 收集所有键
    all_keys = set()
    for item in all_items:
        all_keys.update(item.keys())

    # 对每个键进行批处理
    for key in all_keys:
        values = []
        for item in all_items:
            if key in item:
                values.append(item[key])
            else:
                values.append(None)

        # 如果所有值都是None，跳过
        if all(v is None for v in values):
            continue

        # 如果是tensor类型的数据，尝试stack
        if key in ['image', 'control', 'mask', 'pixel_latent', 'control_latent', 'prompt_embed', 'prompt_embeds_mask']:
            try:
                # 过滤掉None值
                non_none_values = [v for v in values if v is not None]
                if non_none_values:
                    # 转换为torch tensor
                    tensor_values = []
                    for v in non_none_values:
                        if not isinstance(v, torch.Tensor):
                            v = torch.tensor(v)
                        tensor_values.append(v)

                    # 尝试stack
                    if len(tensor_values) > 0:
                        batch_dict[key] = torch.stack(tensor_values, dim=0)
                    else:
                        batch_dict[key] = values
                else:
                    batch_dict[key] = values
            except Exception:
                # 如果stack失败，保持原始列表
                batch_dict[key] = values
        else:
            # 非tensor数据保持列表形式
            batch_dict[key] = values

    return batch_dict


def loader(
        class_path: str,
        init_args: dict,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True) -> DataLoader:
    """
    动态加载数据集类并创建DataLoader
    Args:
        class_path: 类的完整路径，如 'src.data.dataset.ImageDataset'
        init_args: 用于初始化数据集类的参数字典
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        shuffle: 是否打乱数据
    Returns:
        DataLoader对象
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

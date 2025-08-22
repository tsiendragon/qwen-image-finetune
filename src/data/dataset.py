import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from typing import Optional, Dict, Any
from .cache_manager import EmbeddingCacheManager


class ImageDataset(Dataset):
    def __init__(self, data_config):
        """
        初始化数据集
        Args:
            data_config: 包含数据集路径和配置的字典
                - dataset_path: 数据集根目录路径
                - image_size: 图像尺寸 [img_w, img_h]，默认为(512, 512)
                - cache_dir: 缓存目录路径，如果提供则启用缓存
                - use_cache: 是否使用缓存，默认为 True
        """
        self.data_config = data_config
        self.dataset_path = data_config.get('dataset_path', './dataset')
        self.image_size = data_config.get('image_size', (512, 512))

        # 缓存相关配置
        self.cache_dir = data_config.get('cache_dir', None)
        self.use_cache = data_config.get('use_cache', True) and self.cache_dir is not None

        # 初始化缓存管理器
        if self.use_cache:
            self.cache_manager = EmbeddingCacheManager(self.cache_dir)
            print(f"缓存已启用，缓存目录: {self.cache_dir}")
        else:
            self.cache_manager = None
            print("缓存未启用")

        # 图像路径 - 支持多种命名方式
        self.images_dir = self._find_images_directory()
        self.control_dir = self._find_control_directory()

        # 检查目录是否存在
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.control_dir):
            raise ValueError(f"Control directory not found: {self.control_dir}")

        # 扫描所有图像文件
        self.image_files = self._scan_image_files()

        # 图像预处理变换 - 将[img_w, img_h]转换为transforms.Resize期望的(height, width)格式
        if isinstance(self.image_size, (tuple, list)) and len(self.image_size) == 2:
            # 将[img_w, img_h]转换为(img_h, img_w)供PyTorch使用
            resize_size = (self.image_size[1], self.image_size[0])  # (height, width)
        else:
            resize_size = self.image_size

        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
        ])

    def _find_images_directory(self):
        """查找图像目录，支持多种命名方式"""
        possible_names = ['training_images', 'images', 'target_images']

        for name in possible_names:
            path = os.path.join(self.dataset_path, name)
            if os.path.exists(path):
                print(f"Found images directory: {path}")
                return path

        # 如果都没找到，默认返回第一个选项
        return os.path.join(self.dataset_path, 'training_images')

    def _find_control_directory(self):
        """查找控制图像目录，支持多种命名方式"""
        possible_names = ['control_images', 'control', 'condition_images']

        for name in possible_names:
            path = os.path.join(self.dataset_path, name)
            if os.path.exists(path):
                print(f"Found control directory: {path}")
                return path

        # 如果都没找到，默认返回第一个选项
        return os.path.join(self.dataset_path, 'control_images')

    def _scan_image_files(self):
        """扫描数据集中的所有图像文件"""
        image_files = []

        # 获取images目录下的所有图像文件
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for pattern in image_patterns:
            files = glob.glob(os.path.join(self.images_dir, pattern))
            image_files.extend(files)

        # 过滤掉没有对应control图像和caption的文件
        valid_files = []
        for image_file in image_files:
            # 获取文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(image_file))[0]

            # 检查是否存在对应的control图像
            control_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                control_path = os.path.join(self.control_dir, base_name + ext)
                if os.path.exists(control_path):
                    control_file = control_path
                    break

            # 检查是否存在对应的caption文件
            caption_file = os.path.join(self.images_dir, base_name + '.txt')

            if control_file and os.path.exists(caption_file):
                valid_files.append({
                    'image': image_file,
                    'control': control_file,
                    'caption': caption_file
                })
            else:
                print(f"Warning: Skipping {image_file} - missing control image or caption file")

        print(f"Found {len(valid_files)} valid image pairs")
        return valid_files

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

        # 如果启用缓存，尝试加载缓存的嵌入
        if self.use_cache and self.cache_manager:
            # 生成文件哈希
            image_hash = self.cache_manager.get_file_hash_for_image(file_info['image'])
            control_hash = self.cache_manager.get_file_hash_for_image(file_info['control'])
            prompt_hash = self.cache_manager.get_file_hash_for_prompt(file_info['image'], prompt)

            # 检查缓存是否存在
            cached_data = {}
            cache_complete = True

            for cache_type, file_hash in [
                ('pixel_latent', image_hash),
                ('control_latent', control_hash),
                ('prompt_embed', prompt_hash),
                ('prompt_embeds_mask', prompt_hash)
            ]:
                cached_embedding = self.cache_manager.load_cache(cache_type, file_hash)
                if cached_embedding is not None:
                    cached_data[cache_type] = cached_embedding
                else:
                    cache_complete = False

            # 如果所有缓存都存在，返回缓存数据
            if cache_complete:
                return {
                    'cached': True,
                    'pixel_latent': cached_data['pixel_latent'],
                    'control_latent': cached_data['control_latent'],
                    'prompt_embed': cached_data['prompt_embed'],
                    'prompt_embeds_mask': cached_data['prompt_embeds_mask'],
                    'prompt': prompt,
                    'file_hashes': {
                        'image_hash': image_hash,
                        'control_hash': control_hash,
                        'prompt_hash': prompt_hash
                    }
                }

        # 如果没有缓存或缓存不完整，返回原始数据
        image = Image.open(file_info['image']).convert('RGB')
        image_tensor = self.transform(image)

        # 加载控制图像
        control_image = Image.open(file_info['control']).convert('RGB')
        control_tensor = self.transform(control_image)

        return {
            'cached': False,
            'image': image_tensor,
            'control': control_tensor,
            'prompt': prompt,
            'file_paths': file_info,
            'file_hashes': {
                'image_hash': self.cache_manager.get_file_hash_for_image(file_info['image']) if self.cache_manager else None,
                'control_hash': self.cache_manager.get_file_hash_for_image(file_info['control']) if self.cache_manager else None,
                'prompt_hash': self.cache_manager.get_file_hash_for_prompt(file_info['image'], prompt) if self.cache_manager else None
            }
        }

    def save_embeddings_to_cache(self, idx: int, embeddings: Dict[str, torch.Tensor]) -> None:
        """
        保存计算出的嵌入到缓存
        Args:
            idx: 样本索引
            embeddings: 包含各种嵌入的字典
        """
        if not self.use_cache or not self.cache_manager:
            return

        file_info = self.image_files[idx]

        # 读取提示文本
        with open(file_info['caption'], 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

        # 生成哈希值
        image_hash = self.cache_manager.get_file_hash_for_image(file_info['image'])
        control_hash = self.cache_manager.get_file_hash_for_image(file_info['control'])
        prompt_hash = self.cache_manager.get_file_hash_for_prompt(file_info['image'], prompt)

        # 保存嵌入
        cache_mappings = {
            'pixel_latent': image_hash,
            'control_latent': control_hash,
            'prompt_embed': prompt_hash,
            'prompt_embeds_mask': prompt_hash
        }

        for embedding_type, file_hash in cache_mappings.items():
            if embedding_type in embeddings:
                self.cache_manager.save_cache(embedding_type, file_hash, embeddings[embedding_type])

    def get_cache_stats(self) -> Optional[Dict[str, int]]:
        """
        获取缓存统计信息
        Returns:
            缓存统计字典，如果未启用缓存则返回 None
        """
        if self.cache_manager:
            return self.cache_manager.get_cache_stats()
        return None


def loader(dataset_path: str,
          image_size: tuple = (832, 576),
          batch_size: int = 1,
          num_workers: int = 0,
          cache_dir: Optional[str] = None,
          use_cache: bool = True,
          shuffle: bool = True) -> DataLoader:
    """
    创建数据加载器
    Args:
        dataset_path: 数据集路径
        image_size: 图像尺寸
        batch_size: 批量大小
        num_workers: 工作进程数
        cache_dir: 缓存目录路径
        use_cache: 是否使用缓存
        shuffle: 是否打乱数据
    Returns:
        DataLoader 实例
    """
    data_config = {
        'dataset_path': dataset_path,
        'image_size': image_size,
        'cache_dir': cache_dir,
        'use_cache': use_cache
    }

    dataset = ImageDataset(data_config)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == "__main__":
    data_config = {
        'dataset_path': '/data/kyc_gen/id_card/',
        'image_size': (512, 312)
    }
    dataloader = loader(data_config)
    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, v)
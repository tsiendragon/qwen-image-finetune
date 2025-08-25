import os
import random
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
import importlib
from typing import Optional, Dict
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
                - dataset_path: 数据集根目录路径
                - image_size: 图像尺寸 [img_w, img_h]，默认为(512, 512)
                - cache_dir: 缓存目录路径，如果提供则启用缓存
                - use_cache: 是否使用缓存，默认为 True

        """
        self.data_config = data_config
        self.dataset_path = data_config.get('dataset_path', './dataset')
        self.image_size = data_config.get('image_size', (512, 512))
        self.cache_dir = data_config.get('cache_dir', None)
        self.use_cache = data_config.get('use_cache', True)

        # 初始化缓存管理器
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_manager = EmbeddingCacheManager(self.cache_dir)
            print(f"缓存已启用，缓存目录: {self.cache_dir}")
        else:
            self.cache_manager = None
            print("缓存未启用")

        self.cache_exists = check_cache_exists(self.cache_dir)

        self.images_dir = self._find_images_directory()
        self.control_dir = self._find_control_directory()

        # 检查目录是否存在
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not os.path.exists(self.control_dir):
            raise ValueError(f"Control directory not found: {self.control_dir}")

        # 扫描所有图像文件
        self.image_files = self._scan_image_files()
        # 图像预处理变换 - 将[img_w, img_h] 转换为transforms.Resize期望的 (height, width) 格式
        if isinstance(self.image_size, (tuple, list)) and len(self.image_size) == 2:
            # 将[img_w, img_h]转换为(img_h, img_w)供PyTorch使用
            self.resize_size = (self.image_size[1], self.image_size[0])  # (height, width)
        else:
            self.resize_size = self.image_size

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        img = img[:, :, ::-1]
        img = cv2.resize(img, self.resize_size)
        img = img.transpose(2, 0, 1)
        return img

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

        image_numpy = self.preprocess(file_info['image'])

        # 加载控制图像
        control_numpy = self.preprocess(file_info['control'])

        if self.use_cache:
            image_hash = self.cache_manager.get_file_hash_for_image(file_info['image'])
            control_hash = self.cache_manager.get_file_hash_for_image(file_info['control'])
            prompt_hash = self.cache_manager.get_file_hash_for_prompt(file_info['image'], prompt)
            empty_prompt_hash = self.cache_manager.get_file_hash_for_prompt(file_info['image'], "empty")

        if self.cache_exists:
            # 如果启用缓存，尝试加载缓存的嵌入
            # 检查缓存是否存在
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
                prompt_embed = cached_data['empty_prompt_embed']
                prompt_embeds_mask = cached_data['empty_prompt_embeds_mask']
            else:
                prompt_embed = cached_data['prompt_embed']
                prompt_embeds_mask = cached_data['prompt_embeds_mask']
            # 如果所有缓存都存在，返回缓存数据
            data = {
                'cached': True,
                'image': image_numpy,
                'control': control_numpy,
                'pixel_latent': cached_data['pixel_latent'],
                'control_latent': cached_data['control_latent'],
                'prompt_embed': prompt_embed,
                'prompt_embeds_mask': prompt_embeds_mask,
                'prompt': prompt,
                'file_hashes': {
                    'image_hash': image_hash,
                    'control_hash': control_hash,
                    'prompt_hash': prompt_hash,
                    'empty_prompt_hash': empty_prompt_hash
                }
            }
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
        pin_memory=True
    )
    setattr(dataloader, 'cache_manager', cache_manager)
    return dataloader


if __name__ == "__main__":
    from src.data.config import load_config_from_yaml
    import torch
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

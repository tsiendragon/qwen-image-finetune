import os
import torch
import hashlib
import pickle
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


class EmbeddingCacheManager:
    """嵌入缓存管理器，用于保存和加载预计算的嵌入"""

    def __init__(self, cache_root: str):
        """
        初始化缓存管理器
        Args:
            cache_root: 缓存根目录
        """
        self.cache_root = Path(cache_root)
        self.cache_dirs = {
            'pixel_latent': self.cache_root / 'pixel_latent',
            'control_latent': self.cache_root / 'control_latent',
            'prompt_embed': self.cache_root / 'prompt_embed',
            'prompt_embeds_mask': self.cache_root / 'prompt_embeds_mask'
        }

        # 创建缓存目录
        for cache_dir in self.cache_dirs.values():
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_hash(self, file_path: str, prompt: str = "") -> str:
        """
        生成文件的唯一哈希值
        Args:
            file_path: 文件路径
            prompt: 提示文本（用于 prompt embedding）
        Returns:
            唯一哈希值
        """
        # 使用文件路径 + 修改时间 + prompt 生成哈希
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, cache_type: str, file_hash: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dirs[cache_type] / f"{file_hash}.pt"

    def save_cache(self, cache_type: str, file_hash: str, data: torch.Tensor) -> None:
        """
        保存缓存数据
        Args:
            cache_type: 缓存类型 (pixel_latent, control_latent, prompt_embed, prompt_embeds_mask)
            file_hash: 文件哈希值
            data: 要缓存的张量数据
        """
        cache_path = self._get_cache_path(cache_type, file_hash)
        torch.save(data.cpu(), cache_path)

    def load_cache(self, cache_type: str, file_hash: str) -> Optional[torch.Tensor]:
        """
        加载缓存数据
        Args:
            cache_type: 缓存类型
            file_hash: 文件哈希值
        Returns:
            缓存的张量数据，如果不存在则返回 None
        """
        cache_path = self._get_cache_path(cache_type, file_hash)
        if cache_path.exists():
            return torch.load(cache_path, map_location='cpu')
        return None

    def cache_exists(self, cache_type: str, file_hash: str) -> bool:
        """检查缓存是否存在"""
        cache_path = self._get_cache_path(cache_type, file_hash)
        return cache_path.exists()

    def get_file_hash_for_image(self, image_path: str) -> str:
        """为图像文件生成哈希值"""
        return self._get_file_hash(image_path)

    def get_file_hash_for_prompt(self, image_path: str, prompt: str) -> str:
        """为提示文本生成哈希值（基于图像路径和提示内容）"""
        return self._get_file_hash(image_path, prompt)

    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        清理缓存
        Args:
            cache_type: 指定清理的缓存类型，如果为 None 则清理所有缓存
        """
        if cache_type:
            cache_dirs = [self.cache_dirs[cache_type]]
        else:
            cache_dirs = list(self.cache_dirs.values())

        for cache_dir in cache_dirs:
            for cache_file in cache_dir.glob("*.pt"):
                cache_file.unlink()

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        stats = {}
        for cache_type, cache_dir in self.cache_dirs.items():
            stats[cache_type] = len(list(cache_dir.glob("*.pt")))
        return stats

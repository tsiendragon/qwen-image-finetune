import os
import torch
from typing import Dict, Optional, List
from pathlib import Path
from src.utils.tools import extract_file_hash, hash_string_md5


def check_cache_exists(cache_root: str, cache_types: list) -> Dict[str, bool]:
    """
    检查缓存根目录下是否存在不同类型的缓存文件
    Args:
        cache_root: 缓存根目录路径
    Returns:
        包含各种缓存类型存在状态的字典
    """
    cache_root_path = Path(cache_root)

    result = {}
    for cache_type in cache_types:
        cache_dir = cache_root_path / cache_type
        # 检查目录是否存在且包含 .pt 文件
        has_cache = cache_dir.exists() and len(list(cache_dir.glob("*.pt"))) > 0
        result[cache_type] = has_cache

    return all(result.values())


class EmbeddingCacheManager:
    """嵌入缓存管理器，用于保存和加载预计算的嵌入"""

    def __init__(self, cache_root: str, cache_types: List[str]=None):
        """
        初始化缓存管理器
        Args:
            cache_root: 缓存根目录
        """
        self.cache_root = Path(cache_root)
        if cache_types is None:
            self.cache_dirs = {
                'pixel_latent': self.cache_root / 'pixel_latent',
                'control_latent': self.cache_root / 'control_latent',
                'prompt_embed': self.cache_root / 'prompt_embed',
                # 'prompt_embeds_mask': self.cache_root / 'prompt_embeds_mask',
                'empty_prompt_embed': self.cache_root / 'empty_prompt_embed',
                # 'empty_prompt_embeds_mask': self.cache_root / 'empty_prompt_embeds_mask',
                'pooled_prompt_embed': self.cache_root / 'pooled_prompt_embed',
                'empty_pooled_prompt_embed': self.cache_root / 'empty_pooled_prompt_embed',
                # 'control_latent_1': self.cache_root / 'control_latent_1',
                # 'control_latent_2': self.cache_root / 'control_latent_2',
                # 'control_latent_3': self.cache_root / 'control_latent_3',
                # 'control_latent_4': self.cache_root / 'control_latent_4',
                # 'control_latent_5': self.cache_root / 'control_latent_5',
                # 'control_latent_6': self.cache_root / 'control_latent_6',
                # 'control_latent_7': self.cache_root / 'control_latent_7',
            }
            # default is for the Flux Kontext
        else:
            self.cache_dirs = {cache_type: self.cache_root / cache_type for cache_type in cache_types}
        # all possible cache keys

    def get_hash(self, file_path: str, prompt: str = "") -> str:
        if prompt:
            return extract_file_hash(file_path) + hash_string_md5(prompt)
        else:
            return extract_file_hash(file_path)

    def _get_cache_path(self, cache_type: str, file_hash: str) -> Path:
        """获取缓存文件路径"""
        if cache_type in self.cache_dirs:  # original style
            return self.cache_dirs[cache_type] / f"{file_hash}.pt"
        else:
            raise ValueError(f"Invalid cache type: {cache_type}"
                f"supported cache types: {list(self.cache_dirs.keys())}"
            )

    def save_cache(self, cache_type: str, file_hash: str, data: torch.Tensor) -> None:
        """
        保存缓存数据
        Args:
            cache_type: 缓存类型 (pixel_latent, control_latent, prompt_embed, prompt_embeds_mask)
            file_hash: 文件哈希值
            data: 要缓存的张量数据
        """
        cache_path = self._get_cache_path(cache_type, file_hash)
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))
        # 确保tensor没有梯度信息，避免多进程序列化问题
        data_to_save = data.detach().cpu().to(torch.float16)
        torch.save(data_to_save, cache_path)

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
            loaded_data = torch.load(cache_path, map_location='cpu', weights_only=False)
            # 确保加载的数据没有梯度信息
            return loaded_data.detach() if loaded_data is not None else None
        return None

    def cache_exists(self, cache_type: str, file_hash: str) -> bool:
        """检查缓存是否存在"""
        cache_path = self._get_cache_path(cache_type, file_hash)
        return cache_path.exists()

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        stats = {}
        for cache_type, cache_dir in self.cache_dirs.items():
            stats[cache_type] = len(list(cache_dir.glob("*.pt")))
        return stats

    def exist(self):
        for cache_type, cache_dir in self.cache_dirs.items():
            if not cache_dir.exists():
                return False
        return check_cache_exists(self.cache_root, list(self.cache_dirs.keys()))


if __name__ == "__main__":
    cache_root = "/data/lilong/experiment/id_card_qwen_image_lora/cache"
    print(check_cache_exists(cache_root))

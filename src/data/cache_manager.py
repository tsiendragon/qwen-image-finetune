import os
import torch
from typing import Dict, Optional, List
from pathlib import Path
import json
import glob
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
        self.metadata_dir = self.cache_root / 'metadata'

    def create_folders(self):
        # all possible cache keys
        os.makedirs(self.metadata_dir, exist_ok=True)
        for dir in self.cache_dirs:
            os.makedirs(self.metadata_dir / dir, exist_ok=True)

    def get_hash(self, file_path: str, prompt: str = "") -> str:
        if prompt:
            return extract_file_hash(file_path) + hash_string_md5(prompt)
        else:
            return extract_file_hash(file_path)

    def get_metadata_path(self, main_hash: str) -> str:
        return os.path.join(str(self.cache_root), f'{main_hash}_metadata.json')

    def get_cache_embedding_path(self, embedding_key: str, hash_value: str) -> str:
        return os.path.join(str(self.cache_root), embedding_key, f'{hash_value}.pt')

    def save_cache_embedding(self, data:dict, hash_maps:dict, file_hashes:dict):
        """save cache embedding
        data: dict[k, embedding]
            keys like: image_latent, prompt_embedding, pooled_prompt_embedding, etc.
        hash_maps: dict[k, hash_type]. Hash types support
            - image_hash
            - control_hash
            - prompt_hash
            - empty_prompt_hash
            - control_prompt_hash
            - control_empty_prompt_hash
            - control_1_hash
            - control_2_hash
        """
        assert set(hash_maps.keys()) == set(data.keys()), "hash_maps and data keys must be the same"
        assert set(hash_maps.values()).issubset(set(file_hashes.keys())), "hash_maps values must be a subset of file_hashes keys"
        main_hash = file_hashes['image_hash']
        metadata_path = self.get_metadata_path(main_hash)
        metadata = {}

        for key in data.keys():
            hash_type = hash_maps[key]
            hash_value = file_hashes[hash_type]
            embedding = data[key].detach().cpu().to(torch.float16)
            cache_path = self.get_cache_embedding_path(key, hash_value)
            torch.save(embedding, cache_path)
            metadata[key] = hash_value

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    def load_cache(self, data):
        main_hash = data['file_hashes']['image_hash']
        metadata_path = self.get_metadata_path(main_hash)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        for embedding_key, hash_value in metadata.items():
            cache_path = self.get_cache_embedding_path(embedding_key, hash_value)
            embedding = torch.load(cache_path, map_location='cpu', weights_only=False)
            data[embedding_key] = embedding
        return data

    def exist(self):
        """check if metadata exists"""
        meta_folder = self.metadata_dir
        if os.path.exists(meta_folder):
            metadata_files = glob.glob(os.path.join(meta_folder, "*.json"))
            return len(metadata_files) > 0
        return False

if __name__ == "__main__":
    cache_root = "/data/lilong/experiment/id_card_qwen_image_lora/cache"
    print(check_cache_exists(cache_root))

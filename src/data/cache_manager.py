import os
import torch
from typing import List
from pathlib import Path
import json
import glob
from src.utils.tools import extract_file_hash, hash_string_md5


class EmbeddingCacheManager:
    """嵌入缓存管理器，用于保存和加载预计算的嵌入"""

    def __init__(self, cache_root: str):
        """
        初始化缓存管理器
        Args:
            cache_root: 缓存根目录
        """
        self.cache_root = Path(cache_root)
        self.metadata_dir = self.cache_root / "metadata"

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

    @classmethod
    def get_metadata_path(cls, cache_root, main_hash: str) -> str:
        return os.path.join(str(cache_root), f"{main_hash}_metadata.json")

    def get_cache_embedding_path(self, embedding_key: str, hash_value: str) -> str:
        return os.path.join(str(self.cache_root), embedding_key, f"{hash_value}.pt")

    def save_cache_embedding(self, data: dict, hash_maps: dict, file_hashes: dict):
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
        assert set(hash_maps.keys()) == set(
            data.keys()
        ), "hash_maps and data keys must be the same"
        assert set(hash_maps.values()).issubset(
            set(file_hashes.keys())
        ), "hash_maps values must be a subset of file_hashes keys"
        main_hash = file_hashes["image_hash"]
        metadata_path = self.get_metadata_path(self.cache_root, main_hash)
        metadata = {}

        for key in data.keys():
            hash_type = hash_maps[key]
            hash_value = file_hashes[hash_type]
            embedding = data[key].detach().cpu().to(torch.float16)
            cache_path = self.get_cache_embedding_path(key, hash_value)
            torch.save(embedding, cache_path)
            metadata[key] = hash_value

        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def load_cache(self, data, replace_empty_embeddings: bool = False, prompt_empty_drop_keys: List[str] = None):
        main_hash = data["file_hashes"]["image_hash"]
        metadata_path = self.get_metadata_path(self.cache_root, main_hash)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        for embedding_key, hash_value in metadata.items():
            if embedding_key.startswith("empty_"):
                continue
            cache_path = self.get_cache_embedding_path(embedding_key, hash_value)
            embedding = torch.load(cache_path, map_location="cpu", weights_only=False)
            data[embedding_key] = embedding
        if replace_empty_embeddings:
            for key in prompt_empty_drop_keys:
                original_key = key.replace("empty_", "")
                hash_value = metadata[key]
                empty_cache_path = self.get_cache_embedding_path(key, hash_value)
                empty_embedding = torch.load(empty_cache_path, map_location="cpu", weights_only=False)
                data[original_key] = empty_embedding
        return data

    @classmethod
    def exist(cls, cache_root: str):
        """check if metadata exists"""
        meta_folder = cls.get_metadata_path(cache_root, "image_hash")
        if os.path.exists(meta_folder):
            metadata_files = glob.glob(os.path.join(meta_folder, "*.json"))
            return len(metadata_files) > 0
        return False

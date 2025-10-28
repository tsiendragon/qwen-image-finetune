import glob
import json
import os
from pathlib import Path

import torch

from qflux.utils.tools import extract_file_hash, hash_string_md5


class EmbeddingCacheManager:
    """嵌入缓存管理器，用于保存和加载预计算的嵌入"""

    CACHE_VERSION = "2.0"

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
        return os.path.join(str(cache_root), "metadata", f"{main_hash}.json")

    def get_cache_embedding_path(self, embedding_key: str, hash_value: str) -> str:
        return os.path.join(str(self.cache_root), embedding_key, f"{hash_value}.pt")

    def save_cache_embedding(self, data: dict, hash_maps: dict, file_hashes: dict, img_shapes=None):
        """save cache embedding with version management

        Args:
            data: dict[k, embedding]
                keys like: image_latent, prompt_embedding, pooled_prompt_embedding, etc.
            hash_maps: dict[k, hash_type]. Hash types support
                - main_hash: the hash is the sum of image_hash+image_hash+prompt_hash
                - image_hash
                - control_hash
                - prompt_hash
                - empty_prompt_hash
                - control_prompt_hash
                - control_empty_prompt_hash
                - control_1_hash
                - control_2_hash
            file_hashes: dict of file hashes
            img_shapes: Optional tensor of image shapes [(C, H, W), ...] for multi-resolution
        """
        assert set(hash_maps.keys()) == set(data.keys()), "hash_maps and data keys must be the same"
        assert set(hash_maps.values()).issubset(set(file_hashes.keys())), (
            f"hash_maps {hash_maps.values()} must be a subset of file_hashes keys {file_hashes.keys()}"
        )
        file_hashes = {k: v[0] if isinstance(v, list) else v for k, v in file_hashes.items()}
        main_hash = file_hashes["main_hash"]
        metadata_path = self.get_metadata_path(self.cache_root, main_hash)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        metadata = {
            "version": self.CACHE_VERSION,  # Add version info
        }

        for key in data.keys():
            hash_type = hash_maps[key]
            hash_value = file_hashes[hash_type]
            embedding = data[key].detach().cpu().to(torch.float16)
            cache_path = self.get_cache_embedding_path(key, hash_value)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(embedding, cache_path)
            metadata[key] = hash_value

        # Save img_shapes if provided (for multi-resolution support)
        if img_shapes is not None:
            if isinstance(img_shapes, torch.Tensor):
                metadata["img_shapes"] = img_shapes.tolist()
            else:
                metadata["img_shapes"] = img_shapes

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_cache(self, data, replace_empty_embeddings: bool = False, prompt_empty_drop_keys: list[str] = None):
        """Load cache embeddings and img_shapes"""
        main_hash = data["file_hashes"]["main_hash"]
        metadata_path = self.get_metadata_path(self.cache_root, main_hash)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Load embeddings
        for embedding_key, hash_value in metadata.items():
            # Skip metadata fields
            if embedding_key in ["version", "img_shapes"]:
                continue
            if embedding_key.startswith("empty_"):
                continue

            cache_path = self.get_cache_embedding_path(embedding_key, hash_value)
            embedding = torch.load(cache_path, map_location="cpu", weights_only=False)
            data[embedding_key] = embedding

        # # Load img_shapes
        # if "img_shapes" in metadata:
        #     data["img_shapes"] = torch.tensor(metadata["img_shapes"])

        if replace_empty_embeddings and prompt_empty_drop_keys:
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
        metadatas = glob.glob(os.path.join(cache_root, "metadata", "*.json"))
        return len(metadatas) > 0

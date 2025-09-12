import torch
from blake3 import blake3
from PIL import Image, ImageOps
import imagehash
import PIL
import os
from typing import Union
import hashlib


def sample_indices_per_rank(accelerator, dataset_size: int, num_samples: int,
                            *, seed: int = 0, replacement: bool = False,
                            global_shuffle: bool = True):
    """
    返回当前 rank 的 num_samples 个索引（默认无放回、不与其他 rank 重叠）。
    global_shuffle=True 时先做全局 randperm 再 stride 切片以避免排序偏置。
    """
    rank = accelerator.process_index
    world = accelerator.num_processes

    # 1) 构造该 rank 的候选池（不重叠）
    if global_shuffle:
        g0 = torch.Generator().manual_seed(seed)         # 所有 rank 相同 -> 相同 perm
        perm = torch.randperm(dataset_size, generator=g0)
        pool = perm[rank::world]
    else:
        pool = torch.arange(rank, dataset_size, world)

    # 2) 在各自池内抽样
    g = torch.Generator().manual_seed(seed + rank)       # 各 rank 不同打乱
    if replacement:
        idx = pool[torch.randint(len(pool), (num_samples,), generator=g)]
    else:
        if num_samples > len(pool):
            raise ValueError(f"rank{rank}: need {num_samples}, but only {len(pool)} available. "
                             f"Set replacement=True or reduce num_samples.")
        perm_local = torch.randperm(len(pool), generator=g)
        idx = pool[perm_local[:num_samples]]

    return idx.tolist()


def content_hash_blake3(path, chunk_size=1 << 20):
    h = blake3()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()  # 64位 hex


def phash_hex_from_image(img: Image.Image) -> str:
    im = ImageOps.exif_transpose(img)   # 纠正EXIF方向
    return str(imagehash.phash(im))     # 16 hex = 64 bit


def hash_string_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def extract_file_hash(image_path: Union[str, PIL.Image.Image]) -> str:
    if isinstance(image_path, PIL.Image.Image):
        return phash_hex_from_image(image_path)
    elif os.path.exists(image_path):
        return content_hash_blake3(image_path)
    else:
        raise ValueError(f"Invalid image path: {image_path}")

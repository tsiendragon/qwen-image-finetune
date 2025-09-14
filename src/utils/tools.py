import torch
from blake3 import blake3
from PIL import Image, ImageOps
import imagehash
import PIL
import os
from typing import Union
import hashlib
import importlib
import subprocess


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


def calculate_md5(file_path):
    # Open the file in binary mode
    with open(file_path, 'rb') as file:
        # Create an MD5 hash object
        md5_hash = hashlib.md5()

        # Read the file in chunks to handle large files efficiently
        chunk_size = 65536  # 64 KB
        while chunk := file.read(chunk_size):
            md5_hash.update(chunk)

        # Return the hexadecimal representation of the MD5 hash
        return md5_hash.hexdigest()


def phash_hex_from_image(img: Image.Image) -> str:
    im = ImageOps.exif_transpose(img)   # 纠正EXIF方向
    return str(imagehash.phash(im))     # 16 hex = 64 bit


def hash_string_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def extract_file_hash(image_path: Union[str, PIL.Image.Image]) -> str:
    if isinstance(image_path, PIL.Image.Image):
        return phash_hex_from_image(image_path)
    elif os.path.exists(image_path):
        return calculate_md5(image_path)
    else:
        raise ValueError(f"Invalid image path: {image_path}")


def _git(cmd, default=""):
    try:
        return subprocess.check_output(["git"] + cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def get_git_info():
    # commit hash
    commit = _git(["rev-parse", "HEAD"])
    short_commit = _git(["rev-parse", "--short", "HEAD"])

    # branch (detached HEAD -> empty)
    branch = _git(["symbolic-ref", "--short", "-q", "HEAD"])

    # remote url (try origin, else first remote)
    remote = _git(["remote", "get-url", "origin"])
    if not remote:
        remotes = _git(["remote"]).splitlines()
        if remotes:
            remote = _git(["remote", "get-url", remotes[0]])

    # repo root (optional)
    root = _git(["rev-parse", "--show-toplevel"])

    return {
        "commit": commit,
        "short_commit": short_commit,
        "branch": branch or None,  # None if detached
        "remote": remote or None,
        "root": root or None,
    }


def instantiate_class(class_path, init_args):
    """load processor"""
    module_path, module_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    instance = getattr(module, module_name)(init_args)
    return instance


if __name__ == "__main__":
    print(get_git_info())

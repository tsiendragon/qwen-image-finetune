import hashlib
import importlib
import os
import subprocess
from typing import Literal

import imagehash
import PIL
import torch
from blake3 import blake3
from PIL import Image, ImageOps
from torch.nn import functional as F  # NOQA


Layout = Literal["HW", "CHW", "HWC", "BCHW", "BHWC"]
Range = Literal["0-1", "-1-1", "0-255", "unknown"]


def sample_indices_per_rank(
    accelerator,
    dataset_size: int,
    num_samples: int,
    *,
    seed: int = 0,
    replacement: bool = False,
    global_shuffle: bool = True,
):
    """
    返回当前 rank 的 num_samples 个索引（默认无放回、不与其他 rank 重叠）。
    global_shuffle=True 时先做全局 randperm 再 stride 切片以避免排序偏置。
    """
    rank = accelerator.process_index
    world = accelerator.num_processes

    # 1) 构造该 rank 的候选池（不重叠）
    if global_shuffle:
        g0 = torch.Generator().manual_seed(seed)  # 所有 rank 相同 -> 相同 perm
        perm = torch.randperm(dataset_size, generator=g0)
        pool = perm[rank::world]
    else:
        pool = torch.arange(rank, dataset_size, world)

    # 2) 在各自池内抽样
    g = torch.Generator().manual_seed(seed + rank)  # 各 rank 不同打乱
    if replacement:
        idx = pool[torch.randint(len(pool), (num_samples,), generator=g)]
    else:
        if num_samples > len(pool):
            raise ValueError(
                f"rank{rank}: need {num_samples}, but only {len(pool)} available. "
                f"Set replacement=True or reduce num_samples."
            )
        perm_local = torch.randperm(len(pool), generator=g)
        idx = pool[perm_local[:num_samples]]

    return idx.tolist()


def content_hash_blake3(path, chunk_size=1 << 20):
    h = blake3()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()  # 64位 hex


def calculate_md5(file_path):
    # Open the file in binary mode
    with open(file_path, "rb") as file:
        # Create an MD5 hash object
        md5_hash = hashlib.md5()

        # Read the file in chunks to handle large files efficiently
        chunk_size = 65536  # 64 KB
        while chunk := file.read(chunk_size):
            md5_hash.update(chunk)

        # Return the hexadecimal representation of the MD5 hash
        return md5_hash.hexdigest()


def phash_hex_from_image(img: Image.Image) -> str:
    im = ImageOps.exif_transpose(img)  # 纠正EXIF方向
    return str(imagehash.phash(im))  # 16 hex = 64 bit


def hash_string_md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def extract_file_hash(image_path: str | PIL.Image.Image) -> str:
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


def instantiate_class(class_path: str, init_args):
    """load processor"""
    module_path, module_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    if isinstance(init_args, dict):
        instance = getattr(module, module_name)(**init_args)
    else:
        instance = getattr(module, module_name)(init_args)
    return instance


def _looks_like_hw(x: int) -> bool:
    # typical image spatial size
    return 8 <= x <= 32768


def _infer_layout(t: torch.Tensor) -> Layout | None:
    s = tuple(t.shape)
    nd = t.ndim
    chans = {1, 3, 4}

    if nd == 2:
        return "HW"

    if nd == 3:
        c_first = s[0] in chans and _looks_like_hw(s[1]) and _looks_like_hw(s[2])
        c_last = s[2] in chans and _looks_like_hw(s[0]) and _looks_like_hw(s[1])
        if c_first and not c_last:
            return "CHW"
        if c_last and not c_first:
            return "HWC"
        # tie-break: prefer CHW (more common in PyTorch)
        if c_first and c_last:
            return "CHW"
        return None

    if nd == 4:
        c_first = s[1] in chans and _looks_like_hw(s[2]) and _looks_like_hw(s[3])
        c_last = s[3] in chans and _looks_like_hw(s[1]) and _looks_like_hw(s[2])
        if c_first and not c_last:
            return "BCHW"
        if c_last and not c_first:
            return "BHWC"
        # tie-break: prefer BCHW
        if c_first and c_last:
            return "BCHW"
        return None

    return None


def _infer_range(t: torch.Tensor) -> Range:
    # quick rules based on dtype and min/max
    if t.dtype in (torch.uint8,):
        return "0-255"

    # sample to avoid huge reductions
    with torch.no_grad():
        v = t
        if v.numel() > 2_000_000:
            idx = torch.randperm(v.numel(), device=v.device)[:2_000_000]
            v = v.reshape(-1)[idx]
        vmin = torch.min(v.float()).item()
        vmax = torch.max(v.float()).item()

    def within(x, low, high):  # inclusive slack
        return low <= x <= high

    # tolerate small numeric noise
    if within(vmin, -1.05, -0.4) and within(vmax, 0.4, 1.05):
        return "-1-1"
    if within(vmin, -1e-6, 0.1) and within(vmax, 0.9, 1.05):
        return "0-1"
    if within(vmin, -1e-6, 0.6) and within(vmax, 0.2, 1 + 1e-6):
        return "0-1"
    if within(vmin, -1e-6, 5.0) and within(vmax, 1.5, 260.0):
        return "0-255"
    print("vmin, vmax", vmin, vmax, t.shape)
    return "unknown"


def infer_image_tensor(t: torch.Tensor) -> dict[str, object]:
    """
    Infer image tensor layout and numeric range.

    Returns:
      {
        'layout': 'BCHW'|'BHWC'|'CHW'|'HWC'|'HW'|None,
        'batch': int|None,
        'channels': int|None,
        'height': int|None,
        'width': int|None,
        'dtype': torch.dtype,
        'range': '0-1'|'-1-1'|'0-255'|'unknown'
      }
    """
    if not torch.is_tensor(t):
        raise TypeError("Expected a torch.Tensor")

    layout = _infer_layout(t)
    h = w = c = b = None
    s = tuple(t.shape)

    if layout == "HW":
        h, w = s
    elif layout == "CHW":
        c, h, w = s
    elif layout == "HWC":
        h, w, c = s
    elif layout == "BCHW":
        b, c, h, w = s
    elif layout == "BHWC":
        b, h, w, c = s

    rng = _infer_range(t)

    return {
        "layout": layout,
        "batch": b,
        "channels": c,
        "height": h,
        "width": w,
        "dtype": t.dtype,
        "range": rng,
    }


def calculate_sha256_file(filepath):
    sha256_hash = hashlib.sha256()
    # Open the file in binary read mode ('rb')
    with open(filepath, "rb") as f:
        # Read in chunks
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def extract_batch_field(embeddings: dict, key: str, batch_idx: int):
    """Extract a field value for a specific batch index from embeddings

    This helper function handles different data types (list, tensor, scalar)
    uniformly, making it easy to extract per-sample values from batch data.
    It's commonly used in multi-resolution training to handle variable-sized
    batches where different samples may have different dimensions.

    Args:
        embeddings: Dictionary containing batch data
        key: Field name to extract
        batch_idx: Index of the sample in the batch (0-based)

    Returns:
        The value for the specified sample. Type depends on the field:
        - For list/tuple: returns embeddings[key][batch_idx]
        - For multi-element tensor: returns embeddings[key][batch_idx].item()
        - For scalar: returns embeddings[key] (same for all samples)

    Examples:
        >>> # List: different values per sample
        >>> embeddings = {"height": [512, 640, 768], "width": 512}
        >>> extract_batch_field(embeddings, "height", 0)  # 512
        >>> extract_batch_field(embeddings, "height", 1)  # 640
        >>> extract_batch_field(embeddings, "height", 2)  # 768

        >>> # Scalar: same value for all samples
        >>> extract_batch_field(embeddings, "width", 0)   # 512
        >>> extract_batch_field(embeddings, "width", 1)   # 512

        >>> # Tensor: extract specific index
        >>> embeddings = {"height": torch.tensor([512, 640, 768])}
        >>> extract_batch_field(embeddings, "height", 1)  # 640

    Note:
        This function is particularly useful in multi-resolution training
        where batch samples may have different resolutions, and we need
        to extract per-sample metadata (height, width, n_controls, etc.).
    """
    value = embeddings[key]
    if isinstance(value, (list, tuple)):
        return value[batch_idx]
    elif isinstance(value, torch.Tensor) and value.numel() > 1:
        return value[batch_idx].item()
    else:
        return value  # Scalar - same for all samples


def pad_latents_for_multi_res(
    latents: list[torch.Tensor],
    max_seq_len: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad latents to uniform sequence length for multi-resolution training

    This function takes a list of latent tensors with varying sequence lengths
    and pads them to a uniform maximum length. This is essential for batch
    processing in multi-resolution training where different images have
    different resolutions and thus different latent sequence lengths.

    Args:
        latents: List of latent tensors, each with shape [seq_i, C] where seq_i varies
        max_seq_len: Maximum sequence length to pad to

    Returns:
        A tuple of (padded_latents, attention_mask):
        - padded_latents: Padded latents tensor [B, max_seq, C]
        - attention_mask: Binary mask [B, max_seq] where True=valid, False=padded

    Raises:
        ValueError: If latents list is empty
        ValueError: If any latent tensor is not 2D
        ValueError: If channel dimensions don't match across latents
        ValueError: If any latent sequence length exceeds max_seq_len

    Note:
        - Device and dtype are automatically inferred from the first latent tensor
        - All latents will be moved to the same device and dtype as the first one
        - The attention mask is critical for preventing padded positions from
          affecting model predictions and loss calculations

    Examples:
        >>> # Basic usage with varying sequence lengths
        >>> latents = [torch.randn(100, 64), torch.randn(150, 64), torch.randn(120, 64)]
        >>> padded, mask = pad_latents_for_multi_res(latents, 150)
        >>> padded.shape  # torch.Size([3, 150, 64])
        >>> mask.shape    # torch.Size([3, 150])
        >>> mask.sum(dim=1)  # tensor([100, 150, 120]) - number of valid tokens per sample

        >>> # Using mask to compute loss only on valid tokens
        >>> loss = F.mse_loss(pred[mask], target[mask])

        >>> # Or mask before computing loss
        >>> masked_pred = pred * mask.unsqueeze(-1)
        >>> masked_target = target * mask.unsqueeze(-1)
    """
    batch_size = len(latents)
    if batch_size == 0:
        raise ValueError("Cannot pad empty latent list")
    if max_seq_len is None:
        max_seq_len = max(lat.shape[0] for lat in latents)

    # Infer device and dtype from first latent
    device = latents[0].device
    dtype = latents[0].dtype
    channels = latents[0].shape[-1]

    # Validate input dimensions
    for i, lat in enumerate(latents):
        if lat.dim() != 2:
            raise ValueError(f"Expected 2D latent tensor [seq, C], got shape {lat.shape} at index {i}")
        if lat.shape[-1] != channels:
            raise ValueError(f"Channel mismatch: expected {channels}, got {lat.shape[-1]} at index {i}")
        if lat.shape[0] > max_seq_len:
            raise ValueError(f"Latent sequence length {lat.shape[0]} exceeds max_seq_len {max_seq_len} at index {i}")

    # Initialize output tensors
    padded_latents = torch.zeros(batch_size, max_seq_len, channels, device=device, dtype=dtype)
    attention_mask = torch.zeros(batch_size, max_seq_len, device=device, dtype=torch.bool)

    # Fill in valid data and create mask
    for i, lat in enumerate(latents):
        seq_len = lat.shape[0]
        padded_latents[i, :seq_len] = lat.to(device=device, dtype=dtype)
        attention_mask[i, :seq_len] = True

    return padded_latents, attention_mask


def pad_to_max_shape(tensors: list[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    """
    将一组不同形状但维度数一致的张量沿各维右侧填充到最大形状后堆叠。
    Args:
        tensors: List[Tensor]，形状可能不同
        padding_value: 填充值，默认 0
    Returns:
        Tensor: 形状为 (N, ...) 的批次张量

    示例:
    ```python
    x = [torch.ones(2,3), torch.zeros(4,1)]
    y = pad_to_max_shape(x, padding_value=0)  # -> (2,4,3)
    ```
    """
    # Find maximum shape
    max_shape = [max(sizes) for sizes in zip(*[t.shape for t in tensors], strict=False)]
    padded = []
    for t in tensors:
        pad_sizes = []
        for i in range(len(max_shape) - 1, -1, -1):
            diff = max_shape[i] - t.shape[i]
            pad_sizes.extend([0, diff])  # (left, right) per dimension
        padded_tensor = F.pad(t, pad_sizes, value=padding_value)
        padded.append(padded_tensor)

    return torch.stack(padded, dim=0)


if __name__ == "__main__":
    print(get_git_info())

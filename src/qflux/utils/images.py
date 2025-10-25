import math

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F  # NOQA


def resize_bhw(x: torch.Tensor, h: int, w: int, mode="bilinear"):
    x = x.unsqueeze(1)
    # [B, 1, H, W]
    x = F.interpolate(
        x,
        size=(h, w),
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
        antialias=True if mode in {"bilinear", "bicubic"} and (h < x.shape[-2] or w < x.shape[-1]) else False,
    )
    return x.squeeze(1)


def make_image_shape_devisible(width, height, vae_scale_factor: int) -> tuple[int, int]:
    """make width and height devisible by vae_scale_factor * 2"""
    multiple_of = vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of
    return width, height


def make_image_devisible(image: torch.Tensor | PIL.Image.Image | np.ndarray, vae_scale_factor: int) -> torch.Tensor:
    """make image devisible by vae_scale_factor * 2, suppose image is B,C,H,W"""

    if isinstance(image, torch.Tensor):
        height, width = image.shape[2:]
        width, height = make_image_shape_devisible(width, height, vae_scale_factor)
        image = F.interpolate(image, size=(width, height), mode="bilinear")
    elif isinstance(image, PIL.Image.Image):
        width, height = image.size
        width, height = make_image_shape_devisible(width, height, vae_scale_factor)
        image = image.resize((width, height))
    elif isinstance(image, np.ndarray):  # H,W,C
        height, width = image.shape[:2]
        width, height = make_image_shape_devisible(width, height, vae_scale_factor)
        image = cv2.resize(image, (width, height))
    return image


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height, None


def calculate_best_resolution(width: int, height: int, best_resolution: int = 1024 * 1024):
    calculated_width, calculated_height, _ = calculate_dimensions(best_resolution, width / height)
    return calculated_width, calculated_height


def image_adjust_best_resolution(image: torch.Tensor | PIL.Image.Image | np.ndarray):
    """Preprocess images for caching, suppose image is B,C,H,W
    tensor: B,C,H,W
    PIL.Image.Image: H,W,C
    np.ndarray: H,W,C
    """
    if isinstance(image, torch.Tensor):
        height, width = image.shape[2:]
    elif isinstance(image, PIL.Image.Image):
        width, height = image.size
    elif isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    calculated_width, calculated_height = calculate_best_resolution(width, height)
    if isinstance(image, torch.Tensor):
        new_image = F.interpolate(image, size=(calculated_height, calculated_width), mode="bilinear")
    elif isinstance(image, PIL.Image.Image):
        new_image = image.resize((calculated_width, calculated_height), PIL.Image.Resampling.BICUBIC)
    elif isinstance(image, np.ndarray):
        new_image = cv2.resize(image, (calculated_width, calculated_height), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    return new_image

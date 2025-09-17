import torch.nn.functional as F
import torch
from typing import Tuple, Union
import PIL
import numpy as np
import cv2


def resize_bhw(x: torch.Tensor, h: int, w: int, mode="bilinear"):
    x = x.unsqueeze(1)
    # [B, 1, H, W]
    x = F.interpolate(
        x, size=(h, w), mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
        antialias=True if mode in {"bilinear", "bicubic"} and (h < x.shape[-2] or w < x.shape[-1]) else False,
    )
    return x.squeeze(1)


def make_image_shape_devisible(width, height, vae_scale_factor: int) -> Tuple[int, int]:
    """make width and height devisible by vae_scale_factor * 2"""
    multiple_of = vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of
    return width, height


def make_image_devisible(image: Union[torch.Tensor, PIL.Image, np.ndarray], vae_scale_factor: int) -> torch.Tensor:
    """make image devisible by vae_scale_factor * 2, suppose image is B,C,H,W"""
    width, height = image.shape[2:]
    width, height = make_image_shape_devisible(width, height, vae_scale_factor)
    if isinstance(image, torch.Tensor):
        image = F.interpolate(image, size=(width, height), mode='bilinear')
    elif isinstance(image, PIL.Image):
        image = image.resize((width, height))
    elif isinstance(image, np.ndarray):
        image = cv2.resize(image, (width, height))
    return image
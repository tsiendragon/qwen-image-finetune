import torch.nn.functional as F
import torch


def resize_bhw(x: torch.Tensor, h: int, w: int, mode="bilinear"):
    x = x.unsqueeze(1)                        # [B, 1, H, W]
    x = F.interpolate(
        x, size=(h, w), mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
        antialias=True if mode in {"bilinear", "bicubic"} and (h < x.shape[-2] or w < x.shape[-1]) else False,
    )
    return x.squeeze(1)

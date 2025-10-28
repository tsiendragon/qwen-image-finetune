"""
Loss functions for multi-resolution training
"""

from qflux.losses.attention_mask_loss import AttentionMaskMseLoss
from qflux.losses.edit_mask_loss import MaskEditLoss
from qflux.losses.mse_loss import MseLoss


__all__ = [
    "MaskEditLoss",
    "AttentionMaskMseLoss",
    "MseLoss",
]

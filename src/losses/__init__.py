"""
Loss functions for multi-resolution training
"""

from src.losses.edit_mask_loss import MaskEditLoss
from src.losses.attention_mask_loss import AttentionMaskMseLoss
from src.losses.mse_loss import MseLoss


__all__ = [
    'MaskEditLoss',
    'AttentionMaskMseLoss',
    'MseLoss',
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def map_mask_to_latent(image_mask: Tensor) -> Tensor:
    """
    Args:
        image_mask: [B, H, W] - Binary mask in image space
    Returns:
        latent_mask: [B, seq_len] - Weights for packed latent
    """
    B, H, W = image_mask.shape

    # Step 1: VAE-aligned downsampling
    # [B, H, W] → [B, H/8, W/8]
    latent_h, latent_w = H // 8, W // 8
    mask_latent = F.avg_pool2d(image_mask.float().unsqueeze(1), kernel_size=8, stride=8).squeeze(
        1
    )  # [B, latent_h, latent_w]

    # Step 2: Packing simulation
    # [B, latent_h, latent_w] → [B, latent_h//2, latent_w//2, 4]
    # First reshape to separate 2x2 patches, then fold
    patches = mask_latent.reshape(B, latent_h // 2, 2, latent_w // 2, 2)
    patches = patches.permute(0, 1, 3, 2, 4).contiguous().view(B, latent_h // 2, latent_w // 2, 4)

    # Step 3: Patch-wise maximum (preserve text regions)
    # [B, latent_h//2, latent_w//2, 4] → [B, latent_h//2, latent_w//2]
    packed_mask = patches.max(dim=-1)[0]

    # Step 4: Flatten to sequence
    # [B, latent_h//2, latent_w//2] → [B, seq_len]
    seq_len = (latent_h // 2) * (latent_w // 2)
    return packed_mask.view(B, seq_len)


class MaskEditLoss(nn.Module):
    def __init__(self, forground_weight=2.0, background_weight=1.0):
        super().__init__()
        self.forground_weight = forground_weight
        self.background_weight = background_weight

    def forward(self, model_pred, target, weighting=None, edit_mask=None, reduction="mean", **kwargs):
        """
        计算mask加权的loss
        Args:
            edit_mask: [B, seq_len] - 二进制掩码，1表示修改区域，0表示背景区域
            model_pred: [B, seq_len, channels] - 模型预测结果
            target: [B, seq_len, channels] - 目标值
            weighting: [B, seq_len, 1] - 可选的时间步权重
            reduction: str - 'none', 'mean', or 'sum'
                'none': 返回 [B, seq_len, channels]
                'mean': 返回标量，按所有元素求均值
                'sum': 返回标量，按所有元素求和
            **kwargs: Additional arguments (ignored for compatibility with other loss functions)

        Returns:
            torch.Tensor - 加权后的loss值
        """
        # 计算基础element-wise loss
        element_loss = (model_pred.float() - target.float()) ** 2

        # 如果有weighting，应用到element_loss
        B, T, C = model_pred.shape
        if weighting is not None:
            element_loss = weighting.float() * element_loss
        if edit_mask is None:
            edit_mask = torch.ones((B, T), dtype=torch.float32, device=model_pred.device)

        # 创建权重掩码：文本区域权重更高
        # edit_mask: [B, seq_len] -> weight_mask: [B, seq_len, 1]
        weight_mask = edit_mask.float() * self.forground_weight + (1 - edit_mask.float()) * self.background_weight
        weight_mask = weight_mask.unsqueeze(-1)  # [B, seq_len, 1]

        # 应用mask权重
        weighted_loss = element_loss * weight_mask

        # 根据 reduction 参数返回不同结果
        if reduction == "none":
            return weighted_loss  # [B, seq_len, channels]
        elif reduction == "sum":
            return weighted_loss.sum()
        elif reduction == "mean":
            # 聚合loss：先按序列维度求均值，再按batch维度求均值
            loss = torch.mean(weighted_loss.reshape(target.shape[0], -1), 1).mean()
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be 'none', 'mean', or 'sum'.")


if __name__ == "__main__":
    original_mask = torch.randn(1, 832, 576)
    mask = torch.randn(1, 1872)
    model_pred = torch.randn(1, 1872, 64)
    target = torch.randn(1, 1872, 64)
    mask2 = map_mask_to_latent(original_mask)
    print(mask2.shape)
    criterion = MaskEditLoss()
    loss = criterion(mask2, model_pred, target)
    print(loss)

    loss = criterion(mask, model_pred, target)
    print(loss)

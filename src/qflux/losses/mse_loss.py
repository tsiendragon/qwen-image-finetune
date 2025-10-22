"""
MSE Loss with optional weighting support
"""

import torch
import torch.nn as nn


class MseLoss(nn.Module):
    """
    Mean Squared Error Loss with optional weighting support.

    This is a wrapper around torch.nn.functional.mse_loss that supports
    element-wise weighting and is compatible with the auto-feed system.

    Args:
        reduction (str): Reduction method: 'mean' (default), 'sum', or 'none'

    Forward Args:
        model_pred (torch.Tensor): Model predictions [B, T, C]
        target (torch.Tensor): Target values [B, T, C]
        weighting (torch.Tensor, optional): Element-wise weights [B, T, 1] or [B, T, C]
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        torch.Tensor: Loss value

    Examples:
        >>> loss_fn = MseLoss(reduction='mean')
        >>> model_pred = torch.randn(2, 100, 64)
        >>> target = torch.randn(2, 100, 64)
        >>> loss = loss_fn(model_pred, target)
        >>> print(loss.item())

        >>> # With weighting
        >>> weighting = torch.ones(2, 100, 1)
        >>> loss = loss_fn(model_pred, target, weighting=weighting)
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction '{reduction}': must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor = None,
        **kwargs,  # Accept but ignore extra arguments for compatibility
    ) -> torch.Tensor:
        """
        Compute MSE loss with optional weighting.

        Args:
            model_pred: Model predictions [B, T, C] or any shape
            target: Target values, same shape as model_pred
            weighting: Optional element-wise weights, broadcastable to model_pred shape
            **kwargs: Additional arguments (ignored for compatibility with other loss functions)

        Returns:
            Loss tensor (scalar if reduction='mean'/'sum', same shape as input if reduction='none')
        """
        if model_pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: model_pred {model_pred.shape} vs target {target.shape}")

        if weighting is None:
            # Standard MSE loss
            loss = torch.nn.functional.mse_loss(model_pred, target, reduction=self.reduction)
        else:
            # Weighted MSE loss
            element_loss = (model_pred.float() - target.float()) ** 2
            weighted_loss = weighting.float() * element_loss

            if self.reduction == "none":
                loss = weighted_loss
            elif self.reduction == "sum":
                loss = weighted_loss.sum()
            else:  # reduction == 'mean'
                # Mean over all elements
                loss = torch.mean(weighted_loss.reshape(target.shape[0], -1), dim=1).mean()

        return loss

    def extra_repr(self) -> str:
        """Return string representation of module parameters"""
        return f"reduction='{self.reduction}'"

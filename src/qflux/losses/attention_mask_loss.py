"""
Channel-Invariant Token Loss for Multi-Resolution Training

This module implements a loss function where:
1. Loss is computed element-wise (MSE)
2. Optional element-wise weighting is applied
3. Edit mask provides foreground/background weighting
4. Attention mask filters out padding tokens
5. Loss is averaged over channels first (token-level), then over valid tokens

Mathematical Formulation
========================

Let:
- ŷ_{b,t,c}: model prediction
- y_{b,t,c}: target
- a_{b,t} ∈ {0,1}: attention mask (1 = valid token)
- m_{b,t} ∈ {0,1}: edit mask (1 = foreground/edit region; if None, treated as all 1s)
- ω_{b,t,c}: optional element-wise weighting (default: 1)
- fg, bg: foreground/background weights

Step 1: Element-wise loss
    ℓ_{b,t,c} = (ŷ_{b,t,c} - y_{b,t,c})²

Step 2: Edit weighting
    w^{edit}_{b,t} = fg · m_{b,t} + bg · (1 - m_{b,t})

Step 3: Token-level loss (channel mean)
    ℓ̄_{b,t} = (1/C) Σ_c ℓ_{b,t,c} · ω_{b,t,c} · w^{edit}_{b,t}

Step 4: Final loss (average over valid tokens only)
    L = [Σ_b Σ_t a_{b,t} · ℓ̄_{b,t}] / [Σ_b Σ_t a_{b,t} + ε]

Key Properties
==============
1. **Channel-Invariant**: Loss scale is independent of channel dimension C
2. **Token-First**: Each token contributes equally regardless of its channel dimension
3. **Padding-Aware**: Only valid tokens (attention_mask=1) contribute to loss
4. **Edit-Aware**: Foreground regions can be weighted more heavily than background
5. **Flexible Weighting**: Supports element-wise weighting (e.g., timestep weights)

Example Usage
=============
    >>> loss_fn = ChannelInvariantTokenLoss(
    ...     foreground_weight=2.0,
    ...     background_weight=1.0
    ... )
    >>>
    >>> # Basic usage with attention mask only
    >>> loss = loss_fn(model_pred, target, attention_mask)
    >>>
    >>> # With edit mask for region-specific weighting
    >>> loss = loss_fn(model_pred, target, attention_mask, edit_mask)
    >>>
    >>> # With additional timestep weighting
    >>> loss = loss_fn(model_pred, target, attention_mask, edit_mask, weighting)

Comparison with Standard MSE
=============================
Standard MSE averages over all elements (B×T×C), making loss scale dependent on C:
    L_MSE = (1/BTC) Σ_{b,t,c} ℓ_{b,t,c}

Channel-Invariant Token Loss averages channels first, then tokens:
    L_token = (1/Σa) Σ_{b,t} a_{b,t} · [(1/C) Σ_c ℓ_{b,t,c}]

Benefits:
- Consistent loss scale across different channel dimensions
- Each token contributes equally to the loss
- More interpretable: loss represents average per-token error
"""

import torch
import torch.nn as nn


class AttentionMaskMseLoss(nn.Module):
    """Channel-invariant token loss with edit mask and attention mask support

    This loss function computes MSE at the element level, then:
    1. Averages across channels for each token (channel-invariant)
    2. Applies edit mask weighting (foreground vs background)
    3. Filters out padding using attention mask
    4. Averages over valid tokens only

    The resulting loss scale is independent of the channel dimension C,
    making it more stable and interpretable across different architectures.

    Args:
        foreground_weight: Weight for foreground/edit regions (default: 2.0)
        background_weight: Weight for background regions (default: 1.0)
        eps: Small constant for numerical stability (default: 1e-12)
        reduction: Reduction method: 'mean' (default), 'sum', or 'none'
                   - 'mean': Returns scalar loss (average over valid tokens)
                   - 'sum': Returns sum of token losses
                   - 'none': Returns per-token losses [B, T]

    Shape:
        - model_pred: [B, T, C] where B=batch, T=tokens, C=channels
        - target: [B, T, C]
        - attention_mask: [B, T] with values in {0, 1}, 1=valid token
        - edit_mask: [B, T] with values in {0, 1}, 1=foreground (optional)
        - weighting: [B, T, 1] or [B, T, C], broadcastable (optional)
        - Output: scalar (reduction='mean'/'sum') or [B, T] (reduction='none')

    Examples:
        >>> loss_fn = AttentionMaskMseLoss(
        ...     foreground_weight=2.0,
        ...     background_weight=1.0
        ... )
        >>>
        >>> # Multi-resolution batch with variable-length sequences
        >>> model_pred = torch.randn(4, 150, 64)  # 4 samples, max 150 tokens, 64 channels
        >>> target = torch.randn(4, 150, 64)
        >>>
        >>> # Attention mask: first 2 samples have 100 and 120 valid tokens
        >>> attention_mask = torch.zeros(4, 150, dtype=torch.bool)
        >>> attention_mask[0, :100] = True
        >>> attention_mask[1, :120] = True
        >>> attention_mask[2, :80] = True
        >>> attention_mask[3, :150] = True
        >>>
        >>> # Edit mask: marks foreground regions
        >>> edit_mask = torch.rand(4, 150) > 0.5
        >>>
        >>> # Compute loss
        >>> loss = loss_fn(model_pred, target, attention_mask, edit_mask)
        >>> loss.backward()  # Gradients only flow to valid tokens
    """

    def __init__(
        self,
        foreground_weight: float = 2.0,
        background_weight: float = 1.0,
        eps: float = 1e-12,
        reduction: str = "mean",
    ):
        super().__init__()
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight
        self.eps = eps

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction '{reduction}': must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor | None = None,
        attention_mask: torch.Tensor = None,
        edit_mask: torch.Tensor | None = None,
        **kwargs,  # Accept but ignore extra arguments for compatibility
    ) -> torch.Tensor:
        """Compute channel-invariant token loss

        Args:
            model_pred: Model predictions [B, T, C]
            target: Target values [B, T, C]
            attention_mask: Binary mask [B, T] where 1=valid token, 0=padding
            edit_mask: Optional binary mask [B, T] where 1=foreground, 0=background
                       If None, all tokens treated as foreground
            weighting: Optional element-wise weights [B, T, 1] or [B, T, C]

        Returns:
            Loss value (scalar if reduction='mean'/'sum', [B, T] if reduction='none')
        """
        # Validate shapes
        if model_pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: model_pred {model_pred.shape} vs target {target.shape}")

        B, T, C = model_pred.shape

        if attention_mask is None:
            attention_mask = torch.ones((B, T), dtype=torch.bool)

        if attention_mask.shape != (B, T):
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} incompatible with model_pred shape [{B}, {T}, {C}]"
            )
        # Step 1: Element-wise MSE
        element_loss = (model_pred.float() - target.float()) ** 2  # [B, T, C]

        # Step 2: Optional element-wise weighting
        if weighting is not None:
            element_loss = element_loss * weighting.float()  # Broadcasts to [B, T, C]

        # Step 3: Edit mask -> foreground/background weighting
        if edit_mask is None:
            # Treat all tokens as foreground
            # m = torch.ones_like(attention_mask, dtype=torch.float32, device=model_pred.device)
            edit_weight = torch.ones_like(attention_mask, dtype=torch.float32, device=model_pred.device)
            edit_weight = edit_weight.unsqueeze(-1)  # [B, T, 1]
        else:
            if edit_mask.shape != (B, T):
                raise ValueError(
                    f"edit_mask shape {edit_mask.shape} incompatible with attention_mask shape {attention_mask.shape}"
                )
            m = edit_mask.float().to(model_pred.device)
            edit_weight = (m * self.foreground_weight + (1.0 - m) * self.background_weight).unsqueeze(-1)  # [B, T, 1]
        # Compute edit weights: foreground gets higher weight

        weighted_loss = element_loss * edit_weight  # [B, T, C]

        # Step 4: Apply attention mask to filter padding
        attn = attention_mask.float().unsqueeze(-1).to(model_pred.device)  # [B, T, 1]
        masked_loss = weighted_loss * attn  # [B, T, C]

        # Step 5: Average over channels first -> token-level loss
        token_loss = masked_loss.mean(dim=2).to(model_pred.device)  # [B, T]

        # Step 6: Return based on reduction mode
        if self.reduction == "none":
            return token_loss

        # For 'mean' or 'sum', need to account for valid tokens
        num_valid_tokens = attention_mask.sum().to(model_pred.dtype)

        if num_valid_tokens.item() <= 0:
            # No valid tokens in batch
            return torch.zeros((), device=model_pred.device, dtype=model_pred.dtype)

        if self.reduction == "sum":
            return token_loss.sum()
        else:  # reduction == 'mean'
            return token_loss.sum() / (num_valid_tokens + self.eps)

    def extra_repr(self) -> str:
        """Return string representation of module parameters"""
        return (
            f"foreground_weight={self.foreground_weight}, "
            f"background_weight={self.background_weight}, "
            f"eps={self.eps}, "
            f"reduction='{self.reduction}'"
        )

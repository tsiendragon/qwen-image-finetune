# Attention Mask MSE Loss

## Overview

`AttentionMaskMseLoss` is a loss function designed for multi-resolution training where the loss scale should be independent of the channel dimension. This is particularly useful when:

1. Training models with different channel dimensions
2. Ensuring consistent loss scales across architectures
3. Making per-token contributions equal regardless of channel count
4. Training with variable-length sequences (padding support)

## Motivation

### Problem with Standard MSE Loss

Standard MSE loss averages over all elements (batch × tokens × channels):

```
L_MSE = (1/BTC) Σ_{b,t,c} (ŷ_{b,t,c} - y_{b,t,c})²
```

**Issues:**
- Loss scale depends on channel dimension C
- Doubling channels halves the loss magnitude
- Inconsistent loss scales make hyperparameter tuning difficult
- Each token's contribution varies with channel count

### Channel-Invariant Token Loss Solution

Our loss averages over channels first (token-level), then over tokens:

```
L_token = (1/T_valid) Σ_{valid tokens} [(1/C) Σ_c (ŷ - y)²]
```

**Benefits:**
- Loss scale is independent of C
- Each token contributes equally
- Consistent across different architectures
- More interpretable: represents average per-token error

## Mathematical Formulation

### Notation

- `ŷ_{b,t,c}`: Model prediction
- `y_{b,t,c}`: Target value
- `a_{b,t} ∈ {0,1}`: Attention mask (1 = valid token, 0 = padding)
- `m_{b,t} ∈ {0,1}`: Edit mask (1 = foreground, 0 = background)
- `ω_{b,t,c}`: Optional element-wise weighting (e.g., timestep weights)
- `fg`, `bg`: Foreground/background weights

### Step-by-Step Computation

**Step 1: Element-wise Loss**
```
ℓ_{b,t,c} = (ŷ_{b,t,c} - y_{b,t,c})²
```

**Step 2: Edit Weighting**
```
w^{edit}_{b,t} = fg · m_{b,t} + bg · (1 - m_{b,t})
```

**Step 3: Token-Level Loss (Channel Mean)**
```
ℓ̄_{b,t} = (1/C) Σ_c [ℓ_{b,t,c} · ω_{b,t,c} · w^{edit}_{b,t}]
```

**Step 4: Final Loss (Average Over Valid Tokens)**
```
L = [Σ_b Σ_t a_{b,t} · ℓ̄_{b,t}] / [Σ_b Σ_t a_{b,t} + ε]
```

## Usage Examples

### Basic Usage

```python
from qflux.losses import AttentionMaskMseLoss

# Create loss function
loss_fn = AttentionMaskMseLoss(
    foreground_weight=2.0,
    background_weight=1.0
)

# Model predictions and targets
model_pred = model(input)  # [B, T, C]
target = ...               # [B, T, C]

# Attention mask: 1 for valid tokens, 0 for padding
attention_mask = ...       # [B, T]

# Compute loss
loss = loss_fn(model_pred, target, attention_mask)
loss.backward()
```

### With Edit Mask

```python
# Edit mask: 1 for foreground/edit regions, 0 for background
edit_mask = ...  # [B, T], values in {0, 1}

loss = loss_fn(
    model_pred,
    target,
    attention_mask,
    edit_mask  # Foreground gets 2x weight, background 1x
)
```

### With Additional Weighting

```python
# Optional element-wise weighting (e.g., timestep weights)
weighting = ...  # [B, T, 1] or [B, T, C]

loss = loss_fn(
    model_pred,
    target,
    attention_mask,
    edit_mask,
    weighting
)
```

### Functional Interface

```python
from qflux.losses import masked_edit_token_mean_loss

loss = masked_edit_token_mean_loss(
    model_pred,
    target,
    attention_mask,
    edit_mask=None,  # Optional
    weighting=None,  # Optional
    foreground_weight=2.0,
    background_weight=1.0
)
```

### Multi-Resolution Training Example

```python
from qflux.losses import AttentionMaskMseLoss
from qflux.utils.tools import pad_latents_for_multi_res

# Create loss function
loss_fn = AttentionMaskMseLoss(
    foreground_weight=2.0,
    background_weight=1.0
)

# Variable-length latents
latents = [
    torch.randn(100, 64),  # Sample 1: 100 tokens
    torch.randn(150, 64),  # Sample 2: 150 tokens
    torch.randn(120, 64),  # Sample 3: 120 tokens
]

# Pad to uniform length
max_len = 150
padded_latents, attention_mask = pad_latents_for_multi_res(latents, max_len)
# padded_latents: [3, 150, 64]
# attention_mask: [3, 150], marks valid tokens

# Forward pass
model_pred = model(padded_latents)
target = ...

# Compute loss (only valid tokens contribute)
loss = loss_fn(model_pred, target, attention_mask)
```

## Key Features

### 1. Channel Invariance

```python
# Loss scale is consistent regardless of channel dimension
pred_32ch = torch.randn(B, T, 32)
pred_64ch = torch.randn(B, T, 64)

# Both have similar loss scales
loss_32 = loss_fn(pred_32ch, target_32ch, mask)
loss_64 = loss_fn(pred_64ch, target_64ch, mask)
# loss_32 and loss_64 have comparable magnitudes
```

### 2. Padding Support

```python
# Attention mask filters out padding tokens
attention_mask = torch.zeros(B, T, dtype=torch.bool)
attention_mask[:, :valid_len] = True

# Only valid tokens contribute to loss and gradients
loss = loss_fn(pred, target, attention_mask)
```

### 3. Edit-Aware Weighting

```python
# Emphasize foreground regions (e.g., edited areas)
loss_fn = AttentionMaskMseLoss(
    foreground_weight=2.0,  # Foreground gets 2x weight
    background_weight=1.0   # Background gets 1x weight
)

# edit_mask marks regions to emphasize
loss = loss_fn(pred, target, attention_mask, edit_mask)
```

### 4. Reduction Modes

```python
# Mean: returns scalar (average over valid tokens)
loss_fn_mean = ChannelInvariantTokenLoss(reduction='mean')
loss = loss_fn_mean(pred, target, mask)  # Scalar

# Sum: returns scalar (sum over valid tokens)
loss_fn_sum = ChannelInvariantTokenLoss(reduction='sum')
loss = loss_fn_sum(pred, target, mask)  # Scalar

# None: returns per-token losses
loss_fn_none = ChannelInvariantTokenLoss(reduction='none')
loss = loss_fn_none(pred, target, mask)  # [B, T]
```

## Comparison with Other Losses

### vs. Standard MSE Loss

| Aspect | Standard MSE | Channel-Invariant Token Loss |
|--------|--------------|------------------------------|
| Averaging | Over all elements (B×T×C) | Channels first, then tokens |
| Channel sensitivity | Loss scales with C | Independent of C |
| Token contribution | Varies with C | Equal per token |
| Interpretability | Element-level | Token-level |

**Example:**
```python
# Standard MSE
loss_mse = F.mse_loss(pred, target)  # Scales with C

# Channel-Invariant
loss_ci = loss_fn(pred, target, mask)  # Independent of C
```

### vs. MaskEditLoss

| Aspect | MaskEditLoss | Channel-Invariant Token Loss |
|--------|--------------|------------------------------|
| Averaging | All elements equally | Channels first, then tokens |
| Channel sensitivity | Scales with C | Independent of C |
| Edit weighting | ✓ | ✓ |
| Attention masking | Manual | Built-in |
| Multi-resolution | Requires wrapper | Native support |

### vs. compute_loss_multi_resolution

`compute_loss_multi_resolution` is a utility that can use either:
1. Standard MSE (channel-sensitive)
2. MaskEditLoss (channel-sensitive)

`AttentionMaskMseLoss` provides:
- Channel-invariant alternative
- Same edit weighting capabilities
- Same attention masking support
- More consistent loss scales

## Implementation Details

### Shape Requirements

- `model_pred`: `[B, T, C]` - Batch, Tokens, Channels
- `target`: `[B, T, C]`
- `attention_mask`: `[B, T]` - Boolean or 0/1 values
- `edit_mask`: `[B, T]` (optional) - 0/1 values
- `weighting`: `[B, T, 1]` or `[B, T, C]` (optional) - Broadcastable

### Gradient Flow

Only valid tokens (attention_mask=1) receive gradients:

```python
pred = torch.randn(B, T, C, requires_grad=True)
loss = loss_fn(pred, target, attention_mask)
loss.backward()

# pred.grad is non-zero only where attention_mask=1
assert torch.all(pred.grad[:, padding_positions] == 0)
```

### Numerical Stability

- Uses `eps` parameter to avoid division by zero
- Converts to float32 for computation
- Clamps denominators to prevent inf/nan

## When to Use

**Use Attention Mask MSE Loss when:**
- Training with multiple channel dimensions
- Need consistent loss scales across architectures
- Working with multi-resolution training
- Each token should contribute equally
- Want interpretable per-token errors

**Use Standard MSE when:**
- Channel dimension is fixed
- Fine-grained element-level loss is needed
- Backward compatibility is required

**Use MaskEditLoss when:**
- Edit-aware weighting is needed
- Channel sensitivity is acceptable
- Not using multi-resolution training

## Code Reference

- Implementation: `src/losses/attention_mask_loss.py`
- Tests: `tests/loss/test_attention_mask_loss.py`
- Integration: `src/losses/__init__.py`

## References

- Multi-Resolution Training: See `docs/cache-system.md`
- Padding Utilities: See `src/utils/tools.py` (`pad_latents_for_multi_res`)
- Edit Mask Loss: See `src/losses/edit_mask_loss.py`

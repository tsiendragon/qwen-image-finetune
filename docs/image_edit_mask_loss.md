# Image Edit Mask Loss Enhancement

## Overview

This document describes an optional mask-weighted loss component designed to enhance image editing performance by focusing training attention on edit regions while maintaining full backward compatibility with existing training pipelines.

![Mask Loss Overview](images/image-36.png)

A dataset that supports mask-based training is available at TsienDragon/character-composition: https://huggingface.co/datasets/TsienDragon/character-composition

## Problem Statement

Current image editing training approaches treat all image regions equally during loss computation, leading to suboptimal performance in scenarios where only specific edit regions differ between source and target images. This uniform weighting dilutes the training signal from critical edit areas, resulting in slower convergence and reduced editing quality.

## Solution Architecture

### 1. Mask Processing Pipeline

The mask processing pipeline transforms image-space binary masks into latent-space sequence weights that align with the model's packed latent representation.

#### Step-by-Step Transformation (Example: 832×576 → 1024 sequence)

```
Step 1: Input Image Space Mask
├─ Shape: [B, 832, 576]
├─ Format: Binary mask (0=background, 1=edit_region)
└─ Purpose: Edit regions marked as 1, background as 0

Step 2: VAE Downsampling Simulation
├─ Operation: F.avg_pool2d(mask, kernel_size=8, stride=8)
├─ Shape: [B, 832, 576] → [B, 104, 72]
├─ Rationale: Match VAE encoder's 8x spatial compression
├─ Preservation: Average pooling maintains edit region density
└─ Result: Latent-space aligned mask

Step 3: Packing Simulation (2×2 Patch Merge)
├─ Operation: reshape + permute + max_pooling
├─ Shape: [B, 104, 72] → [B, 52, 36] → [B, 1872]
├─ Process:
│   ├─ Reshape: [B, 104, 72] → [B, 52, 2, 36, 2] → [B, 52, 36, 4]
│   ├─ Max pooling: Take max value per 2×2 patch → [B, 52, 36]
│   └─ Flatten: [B, 52, 36] → [B, 1872]
├─ Rationale: Simulate transformer's 2×2 patchify operation
└─ Result: seq_len = (104÷2) × (72÷2) = 52 × 36 = 1872

Step 4: Alignment with Packed Latents
├─ Image latent: [B, 16, 104, 72] → pack → [B, 1872, 64]
├─ Mask latent: [B, 832, 576] → process → [B, 1872]
├─ Alignment: Both sequences have matching length (seq_len = 1872)
└─ Usage: mask[i] weights latent[i, :] across all 64 channels
```

#### Mathematical Formulation

```python
def map_mask_to_latent(image_mask: Tensor) -> Tensor:
    """
    Transform image-space mask to latent-space sequence weights.

    Args:
        image_mask: [B, H, W] - Binary mask in image space
    Returns:
        latent_mask: [B, seq_len] - Weights for packed latent
    """
    B, H, W = image_mask.shape

    # Step 1: VAE-aligned downsampling
    # [B, H, W] → [B, H/8, W/8]
    latent_h, latent_w = H // 8, W // 8
    mask_latent = F.avg_pool2d(
        image_mask.float().unsqueeze(1),
        kernel_size=8, stride=8
    ).squeeze(1)  # [B, latent_h, latent_w]

    # Step 2: Packing simulation
    # [B, latent_h, latent_w] → [B, latent_h//2, latent_w//2, 4]
    # Reshape to separate 2x2 patches, then fold
    patches = mask_latent.reshape(B, latent_h//2, 2, latent_w//2, 2)
    patches = patches.permute(0, 1, 3, 2, 4).contiguous().view(B, latent_h//2, latent_w//2, 4)

    # Step 3: Patch-wise maximum (preserve edit regions)
    # [B, latent_h//2, latent_w//2, 4] → [B, latent_h//2, latent_w//2]
    packed_mask = patches.max(dim=-1)[0]

    # Step 4: Flatten to sequence
    # [B, latent_h//2, latent_w//2] → [B, seq_len]
    seq_len = (latent_h // 2) * (latent_w // 2)
    return packed_mask.view(B, seq_len)
```

### 2. Dual Loss Computation

The system employs a dual loss strategy that combines:
- **Original Loss**: Standard flow matching loss (unchanged for backward compatibility)
- **Mask Loss**: Weighted loss emphasizing edit regions with higher importance
- **Combined Loss**: `final_loss = (1-α) × original_loss + α × mask_loss`

### 3. Implementation Strategy

#### Configuration (Incremental)

```yaml
loss:
    mask_loss: false          # Enable mask-weighted loss computation
    foreground_weight: 2.0    # Weight multiplier for edit regions
    background_weight: 1.0    # Weight multiplier for background regions

data:
  init_args:
    use_edit_mask: false      # Enable edit mask loading from dataset
```

#### Core Function Signature

```python
class MaskEditLoss:
    def __init__(self, foreground_weight=2.0, background_weight=1.0):
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight

    def forward(self, mask, model_pred, target, weighting=None):
        """
        Compute mask-weighted loss for image editing training.

        Args:
            mask: [B, seq_len] - Binary mask (1=edit region, 0=background)
            model_pred: [B, seq_len, channels] - Model predictions
            target: [B, seq_len, channels] - Target values
            weighting: [B, seq_len, 1] - Optional timestep weighting

        Returns:
            torch.Tensor - Weighted loss value
        """
        # Compute element-wise MSE loss
        element_loss = (model_pred.float() - target.float()) ** 2

        # Apply optional timestep weighting
        if weighting is not None:
            element_loss = weighting.float() * element_loss

        # Create weight mask: higher weights for edit regions
        # mask: [B, seq_len] -> weight_mask: [B, seq_len, 1]
        weight_mask = (mask * self.foreground_weight + (1 - mask) * self.background_weight)
        weight_mask = weight_mask.unsqueeze(-1)  # [B, seq_len, 1]

        # Apply mask weighting
        weighted_loss = element_loss * weight_mask

        # Aggregate loss: mean over sequence dimension, then batch dimension
        loss = torch.mean(weighted_loss.reshape(target.shape[0], -1), 1).mean()
        return loss
```

## Dataset Structure with Edit Masks

The dataset structure supports optional edit masks alongside existing image pairs and text prompts:

```bash
data/your_dataset/
├── control_images/          # Input/control images
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── training_images/         # Target images, text prompts, and edit masks
    ├── image_001.png        # Target image
    ├── image_001.txt        # Text prompt for image_001
    ├── image_001_mask.png   # Edit mask for image_001 (optional)
    ├── image_002.png
    ├── image_002.txt
    ├── image_002_mask.png   # Edit mask for image_002 (optional)
    └── ...
```

## Configuration

The mask loss feature is controlled by a new configuration block. When this configuration is missing, the system falls back to traditional MSE loss computation:

```yaml
loss:
  mask_loss: true               # Enable mask-weighted loss computation
  foreground_weight: 2.0        # Weight multiplier for edit regions
  background_weight: 1.0        # Weight multiplier for background regions
```

**Key Benefits:**
- **Focused Training**: Higher weights on edit regions improve convergence on critical areas
- **Backward Compatibility**: Existing training pipelines work unchanged when `mask_loss: false`
- **Flexible Weighting**: Configurable foreground/background weight ratios for different editing scenarios

# Image Edit Mask Loss Enhancement

## Overview
Add an optional mask-weighted loss component to improve text editing performance by focusing training on text regions while maintaining backward compatibility.

## Problem Statement
Current text editing training treats all image regions equally, leading to suboptimal performance on text-specific edits where only text regions differ between source and target images.

## Solution Architecture

### 1. Mask Processing Pipeline

#### Step-by-Step Transformation (Example: 832×576 → 1024 sequence)

```
Step 1: Input Image Space Mask
├─ Shape: [B, 832, 576]
├─ Format: Binary mask (0=background, 1=text_region)
├─ Source: OCR detection or manual annotation
└─ Example: Text regions marked as 1, background as 0

Step 2: VAE Downsampling Simulation
├─ Operation: F.avg_pool2d(mask, kernel_size=8, stride=8)
├─ Shape: [B, 832, 576] → [B, 104, 72]
├─ Rationale: Match VAE encoder 8x compression
├─ Preservation: avg_pool maintains text region density
└─ Result: Latent-space aligned mask

Step 3: Packing Simulation (2×2 Patch Merge)
├─ Operation: unfold(2,2) + reshape + max_pooling
├─ Shape: [B, 104, 72] → [B, 52, 36] → [B, 1872]
├─ Process:
│   ├─ unfold: [B, 104, 72] → [B, 52, 36, 4]
│   ├─ max: Take max value per 2×2 patch → [B, 52, 36]
│   └─ flatten: [B, 52, 36] → [B, 1872]
├─ Rationale: Simulate transformer's 2×2 patchify operation
└─ Final: seq_len = (104÷2) × (72÷2) = 52 × 36 = 1872

Step 4: Alignment with Packed Latents
├─ Image latent: [B, 16, 104, 72] → pack → [B, 1872, 64]
├─ Mask latent: [B, 832, 576] → process → [B, 1872]
├─ Alignment: Both have seq_len = 1872
└─ Usage: mask[i] weights latent[i, :] across all 64 channels
```

#### Mathematical Formulation
```python
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
    mask_latent = F.avg_pool2d(
        image_mask.float().unsqueeze(1),
        kernel_size=8, stride=8
    ).squeeze(1)  # [B, latent_h, latent_w]

    # Step 2: Packing simulation
    # [B, latent_h, latent_w] → [B, latent_h//2, latent_w//2, 4]
    patches = mask_latent.unfold(1, 2, 2).unfold(2, 2, 2)
    patches = patches.contiguous().view(B, latent_h//2, latent_w//2, 4)

    # Step 3: Patch-wise maximum (preserve text regions)
    # [B, latent_h//2, latent_w//2, 4] → [B, latent_h//2, latent_w//2]
    packed_mask = patches.max(dim=-1)[0]

    # Step 4: Flatten to sequence
    # [B, latent_h//2, latent_w//2] → [B, seq_len]
    seq_len = (latent_h // 2) * (latent_w // 2)
    return packed_mask.view(B, seq_len)
```

### 2. Dual Loss Computation
- **Original Loss**: Standard flow matching loss (unchanged)
- **Mask Loss**: Weighted loss emphasizing text regions
- **Combined Loss**: `final_loss = (1-α) × original_loss + α × mask_loss`

### 3. Implementation Strategy

#### Configuration (Incremental)
```yaml
mask_loss:
  enabled: false          # Feature toggle
  weight: 0.3             # α in combined loss
  text_region_weight: 2.0 # Text region emphasis
  background_weight: 1.0  # Background region weight

data:
  init_args:
    use_text_mask: false  # Enable mask loading
    mask_dir_name: "text_masks"
```

#### Core Function Signature
```python
def _compute_loss_with_mask(
    self,
    pixel_latents, control_latents,
    prompt_embeds, prompt_embeds_mask,
    height, width,
    text_mask=None  # Optional [B, seq_len] mask
) -> torch.Tensor
```

#### Loss Computation Logic
```python
# Standard loss (unchanged)
element_loss = (model_pred - target) ** 2

if text_mask is not None:
    # Apply region-specific weights
    weight_mask = (text_mask * text_weight +
                   (1 - text_mask) * background_weight)
    weighted_loss = element_loss * weight_mask.unsqueeze(-1)
    mask_loss = weighted_loss.mean()

    # Combine losses
    final_loss = (1 - mask_weight) * original_loss + mask_weight * mask_loss
else:
    final_loss = original_loss  # Fallback to original behavior
```

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Add mask preprocessing utilities
- [ ] Implement mask-to-latent mapping
- [ ] Create dual loss computation function

### Phase 2: Data Pipeline Integration
- [ ] Extend dataset to support optional mask loading
- [ ] Update training step with mask parameter
- [ ] Add configuration options

### Phase 3: Testing & Validation
- [ ] Backward compatibility verification
- [ ] Performance benchmarking
- [ ] Documentation updates

## Key Features

### Backward Compatibility
- **Zero Impact**: When `mask_loss.enabled=false`, behavior identical to original
- **Optional Data**: Mask files not required; graceful degradation when missing
- **Incremental Adoption**: Can be enabled per-dataset or per-experiment

### Flexible Configuration
- **Adjustable Weights**: Fine-tune text vs background emphasis
- **Loss Mixing Ratio**: Control contribution of mask loss vs original loss
- **Dataset Agnostic**: Works with existing data structure

### Performance Considerations
- **Minimal Overhead**: Mask processing only when enabled
- **Memory Efficient**: Mask stored as compact binary format
- **GPU Friendly**: All operations vectorized for parallel processing

## Expected Outcomes
- **Improved Text Editing**: Better preservation and modification of text regions
- **Maintained Generalization**: Background editing capabilities unchanged
- **Training Stability**: Gradual loss weighting prevents training disruption
- **Easy Experimentation**: Quick enable/disable for A/B testing

## Data Requirements
- **Mask Format**: Binary images (0=background, 255=text) matching source image resolution
- **File Structure**: `dataset/text_masks/filename.jpg` parallel to existing structure
- **Generation**: Can be created via OCR, manual annotation, or automated text detection

# DreamOmni2 LoRA Trainer Implementation Plan

## Document Information
- **Created**: 2025-10-30
- **Purpose**: Analysis and implementation plan for DreamOmni2 LoRA finetune trainer
- **Status**: Planning Phase

---

## 1. Model Understanding

### 1.1 DreamOmni2 Overview

DreamOmni2 is a multi-image conditioning diffusion model built on the Flux architecture. It extends the Flux Kontext pipeline to support **multiple reference images** simultaneously, enabling more complex image generation and editing scenarios.

### 1.2 Key Components

The DreamOmni2 pipeline uses the same core components as Flux Kontext:
- **Transformer**: `FluxTransformer2DModel` - Main denoising model
- **VAE**: `AutoencoderKL` - Image encoding/decoding
- **Text Encoders**:
  - `CLIPTextModel` - CLIP text encoder (pooled embeddings)
  - `T5EncoderModel` - T5 text encoder (sequence embeddings)
- **Scheduler**: `FlowMatchEulerDiscreteScheduler` - Flow matching scheduler
- **Tokenizers**: CLIP and T5 tokenizers

### 1.3 Core Innovation: Multi-Image Support

The primary difference from Flux Kontext is the ability to handle **multiple reference images** in a single forward pass:

```python
# Flux Kontext: Single image
images: Optional[PipelineImageInput] = None  # Single image

# DreamOmni2: Multiple images
images: Optional[List[PipelineImageInput]] = None  # List of images
```

---

## 重要发现：核心差异就是偏移量！

**关键结论**：如果 Flux Kontext Trainer 添加了累积偏移量（cumulative offsets），那么在算法上就与 DreamOmni2 Pipeline 完全一致了！

### 核心差异解释

**Flux Kontext Trainer（当前）**：
- 多个控制图像，每个图像的列坐标都从 0 开始
- 图像1: `col=[0..W1-1]`
- 图像2: `col=[0..W2-1]` ⚠️ 重叠了！

**DreamOmni2 Pipeline**：
- 使用累积偏移量，列坐标连续
- 图像1: `col=[0..W1-1]`
- 图像2: `col=[W1..W1+W2-1]` ✅ 连续了！

**为什么重要**：RoPE（Rotary Position Embedding）根据坐标计算位置编码。如果两个图像的列坐标都从 0 开始，模型会认为它们在空间上重叠；如果有偏移，模型知道它们在空间上是连续的。

**最小改动方案**：只需要在 `prepare_embeddings()` 中添加偏移量计算，保持现有的 `control`, `control_1`, ... 命名约定即可！

---

## 2.1 核心差异：偏移量分析

### 为什么偏移量重要？

#### RoPE（Rotary Position Embedding）的工作原理

Flux Transformer 使用 3D RoPE，基于 `[domain, row, col]` 计算位置编码：

```python
# 在 transformer 中
ids = torch.cat([txt_ids, img_ids], dim=0)  # 拼接所有ID
image_rotary_emb = self.pos_embed(ids)  # 基于坐标计算RoPE
```

RoPE 会根据坐标的相对位置计算注意力权重：
- **无偏移**：两个图像的 col 都从 0 开始 → 模型认为它们在空间上重叠
- **有偏移**：第二个图像的 col 从第一个图像的宽度开始 → 模型知道它们在空间上连续

#### 示例对比

假设有两个 64x64 的图像：

**无偏移（Flux Kontext Trainer）**：
```
图像1: domain=1, col=[0, 1, 2, ..., 63]
图像2: domain=2, col=[0, 1, 2, ..., 63]  ⚠️ 重叠了！
```

**有偏移（DreamOmni2 Pipeline）**：
```
图像1: domain=1, col=[0, 1, 2, ..., 63]
图像2: domain=2, col=[64, 65, 66, ..., 127]  ✅ 连续了！
```

### 实现一致性

如果我们在 Flux Kontext Trainer 的 `prepare_embeddings()` 中添加偏移量计算：

```python
# 当前实现（无偏移）
batch["control_latents"] = torch.cat([control_latents, control_1_latents, ...], dim=1)
batch["control_ids"] = torch.cat([control_ids, control_1_ids, ...], dim=0)

# 修改后（有偏移）
w_offset = 0
for i, control_latents in enumerate(all_control_latents):
    if i > 0:
        control_ids[..., 2] += w_offset  # ⭐ 添加偏移量
    w_offset += control_width // 2  # 累积偏移
batch["control_ids"] = torch.cat([control_ids, ...], dim=0)
```

**这样就和 DreamOmni2 Pipeline 的算法一致了！**

---

### 2.2 Architecture Comparison

| Aspect | Flux Kontext | DreamOmni2 |
|--------|--------------|------------|
| **Input Images** | Single image (`image`) in pipeline | Multiple images (`images: List`) in pipeline |
| **Trainer Support** | Multiple controls (`control`, `control_1`, `control_2`, ...) | Multiple images (`images: List[Tensor]`) |
| **Processing Level** | Trainer-level multi-image handling | Pipeline-level multi-image handling |
| **Image IDs** | Fixed: `image_ids[..., 0] = 1` | Dynamic: `image_ids[..., 0] = i+1` where `i` is image index |
| **Position Encoding** | No cumulative offsets | Cumulative offsets (`h_offset`, `w_offset`) |
| **Latent Concatenation** | Trainer concatenates after individual processing | Pipeline concatenates with offsets |
| **Pipeline Class** | `FluxKontextPipeline` | `DreamOmni2Pipeline` |

### 2.2 Key Insight: Flux Kontext Trainer Already Supports Multiple Controls

**Important Discovery**: The existing `FluxKontextTrainer` already supports multiple reference images, but:
- Uses naming convention: `control`, `control_1`, `control_2`, `control_3`, ...
- Processes each control image separately in `prepare_embeddings()`
- Concatenates them at the trainer level (not pipeline level)
- **Does NOT use cumulative offsets** (this is the key difference!)

**Flux Kontext Trainer Multi-Image Handling**:
```python
# In prepare_embeddings()
if "control" in batch:
    control_ids[..., 0] = 1  # First control
    batch["control_latents"] = [control_latents]
    batch["control_ids"] = [control_ids]

for i in range(1, num_additional_controls + 1):
    control_key = f"control_{i}"  # control_1, control_2, ...
    control_ids[..., 0] = i + 1  # Domain IDs: 2, 3, 4, ...
    batch["control_latents"].append(control_latents)
    batch["control_ids"].append(control_ids)

# Concatenate all controls
batch["control_latents"] = torch.cat(batch["control_latents"], dim=1)
batch["control_ids"] = torch.cat(batch["control_ids"], dim=0)
```

**DreamOmni2 Pipeline Multi-Image Handling**:
```python
# In pipeline.prepare_latents()
h_offset = 0
w_offset = 0
for i, image in enumerate(images):
    image_ids[..., 0] = i + 1  # Domain IDs: 1, 2, 3, ...
    image_ids[..., 2] += w_offset  # Cumulative column offset
    h_offset += image_latent_height // 2
    w_offset += image_latent_width // 2
```

**Key Difference**: DreamOmni2 uses **cumulative offsets** for spatial position encoding, while Flux Kontext trainer does not.

### 2.2 Key Implementation Differences

#### 2.2.1 `prepare_latents` Method

**Flux Kontext Trainer (Single Image per Call)**:
```python
def prepare_latents(self, image: Optional[torch.Tensor], ...):
    # Called separately for each control image
    image_latents = self._encode_vae_image(image=image)
    image_latents = self._pack_latents(...)
    image_ids = self._prepare_latent_image_ids(...)
    image_ids[..., 0] = 1  # Fixed domain ID (set to i+1 later in prepare_embeddings)
    return latents, image_latents, latent_ids, image_ids
```

**Note**: Flux Kontext trainer handles multiple images at the trainer level by:
- Calling `prepare_latents()` multiple times (once per control)
- Setting domain IDs in `prepare_embeddings()`: `control_ids[..., 0] = 1` for control, `i+1` for control_i
- Concatenating results: `torch.cat([control_latents, control_1_latents, ...], dim=1)`

**DreamOmni2 Pipeline (Multiple Images in One Call)**:
```python
def prepare_latents(self, images: Optional[List[torch.Tensor]], ...):
    h_offset = 0
    w_offset = 0
    tp_image_latents = []
    tp_image_ids = []

    for i, image in enumerate(images):
        image_latents = self._encode_vae_image(image=image, generator=generator)
        image_latents = self._pack_latents(...)
        image_ids = self._prepare_latent_image_ids(...)
        image_ids[..., 0] = i+1  # Dynamic domain ID (1, 2, 3, ...)
        image_ids[..., 2] += w_offset  # Cumulative column offset ⭐ KEY DIFFERENCE
        tp_image_latents.append(image_latents)
        tp_image_ids.append(image_ids)
        h_offset += image_latent_height // 2
        w_offset += image_latent_width // 2

    # Concatenate all images along sequence dimension
    image_latents = torch.cat(tp_image_latents, dim=1)
    image_ids = torch.cat(tp_image_ids, dim=0)

    return latents, image_latents, latent_ids, image_ids
```

**Key Differences**:
1. **Processing level**: Flux Kontext handles at trainer level, DreamOmni2 at pipeline level
2. **Cumulative offsets**: DreamOmni2 uses `h_offset` and `w_offset` ⭐ **This is the main innovation**
3. **Domain IDs**: Both use `i+1`, but DreamOmni2 sets them in pipeline, Flux Kontext sets in trainer
4. **Single vs multiple calls**: DreamOmni2 processes all images in one call, Flux Kontext calls separately

#### 2.2.2 Image ID Structure

**Flux Kontext Trainer**:
- Format: `[domain, row, col]`
- Domain: `1` for `control`, `2` for `control_1`, `3` for `control_2`, etc.
- **No cumulative offsets**: Each control image has independent coordinates starting from (0,0)
- Example (2 controls):
  - Control: `[1, 0, 0]`, `[1, 0, 1]`, `[1, 1, 0]`, ...
  - Control_1: `[2, 0, 0]`, `[2, 0, 1]`, `[2, 1, 0]`, ... (starts from 0,0)

**DreamOmni2 Pipeline**:
- Format: `[domain, row, col]`
- Domain: `1` for first image, `2` for second image, `3` for third, etc.
- **Cumulative offsets**: Column coordinates accumulate across images (`w_offset` accumulates)
- Example (2 images, where first image has width 64):
  - Image 0: `[1, 0, 0]`, `[1, 0, 1]`, ..., `[1, 0, 63]`
  - Image 1: `[2, 0, 64]`, `[2, 0, 65]`, ..., `[2, 0, 64+63]` (starts from w_offset)

#### 2.2.3 Latent Concatenation

**Flux Kontext**:
```
image_latents: [batch_size, seq_len_single, channels]
```

**DreamOmni2**:
```
image_latents: [batch_size, seq_len_total, channels]
where seq_len_total = sum(seq_len_i for each image i)
```

### 2.3 Training Data Structure

**Flux Kontext Trainer**:
- Multiple control images per sample (already supported!)
- Batch structure:
  ```python
  {
      "control": tensor,              # First control image
      "control_1": tensor,            # Second control image
      "control_2": tensor,            # Third control image
      ...
      "control_latents": tensor,      # Concatenated [B, sum(seq_i), C]
      "control_ids": tensor,          # Concatenated [sum(seq_i), 3]
      "n_controls": int,             # Number of additional controls
  }
  ```
- Processing: Each control processed separately, then concatenated

**DreamOmni2 Trainer** (Expected):
- Multiple images per sample (should match pipeline interface)
- Batch structure options:
  ```python
  # Option 1: Match pipeline interface
  {
      "images": [tensor1, tensor2, ...],  # List of images
      "control_latents": tensor,          # Concatenated with offsets
      "control_ids": tensor,             # Concatenated with offsets
  }

  # Option 2: Keep compatibility with Flux Kontext format
  {
      "control": tensor,
      "control_1": tensor,
      ...
      # Convert to images list internally
  }
  ```
- Processing: Should use pipeline's `prepare_latents()` which handles offsets automatically

---

## 3. Implementation Plan

### 3.1 File Structure

```
src/qflux/trainer/
├── dreamomni2_trainer.py       # New trainer implementation
├── flux_kontext_trainer.py      # Reference implementation
└── base_trainer.py              # Base abstract class
```

### 3.2 Implementation Steps

#### Phase 1: Core Trainer Class Setup

**Step 1.1**: Create `DreamOmni2LoraTrainer` class
- Inherit from `BaseTrainer`
- Copy initialization from `FluxKontextLoraTrainer`
- Update model component references to use DreamOmni2 pipeline

**Step 1.2**: Update `get_pipeline_class()`
```python
def get_pipeline_class(self):
    from qflux.models.pipeline_dreamomni2 import DreamOmni2Pipeline
    return DreamOmni2Pipeline
```

**Step 1.3**: Update model loading
- Reuse `load_flux_kontext_*` functions (same components)
- Ensure all components are compatible

#### Phase 2: Multi-Image Latent Preparation

**Step 2.1**: Override `prepare_latents()` method
- **Option A**: Accept `images: Optional[List[torch.Tensor]]` to match pipeline interface
- **Option B**: Keep current trainer convention (`control`, `control_1`, ...) and convert internally
- Use pipeline's `prepare_latents()` method (which handles offsets) OR reimplement with:
  - Loop over images
  - Individual VAE encoding
  - Packing for each image
  - Dynamic image ID generation (`i+1`)
  - **Cumulative offset tracking** (`h_offset`, `w_offset`) ⭐ **Key feature**
  - Concatenation along sequence dimension

**Step 2.2**: Handle batch expansion
- Support multiple images per batch sample
- Handle batch size expansion correctly
- Ensure offset calculation works across batch dimension
- **Important**: Offsets should be calculated per sample, not globally

#### Phase 3: Training Data Preparation

**Step 3.1**: Update `prepare_embeddings()` method
- **Decision needed**: Match pipeline interface (`images: List`) or keep trainer convention (`control`, `control_1`, ...)?
- If keeping trainer convention: Convert `control`, `control_1`, ... to `images` list, then call pipeline's `prepare_latents()`
- If using pipeline interface: Accept `images: List[Tensor]` directly
- **Key difference**: Use pipeline's `prepare_latents()` which handles cumulative offsets automatically
- Maintain compatibility with existing loss computation

**Step 3.2**: Update `prepare_cached_embeddings()` method
- Ensure cached embeddings support multi-image format with offsets
- Handle batch dimension correctly
- **Important**: Cache format must include offset information or be recalculated

**Step 3.3**: Update `cache_step()` method
- Save multiple control latents per sample
- Store concatenated control IDs **with offsets** (if needed)
- Or regenerate offsets during cache loading

#### Phase 4: Loss Computation

**Step 4.1**: Verify `_compute_loss()` compatibility
- Check if existing loss computation handles concatenated latents correctly
- Test with multi-image batches
- Ensure attention masks work correctly

**Step 4.2**: Multi-resolution mode support
- Check compatibility with existing multi-resolution implementation
- Ensure padding and masking work with multi-image concatenation

#### Phase 5: Prediction/Inference

**Step 5.1**: Update `prepare_predict_batch_data()`
- Accept multiple images as input
- Support both single and multi-image modes
- Handle image preprocessing for multiple images

**Step 5.2**: Update `sampling_from_embeddings()`
- Handle concatenated image latents
- Ensure scheduler steps work correctly
- Support multi-resolution mode

**Step 5.3**: Update `decode_vae_latent()`
- Ensure decoding works with multi-image latents (if needed)
- Handle output correctly

#### Phase 6: Testing and Validation

**Step 6.1**: Unit tests
- Test `prepare_latents()` with 1, 2, 3+ images
- Test batch expansion
- Test offset calculation

**Step 6.2**: Integration tests
- Test full training loop with multi-image data
- Test caching with multi-image data
- Test prediction with multiple images

**Step 6.3**: Validation
- Compare outputs with reference DreamOmni2 pipeline
- Verify loss computation consistency
- Test multi-resolution mode

---

## 4. Code Changes Summary

### 4.1 New Methods to Implement

1. **`prepare_latents()`** - Override with multi-image support **using cumulative offsets**
   - Accept `images: Optional[List[torch.Tensor]]` OR convert from `control`, `control_1`, ... format
   - Implement cumulative offset tracking (`h_offset`, `w_offset`)
   - Match DreamOmni2Pipeline's behavior

2. **`prepare_embeddings()`** - Update to use pipeline's prepare_latents with offsets
   - Convert trainer format (`control`, `control_1`, ...) to pipeline format (`images: List`)
   - OR update to accept `images: List` directly
   - Call pipeline's `prepare_latents()` which handles offsets

3. **`prepare_predict_batch_data()`** - Update for multi-image input
   - Support both single and multi-image modes
   - Handle image preprocessing for multiple images

### 4.2 Methods to Modify

1. **`cache_step()`** - Handle multiple control images **with offset information**
   - Store multiple control latents separately (recommended)
   - Store offset metadata or regenerate during loading

2. **`prepare_cached_embeddings()`** - Handle cached multi-image format **with offsets**
   - Regenerate offsets from cached latents if needed
   - Or load offset information from cache

3. **`sampling_from_embeddings()`** - Verify compatibility with concatenated latents **with offsets**
   - Ensure offsets are preserved during sampling
   - Test with multi-image batches

### 4.3 Methods to Inherit (No Changes)

1. **`encode_prompt()`** - Same as Flux Kontext
2. **`get_clip_prompt_embeds()`** - Same as Flux Kontext
3. **`get_t5_prompt_embeds()`** - Same as Flux Kontext
4. **`encode_vae_image()`** - Same as Flux Kontext
5. **`_pack_latents()`** - Static method, same implementation
6. **`_unpack_latents()`** - Static method, same implementation
7. **`_prepare_latent_image_ids()`** - Static method, same implementation
8. **`setup_model_device_train_mode()`** - Same device management
9. **`configure_optimizers()`** - Same optimizer setup
10. **`_compute_loss()`** - Should work with concatenated latents (verify with offsets)

### 4.4 Key Implementation Detail: Cumulative Offsets

The main difference from Flux Kontext trainer is the **cumulative offset calculation**:

```python
# DreamOmni2 style (with offsets)
h_offset = 0
w_offset = 0
for i, image in enumerate(images):
    image_ids[..., 2] += w_offset  # Apply cumulative column offset
    # ... accumulate for next image
    w_offset += image_latent_width // 2
```

This must be implemented correctly to match DreamOmni2Pipeline behavior.

---

## 5. Key Implementation Details

### 5.1 Image ID Domain Mapping

```python
# Single image (Flux Kontext style)
image_ids[..., 0] = 1

# Multiple images (DreamOmni2 style)
for i, image in enumerate(images):
    image_ids[..., 0] = i + 1  # Domain IDs: 1, 2, 3, ...
```

### 5.2 Offset Accumulation

```python
h_offset = 0
w_offset = 0

for i, image in enumerate(images):
    image_latent_height, image_latent_width = image_latents.shape[2:]

    # Apply column offset
    image_ids[..., 2] += w_offset

    # Accumulate offsets for next image
    h_offset += image_latent_height // 2
    w_offset += image_latent_width // 2
```

**Note**: The offset mechanism helps the RoPE position encoding understand spatial relationships between multiple images when they are concatenated.

### 5.3 Latent Concatenation

```python
# Individual images
tp_image_latents = [latent_0, latent_1, latent_2, ...]  # Each: [B, seq_i, C]

# Concatenated along sequence dimension
image_latents = torch.cat(tp_image_latents, dim=1)  # [B, sum(seq_i), C]

# Same for IDs
image_ids = torch.cat(tp_image_ids, dim=0)  # [sum(seq_i), 3]
```

### 5.4 Batch Handling

```python
# Support batch expansion
if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
    additional_image_per_prompt = batch_size // image_latents.shape[0]
    image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
```

---

## 6. Data Format Requirements

### 6.1 Training Data Format

**Single Image (Flux Kontext)**:
```python
batch = {
    "control": torch.Tensor,  # [B, C, H, W]
    "control_latents": torch.Tensor,  # [B, seq, C]
    "control_ids": torch.Tensor,  # [seq, 3]
    ...
}
```

**Multiple Images (DreamOmni2)**:
```python
batch = {
    "control": List[torch.Tensor],  # [tensor1, tensor2, ...] or [B, C, H, W] if single
    "control_latents": torch.Tensor,  # [B, sum(seq_i), C] - concatenated
    "control_ids": torch.Tensor,  # [sum(seq_i), 3] - concatenated with offsets
    ...
}
```

### 6.2 Dataset Compatibility

- Check if existing dataset classes need updates
- Ensure data collator handles multi-image format
- Verify cache format compatibility

---

## 7. Potential Challenges and Solutions

### 7.1 Challenge: Offset Calculation in Batch Mode

**Problem**: When processing batches, offsets need to be calculated per sample, not globally. The cumulative offset mechanism in DreamOmni2 requires careful handling.

**Solution**:
- Calculate offsets per sample in the batch
- Ensure offset logic works correctly with batch dimension
- Test with various batch sizes
- **Key insight**: Each sample in the batch may have different numbers of images or different image sizes, so offsets must be independent per sample

### 7.1.1 Challenge: Trainer vs Pipeline Interface Mismatch

**Problem**: Flux Kontext trainer uses `control`, `control_1`, ... convention, but DreamOmni2 pipeline expects `images: List[Tensor]`.

**Solution**:
- **Option A**: Convert trainer format to pipeline format internally
  ```python
  # In prepare_embeddings()
  images = []
  if "control" in batch:
      images.append(batch["control"])
  for i in range(1, num_additional_controls + 1):
      images.append(batch[f"control_{i}"])
  # Then use pipeline's prepare_latents(images, ...)
  ```
- **Option B**: Update trainer to use `images: List` format directly
- **Recommendation**: Option A for backward compatibility

### 7.2 Challenge: Loss Computation with Concatenated Latents

**Problem**: Loss computation might need to handle variable-length sequences per image.

**Solution**:
- Verify existing loss functions handle concatenated latents correctly
- Check if attention masks are needed
- Consider multi-resolution mode which already handles padding

### 7.3 Challenge: Multi-Resolution Mode Compatibility

**Problem**: Multi-resolution mode uses padding and masking - need to ensure it works with multi-image concatenation **and cumulative offsets**.

**Solution**:
- Test multi-resolution mode with multi-image data
- Ensure padding logic accounts for concatenated structure
- Verify attention masks cover all image tokens correctly
- **Important**: Offsets must be calculated correctly even when images have different resolutions
- Consider if offsets should be relative to the largest image or per-image

### 7.4 Challenge: Cache Format Compatibility

**Problem**: Existing cache format might not support multi-image structure with cumulative offsets.

**Solution**:
- Review cache format in `cache_step()` and `prepare_cached_embeddings()`
- **Option A**: Store multiple control latents separately, regenerate offsets during loading
- **Option B**: Store concatenated latents with offset information
- Ensure backward compatibility if possible
- **Recommendation**: Store separately and regenerate offsets (simpler, more flexible)

---

## 8. Testing Strategy

### 8.1 Unit Tests

1. **Test `prepare_latents()` with 1 image**
   - Should match Flux Kontext behavior

2. **Test `prepare_latents()` with 2+ images**
   - Verify correct domain IDs (1, 2, 3, ...)
   - Verify offset accumulation
   - Verify concatenation shape

3. **Test batch expansion**
   - Verify batch size expansion works correctly
   - Test with various batch sizes

### 8.2 Integration Tests

1. **End-to-end training**
   - Single epoch training with multi-image data
   - Verify loss decreases
   - Verify checkpoint saving/loading

2. **Cache mode**
   - Test caching with multi-image data
   - Test loading cached embeddings
   - Verify correctness

3. **Prediction mode**
   - Test inference with multiple images
   - Compare outputs with reference pipeline
   - Test multi-resolution mode

### 8.3 Validation Tests

1. **Compare with reference pipeline**
   - Generate same images using DreamOmni2Pipeline directly
   - Compare latents, IDs, and outputs
   - Verify numerical consistency

---

## 9. Migration Path

### 9.1 Backward Compatibility

- Consider supporting both single and multi-image modes
- Detect input format automatically
- Allow gradual migration

### 9.2 Configuration Updates

- Update config format to support multi-image specification
- Maintain compatibility with existing Flux Kontext configs
- Add new config options if needed

---

## 10. Timeline Estimate

- **Phase 1-2**: Core implementation (2-3 days)
- **Phase 3**: Training data preparation (1-2 days)
- **Phase 4**: Loss computation verification (1 day)
- **Phase 5**: Prediction/inference (1-2 days)
- **Phase 6**: Testing and validation (2-3 days)

**Total**: ~7-11 days

---

## 11. Dependencies

- `qflux.models.pipeline_dreamomni2.DreamOmni2Pipeline` - Must be available
- All existing Flux Kontext loader utilities
- Same base trainer infrastructure

---

## 12. Success Criteria

1. ✅ Training loop runs successfully with multi-image data
2. ✅ Loss computation works correctly
3. ✅ Prediction/inference produces correct outputs
4. ✅ Multi-resolution mode works (if applicable)
5. ✅ Cache mode works correctly
6. ✅ Unit tests pass
7. ✅ Integration tests pass
8. ✅ Outputs match reference DreamOmni2 pipeline

---

## Appendix: Code Reference

### Key Differences Summary

| Component | Flux Kontext | DreamOmni2 |
|-----------|--------------|------------|
| `prepare_latents` | Single image, `image_ids[..., 0] = 1` | Multiple images, `image_ids[..., 0] = i+1`, offsets |
| Input format | `image: Tensor` | `images: List[Tensor]` |
| Latent shape | `[B, seq, C]` | `[B, sum(seq_i), C]` |
| ID shape | `[seq, 3]` | `[sum(seq_i), 3]` |
| Domain IDs | Fixed `1` | Dynamic `1, 2, 3, ...` |

---

**End of Document**

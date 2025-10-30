# Qwen Multi-Resolution Training Implementation Summary

## Overview

本文档总结了为 Qwen Image Edit 模型实现多分辨率 padding 模式训练的完整方案。

## 问题分析

### Qwen 原始设计的限制

Qwen 的 RoPE 实现采用 **sequence concatenation** 模式：
- `QwenEmbedRope.forward` 为每个样本生成 RoPE 频率，然后 concat 成一个长序列
- 要求所有样本的 tokens 连续排列：`[sample0_tokens, sample1_tokens, ...]`
- **不支持独立的 batch 维度和 padding**

这导致无法直接进行多分辨率批处理训练（不同尺寸的图像无法在同一个 batch 中）。

## 解决方案

### 1. 核心思路

实现 **padding 模式**的多分辨率支持：
- 为每个样本生成独立的 per-sample RoPE
- Padding 到相同长度形成批次
- 使用 attention_mask 忽略 padding 区域
- 验证 batch inference 与逐个样本 inference 的一致性

### 2. 实现文件

#### 2.1 `src/models/qwen_multi_resolution_patch.py` (新增)

**功能**：提供 Qwen 模型的多分辨率支持 patch

**核心组件**：

1. **`apply_rotary_emb_qwen_batched`**
   - 支持 2D RoPE (concatenation 模式) 和 3D RoPE (padding 模式)
   - 自动检测 freqs_cis 维度并适配

2. **`QwenDoubleStreamAttnProcessor2_0_MultiRes`**
   - 增强版 attention processor
   - 支持预计算的 per-sample RoPE (B, max_seq, rope_dim)
   - 兼容原始 concatenation 模式

3. **`patch_qwen_forward_for_multi_resolution`**
   - Patch 模型的 forward 方法
   - 添加 `image_rotary_emb` 参数支持预计算 RoPE
   - 保持向后兼容（`img_shapes` 仍然有效）

4. **`patch_qwen_model_for_multi_resolution`**
   - 一键 patch 函数
   - 替换所有 attention processors
   - 修改 forward 方法

**使用示例**：
```python
from qflux.models.transformer_qwenimage import QwenImageTransformer2DModel
from qflux.models.qwen_multi_resolution_patch import patch_qwen_model_for_multi_resolution

model = QwenImageTransformer2DModel(...)
patch_qwen_model_for_multi_resolution(model)

# 使用预计算的 per-sample RoPE
img_freqs_batched = ...  # (B, max_seq, rope_dim)
txt_freqs = ...          # (txt_len, rope_dim)
image_rotary_emb = (img_freqs_batched, txt_freqs)

output = model(
    hidden_states=padded_hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    image_rotary_emb=image_rotary_emb,  # 预计算的 RoPE
    attention_kwargs={'attention_mask': attention_mask_4d},
    return_dict=True,
)
```

#### 2.2 `src/utils/tools.py` (更新)

**新增函数**: `extract_batch_field`
- 从 embeddings 字典中提取指定 batch index 的字段值
- 统一处理 list, tensor, scalar 三种数据类型
- 用于多分辨率训练中提取 per-sample 参数

```python
def extract_batch_field(embeddings: dict, key: str, batch_idx: int):
    """Extract a field value for a specific batch index"""
    value = embeddings[key]
    if isinstance(value, (list, tuple)):
        return value[batch_idx]
    elif isinstance(value, torch.Tensor) and value.numel() > 1:
        return value[batch_idx].item()
    else:
        return value  # Scalar - same for all samples
```

#### 2.3 `src/trainer/qwen_image_edit_trainer.py` (更新)

**新增方法**: `_get_image_shapes_multi_resolution`
- 生成 per-sample 的 img_shapes
- 支持多控制图像（target + controls）
- 处理嵌套列表格式：`List[List[Tuple[int, int, int]]]`

```python
def _get_image_shapes_multi_resolution(
    self, embeddings: dict, batch_size: int
) -> List[List[Tuple[int, int, int]]]:
    """
    Generate per-sample img_shapes for multi-resolution training.

    Returns:
        List[List[Tuple]]:
        - Outer list: batch samples
        - Inner list: images per sample (target + controls)
        - Tuple: (channels, height, width) in latent space
    """
    # Priority 1: Use pre-computed img_shapes if available
    if "img_shapes" in embeddings:
        ...

    # Priority 2: Reconstruct from height/width fields
    result = []
    for b in range(batch_size):
        img_shapes_sample = []

        # Target image
        height = extract_batch_field(embeddings, "height", b)
        width = extract_batch_field(embeddings, "width", b)
        img_shapes_sample.append((1, height // 16, width // 16))

        # Control images (if exist)
        if "height_control" in embeddings:
            ...

        result.append(img_shapes_sample)

    return result
```

#### 2.4 `tests/trainer/test_qwen_multi_resolution.py` (新增)

**测试用例**：

1. **`test_get_image_shapes_multi_resolution_from_img_shapes`**
   - 验证从 `img_shapes` 字段提取

2. **`test_get_image_shapes_multi_resolution_from_height_width`**
   - 验证从 `height/width` 字段重建

3. **`test_get_image_shapes_with_additional_controls`**
   - 验证多控制分支支持

4. **`test_extract_batch_field_*`** (3个测试)
   - 验证 list, tensor, scalar 三种类型处理

5. **`test_qwen_multi_resolution_inference_consistency`** ⭐ **核心测试**
   - 验证多分辨率批处理推理一致性
   - 比较 batch padding 模式 vs 逐个样本推理
   - 使用相对误差 (relative error) 判断一致性

**测试流程**：
```
Step 1: Individual inference (batch_size=1 for each sample)
  - Sample 0: 16x32 (seq_len=512)
  - Sample 1: 24x24 (seq_len=576)
  - Sample 2: 20x28 (seq_len=560)

Step 2: Batched inference with padding and per-sample RoPE
  - Pad to max_seq = 576
  - Generate per-sample RoPE and pad
  - Use attention_mask to ignore padding

Step 3: Comparing individual vs batched outputs
  - Extract valid regions (remove padding)
  - Calculate relative error: ||batch - individual|| / ||individual||
  - Assert: relative_error < 1e-3 (0.1%)
```

### 3. 关键修改点

#### 3.1 RoPE 格式支持

**原始**：2D RoPE (concatenation)
```python
img_freqs: (total_seq, rope_dim)
# where total_seq = sum(h*w for all samples)
```

**新增**：3D RoPE (padding)
```python
img_freqs: (B, max_seq, rope_dim)
# Per-sample RoPE with padding
```

#### 3.2 Attention Mask 格式

```python
# 2D boolean mask (B, joint_seq)
attention_mask = torch.zeros(B, txt_len + max_seq, dtype=torch.bool)
for i in range(B):
    seq_len_i = h_i * w_i
    attention_mask[i, :txt_len] = True  # text tokens
    attention_mask[i, txt_len:txt_len + seq_len_i] = True  # image tokens
    # padding 区域保持 False

# 转换为 4D mask for scaled_dot_product_attention
attention_mask_4d = attention_mask.view(B, 1, 1, joint_seq)
```

#### 3.3 Trainer 使用示例

```python
# 在 QwenImageEditTrainer._compute_loss 中
if self.multi_resolution_mode:
    # 1. 为每个样本生成 RoPE
    img_freqs_list = []
    for b in range(batch_size):
        img_freqs_b, txt_freqs = self.dit.pos_embed(
            [img_shapes[b]], [txt_seq_lens[b]], device
        )
        # Pad to max_seq
        img_freqs_padded = torch.zeros(max_seq, rope_dim, device=device, dtype=img_freqs_b.dtype)
        img_freqs_padded[:img_freqs_b.shape[0]] = img_freqs_b
        img_freqs_list.append(img_freqs_padded)

    img_freqs_batched = torch.stack(img_freqs_list, dim=0)  # (B, max_seq, rope_dim)
    image_rotary_emb = (img_freqs_batched, txt_freqs)

    # 2. 调用模型
    model_pred = self.dit(
        hidden_states=padded_latents,
        encoder_hidden_states=prompt_embeds,
        image_rotary_emb=image_rotary_emb,  # 预计算的 RoPE
        attention_kwargs={'attention_mask': attention_mask_4d},
        return_dict=False,
    )[0]
```

## 文档更新

### `docs/prd/multi-resolution-padding-mask-training.plan.md`

**Section 5.2.7**: 新增 "支持预计算 Per-Sample RoPE（Multi-Resolution Padding 模式）"
- 问题分析
- 解决方案详解
- 实现步骤
- 兼容性保证
- 与 Flux 实现对比

## 测试验证

### 单元测试覆盖

- ✅ img_shapes 格式处理 (2 tests)
- ✅ extract_batch_field 工具函数 (3 tests)
- ✅ 多控制分支支持 (1 test)
- ✅ **端到端推理一致性** (1 test)

### 验证标准

使用相对误差（Relative Error）作为判断标准：
```python
relative_error = ||batched_output - individual_output|| / ||individual_output||

# 判断条件
if relative_error > 1e-3 and max_absolute_diff > 1e-4:
    FAILED
else:
    PASSED
```

**原理**：
- 相对误差 < 0.1% (1e-3) 表示数值一致性良好
- 同时检查绝对误差以处理接近零的情况
- 确保 padding 模式与单独推理在数值上等价

## 兼容性

### 向后兼容

- ✅ 原有 API 不变（`img_shapes` 参数仍然有效）
- ✅ concatenation 模式继续工作
- ✅ 不使用 multi-resolution 时无影响

### 新功能

- ✅ 支持预计算 `image_rotary_emb` 参数
- ✅ 支持 padding 模式的多分辨率训练
- ✅ 支持 attention_mask 忽略 padding

## 下一步工作

### Phase 5.2: QwenImageEditTrainer Loss 计算更新
- 添加 multi-resolution 判断逻辑
- 实现 padding 和 loss 归一化
- 处理 edit_mask 的 padding

### Phase 5.3: 缓存支持 v2.0
- 更新 `prepare_cached_embeddings` 支持 v2.0
- 保存 img_shapes 到缓存
- 向后兼容 v1.0 缓存

### Phase 5.5: QwenImageEditPlusTrainer
- 多控制分支的完整支持
- 处理不同控制图像分辨率

## 参考资料

- [Flux Per-Sample RoPE Implementation](./flux-per-sample-rope-implementation.md)
- [Multi-Resolution Padding Mask Training Plan](./multi-resolution-padding-mask-training.plan.md)
- [Test: test_flux_per_sample_rope.py](../../tests/models/test_flux_per_sample_rope.py)

## 总结

本次实现成功为 Qwen 模型添加了多分辨率 padding 模式支持：

1. ✅ **核心功能**：支持不同分辨率图像在同一 batch 中训练
2. ✅ **代码质量**：完整的单元测试和端到端验证
3. ✅ **兼容性**：保持向后兼容，不影响现有功能
4. ✅ **文档**：详细的设计文档和使用说明
5. ✅ **可扩展**：为 Trainer 层的集成奠定基础

**关键创新点**：
- 自动检测和适配 2D/3D RoPE 格式
- 统一的 `extract_batch_field` 工具
- 基于相对误差的严格验证标准
- 完整的 patch 机制，无需修改原始模型代码

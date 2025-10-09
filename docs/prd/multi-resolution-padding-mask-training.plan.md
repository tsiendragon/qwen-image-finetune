<!-- dec23b68-366b-4ffe-a19f-3400695dabd5 / ea430290-a4f1-48cd-9cea-59bd5f31cb4d -->
# Multi-resolution Training with Per-Sample IDs
Target:
1. 支持多种分辨率的训练
2. 单个样本的inference走之前的逻辑，也就是和之前的预测结果应该保持一致
3. 使用 padding 和attention mask 的思路去实现这个想法
4. 最小代码改动，不破坏原本的逻辑，可以新增不同的 branch
5. 拆分不同 trainer 共用的代码和每个 trainer 需要特定修改的内容
## 0. 概述与当前进度

目标：允许同一 batch 内的样本使用不同的目标分辨率，确保数据流（配置 → 预处理 → 缓存 → collate → trainer → 模型 → loss）全链路正确处理 padding，并保证 pad token 的 attention/output 被屏蔽。

当前完成情况：

- ✅ 数据管线（配置、预处理、collate）规划已明确；
- ✅ `FluxTransformer2DModel` 已支持 `attention_mask`（零出 padding token，生成 additive mask）；
- ✅ `tests/models/test_flux_transformer_padding.py` 覆盖等价性测试；
- ⏳ `FluxTransformer2DModel` 仍未实现 **per-sample RoPE**（仍复用 diffusers 的 shared `img_ids` 逻辑）；
- ⏳ Trainer 端 per-sample latents/ids、Qwen Trainer 迁移均待完成。

以下内容恢复原有的完整计划，按组件说明所有改动及 TODO。重点更新了「模型层 / Flux」章节，标记现状与下一步。

### 0.1 数据格式规范（Type Definitions）

统一的类型定义和数据约定：

```python
from typing import List, Tuple
import torch

# 数据格式规范
ImgShapes = List[Tuple[int, int, int]]  # [(C, H', W'), ...] 每个样本的 latent shape
# 示例: [(1, 52, 104), (1, 64, 64)] 表示两个样本，一个 52x104，一个 64x64

ImgIdsBatched = torch.Tensor  # shape: (B, seq, 3) for per-sample mode
ImgIdsShared = torch.Tensor   # shape: (seq, 3) for shared resolution mode
# img_ids[..., 0] = batch_idx (在 shared mode 时为 0)
# img_ids[..., 1] = h_idx (height position index)
# img_ids[..., 2] = w_idx (width position index)

AttentionMask = torch.Tensor  # shape: (B, seq_txt + seq_img), bool or float
# True/1 表示有效 token，False/0 表示 padding token
```

### 0.2 数据流约定

| 阶段 | img_shapes 格式 | img_ids 格式 | 说明 |
|------|----------------|-------------|------|
| **预处理输出** | N/A | N/A | 返回 `height`, `width` 标量 |
| **Dataset.__getitem__** | N/A | N/A | 单样本，返回标量维度 |
| **缓存文件** | `torch.tensor([(1, H', W'), ...])` | N/A | 保存为 tensor，无 batch 维度 |
| **Collate 输出** | `List[List[(1, H', W'), ...]]` | N/A | 外层 list 长度为 batch_size |
| **Trainer prepare_embeddings** | `List[List[(1, H', W'), ...]]` | 根据模式生成 | 判断是否多分辨率 |
| **模型 forward (shared)** | N/A | `(seq, 3)` | 单分辨率模式 |
| **模型 forward (per-sample)** | N/A | `(B, seq, 3)` | 多分辨率模式 |

---

## 1. 配置层 (`src/data/config.py`)

### 1.1 `ImageProcessorInitArgs`

- 新增字段：
  ```python
  multi_resolutions: Optional[Union[List[int], List[str]]] = None
  ```
- Validator：解析 `"512*768"` 等字符串为整数面积；保留旧字段兼容；
- **多分辨率判断**：只要 `multi_resolutions` 存在且非空，即启用多分辨率模式，无需额外开关；
- 优先级：
  1. `target_size`（固定尺寸，禁用多分辨率）
  2. `multi_resolutions`（多分辨率模式）
  3. `target_pixels`（单一面积，禁用多分辨率）

### 1.2 配置示例

#### 1.2.1 多分辨率配置

```yaml
data:
  init_args:
    processor:
      class_path: src.data.preprocess.ImageProcessor
      init_args:
        process_type: fixed_pixels
        # 存在 multi_resolutions 即自动启用多分辨率
        multi_resolutions:
          - "512*512"    # 262144
          - "640*640"    # 409600
          - "768*512"    # 393216
          - "832*576"    # 479232
        max_aspect_ratio: 3.0  # 可选：最大宽高比限制

    # Collate 配置
    batch_size: 4
    # 注意：多分辨率下，实际 batch token 数会增加（padding）
    # 建议根据 max_resolution 调整 batch_size

loss:
  use_mask_loss: true
  mask_loss_fn:
    foreground_weight: 2.0
    background_weight: 1.0
  normalize_by_valid_tokens: true  # 新增：按实际 token 数归一化
```

#### 1.2.2 单分辨率配置（向后兼容）

```yaml
data:
  init_args:
    processor:
      init_args:
        # 方式 1: 使用固定尺寸
        target_size: [512, 512]

        # 方式 2: 使用单一面积（不提供 multi_resolutions）
        # target_pixels: 262144  # 512*512
```

---

## 2. 数据预处理 (`src/data/preprocess.py`)

### 2.1 选择候选面积

- 在 `ImageProcessor.__init__` 中保存 `self.multi_resolutions`；
- 新增 `_select_pixels_candidate(orig_w, orig_h)`：
  - `orig_area = orig_w * orig_h`；
  - 对每个候选 `A_i` 计算 `err = abs(A_i - orig_area) / orig_area`；
  - 误差相同时选面积更小者；
  - 如仍并列，比较 `calculate_best_resolution` 得到的尺寸与原比例的差异。

### 2.2 `_process_image`

```python
if self.multi_resolutions:
    best_pixels = self._select_pixels_candidate(w, h)
    new_w, new_h = calculate_best_resolution(w, h, best_pixels)
elif self.target_size:
    ...
```

- target image、mask、control 共用同一 `(new_w, new_h)`；
- 多控制分支逐个调用；
- `preprocess` 返回 `height`, `width`, `height_control`, `width_control`, `height_control_i`, `width_control_i` 供 Trainer 使用；
- mask 输出保持 `[H, W]`，值域 `[0, 1]`。

---

## 3. 数据集与缓存 (`src/data/dataset.py`)

### 3.1 缓存策略与兼容性

#### 3.1.1 缓存版本管理

在 `cache_step` 保存时添加版本标记：

```python
cache_dict = {
    "version": "2.0",  # 新增：标记支持多分辨率
    "prompt_embeds": ...,
    "pooled_prompt_embeds": ...,
    "img_shapes": torch.tensor(img_shapes_list),  # [(1, H', W'), ...]
    # 旧版本（v1.0）没有 version 和 img_shapes 字段
}
```

需保存的字段：
  - `img_shapes`: `torch.tensor([(1, H', W'), ...])`；
  - 控制分支的形状 `control_shapes`, `control_i_shapes`；
  - 若启用 mask loss，保存 latent 尺寸的 mask。

#### 3.1.2 加载时的兼容处理

在 `prepare_cached_embeddings` 中：

```python
def prepare_cached_embeddings(self, batch: dict):
    cache_version = batch.get("version", "1.0")

    if cache_version == "1.0":
        # 旧缓存格式：没有 img_shapes，从 height/width 重建
        batch_size = batch["prompt_embeds"].shape[0]
        height = batch.get("height", self.config.data.default_height)
        width = batch.get("width", self.config.data.default_width)

        # 生成统一的 img_shapes
        latent_h = height // self.vae_scale_factor // 2
        latent_w = width // self.vae_scale_factor // 2
        img_shapes = [[(1, latent_h, latent_w)]] * batch_size
        batch["image_latents_shapes"] = img_shapes

        logging.info(f"Loaded v1.0 cache, reconstructed shapes: {img_shapes[0]}")

    elif cache_version == "2.0":
        # 新缓存格式：直接使用 img_shapes
        img_shapes_tensor = batch["img_shapes"]  # [N, 3]
        batch_size = batch["prompt_embeds"].shape[0]
        img_shapes = img_shapes_tensor.tolist()
        batch["image_latents_shapes"] = [img_shapes] * batch_size

    else:
        raise ValueError(f"Unsupported cache version: {cache_version}")

    # mask 处理（如果存在）
    if "mask" in batch:
        if batch["mask"].dim() == 2:  # [H, W] - 旧格式
            # 需要 latent 化
            mask_bhw = batch["mask"].unsqueeze(0)  # [1, H, W]
            mask_latent = map_mask_to_latent(mask_bhw).squeeze(0)  # [seq]
            batch["mask"] = mask_latent
        # else: 已是 [seq] 格式，直接使用
```

#### 3.1.3 缓存迁移工具（可选）

提供脚本 `scripts/migrate_cache_v1_to_v2.py`：

```python
def migrate_cache(old_path: str, new_path: str):
    """迁移 v1.0 缓存到 v2.0 格式"""
    data = torch.load(old_path)

    # 添加 version 字段
    data["version"] = "2.0"

    # 重建 img_shapes（如果缺失）
    if "img_shapes" not in data:
        height = data.get("height", 512)
        width = data.get("width", 512)
        # 假设 vae_scale_factor=8, packing=2
        latent_h = height // 8 // 2
        latent_w = width // 8 // 2
        data["img_shapes"] = torch.tensor([(1, latent_h, latent_w)])

    torch.save(data, new_path)
    logging.info(f"Migrated {old_path} -> {new_path}")
```

### 3.2 Collate Function

```python
from src.loss.edit_mask_loss import map_mask_to_latent

def collate_fn(batch):
    ...
    if key == "mask":
        latent_masks = []
        for mask in batch_dict[key]:
            if mask.dim() == 2:
                latent_mask = map_mask_to_latent(mask.unsqueeze(0)).squeeze(0)
            else:
                latent_mask = mask  # 缓存模式
            latent_masks.append(latent_mask)
        batch_dict[key] = pad_to_max_shape(latent_masks)
        continue
    ...
```

- 记录 `image_latents_shapes` / `control_latents_shapes`；
- 嵌套 dict 继续递归处理。

---

## 4. Trainer 层（以 `FluxKontextLoraTrainer` 为例）

### 4.1 `prepare_embeddings` 与边界情况处理

#### 4.1.1 `_should_use_multi_resolution_mode` 完整实现

```python
def _should_use_multi_resolution_mode(self, batch: dict) -> bool:
    """判断是否需要启用多分辨率模式

    返回 False 的情况：
    1. batch_size == 1（单样本直接走原逻辑）
    2. 所有样本尺寸完全一致（不需要 padding）
    3. 未配置 multi_resolutions（单分辨率配置）

    返回 True 的情况：
    1. 配置了 multi_resolutions
    2. batch_size > 1
    3. batch 中存在不同尺寸的样本
    """
    batch_size = batch["prompt_embeds"].shape[0]
    if batch_size == 1:
        return False

    # 检查是否有 _shapes 字段（collate 输出）
    if "image_latents_shapes" in batch:
        shapes = batch["image_latents_shapes"]
        if len(shapes) != batch_size:
            raise ValueError(
                f"image_latents_shapes length {len(shapes)} != batch_size {batch_size}"
            )
        # 检查所有样本是否同尺寸
        first_shape = shapes[0]
        if all(s == first_shape for s in shapes):
            logging.debug("All samples have identical resolution, using shared mode")
            return False
        return True

    # 回退：检查 height/width 字段
    if "height" in batch and "width" in batch:
        heights = batch["height"] if isinstance(batch["height"], list) else [batch["height"]] * batch_size
        widths = batch["width"] if isinstance(batch["width"], list) else [batch["width"]] * batch_size

        if len(set(zip(heights, widths))) == 1:
            return False  # 所有样本同尺寸
        return True

    # 无法判断，默认 False（保守策略）
    logging.warning("Cannot determine resolution mode, falling back to shared mode")
    return False
```

**关键简化**：
- 是否多分辨率由配置 `multi_resolutions` 决定（在预处理阶段已确定）
- 运行时仅检查 batch 内样本尺寸是否一致

#### 4.1.2 同尺寸优化

当检测到所有样本尺寸一致时：
- 不执行 padding
- 使用原 batched `prepare_latents`
- 不传递 `attention_mask`
- 生成 shared `img_ids` (seq, 3)

#### 4.1.3 `prepare_embeddings` 流程

1. 调用 `_should_use_multi_resolution_mode(batch)` 判断模式；
2. 若多分辨率：
   - 对每样本调用 `prepare_latents(single_image, 1, ...)`，收集 `image_latents_list`, `image_ids_list`；
   - 记录 `image_latents_shapes = [lat.shape for lat in image_latents_list]`；
   - 控制分支同理（设 `type_id`）；
   - `pad_to_max_shape` 统一尺寸；
   - mask collate 后已是 `[seq]`，直接使用；
3. 单分辨率 → 保留旧逻辑。

### 4.2 `_pad_latents_for_multi_res`

```python
def _pad_latents_for_multi_res(latents_list, ids_list=None):
    ...
    padded_latents = torch.zeros(B, max_seq, C, ...)
    attention_mask = torch.zeros(B, max_seq, dtype=torch.bool)
    if ids_list:
        padded_ids = torch.zeros(B, max_seq, ids_dim, ...)
    ...
    if ids_list:
        return padded_latents, attention_mask, padded_ids
    return padded_latents, attention_mask
```

### 4.3 `_compute_loss_multi_resolution` 与 Loss 归一化

#### 4.3.1 当前问题

padding 后的 loss 计算会包含无效 token，导致：
- 梯度被稀释（loss 被 padding token 的零梯度平均）
- 不同 batch 的 loss scale 不一致

#### 4.3.2 完整实现流程

1. 根据 `_shapes` 切片真实序列；
2. 拼接 image/control latent，调用 `_pad_latents_for_multi_res`；
3. 构造 additive mask：
   ```python
   additive_mask = latent_model_input.new_full(attention_mask.shape, 0.0)
   additive_mask[~attention_mask] = float("-inf")
   ```
4. 调用模型：
   ```python
   model_pred = self.dit(
       hidden_states=latent_model_input.to(self.weight_dtype),
       encoder_hidden_states=prompt_embeds.to(self.weight_dtype),
       timestep=t.to(self.weight_dtype),
       img_shapes=img_shapes_per_sample,           # TODO: per-sample RoPE 完成后生效
       joint_attention_kwargs={'attention_mask': additive_mask},
       ...
   )
   ```

#### 4.3.3 Loss 计算（修正版）

**关键**：需要按实际有效 token 数归一化，避免 padding 稀释梯度。

```python
# Step 5: 计算 loss（修正版）
if self.config.loss.get("use_mask_loss", False):
    # MaskEditLoss 内部已按序列长度归一化，但需调整为实际长度
    loss_unreduced = self.mask_loss_fn(
        mask=mask_latent,
        model_pred=model_pred,
        target=target,
        weighting=weighting,
        reduction='none'  # 返回 [B, seq, C]
    )
    # 仅保留真实 token 的 loss
    valid_mask = attention_mask[:, seq_txt:].unsqueeze(-1)  # [B, seq_img, 1]
    loss_masked = loss_unreduced * valid_mask.float()

    # 按实际 token 数归一化
    num_valid_tokens = valid_mask.sum()
    loss = loss_masked.sum() / num_valid_tokens.clamp(min=1)
else:
    # MSE loss 同样需要 mask
    mse = F.mse_loss(model_pred, target, reduction='none')  # [B, seq, C]
    valid_mask = attention_mask[:, seq_txt:].unsqueeze(-1)
    mse_masked = mse * valid_mask.float()

    num_valid_tokens = valid_mask.sum() * mse.shape[-1]  # 包含通道维度
    loss = mse_masked.sum() / num_valid_tokens.clamp(min=1)

# 可选：记录 padding ratio 用于监控
padding_ratio = 1.0 - (num_valid_tokens.item() / (attention_mask.shape[0] * attention_mask.shape[1]))
if padding_ratio > 0.3:  # 超过 30% padding 时警告
    logging.debug(f"High padding ratio: {padding_ratio:.2%}")
```

**关键点**：
- `attention_mask[:, seq_txt:]` 提取图像部分的 mask
- `num_valid_tokens.clamp(min=1)` 避免除零
- 需要修改 `MaskEditLoss` 支持 `reduction='none'`

#### 4.3.4 其它注意事项

- cache / 非 cache 流程保持一致；
- `MaskEditLoss` 仅作用真实 image token。

### 4.4 其它注意事项

- `prepare_cached_embeddings`：恢复 `_shapes`，对 mask 做 latent 化；
- `_should_use_multi_resolution` → False 时保留 legacy 路径；
- Trainer 需根据模型层的进展决定 `img_ids` 的结构（当前仍需共享尺寸，待模型更新后输出 per-sample）。

---

## 5. 模型层

### 5.1 FluxTransformer2DModel（`src/models/transformer_flux_custom.py`）

**已完成**：

- `attention_mask` 处理（bool/float 兼容、零出 padding token、生成 additive mask）；
- 单元测试 `test_padding_equivalence_single_sample` / `test_attention_mask_accepts_float`。

**仍需完成**：per-sample RoPE / img_ids。

**计划与实现细节**：

#### 5.1.1 保留原样参考
`src/models/transformer_flux.py` 存放 diffusers 原实现作为对比。

#### 5.1.2 Trainer 端 img_ids 生成

在 `prepare_latents` 中为每个样本生成 img_ids：

     ```python
def _prepare_latent_image_ids_batched(
    self,
    shapes: List[Tuple[int, int, int]],  # [(1, H', W'), ...]
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, int]:
    """生成 batched img_ids 用于 per-sample RoPE

    Returns:
        img_ids: (B, max_seq, 3) - 包含 batch_idx, h_idx, w_idx
        max_seq: 最大序列长度（padding 后）
    """
    B = len(shapes)
    ids_list = []

    for b, (_, height, width) in enumerate(shapes):
        # 为单个样本生成 img_ids
        ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        ids[..., 0] = b  # batch index
        ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
        ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
        ids_list.append(ids.view(-1, 3))  # (H'*W', 3)

    # Padding 到最大长度
    max_seq = max(ids.shape[0] for ids in ids_list)
    padded_ids = torch.zeros(B, max_seq, 3, device=device, dtype=dtype)

    for b, ids in enumerate(ids_list):
        padded_ids[b, :ids.shape[0]] = ids

    return padded_ids, max_seq
```

#### 5.1.3 模型端 per-sample RoPE

在 `FluxTransformer2DModel` 中：

```python
def _compute_per_sample_rope(
    self,
    img_ids: torch.Tensor  # (B, seq, 3)
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """为每个样本计算独立的 RoPE 频率

    Returns:
        List of (txt_freqs, img_freqs) for each sample in batch
    """
    B = img_ids.shape[0]
    result = []

    for b in range(B):
        img_ids_b = img_ids[b]  # (seq, 3)

        # 从 img_ids 提取实际尺寸
        # img_ids_b[:, 1] 是 h_idx，最大值 + 1 = height
        # img_ids_b[:, 2] 是 w_idx，最大值 + 1 = width
        h_max = img_ids_b[:, 1].max().item() + 1
        w_max = img_ids_b[:, 2].max().item() + 1

        # 生成该样本的 RoPE 频率
        # （具体实现参考 diffusers 的 rope 计算逻辑）
        # txt_freqs_b = ... (根据 txt_ids 计算)
        # img_freqs_b = ... (根据 h_max, w_max 计算)

        # 缓存该尺寸的频率（避免重复计算）
        cache_key = (b, h_max, w_max)
        if cache_key not in self._rope_cache:
            txt_freqs_b = self._compute_text_rope(...)
            img_freqs_b = self._compute_image_rope(h_max, w_max, ...)
            self._rope_cache[cache_key] = (txt_freqs_b, img_freqs_b)
        else:
            txt_freqs_b, img_freqs_b = self._rope_cache[cache_key]

        result.append((txt_freqs_b, img_freqs_b))

    return result
```

#### 5.1.4 forward 中判断模式

```python
def forward(self, ..., img_ids, ...):
    # 判断 img_ids 是否包含 batch 维
    if img_ids is not None and img_ids.ndim == 3:
        # Per-sample mode: img_ids shape (B, seq, 3)
        image_rotary_emb = self._compute_per_sample_rope(img_ids)
         per_sample_mode = True
     else:
        # Shared mode: img_ids shape (seq, 3)
        image_rotary_emb = self._compute_shared_rope(img_ids)
         per_sample_mode = False

    # 传递给 attention blocks
    ...
     ```

#### 5.1.5 自定义 Attention Processor

     ```python
class FluxAttnProcessorPerSample:
    """支持 per-sample RoPE 的 attention processor"""

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb,  # List[(txt_freqs, img_freqs)] or Tuple
        **kwargs
    ):
        # 判断是否 per-sample 模式
        if isinstance(image_rotary_emb, list):
            # Per-sample mode
            B = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]

            # 对每个样本单独应用 RoPE
            rotated_hidden = []
            for b in range(B):
                txt_freqs, img_freqs = image_rotary_emb[b]
                hs_b = hidden_states[b:b+1]  # (1, seq, dim)
                # 应用 RoPE
                hs_b_rotated = apply_rope(hs_b, txt_freqs, img_freqs)
                rotated_hidden.append(hs_b_rotated)

            hidden_states = torch.cat(rotated_hidden, dim=0)
        else:
            # Shared mode - 原逻辑
            hidden_states = apply_rope(hidden_states, *image_rotary_emb)

        # 后续 attention 计算（与 additive mask 协同）
        return original_flux_attn_processor(...)
```

#### 5.1.6 兼容性保证

- 若 `img_ids` 没有 batch 维（ndim == 2），走原逻辑；
- 单样本 inference / 同尺寸 batch 不受影响；
- 与 additive mask 协同工作（加 -inf 屏蔽 padding）。

**TODO 列表**：

- [ ] `_prepare_latent_image_ids_batched` 的实现（Trainer 端）；
- [ ] `_compute_per_sample_rope` 的实现（模型端）；
- [ ] `FluxAttnProcessorPerSample` 实现；
- [ ] forward 中根据模式选择 processor；
- [ ] 新增 `test_flux_rope_per_sample`、`test_flux_multi_sample_padding`；
- [ ] 与 Trainer 对接（输出 per-sample `img_ids` / shapes）。

### 5.2 QwenImageTransformer2DModel（`src/models/transformer_qwenimage.py`）

保持上一版计划：

- `QwenEmbedRope.forward` 支持 per-sample；
- `QwenImageTransformer2DModel.forward` 根据 `img_shapes` 调整；
- `QwenDoubleStreamAttnProcessor2_0` 按样本拆分 RoPE；
- 与 Flux 实现一致的测试矩阵。

---

## 6. 测试计划

### 6.1 单元测试矩阵

| Test Case | 文件 | 状态 | 说明 |
|-----------|------|------|------|
| **模型层** |
| Padding 等价性（单样本） | `test_flux_transformer_padding.py` | ✅ | 已完成 |
| Float mask 兼容性 | `test_flux_transformer_padding.py` | ✅ | 已完成 |
| Per-sample RoPE 正确性 | `test_flux_transformer_padding.py` | ⏳ | **新增** |
| 混合尺寸 forward | `test_flux_transformer_padding.py` | ⏳ | **新增** |
| **Loss 计算** |
| 带 mask 的 loss 归一化 | `tests/loss/test_mask_loss.py` | ⏳ | **新增** |
| Padding token 零梯度验证 | `tests/loss/test_mask_loss.py` | ⏳ | **新增** |
| **Trainer 层** |
| 多分辨率 collate | `tests/trainer/test_flux_kontext_trainer.py` | ⏳ | **新增** |
| 混合尺寸训练 step | `tests/trainer/test_flux_kontext_trainer.py` | ⏳ | **新增** |
| 同尺寸 fallback | `tests/trainer/test_flux_kontext_trainer.py` | ⏳ | **新增** |
| **缓存兼容性** |
| v1.0 缓存加载 | `tests/data/test_cache_compatibility.py` | ⏳ | **新增** |
| v2.0 缓存保存/加载 | `tests/data/test_cache_compatibility.py` | ⏳ | **新增** |
| **Qwen 相关** |
| Qwen 模型迁移测试 | 待定 | ⏳ | Flux 完成后迁移 |

### 6.2 测试用例完整代码

#### 6.2.1 模型层测试

##### test_padding_equivalence_single_sample（已完成）

```python
def test_padding_equivalence_single_sample():
    """验证单样本 padding 等价性"""
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = _build_model().to(device)
    model.eval()

    batch_size = 1
    seq_txt = 4
    height, width = 2, 4  # seq_img = 8
    seq_img = height * width
    pad_tokens = 4

    hidden_states = torch.randn(batch_size, seq_img, 64, device=device)
    encoder_hidden_states = torch.randn(batch_size, seq_txt, 32, device=device)
    pooled = torch.randn(batch_size, 16, device=device)
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    img_ids = _prepare_latent_image_ids(height, width, device, torch.float32)
    txt_ids = torch.stack(
        [torch.tensor([0.0, float(i), 0.0], device=device) for i in range(seq_txt)],
        dim=0,
    )

    # Baseline: 无 padding
    with torch.no_grad():
        out_base = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample

    # 添加 padding
    hidden_states_padded = torch.cat(
        [hidden_states, torch.zeros(batch_size, pad_tokens, 64, device=device)], dim=1
    )
    extra_ids = torch.zeros(pad_tokens, 3, device=device)
    extra_ids[:, 1] = height + torch.arange(pad_tokens, device=device)
    img_ids_padded = torch.cat([img_ids, extra_ids], dim=0)

    mask = torch.ones(batch_size, seq_txt + seq_img + pad_tokens, device=device, dtype=torch.bool)
    mask[:, seq_txt + seq_img :] = False

    with torch.no_grad():
        out_padded = model(
            hidden_states=hidden_states_padded,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids_padded,
            txt_ids=txt_ids,
            attention_mask=mask,
        ).sample

    # 验证：非 padding 部分应该一致
    assert torch.allclose(out_base, out_padded[:, :seq_img], atol=1e-5)
```

##### test_attention_mask_accepts_float（已完成）

```python
def test_attention_mask_accepts_float():
    """验证 attention_mask 支持 float 和 bool"""
    torch.manual_seed(1)
    device = torch.device("cpu")

    model = _build_model().to(device)
    model.eval()

    batch_size = 1
    seq_txt = 2
    height, width = 2, 2
    seq_img = height * width

    hidden_states = torch.randn(batch_size, seq_img, 64, device=device)
    encoder_hidden_states = torch.randn(batch_size, seq_txt, 32, device=device)
    pooled = torch.randn(batch_size, 16, device=device)
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    img_ids = _prepare_latent_image_ids(height, width, device, torch.float32)
    txt_ids = torch.stack(
        [torch.tensor([0.0, float(i), 0.0], device=device) for i in range(seq_txt)],
        dim=0,
    )

    hidden_states_padded = torch.cat(
        [hidden_states, torch.zeros(batch_size, 2, 64, device=device)], dim=1
    )
    img_ids_padded = torch.cat([img_ids, torch.zeros(2, 3, device=device)], dim=0)

    mask_float = torch.ones(batch_size, seq_txt + seq_img + 2, device=device)
    mask_float[:, seq_txt + seq_img :] = 0.0
    mask_bool = mask_float.bool()

    with torch.no_grad():
        out_float = model(
            hidden_states=hidden_states_padded,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids_padded,
            txt_ids=txt_ids,
            attention_mask=mask_float,
        ).sample

        out_bool = model(
            hidden_states=hidden_states_padded,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids_padded,
            txt_ids=txt_ids,
            attention_mask=mask_bool,
        ).sample

    # 验证：float 和 bool mask 结果一致
    assert torch.allclose(out_float, out_bool, atol=1e-5)
```

##### test_flux_rope_per_sample（新增）

```python
def test_flux_rope_per_sample():
    """验证不同尺寸样本的 RoPE 计算正确性"""
    torch.manual_seed(2)
    device = torch.device("cpu")

    model = _build_model().to(device)
    model.eval()

    # 两个不同尺寸的样本
    batch_size = 2
    seq_txt = 3
    shapes = [(1, 2, 4), (1, 3, 3)]  # seq_len = 8 和 9
    max_seq = 9

    # 生成 batched img_ids (B, seq, 3)
    img_ids_batched = torch.zeros(batch_size, max_seq, 3, device=device)
    for b, (_, h, w) in enumerate(shapes):
        ids = torch.zeros(h, w, 3, device=device)
        ids[..., 0] = b
        ids[..., 1] = torch.arange(h, device=device)[:, None]
        ids[..., 2] = torch.arange(w, device=device)[None, :]
        img_ids_batched[b, :h*w] = ids.view(-1, 3)

    # 生成 txt_ids (shared)
    txt_ids = torch.stack(
        [torch.tensor([0.0, float(i), 0.0], device=device) for i in range(seq_txt)],
        dim=0,
    )

    # Hidden states 和 mask
    hidden_states = torch.randn(batch_size, max_seq, 64, device=device)
    encoder_hidden_states = torch.randn(batch_size, seq_txt, 32, device=device)
    pooled = torch.randn(batch_size, 16, device=device)
    timestep = torch.randint(0, 1000, (batch_size,), device=device)

    attention_mask = torch.zeros(batch_size, seq_txt + max_seq, dtype=torch.bool, device=device)
    attention_mask[0, :seq_txt + 8] = True  # 第一个样本 8 个 token
    attention_mask[1, :seq_txt + 9] = True  # 第二个样本 9 个 token

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids_batched,  # (B, seq, 3) 触发 per-sample mode
            txt_ids=txt_ids,
            attention_mask=attention_mask,
        )

    # 验证 1: padding 部分应该全零
    assert torch.allclose(output.sample[0, 8:], torch.zeros_like(output.sample[0, 8:]), atol=1e-5)

    # 验证 2: 输出 shape 正确
    assert output.sample.shape == (batch_size, max_seq, 64)

    # 验证 3: 非 padding 部分不全为零
    assert not torch.allclose(output.sample[0, :8], torch.zeros_like(output.sample[0, :8]))
    assert not torch.allclose(output.sample[1, :9], torch.zeros_like(output.sample[1, :9]))
```

##### test_flux_multi_sample_padding（新增）

```python
def test_flux_multi_sample_padding():
    """验证混合尺寸 batch 的 forward 正确性"""
    torch.manual_seed(3)
    device = torch.device("cpu")

    model = _build_model().to(device)
    model.eval()

    batch_size = 3
    seq_txt = 2
    shapes = [(1, 2, 2), (1, 3, 4), (1, 2, 3)]  # seq_len = 4, 12, 6
    max_seq = 12

    # 构造 batched 输入
    img_ids_batched = torch.zeros(batch_size, max_seq, 3, device=device)
    hidden_states = torch.randn(batch_size, max_seq, 64, device=device)
    attention_mask = torch.zeros(batch_size, seq_txt + max_seq, dtype=torch.bool, device=device)

    for b, (_, h, w) in enumerate(shapes):
        seq_len = h * w
        ids = torch.zeros(h, w, 3, device=device)
        ids[..., 0] = b
        ids[..., 1] = torch.arange(h, device=device)[:, None]
        ids[..., 2] = torch.arange(w, device=device)[None, :]
        img_ids_batched[b, :seq_len] = ids.view(-1, 3)
        attention_mask[b, :seq_txt + seq_len] = True

    txt_ids = torch.stack(
        [torch.tensor([0.0, float(i), 0.0], device=device) for i in range(seq_txt)],
        dim=0,
    )
    encoder_hidden_states = torch.randn(batch_size, seq_txt, 32, device=device)
    pooled = torch.randn(batch_size, 16, device=device)
    timestep = torch.randint(0, 1000, (batch_size,), device=device)

    with torch.no_grad():
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids_batched,
            txt_ids=txt_ids,
            attention_mask=attention_mask,
        )

    # 验证：每个样本的 padding 部分全零
    assert torch.allclose(output.sample[0, 4:], torch.zeros_like(output.sample[0, 4:]), atol=1e-5)
    assert torch.allclose(output.sample[2, 6:], torch.zeros_like(output.sample[2, 6:]), atol=1e-5)
    # 样本 1 没有 padding

    # 验证：非 padding 部分有值
    assert output.sample[0, :4].abs().sum() > 0
    assert output.sample[1, :12].abs().sum() > 0
    assert output.sample[2, :6].abs().sum() > 0
```

#### 6.2.2 Loss 计算测试

##### test_mask_loss_with_reduction_none（新增）

```python
def test_mask_loss_with_reduction_none():
    """验证 MaskEditLoss 支持 reduction='none'"""
    from src.loss.edit_mask_loss import MaskEditLoss

    loss_fn = MaskEditLoss(foreground_weight=2.0, background_weight=1.0)

    B, seq, C = 2, 10, 64
    mask = torch.rand(B, seq) > 0.5  # 随机 mask
    pred = torch.randn(B, seq, C)
    target = torch.randn(B, seq, C)

    # 测试 reduction='none'
    loss_unreduced = loss_fn(mask, pred, target, reduction='none')

    # 验证 shape
    assert loss_unreduced.shape == (B, seq, C)

    # 验证与手动计算一致
    element_loss = (pred - target) ** 2
    weight_mask = (mask.float() * 2.0 + (~mask).float() * 1.0).unsqueeze(-1)
    expected = element_loss * weight_mask

    assert torch.allclose(loss_unreduced, expected, atol=1e-5)
```

##### test_loss_normalization_with_padding（新增）

```python
def test_loss_normalization_with_padding():
    """验证 padding 情况下 loss 正确归一化"""
    import torch.nn.functional as F

    # 两个样本：8 和 6 个有效 token
    B, max_seq, C = 2, 10, 64
    seq_txt = 2

    pred = torch.randn(B, max_seq, C)
    target = torch.randn(B, max_seq, C)

    # Attention mask: [B, seq_txt + max_seq]
    attention_mask = torch.zeros(B, seq_txt + max_seq, dtype=torch.bool)
    attention_mask[0, :seq_txt + 8] = True  # 样本 0: 8 个有效 image token
    attention_mask[1, :seq_txt + 6] = True  # 样本 1: 6 个有效 image token

    # 提取 image 部分的 mask
    valid_mask = attention_mask[:, seq_txt:].unsqueeze(-1)  # [B, max_seq, 1]

    # 计算 masked MSE loss
    mse = F.mse_loss(pred, target, reduction='none')  # [B, max_seq, C]
    mse_masked = mse * valid_mask.float()
    num_valid_tokens = valid_mask.sum() * C
    loss = mse_masked.sum() / num_valid_tokens.clamp(min=1)

    # 手动验证
    valid_elements_0 = pred[0, :8] - target[0, :8]  # 样本 0
    valid_elements_1 = pred[1, :6] - target[1, :6]  # 样本 1
    expected = (valid_elements_0 ** 2).sum() + (valid_elements_1 ** 2).sum()
    expected = expected / ((8 + 6) * C)

    assert torch.allclose(loss, expected, atol=1e-5)

    # 验证：如果不做归一化，loss 会被稀释
    naive_loss = mse.mean()
    assert naive_loss < loss  # naive loss 被 padding 稀释
```

##### test_padding_token_zero_gradient（新增）

```python
def test_padding_token_zero_gradient():
    """验证 padding token 的梯度为零"""
    B, max_seq, C = 2, 10, 64
    seq_txt = 2

    pred = torch.randn(B, max_seq, C, requires_grad=True)
    target = torch.randn(B, max_seq, C)

    attention_mask = torch.zeros(B, seq_txt + max_seq, dtype=torch.bool)
    attention_mask[0, :seq_txt + 8] = True
    attention_mask[1, :seq_txt + 6] = True

    valid_mask = attention_mask[:, seq_txt:].unsqueeze(-1)

    # 计算 loss
    mse = (pred - target) ** 2
    mse_masked = mse * valid_mask.float()
    loss = mse_masked.sum() / valid_mask.sum().clamp(min=1)

    # 反向传播
    loss.backward()

    # 验证：padding 位置的梯度应该为零
    assert torch.allclose(pred.grad[0, 8:], torch.zeros_like(pred.grad[0, 8:]), atol=1e-6)
    assert torch.allclose(pred.grad[1, 6:], torch.zeros_like(pred.grad[1, 6:]), atol=1e-6)

    # 验证：非 padding 位置有梯度
    assert pred.grad[0, :8].abs().sum() > 0
    assert pred.grad[1, :6].abs().sum() > 0
```

#### 6.2.3 Trainer 层测试

##### test_should_use_multi_resolution_mode（新增）

```python
def test_should_use_multi_resolution_mode():
    """验证 _should_use_multi_resolution_mode 判断逻辑"""
    from src.trainer.flux_kontext_trainer import FluxKontextLoraTrainer

    # Mock config
    config = type('Config', (), {})()
    trainer = FluxKontextLoraTrainer(config)

    # Case 1: batch_size == 1 → False
    batch = {"prompt_embeds": torch.randn(1, 128, 768)}
    assert not trainer._should_use_multi_resolution_mode(batch)

    # Case 2: 所有样本同尺寸 → False
    batch = {
        "prompt_embeds": torch.randn(4, 128, 768),
        "image_latents_shapes": [[(1, 32, 64)]] * 4,  # 全部相同
    }
    assert not trainer._should_use_multi_resolution_mode(batch)

    # Case 3: 不同尺寸 → True
    batch = {
        "prompt_embeds": torch.randn(4, 128, 768),
        "image_latents_shapes": [[(1, 32, 64)], [(1, 48, 48)], [(1, 32, 64)], [(1, 40, 50)]],
    }
    assert trainer._should_use_multi_resolution_mode(batch)

    # Case 4: height/width fallback
    batch = {
        "prompt_embeds": torch.randn(3, 128, 768),
        "height": [512, 640, 512],
        "width": [512, 640, 768],
    }
    assert trainer._should_use_multi_resolution_mode(batch)
```

##### test_pad_latents_for_multi_res（新增）

```python
def test_pad_latents_for_multi_res():
    """验证 latents padding 函数"""
    from src.trainer.flux_kontext_trainer import FluxKontextLoraTrainer

    config = type('Config', (), {})()
    trainer = FluxKontextLoraTrainer(config)

    # 三个不同长度的 latents
    latents_list = [
        torch.randn(1, 8, 16),   # seq=8
        torch.randn(1, 12, 16),  # seq=12
        torch.randn(1, 6, 16),   # seq=6
    ]

    # 对应的 ids
    ids_list = [
        torch.randn(8, 3),
        torch.randn(12, 3),
        torch.randn(6, 3),
    ]

    # Padding
    padded_latents, attention_mask, padded_ids = trainer._pad_latents_for_multi_res(
        latents_list, ids_list
    )

    # 验证 shape
    assert padded_latents.shape == (3, 12, 16)  # max_seq = 12
    assert attention_mask.shape == (3, 12)
    assert padded_ids.shape == (3, 12, 3)

    # 验证 mask
    assert attention_mask[0, :8].all() and not attention_mask[0, 8:].any()
    assert attention_mask[1, :12].all()
    assert attention_mask[2, :6].all() and not attention_mask[2, 6:].any()

    # 验证原始值保留
    assert torch.allclose(padded_latents[0, :8], latents_list[0].squeeze(0))
    assert torch.allclose(padded_latents[1, :12], latents_list[1].squeeze(0))
    assert torch.allclose(padded_ids[2, :6], ids_list[2])

    # 验证 padding 为零
    assert torch.allclose(padded_latents[0, 8:], torch.zeros_like(padded_latents[0, 8:]))
```

##### test_mixed_resolution_training_step（新增）

```python
def test_mixed_resolution_training_step():
    """验证混合分辨率训练的完整 step"""
    import os
    from unittest import mock

    os.environ.setdefault("HF_TOKEN", "dummy")
    with mock.patch("huggingface_hub.login"):
        from src.trainer.flux_kontext_trainer import FluxKontextLoraTrainer

    # 构造 minimal config
    config = type('Config', (), {
        'model': type('Model', (), {
            'pretrained_model_name_or_path': 'black-forest-labs/FLUX.1-Kontext',
        })(),
        'data': {},
        'loss': {'use_mask_loss': False},
    })()

    trainer = FluxKontextLoraTrainer(config)
    trainer.setup()  # 加载模型

    # 构造混合尺寸 batch
    batch = {
        "image": torch.randn(2, 3, 512, 512),  # 会被处理成不同尺寸
        "prompt": ["a cat", "a dog"],
        "image_latents_shapes": [[(1, 32, 64)], [(1, 48, 48)]],
        "prompt_embeds": torch.randn(2, 77, 768),
        "pooled_prompt_embeds": torch.randn(2, 768),
    }

    # 执行训练 step
    trainer.train()
    loss = trainer.training_step(batch, 0)

    # 验证 loss 是标量且可反向传播
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # 验证梯度可以传播
    loss.backward()
    # 检查模型参数有梯度
    has_grad = any(p.grad is not None for p in trainer.dit.parameters() if p.requires_grad)
    assert has_grad
```

##### test_same_resolution_fallback（新增）

```python
def test_same_resolution_fallback():
    """验证同尺寸 batch 自动回退到 shared mode"""
    from src.trainer.flux_kontext_trainer import FluxKontextLoraTrainer

    config = type('Config', (), {})()
    trainer = FluxKontextLoraTrainer(config)

    # 所有样本同尺寸
    batch = {
        "prompt_embeds": torch.randn(4, 128, 768),
        "image_latents_shapes": [[(1, 32, 64)]] * 4,
        "height": [512] * 4,
        "width": [1024] * 4,
    }

    # 应该使用 shared mode
    assert not trainer._should_use_multi_resolution_mode(batch)

    # prepare_embeddings 应该走原逻辑（不做 padding）
    # 这需要 mock 一些方法，这里只验证判断逻辑
```

#### 6.2.4 缓存兼容性测试

##### test_cache_v1_loading（新增）

```python
def test_cache_v1_loading():
    """验证 v1.0 缓存加载兼容性"""
    from src.trainer.flux_kontext_trainer import FluxKontextLoraTrainer

    config = type('Config', (), {
        'data': {'default_height': 512, 'default_width': 512}
    })()
    trainer = FluxKontextLoraTrainer(config)
    trainer.vae_scale_factor = 8

    # 模拟 v1.0 缓存（没有 version 和 img_shapes）
    batch_v1 = {
        "prompt_embeds": torch.randn(2, 77, 768),
        "pooled_prompt_embeds": torch.randn(2, 768),
        "height": 512,
        "width": 512,
        # 注意：没有 "version" 和 "img_shapes"
    }

    # 加载兼容处理
    trainer.prepare_cached_embeddings(batch_v1)

    # 验证：自动重建 img_shapes
    assert "image_latents_shapes" in batch_v1
    shapes = batch_v1["image_latents_shapes"]
    assert len(shapes) == 2  # batch_size = 2

    # 验证尺寸计算正确
    expected_h = 512 // 8 // 2  # 32
    expected_w = 512 // 8 // 2  # 32
    assert shapes[0] == [(1, expected_h, expected_w)]
    assert shapes[1] == [(1, expected_h, expected_w)]
```

##### test_cache_v2_save_and_load（新增）

```python
def test_cache_v2_save_and_load():
    """验证 v2.0 缓存保存和加载"""
    import tempfile
    import os

    from src.data.dataset import ImageEditDataset

    # 构造数据
    cache_data = {
        "version": "2.0",
        "prompt_embeds": torch.randn(5, 77, 768),
        "pooled_prompt_embeds": torch.randn(5, 768),
        "img_shapes": torch.tensor([
            (1, 32, 64),
            (1, 48, 48),
            (1, 40, 50),
            (1, 32, 64),
            (1, 36, 54),
        ]),
    }

    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        cache_path = f.name
        torch.save(cache_data, cache_path)

    try:
        # 加载
        loaded = torch.load(cache_path)

        # 验证版本
        assert loaded["version"] == "2.0"

        # 验证 img_shapes
        assert loaded["img_shapes"].shape == (5, 3)
        assert torch.equal(loaded["img_shapes"][0], torch.tensor([1, 32, 64]))
        assert torch.equal(loaded["img_shapes"][1], torch.tensor([1, 48, 48]))

        # 验证其他字段
        assert loaded["prompt_embeds"].shape == (5, 77, 768)
        assert loaded["pooled_prompt_embeds"].shape == (5, 768)
    finally:
        os.unlink(cache_path)
```

##### test_cache_migration（新增）

```python
def test_cache_migration():
    """验证缓存迁移工具"""
    import tempfile
    import os

    # 模拟 v1.0 缓存
    cache_v1 = {
        "prompt_embeds": torch.randn(3, 77, 768),
        "height": 512,
        "width": 1024,
    }

    # 保存 v1
    with tempfile.NamedTemporaryFile(delete=False, suffix='_v1.pt') as f:
        cache_v1_path = f.name
        torch.save(cache_v1, cache_v1_path)

    with tempfile.NamedTemporaryFile(delete=False, suffix='_v2.pt') as f:
        cache_v2_path = f.name

    try:
        # 执行迁移
        def migrate_cache(old_path, new_path):
            data = torch.load(old_path)
            data["version"] = "2.0"

            if "img_shapes" not in data:
                height = data.get("height", 512)
                width = data.get("width", 512)
                latent_h = height // 8 // 2
                latent_w = width // 8 // 2
                batch_size = data["prompt_embeds"].shape[0]
                data["img_shapes"] = torch.tensor(
                    [(1, latent_h, latent_w)] * batch_size
                )

            torch.save(data, new_path)

        migrate_cache(cache_v1_path, cache_v2_path)

        # 验证迁移结果
        cache_v2 = torch.load(cache_v2_path)
        assert cache_v2["version"] == "2.0"
        assert "img_shapes" in cache_v2
        assert cache_v2["img_shapes"].shape == (3, 3)

        # 验证尺寸计算
        expected_h = 512 // 8 // 2  # 32
        expected_w = 1024 // 8 // 2  # 64
        assert torch.equal(cache_v2["img_shapes"][0], torch.tensor([1, expected_h, expected_w]))
    finally:
        os.unlink(cache_v1_path)
        if os.path.exists(cache_v2_path):
            os.unlink(cache_v2_path)
```

### 6.3 测试优先级

**高优先级（必须通过）**：
- ✅ 单样本等价性（已有）
- 🔴 Per-sample RoPE 正确性
- 🔴 Loss 归一化正确性
- 🔴 缓存兼容性

**中优先级（重要但非阻塞）**：
- 🟡 混合尺寸梯度传播
- 🟡 同尺寸 fallback

**低优先级（可选）**：
- ⚪ 极端尺寸比例测试
- ⚪ 性能基准测试

---

## 7. 其它 Trainer 迁移

| Trainer | 改动要点 | 说明 |
|---------|----------|------|
| `src/trainer/qwen_image_edit_trainer.py` | 复用 `_should_use_multi_resolution_mode`、`_pad_latents_for_multi_res`、`_compute_loss_multi_resolution`；缓存模式转回 list-of-shapes；mask latent 化 | `_get_image_shapes` 需扩展 per-sample；cache / 非 cache 保持一致 |
| `src/trainer/qwen_image_edit_plus_trainer.py` | 同上，额外处理多控制分支 | `n_controls`, `height_control_i`, `width_control_i` 必须参与 |

抽象建议：

- 将 `_should_use_multi_resolution` 等工具提升到 BaseTrainer 或 util；
- Flux 版本就绪后再迁移 Qwen 系列。

---

## 8. 下一步 Checklist

按依赖关系排序的实施计划：

### Phase 1: 基础设施（1-2 天）
1. 添加类型定义和数据格式文档（Section 0.1, 0.2）
2. 实现 `_should_use_multi_resolution_mode`（Section 4.1.1）
3. 更新 Loss 计算逻辑，支持 `reduction='none'`（Section 4.3.3）
4. 实现缓存版本管理（Section 3.1）

### Phase 2: 模型层（2-3 天）
5. 实现 `_prepare_latent_image_ids_batched`（Section 5.1.2）
6. 实现 `_compute_per_sample_rope`（Section 5.1.3）
7. 实现 `FluxAttnProcessorPerSample`（Section 5.1.5）
8. 单元测试：`test_flux_rope_per_sample`、`test_flux_multi_sample_padding`

### Phase 3: Trainer 集成（1-2 天）
9. 更新 `prepare_embeddings` 支持 per-sample 路径
10. 实现 `_pad_latents_for_multi_res`
11. 更新 `_compute_loss_multi_resolution`
12. 测试：`test_training_with_mixed_resolutions`

### Phase 4: 验证与优化（1-2 天）
13. 端到端测试（数据 → 训练 → 推理）
14. 性能基准测试
15. 文档更新与 Migration Guide
16. Code review 与重构

**总计：约 6-9 天**

---

## 9. Migration Guide（迁移指南）

### 9.1 从单分辨率迁移

#### 9.1.1 迁移步骤

1. **更新配置文件**：添加 `multi_resolutions` 即自动启用多分辨率

```yaml
# 旧配置（单分辨率）
data:
  init_args:
    processor:
      init_args:
        target_pixels: 262144  # 512*512

# 新配置（多分辨率）
data:
  init_args:
    processor:
      init_args:
        # 存在 multi_resolutions 即自动启用多分辨率
        multi_resolutions:
          - "512*512"
          - "640*640"
          - "768*512"
```

2. **重新生成缓存**（推荐）或使用兼容模式

3. **验证训练**：先用 batch_size=1 测试单样本一致性

4. **逐步增加 batch_size**

#### 9.1.2 验证检查点

```bash
# 1. 单样本推理一致性
python -m src.main --config configs/test.yaml --mode predict --batch_size 1

# 2. 混合分辨率训练
python -m src.main --config configs/test.yaml --mode fit --batch_size 4

# 3. 检查 loss 是否合理
# - Loss 不应因 batch 中尺寸组合而剧烈波动
# - 相同样本在不同 batch 中的 loss 应接近
```

### 9.2 缓存数据处理

#### 选项 1：重新生成（推荐）

```bash
python -m src.main --config configs/xxx.yaml --mode cache
```

**优点**：完全兼容新格式，性能最佳
**缺点**：需要重新处理所有数据（耗时）

#### 选项 2：使用兼容模式

- 代码会自动检测 v1.0 缓存
- 运行时重建 `img_shapes`
- 性能轻微下降（需动态计算）

**优点**：无需重新生成缓存
**缺点**：运行时开销，不支持真正的多分辨率

#### 选项 3：批量迁移

```bash
python scripts/migrate_cache_v1_to_v2.py \
    --input_dir data/cache/v1/ \
    --output_dir data/cache/v2/
```

**优点**：保留缓存数据，支持多分辨率
**缺点**：需要手动执行迁移脚本

### 9.3 常见问题排查

#### Q1: Loss 突然变大或出现 NaN

**可能原因**：
- Padding token 参与了 loss 计算
- Loss 归一化不正确

**解决方法**：
```python
# 检查 attention_mask 是否正确传递
logging.info(f"attention_mask sum: {attention_mask.sum()}")

# 确认 loss 按有效 token 归一化
num_valid = attention_mask.sum()
logging.info(f"Valid tokens: {num_valid}, Total: {attention_mask.numel()}")
```

#### Q2: 显存占用显著增加

**可能原因**：
- Padding 到最大尺寸导致显存浪费

**解决方法**：
- 减小 batch_size
- 限制 `multi_resolutions` 的范围
- 配置 `max_aspect_ratio` 避免极端尺寸

#### Q3: 单样本推理结果与之前不一致

**可能原因**：
- `_should_use_multi_resolution_mode` 判断错误
- 单样本走了多分辨率路径

**解决方法**：
```python
# 确认单样本使用 shared mode
if batch_size == 1:
    assert not self._should_use_multi_resolution_mode(batch)
```

### 9.4 性能监控建议

虽然文档不包含性能优化策略，但建议记录以下指标用于监控：

```python
# 在训练循环中添加
padding_ratio = (total_tokens - valid_tokens) / total_tokens
logging.info(f"Padding ratio: {padding_ratio:.2%}")

# 每个 epoch 统计
avg_padding = sum(padding_ratios) / len(padding_ratios)
if avg_padding > 0.3:
    logging.warning(f"High average padding ratio: {avg_padding:.2%}")
```

### 9.5 Rollback 方案

如果多分辨率训练出现问题，可以快速回退到单分辨率：

```yaml
# 方式 1：移除 multi_resolutions，使用固定尺寸
data:
  init_args:
    processor:
      init_args:
        target_size: [512, 512]  # 回到固定尺寸

# 方式 2：改用单一面积
data:
  init_args:
    processor:
      init_args:
        target_pixels: 262144  # 512*512，不使用 candidates
```

---

## 附录：关键设计决策

### A0. 配置设计简化

**决策**：存在 `multi_resolutions` 即自动启用多分辨率，无需额外开关。

**理由**：
- 避免配置冗余（`multi_resolutions` + `multi_resolution.enabled`）
- 配置意图明确：有候选列表 = 多分辨率；无候选列表 = 单分辨率
- 简化判断逻辑，减少配置错误（如候选列表存在但 enabled=false 的矛盾情况）
- 向后兼容：旧配置（`target_size` 或 `target_pixels`）自动使用单分辨率模式

### A1. 为什么不使用 Bucketing？

按用户要求，不使用 bucketing 策略。主要考虑：
- 实现复杂度较高
- 可能影响数据随机性
- Padding 方案更通用

### A2. 为什么需要 Per-sample RoPE？

RoPE（Rotary Position Embedding）依赖序列的空间结构（height, width）。不同尺寸的样本必须使用各自的位置编码，否则位置信息会错乱。

### A3. Loss 归一化的必要性

假设 batch 中有两个样本：
- 样本 A：512x512，有效 token = 4096
- 样本 B：1024x1024，padding 到 16384

如果不按有效 token 归一化：
```python
loss = (pred - target).pow(2).mean()  # 错误！
# 样本 A 的 loss 被稀释为 4096/16384 = 25%
```

正确做法：
```python
loss = ((pred - target).pow(2) * valid_mask).sum() / valid_mask.sum()
# 样本 A 和 B 的 loss 权重相同
```

---

此文档详细补充了多分辨率训练的实现细节，包括数据格式、边界情况、缓存兼容性、per-sample RoPE、测试计划和迁移指南，为后续实施提供完整的技术规范。

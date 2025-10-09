<!-- dec23b68-366b-4ffe-a19f-3400695dabd5 ea430290-a4f1-48cd-9cea-59bd5f31cb4d -->
# Multi-resolution Training with Per-Sample IDs

## 0. 概述

目标：在 FluxKontext / Qwen Image Edit / Qwen Image Edit Plus 训练链路上支持同一 batch 内混合分辨率样本。核心思路：

- 配置层允许给定多个候选面积，按原图尺寸自动挑选最合适的目标尺寸；
- 数据预处理到 collate_function 全链路保留原始尺寸信息，并在 collate 时完成 mask → latent 的对齐；
- Trainer 端按样本生成 VAE latent、img_ids 和控制 latents，拼接后统一 padding，使用 attention mask 屏蔽；
- 模型侧（FluxTransformer2DModel、QwenImageTransformer2DModel）支持 per-sample 形状 + additive mask；
- Loss 计算仅在真实 token 范围内进行，兼容 mask loss；
- 补充测试确保 padding 引入后模型输出保持一致。

下文按 **配置 → 数据处理 → 缓存 → Collate → Trainer → 模型 → 测试 → 其它 Trainer 迁移** 顺序说明。

---

## 1. 配置层（`src/data/config.py`）

- `ImageProcessorInitArgs` 新增字段：
  ```python
  target_pixels_candidates: Optional[Union[List[int], List[str]]] = None
  ```
- Validator 需解析 `"512*768"` 等格式为整数面积，保留 `target_size`、`target_pixels` 兼容；
- 优先级：`target_size` > `target_pixels_candidates` > `target_pixels`；
- 若配置有候选面积，需在 docs 中给出写法示例（见附录）。

---

## 2. 数据预处理（`src/data/preprocess.py`）

### 2.1 选择候选面积

- `ImageProcessor.__init__` 保存 `self.target_pixels_candidates`;
- 新增 `_select_pixels_candidate(orig_w, orig_h)`：
  - `orig_area = orig_w * orig_h`;
  - 遍历候选面积 `A_i`，计算误差 `err_i = abs(A_i - orig_area) / orig_area`;
  - 若误差相同，优先面积较小者；再次并列时比较 `calculate_best_resolution` 后 `(new_w, new_h)` 与原始比例的差异；
  - 返回最优候选 `A*`。

### 2.2 `_process_image`

```python
if self.target_pixels_candidates:
    best_pixels = self._select_pixels_candidate(w, h)
    new_w, new_h = calculate_best_resolution(w, h, best_pixels)
else:
    # 原逻辑：target_size 或 target_pixels
```

- main image / mask / control 共用 `(new_w, new_h)`；多控制独立选择；
- `preprocess` 返回 `height`, `width`, `height_control`, `width_control`, `height_control_i`, `width_control_i`；
- mask 输出保持二维 `[H, W]`，值域 `[0,1]`。

---

## 3. 缓存（`cache_step` & `prepare_cached_embeddings`）

- `cache_step` 仍按单样本缓存，但需额外存储：
  - `img_shapes`：`torch.tensor([(1, H', W'), ...])`；
  - `control_shapes` / `control_i_shapes`；
  - `mask` 若启用 mask loss，需保存 latent 形态（见 collate）；
- `prepare_cached_embeddings` 加载后：
  - 将 `img_shapes` 转换为 `List[List[Tuple[int,int,int]]]`；
  - 若 mask 仍是 `[H,W]`，补执行 `resize_bhw + map_mask_to_latent`。

---

## 4. Collate Function（`src/data/dataset.py::collate_fn`）

- 文件顶部新增：
  ```python
  from src.loss.edit_mask_loss import map_mask_to_latent
  ```
- 处理逻辑：
  ```python
  if key == "mask":
      latent_masks = []
      for mask in batch_dict[key]:
          if mask.dim() == 2:
              latent_mask = map_mask_to_latent(mask.unsqueeze(0)).squeeze(0)
          else:
              latent_mask = mask  # 缓存场景已是 latent 大小
          latent_masks.append(latent_mask)
      batch_dict[key] = pad_to_max_shape(latent_masks)
      continue
  ```
- 对 `image_latents` / `control_latents` 记录 `_shapes`（list of shape tuples）；
- 嵌套 dict 递归调用保持已有行为。

---

## 5. Trainer 层：多分辨率准备与 Loss 计算

以下以 `FluxKontextLoraTrainer` 为例，`QwenImageEditTrainer` / `QwenImageEditPlusTrainer` 需按同思路迁移（见第 9 节）。

### 5.1 `prepare_embeddings`

1. 新增 `_should_use_multi_resolution_mode(batch)`：
   - 若 `batch_size == 1` → `False`;
   - 若显式存在 `_shapes` / `img_shapes` 且长度等于 batch → `True`;
   - 否则比较 `height` / `width` 是否完全一致。
2. 当需要多分辨率：
   - 对每个样本单独调用 `prepare_latents(single_image, 1, ...)`，收集 `image_latents_list`、`image_ids_list`；
   - 记录 `image_latents_shapes = [lat.shape for lat in image_latents_list]`;
   - 对主控制图 / 额外控制图重复以上操作，设置 `type_id`（主图=0，control=1，control_i=1+i）；
   - `pad_to_max_shape` 对 `image_latents_list`、`image_ids_list` 做 padding；
   - 若启用 mask loss，则 collate 后的 `[seq]` mask 可直接使用。
3. 非多分辨率 -> 保持旧逻辑（一次性批处理）。

### 5.2 `_pad_latents_for_multi_res`

```python
def _pad_latents_for_multi_res(
    self,
    latents_list: List[torch.Tensor],  # [(1, seq_i, C)]
    ids_list: Optional[List[torch.Tensor]] = None,  # [(seq_i, 3)]
):
    batch_size = len(latents_list)
    max_seq = max(lat.shape[1] for lat in latents_list)
    C = latents_list[0].shape[2]
    device = latents_list[0].device

    padded_latents = torch.zeros(batch_size, max_seq, C, device=device, dtype=latents_list[0].dtype)
    attention_mask = torch.zeros(batch_size, max_seq, device=device, dtype=torch.bool)
    if ids_list is not None:
        padded_ids = torch.zeros(batch_size, max_seq, ids_list[0].shape[-1], device=device, dtype=ids_list[0].dtype)

    for i, lat in enumerate(latents_list):
        seq_len = lat.shape[1]
        padded_latents[i, :seq_len] = lat[0]
        attention_mask[i, :seq_len] = True
        if ids_list is not None:
            ids = ids_list[i]
            padded_ids[i, :ids.shape[0]] = ids
            if ids.shape[0] < max_seq:
                padded_ids[i, ids.shape[0]:] = ids[-1:]

    if ids_list is not None:
        return padded_latents, attention_mask, padded_ids
    return padded_latents, attention_mask
```

### 5.3 `_compute_loss_multi_resolution`

核心流程：
1. 从 `image_latents` / `control_latents` / `_shapes` 取真实序列长度；
2. 对每个样本裁剪噪声后的 latent，拼接 image + control；
3. 调用 `_pad_latents_for_multi_res(..., ids_list)` 获得 `latent_model_input`、`attention_mask`、`padded_ids`；
4. 构造 additive mask：
   ```python
   additive_mask = latent_model_input.new_full(attention_mask.shape, 0.0)
   additive_mask[~attention_mask] = float("-inf")
   ```
5. 调用模型：
   ```python
   model_pred = self.dit(
       hidden_states=latent_model_input.to(self.weight_dtype),
       encoder_hidden_states=prompt_embeds.to(self.weight_dtype),
       timestep=t.to(self.weight_dtype),
       img_shapes=img_shapes_per_sample,   # List[List[(frame, H', W')]]
       guidance=guidance,
       attention_kwargs={'attention_mask': additive_mask},
       return_dict=False,
   )[0]
   ```
6. 对每个样本计算 loss：
   - 未启用 mask → `F.mse_loss(pred_b, target_b, reduction='mean')`;
   - 启用 mask → 使用 `MaskEditLoss`，`mask_b = latent_masks[idx, : img_seq_len]`；
   - 所有样本 loss 求均值。
7. 说明：
   - `img_seq_len` 来自 `image_latents_shapes`，padding token 已剔除；
   - 如需在未来保留 padding 参与加权，可传入 `attention_mask.float().unsqueeze(-1)` 作为 weighting。

### 5.4 其它注意点

- `cache_step` / `prepare_cached_embeddings` 中需要同步处理 `img_shapes`、`mask`；
- `_should_use_multi_resolution` 返回 `False` 时继续走 legacy 路径，保证同尺寸 batch 或单样本预测不变；
- 若 Trainer 需要自定义 Transformer（如 Flux），在初始化阶段加载自定义版本：
  ```python
  from src.models.transformer_flux_custom import FluxTransformer2DModel
  self.dit = FluxTransformer2DModel.from_pretrained(...).to(self.weight_dtype)
  ```

---

## 6. 模型层：FluxTransformer2DModel（`src/models/transformer_flux_custom.py`）

1. 将官方 `FluxTransformer2DModel` 复制到 `src/models/transformer_flux_custom.py`；
2. 改造 `forward`：
   - 接收 `(B, seq_img, C)` 的 `latent_model_input` 和 `(B, max_seq)` 的 attention mask；
   - 支持 `img_ids`、`control_ids` 已经按 batch padding，直接拼接；
   - attention processor 使用 additive mask 屏蔽 padding；
3. Trainer 端通过 `_pad_latents_for_multi_res(..., ids_list)` 构造 `padded_ids` 后传入；
4. 保持与原始接口兼容（单分辨率或无 mask 时逻辑一致）。

---

## 7. 模型层：QwenImageTransformer2DModel（`src/models/transformer_qwenimage.py`）

### 7.1 `QwenEmbedRope.forward`

- 改为遍历 `video_fhw`，为每个样本生成独立的 `(frame, H', W')` 频率：
  ```python
  if not isinstance(video_fhw, (list, tuple)):
      video_fhw = [video_fhw]

  per_sample_freqs = []
  for sample_idx, fhw_list in enumerate(video_fhw):
      if not isinstance(fhw_list, (list, tuple)):
          fhw_list = [fhw_list]
      sample_freqs = []
      for frame, height, width in fhw_list:
          rope_key = (sample_idx, frame, height, width)
          if rope_key not in self.rope_cache:
              self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width)
          sample_freqs.append(self.rope_cache[rope_key].to(device))
      per_sample_freqs.append(torch.cat(sample_freqs, dim=0))
  txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_txt_len, ...]
  return per_sample_freqs, txt_freqs
  ```

### 7.2 `QwenImageTransformer2DModel.forward`

- 接受 `img_shapes` 为 `List[List[Tuple[int,int,int]]]`；
- 调用 `self.pos_embed(img_shapes, txt_seq_lens, device)` → 得到 `(per_sample_img_freqs, txt_freqs)`；
- 对每个样本写入 `(img_freqs_b, txt_freqs_b)`，组成 `image_rotary_emb = [(..., ...), ...]`；
- 若 `img_shapes` 缺失或仅单个 shape → 仍调用 legacy 路径。

### 7.3 `QwenDoubleStreamAttnProcessor2_0.__call__`

```python
if isinstance(image_rotary_emb, list):
    img_query_list, img_key_list = [], []
    txt_query_list, txt_key_list = [], []
    for b, (img_freqs_b, txt_freqs_b) in enumerate(image_rotary_emb):
        img_query_list.append(apply_rotary_emb_qwen(img_query[b:b+1], img_freqs_b, use_real=False))
        img_key_list.append(apply_rotary_emb_qwen(img_key[b:b+1], img_freqs_b, use_real=False))
        txt_query_list.append(apply_rotary_emb_qwen(txt_query[b:b+1], txt_freqs_b, use_real=False))
        txt_key_list.append(apply_rotary_emb_qwen(txt_key[b:b+1], txt_freqs_b, use_real=False))
    img_query = torch.cat(img_query_list, dim=0)
    img_key = torch.cat(img_key_list, dim=0)
    txt_query = torch.cat(txt_query_list, dim=0)
    txt_key = torch.cat(txt_key_list, dim=0)
else:
    img_freqs, txt_freqs = image_rotary_emb
    ...
```

### 7.4 Attention Mask

- `attention_kwargs` 传入 additive mask（`float`，padding 区域为 `-inf`）：
  ```python
  additive_mask = attention_kwargs.get("attention_mask") if attention_kwargs else None
  joint_hidden_states = dispatch_attention_fn(
      joint_query, joint_key, joint_value,
      attn_mask=additive_mask,
      dropout_p=0.0, is_causal=False, backend=self._attention_backend,
  )
  ```
- 保证 mask 与 `[txt_seq + img_seq]` 长度一致。

---

## 8. 测试计划

1. `tests/models/test_transformer_padding.py::test_transformer_padding_equivalence`
   - 构造无 padding 与 padding+mask 两组输入，比较输出是否 `torch.allclose(..., atol=1e-5)`；
   - 验证 Transformer 正确屏蔽 padding token。
2. `test_per_sample_rope_generation`
   - 检查不同 `(H', W')` 下的 RoPE 输出与 legacy 一致。
3. `test_loss_masking_correctness`
   - 构造混合分辨率 batch，确认 mask loss 区分 image / control。
4. `test_training_with_mixed_resolutions`
   - 运行一个训练 step，确保 loss 可回传、无 NaN。

---

## 9. 其它 Trainer 迁移

| Trainer | 主要改动 | 说明 |
|---------|----------|------|
| `src/trainer/qwen_image_edit_trainer.py` | 复用 `_should_use_multi_resolution_mode`、`_pad_latents_for_multi_res`、`_compute_loss_multi_resolution`；缓存流程补齐 `img_shapes` 与 mask latent | 原有 `_get_image_shapes` 需扩展到 per-sample；cache 模式下恢复 list-of-shapes。 |
| `src/trainer/qwen_image_edit_plus_trainer.py` | 同上，额外处理多控制分支：遍历 `control_i` 收集形状、设置 `type_id = i+1` | `n_controls`、`height_control_i`、`width_control_i` 必须参与。 |

迁移建议：
1. 抽象 `_should_use_multi_resolution` / `_pad_latents_for_multi_res` / `_compute_loss_multi_resolution` 到基类或 shared helper；
2. 先完成 FluxKontext 版本并通过测试，再迁移 Qwen 系列；
3. 确保 cache / 非 cache 行为一致。

---

## 10. 附录

### 10.1 配置示例

```yaml
data:
  init_args:
    processor:
      class_path: src.data.preprocess.ImageProcessor
      init_args:
        process_type: fixed_pixels
        target_pixels_candidates:
          - "512*512"
          - "640*640"
          - 786432   # 768*1024
```

### 10.2 验证清单

- [ ] `target_pixels_candidates` 解析及优先级正确；
- [ ] mask 在 collate 前后长度与 `image_latents` 对齐；
- [ ] `_should_use_multi_resolution` 与 `_pad_latents_for_multi_res` 在 cache / 非 cache 下结果一致；
- [ ] `test_transformer_padding_equivalence`、`test_loss_masking_correctness`、`test_training_with_mixed_resolutions` 通过；
- [ ] QwenImageTransformer2DModel 在混合分辨率下生成独立 RoPE 并使用 attention mask。

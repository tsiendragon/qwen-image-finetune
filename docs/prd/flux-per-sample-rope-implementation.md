# Flux Transformer Per-Sample RoPE Implementation Guide

## 问题分析

### 当前实现的限制

#### 1. img_shapes → latent_ids → positional embedding 流程

**当前流程（Shared Mode）**：
```python
# 第728-736行 (src/models/transformer_flux.py)
if img_ids.ndim == 3:
    logger.warning("Passing `img_ids` 3d torch.Tensor is deprecated.")
    img_ids = img_ids[0]  # 强制压缩为 2D

ids = torch.cat((txt_ids, img_ids), dim=0)  # (seq_txt + seq_img, 3)
image_rotary_emb = self.pos_embed(ids)       # 所有样本共享同一个 RoPE
```

**问题所在**：
- `img_ids` 是 2D 的 `(seq, 3)`，所有 batch 样本共享
- `pos_embed(ids)` 生成的 RoPE 对整个 batch 都一样
- 无法处理多分辨率训练中不同样本的不同空间尺寸

#### 2. FluxPosEmbed 的实现

```python
# 第510-538行
class FluxPosEmbed(nn.Module):
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids 形状: (seq, 3) - 只能处理 2D 输入
        n_axes = ids.shape[-1]  # 3 (type, height, width)
        pos = ids.float()

        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],  # 例如: [16, 56, 56]
                pos[:, i],          # 位置索引
                ...
            )

        return freqs_cos, freqs_sin  # 形状: (seq, D)
```

**特点**：
- 输入：`ids` 形状 `(seq, 3)`
- 输出：`(freqs_cos, freqs_sin)` 形状 `(seq, D)`
- **只能生成单个序列的 RoPE，无法区分不同样本**

#### 3. Attention Processor 中的 RoPE 应用

```python
# 第130-132行 (FluxAttnProcessor)
if image_rotary_emb is not None:
    query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
```

**特点**：
- `image_rotary_emb` 是共享的 `(seq, D)`
- 应用到所有样本的 query/key 上
- **无法为不同样本应用不同的 RoPE**

---

## 多分辨率场景的需求

### 场景示例

```python
# Batch 中的 3 个样本
batch = {
    'img_shapes': [
        [(3, 512, 512), ...],   # 样本 0: 512x512 -> latent (32, 32) -> seq=1024
        [(3, 640, 640), ...],   # 样本 1: 640x640 -> latent (40, 40) -> seq=1600
        [(3, 768, 512), ...],   # 样本 2: 768x512 -> latent (48, 32) -> seq=1536
    ]
}
```

### 需要的处理

1. **Per-sample img_ids 生成**（✅ 已实现）：
   ```python
   # src/trainer/flux_kontext_trainer.py 第644-646行
   image_ids_list = self._prepare_latent_image_ids_batched(
       latent_hw_list,  # [(32,32), (40,40), (48,32)]
       device, dtype
   )
   # 返回: [tensor(1024,3), tensor(1600,3), tensor(1536,3)]
   ```

2. **Padding 到统一长度**（✅ 已实现）：
   ```python
   # 第675-692行
   image_ids_batched = torch.stack(image_ids_padded_list, dim=0)
   # 形状: (B=3, max_seq=1600, 3)
   ```

3. **Per-sample RoPE 生成**（❌ 未实现）：
   - 需要为每个样本独立计算 RoPE
   - 不同样本有不同的空间维度 (32x32 vs 40x40 vs 48x32)

4. **Per-sample RoPE 应用**（❌ 未实现）：
   - 在 attention 中为每个样本应用各自的 RoPE
   - Padding 部分不应用 RoPE

---

## 实现方案

### 方案概览

```
┌─────────────────────────────────────────────────────────────────┐
│ Trainer: 生成 per-sample img_ids                                │
│ ✅ 已实现: _prepare_latent_image_ids_batched                     │
│ 输出: List[Tensor(seq_i, 3)] -> Padded Tensor(B, max_seq, 3)  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ FluxTransformer2DModel.forward                                  │
│ ⏳ 待实现: 检测 img_ids 维度并选择模式                          │
│ - 如果 img_ids.ndim == 2: shared mode (当前逻辑)               │
│ - 如果 img_ids.ndim == 3: per-sample mode (新逻辑)             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ FluxPosEmbed (扩展版)                                           │
│ ⏳ 待实现: 支持 batched ids                                      │
│ - forward_batched(ids: Tensor(B, seq, 3)) -> List[Tuple]       │
│ - 为每个样本生成独立的 (freqs_cos, freqs_sin)                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────────────────────┐
│ FluxAttnProcessorPerSample (新建)                               │
│ ⏳ 待实现: per-sample RoPE 应用                                  │
│ - 接收 List[Tuple[freqs_cos, freqs_sin]]                       │
│ - 为每个样本应用各自的 RoPE                                     │
│ - 与 attention_mask 协同工作                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 实现步骤

#### Step 1: 扩展 FluxPosEmbed 支持 batched ids

**位置**：`src/models/transformer_flux_custom.py`（或新建 `src/models/flux_rope_utils.py`）

```python
class FluxPosEmbedBatched(FluxPosEmbed):
    """扩展版的 FluxPosEmbed，支持 per-sample RoPE"""

    def forward_batched(
        self,
        ids: torch.Tensor,  # (B, seq, 3)
        valid_lengths: Optional[torch.Tensor] = None  # (B,) - 每个样本的实际长度
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        为每个样本生成独立的 RoPE

        Args:
            ids: (B, seq, 3) - batched image IDs
            valid_lengths: (B,) - 每个样本的实际序列长度（不含 padding）

        Returns:
            List of (freqs_cos, freqs_sin) tuples, one per sample
            每个 tuple 的形状: (seq_i, D) 或 (seq, D) if padded
        """
        batch_size = ids.shape[0]
        max_seq = ids.shape[1]

        rope_list = []
        for b in range(batch_size):
            ids_b = ids[b]  # (seq, 3)

            # 如果提供了有效长度，只处理有效部分
            if valid_lengths is not None:
                valid_len = valid_lengths[b].item()
                ids_valid = ids_b[:valid_len]  # (valid_len, 3)
            else:
                ids_valid = ids_b

            # 调用原始 forward 方法
            freqs_cos, freqs_sin = super().forward(ids_valid)

            # 如果需要 padding 到 max_seq
            if ids_valid.shape[0] < max_seq:
                # Padding RoPE（用零填充）
                pad_len = max_seq - ids_valid.shape[0]
                freqs_cos = torch.cat([
                    freqs_cos,
                    torch.zeros(pad_len, freqs_cos.shape[-1],
                               device=freqs_cos.device, dtype=freqs_cos.dtype)
                ], dim=0)
                freqs_sin = torch.cat([
                    freqs_sin,
                    torch.zeros(pad_len, freqs_sin.shape[-1],
                               device=freqs_sin.device, dtype=freqs_sin.dtype)
                ], dim=0)

            rope_list.append((freqs_cos, freqs_sin))

        return rope_list

    def forward(self, ids: torch.Tensor, **kwargs):
        """保持向后兼容的 forward 方法"""
        if ids.ndim == 3:
            # Batched mode
            return self.forward_batched(ids, **kwargs)
        else:
            # Original shared mode
            return super().forward(ids)
```

#### Step 2: 创建 Per-Sample Attention Processor

**位置**：`src/models/transformer_flux_custom.py`

```python
class FluxAttnProcessorPerSample:
    """支持 per-sample RoPE 的 Flux Attention Processor"""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0."
            )

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Union[
            Tuple[torch.Tensor, torch.Tensor],           # Shared mode
            List[Tuple[torch.Tensor, torch.Tensor]]      # Per-sample mode
        ]] = None,
    ) -> torch.Tensor:
        """
        Args:
            image_rotary_emb:
                - Shared mode: (freqs_cos, freqs_sin) with shape (seq, D)
                - Per-sample mode: List of (freqs_cos, freqs_sin),
                  each with shape (seq, D)
        """
        query, key, value, encoder_query, encoder_key, encoder_value = \
            _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        # ... QKV reshaping and normalization (same as original) ...
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Apply RoPE - 关键修改点
        if image_rotary_emb is not None:
            if isinstance(image_rotary_emb, list):
                # Per-sample mode
                batch_size = query.shape[0]
                query_list = []
                key_list = []

                for b in range(batch_size):
                    freqs_cos, freqs_sin = image_rotary_emb[b]
                    rope_tuple = (freqs_cos, freqs_sin)

                    # Apply RoPE to single sample
                    q_b = apply_rotary_emb(
                        query[b:b+1], rope_tuple, sequence_dim=1
                    )
                    k_b = apply_rotary_emb(
                        key[b:b+1], rope_tuple, sequence_dim=1
                    )

                    query_list.append(q_b)
                    key_list.append(k_b)

                query = torch.cat(query_list, dim=0)
                key = torch.cat(key_list, dim=0)
            else:
                # Shared mode (original behavior)
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Attention computation (same as original)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = \
                hidden_states.split_with_sizes(
                    [encoder_hidden_states.shape[1],
                     hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                    dim=1
                )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
```

#### Step 3: 修改 FluxTransformer2DModel.forward

**位置**：`src/models/transformer_flux_custom.py`（扩展现有类）

```python
class FluxTransformer2DModel(_FluxTransformer2DModel):
    """Custom Flux transformer with per-sample RoPE support"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 使用扩展版的 pos_embed
        self.pos_embed = FluxPosEmbedBatched(
            theta=10000,
            axes_dim=self.config.axes_dims_rope
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # ✅ 已支持
        **kwargs
    ):
        """Extended forward with per-sample RoPE support"""

        # ... (前面的代码保持不变) ...

        # 关键修改：检测 img_ids 维度并选择模式
        per_sample_mode = False

        if img_ids is not None and img_ids.ndim == 3:
            # Per-sample mode
            per_sample_mode = True
            logging.debug(f"Using per-sample RoPE mode: img_ids shape {img_ids.shape}")

            # 从 attention_mask 提取有效长度
            if attention_mask is not None:
                # attention_mask: (B, seq_txt + seq_img)
                seq_txt = txt_ids.shape[0] if txt_ids.ndim == 2 else txt_ids.shape[1]
                img_mask = attention_mask[:, seq_txt:]  # (B, seq_img)
                valid_lengths = img_mask.sum(dim=1)  # (B,)
            else:
                valid_lengths = None

            # 为每个样本生成 RoPE
            image_rotary_emb_list = self.pos_embed.forward_batched(
                img_ids, valid_lengths=valid_lengths
            )

            # 处理 txt_ids (仍然是 shared 的)
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]

            # 为每个样本创建完整的 rotary_emb (txt + img)
            full_rotary_emb_list = []
            for b in range(img_ids.shape[0]):
                # Text part (shared)
                txt_rope = self.pos_embed.forward(txt_ids)

                # Image part (per-sample)
                img_rope = image_rotary_emb_list[b]

                # Concatenate
                full_freqs_cos = torch.cat([txt_rope[0], img_rope[0]], dim=0)
                full_freqs_sin = torch.cat([txt_rope[1], img_rope[1]], dim=0)

                full_rotary_emb_list.append((full_freqs_cos, full_freqs_sin))

            image_rotary_emb = full_rotary_emb_list

        else:
            # Shared mode (original behavior)
            if txt_ids.ndim == 3:
                txt_ids = txt_ids[0]
            if img_ids is not None and img_ids.ndim == 3:
                img_ids = img_ids[0]

            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = self.pos_embed(ids)

        # 如果是 per-sample mode，需要使用特殊的 attention processor
        if per_sample_mode:
            # 临时替换 attention processors
            original_processors = {}
            for name, module in self.named_modules():
                if isinstance(module, FluxAttention):
                    original_processors[name] = module.processor
                    module.processor = FluxAttnProcessorPerSample()

        try:
            # Pass attention_mask through joint_attention_kwargs
            joint_attention_kwargs = joint_attention_kwargs or {}
            if attention_mask is not None:
                # 转换为 additive mask
                additive_mask = torch.zeros_like(attention_mask, dtype=hidden_states.dtype)
                additive_mask = additive_mask.masked_fill(~attention_mask, float("-inf"))
                joint_attention_kwargs["attention_mask"] = additive_mask

            # ... 调用 transformer blocks (使用 image_rotary_emb) ...
            for block in self.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,  # List or Tuple
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # ... (后续代码保持不变) ...

        finally:
            # 恢复原始 processors
            if per_sample_mode:
                for name, module in self.named_modules():
                    if name in original_processors:
                        module.processor = original_processors[name]

        return output
```

#### Step 4: Trainer 端的集成

**位置**：`src/trainer/flux_kontext_trainer.py`

当前第641-700行已经生成了 `image_ids_batched`，只需要确保传递给模型：

```python
# 第719行之后
# Step 6: Forward pass through transformer
model_pred = self.dit(
    hidden_states=latent_model_input,
    encoder_hidden_states=prompt_embeds,
    pooled_projections=pooled_prompt_embeds,
    timestep=t / 1000.0,
    img_ids=latent_ids,  # ✅ 已经是 (B, max_seq, 3) 形状
    txt_ids=text_ids,    # (seq_txt, 3) - shared
    guidance=guidance,
    attention_mask=attention_mask,  # ✅ 已支持
    joint_attention_kwargs={},
    return_dict=False,
)[0]
```

---

## 性能优化考虑

### 1. RoPE 计算缓存

```python
class FluxPosEmbedBatched(FluxPosEmbed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rope_cache = {}  # 缓存已计算的 RoPE

    def forward_batched(self, ids, valid_lengths=None):
        # ...
        for b in range(batch_size):
            # 生成 cache key
            if valid_lengths is not None:
                h, w = self._infer_hw_from_ids(ids[b], valid_lengths[b])
                cache_key = (h, w)

                if cache_key in self.rope_cache:
                    freqs_cos, freqs_sin = self.rope_cache[cache_key]
                else:
                    freqs_cos, freqs_sin = super().forward(ids_valid)
                    self.rope_cache[cache_key] = (freqs_cos, freqs_sin)
            # ...
```

### 2. Batch RoPE 计算

对于相同尺寸的样本，可以批量计算：

```python
def forward_batched_optimized(self, ids, valid_lengths):
    # 按尺寸分组
    shape_groups = defaultdict(list)
    for b in range(batch_size):
        h, w = self._infer_hw_from_ids(ids[b], valid_lengths[b])
        shape_groups[(h, w)].append(b)

    # 批量计算每组的 RoPE
    rope_dict = {}
    for (h, w), batch_indices in shape_groups.items():
        # 只计算一次
        rope = self._compute_rope(h, w)
        for b in batch_indices:
            rope_dict[b] = rope

    return [rope_dict[b] for b in range(batch_size)]
```

---

## 测试策略

### Test 1: Per-Sample RoPE 正确性

```python
def test_per_sample_rope_correctness():
    """验证 per-sample RoPE 与单独计算的 RoPE 一致"""
    model = FluxTransformer2DModel(...)

    # 三个不同尺寸的样本
    shapes = [(32, 32), (40, 40), (48, 32)]
    img_ids_list = [
        create_img_ids(h, w) for h, w in shapes
    ]

    # 方法 1: 单独计算
    ropes_individual = []
    for ids in img_ids_list:
        rope = model.pos_embed.forward(ids)
        ropes_individual.append(rope)

    # 方法 2: Batched 计算
    max_seq = max(ids.shape[0] for ids in img_ids_list)
    img_ids_batched = pad_ids_list(img_ids_list, max_seq)  # (B, max_seq, 3)
    valid_lengths = torch.tensor([ids.shape[0] for ids in img_ids_list])

    ropes_batched = model.pos_embed.forward_batched(
        img_ids_batched, valid_lengths
    )

    # 验证一致性
    for i in range(len(shapes)):
        valid_len = img_ids_list[i].shape[0]
        torch.testing.assert_close(
            ropes_individual[i][0][:valid_len],
            ropes_batched[i][0][:valid_len],
            msg=f"Sample {i} freqs_cos mismatch"
        )
        torch.testing.assert_close(
            ropes_individual[i][1][:valid_len],
            ropes_batched[i][1][:valid_len],
            msg=f"Sample {i} freqs_sin mismatch"
        )
```

### Test 2: 与 Attention Mask 的协同

```python
def test_per_sample_rope_with_attention_mask():
    """验证 per-sample RoPE 与 attention mask 正确协同"""
    model = FluxTransformer2DModel(...)

    # 构造混合尺寸的 batch
    batch_size, seq_txt = 2, 10
    shapes = [(32, 32), (40, 40)]  # seq_img = 1024, 1600
    max_seq = 1600

    # 构造 attention mask
    attention_mask = torch.zeros(batch_size, seq_txt + max_seq, dtype=torch.bool)
    attention_mask[0, :seq_txt + 1024] = True  # 样本 0: 1024 tokens
    attention_mask[1, :seq_txt + 1600] = True  # 样本 1: 1600 tokens

    # Forward pass
    output = model(
        hidden_states=hidden_states,  # (B, max_seq, C)
        encoder_hidden_states=encoder_hidden_states,
        img_ids=img_ids_batched,  # (B, max_seq, 3)
        txt_ids=txt_ids,  # (seq_txt, 3)
        attention_mask=attention_mask,
    )

    # 验证 padding 部分为零
    assert torch.allclose(
        output.sample[0, 1024:],
        torch.zeros_like(output.sample[0, 1024:]),
        atol=1e-5
    )
```

### Test 3: 向后兼容性

```python
def test_backward_compatibility():
    """确保 shared mode 仍然工作"""
    model = FluxTransformer2DModel(...)

    # 2D img_ids (shared mode)
    img_ids = create_img_ids(32, 32)  # (1024, 3)
    txt_ids = create_txt_ids(10)      # (10, 3)

    # 应该使用原始逻辑
    output = model(
        hidden_states=hidden_states,  # (B, 1024, C)
        encoder_hidden_states=encoder_hidden_states,
        img_ids=img_ids,  # 2D - shared mode
        txt_ids=txt_ids,
    )

    # 验证输出正确
    assert output.sample.shape == (batch_size, 1024, channels)
```

---

## 实施优先级

### Phase 1: 核心功能（高优先级）
1. ✅ FluxPosEmbedBatched 实现
2. ✅ FluxAttnProcessorPerSample 实现
3. ✅ FluxTransformer2DModel 模式检测和切换
4. ✅ 基础测试（正确性、协同性）

### Phase 2: 优化和完善（中优先级）
5. ⏳ RoPE 缓存机制
6. ⏳ Batch 优化（相同尺寸分组）
7. ⏳ 性能基准测试
8. ⏳ 向后兼容性验证

### Phase 3: 文档和工具（低优先级）
9. ⏳ 用户文档更新
10. ⏳ 调试工具（visualize RoPE）
11. ⏳ Migration guide

---

## 总结

### 当前限制
- FluxTransformer 只支持 shared RoPE（所有样本相同的位置编码）
- 无法处理多分辨率训练中不同样本的不同空间尺寸

### 解决方案
- 扩展 FluxPosEmbed 支持 batched 计算
- 创建 FluxAttnProcessorPerSample 为每个样本应用独立 RoPE
- 在 FluxTransformer2DModel.forward 中检测模式并切换

### 优势
- ✅ 向后兼容（shared mode 仍然工作）
- ✅ 与 attention mask 协同（padding 正确处理）
- ✅ 性能可优化（缓存、批量计算）
- ✅ 测试覆盖完整

### 风险
- 计算开销增加（per-sample RoPE 计算）
- 代码复杂度提升（模式切换逻辑）
- 需要充分测试确保正确性

### 替代方案
如果 per-sample RoPE 性能开销过大，可以考虑：
1. **仅使用 attention mask**（当前方案）：效果已经很好
2. **Bucketing**：将相似尺寸的样本分组到同一 batch
3. **Dynamic batching**：运行时动态组织 batch

当前推荐：**先使用 attention mask 方案训练**，如果效果不佳再实现 per-sample RoPE。

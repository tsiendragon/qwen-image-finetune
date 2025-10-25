# Qwen Custom Transformer 测试说明

## 测试文件
- `test_qwen_custom.py`: 测试 Qwen custom transformer 的多分辨率支持和 attention mask 功能

## 测试目标

验证自定义的 `QwenImageTransformer2DModel` 在以下场景下与原始实现保持一致：

1. **参数等效性**: 验证模型参数在同步后完全相同
2. **Split 模式**: 按序列长度分组处理（无 padding，无 attention mask）
3. **Per-sample RoPE 隔离测试**: 使用 attention mask 但无实际 padding
4. **Multi-resolution 完整测试**: 完整 batch + attention mask + per-sample RoPE
5. **Padding 遮盖测试**: 验证 padding 区域被正确清零

## 测试数据设计

### 样本配置（多图像输入）
```python
batch_size = 4
samples = [
    Sample 0: [(1, 32, 32), (1, 32, 32)] → 2048 tokens  # 2 images, 需要 padding
    Sample 1: [(1, 32, 32), (1, 32, 32)] → 2048 tokens  # 2 images, 需要 padding
    Sample 2: [(1, 64, 64), (1, 64, 64)] → 8192 tokens  # 2 images, 无 padding（最大长度）
    Sample 3: [(1, 64, 64), (1, 64, 64)] → 8192 tokens  # 2 images, 无 padding（最大长度）
]
text_len = 77  # 所有样本相同
```

**注意**：测试场景模拟多图像输入，每个样本包含 2 个图像，这是 Qwen 模型的典型使用场景。

### Attention Mask 设计
```python
# 完整的 attention mask: [batch, text_len + max_img_seq]
# 对于多图像输入，img_seq_len 是所有图像 token 的总和
attention_mask[b, :text_len] = True                    # Text 部分全部有效
attention_mask[b, text_len:text_len+img_seq_len] = True  # Image 有效部分（所有图像）
attention_mask[b, text_len+img_seq_len:] = False       # Padding 部分

# 示例：
# Sample 0: 77 (text) + 2048 (2 images) = 2125 有效 tokens
# Sample 2: 77 (text) + 8192 (2 images) = 8269 有效 tokens（最大长度）
```

## 运行测试

### 使用预训练模型（默认）
```bash
# 运行所有测试（使用 ovedrive/Qwen-Image-Edit-2509-4bit 预训练模型）
pytest tests/models/test_qwen_custom.py -v -s
```

**注意**:
- 默认使用 `ovedrive/Qwen-Image-Edit-2509-4bit` 预训练模型（4-bit 量化版本）
- 首次运行会下载模型（约 8GB，相比原版 30GB 大幅减小）
- 需要 GPU 和足够的显存（建议 12GB+，4-bit 量化后显存需求更低）

### 使用小型随机模型（快速测试）
如果想快速测试或在 CI 中运行，可以修改 `use_pretrained` fixture：

```python
# 在 test_qwen_custom.py 中修改
@pytest.fixture(scope="class")
def use_pretrained(self):
    return False  # 改为 False 使用小型随机模型
```

### 运行特定测试
```bash
# 测试参数等效性
pytest tests/models/test_qwen_custom.py::TestQwenTransformerEquivalence::test_model_parameters_equivalence -v -s

# 测试 split 模式等效性
pytest tests/models/test_qwen_custom.py::TestQwenTransformerEquivalence::test_equivalence_split_vs_split -v -s

# 测试完整 multi-resolution
pytest tests/models/test_qwen_custom.py::TestQwenTransformerEquivalence::test_custom_model_full_batch_with_mask -v -s

# 测试 padding 遮盖
pytest tests/models/test_qwen_custom.py::TestQwenTransformerEquivalence::test_padding_is_masked -v -s
```

### 使用不同的 dtype
```bash
# 使用 float32（更高精度）
pytest tests/models/test_qwen_custom.py --dtype=float32 -v -s

# 使用 bfloat16（默认，更快）
pytest tests/models/test_qwen_custom.py --dtype=bfloat16 -v -s
```

### 使用不同的模型
可以通过修改 `repo_id` fixture 来测试不同的预训练模型：

```python
@pytest.fixture(scope="class")
def repo_id(self):
    return "your-custom-model-path"  # 本地路径或 HF repo
```

## 测试流程

```
1. test_model_parameters_equivalence
   ↓
2. test_original_model_split_batches  ← 原始模型基准
   ↓
3. test_custom_model_split_batches    ← Custom 模型 (shared RoPE)
   ↓
4. test_equivalence_split_vs_split    ✓ 验证 original == custom (split)
   ↓
5. test_custom_model_split_with_mask  ← Custom 模型 (per-sample RoPE, 无实际 padding)
   ↓
6. test_equivalence_split_with_vs_without_mask  ✓ 验证 mask 不影响无 padding 样本
   ↓
7. test_custom_model_full_batch_with_mask  ← 完整 multi-resolution
   ↓
8. test_equivalence_split_vs_full     ✓ 验证 split == full (extracted)
   ↓
9. test_equivalence_original_vs_custom_full  ✓ 验证 original == custom (full)
   ↓
10. test_padding_is_masked            ✓ 验证 padding 区域为 0
```

## 关键测试点

### ✅ RoPE 模式检测
- **Shared mode**: `img_shapes = [(1, 32, 32)]` (List[Tuple])
- **Per-sample mode**: `img_shapes = [[(1, 32, 32)], [(1, 48, 48)]]` (List[List[Tuple]])

### ✅ Attention Mask 正确性
- Text tokens: 全部有效
- Image tokens: 按实际长度
- Padding tokens: 被遮盖（输出为 0）

### ✅ 数值一致性
- Original (split) == Custom (split, no mask)
- Custom (split, no mask) == Custom (split, with mask) [无实际 padding]
- Custom (split) == Custom (full, extracted)
- Original (split) == Custom (full, extracted)

### ✅ Padding 处理
- Padding 区域输出必须为 0
- 有效区域输出非零

## 预期结果

所有测试应该通过，相对误差 < 1e-4（bfloat16 精度下）:

```
✓ test_model_parameters_equivalence
✓ test_original_model_split_batches
✓ test_custom_model_split_batches
✓ test_custom_model_split_with_mask
✓ test_custom_model_full_batch_with_mask
✓ test_equivalence_split_vs_split
✓ test_equivalence_split_with_vs_without_mask
✓ test_equivalence_split_vs_full
✓ test_equivalence_original_vs_custom_full
✓ test_padding_is_masked
```

## 与 Flux 测试的主要区别

| 特性 | Flux | Qwen |
|------|------|------|
| **位置编码输入** | `img_ids` + `txt_ids` | `img_shapes` + `txt_seq_lens` |
| **RoPE 生成** | 从 ids 直接计算 | 从 shapes 计算 3D 网格 RoPE |
| **额外输入** | `pooled_projections` | `guidance` (可选) |
| **输出流** | 单流 (image only) | 双流 (image + text, 返回 image) |
| **模型加载** | 预训练模型（从文件） | 预训练模型（默认）或小型随机模型 |
| **测试数据** | 从 resources/ 加载 | 随机生成 |

## 故障排查

### 内存不足
- **使用小型模型**: 设置 `use_pretrained=False` 并减少 `num_layers`
- **使用 CPU**: `pytest --device=cpu`（会很慢）
- **减少 batch size**: 修改 `sequence_info` 中的样本数量

### 模型下载问题
- **网络问题**: 确保可以访问 Hugging Face Hub
- **缓存位置**: 模型会下载到 `~/.cache/huggingface/`
- **手动下载**: 可以先手动下载模型到本地，然后修改 `repo_id`

### 精度问题
- 使用 float32: `pytest --dtype=float32`
- 检查相对误差阈值是否合理（预训练模型应该 < 1e-4）

### 测试失败
1. **参数不匹配**: 检查 original 和 custom 模型是否从同一个 checkpoint 加载
2. **Padding 错误**: 检查 padding 是否正确应用
3. **Attention mask 错误**: 检查 attention mask 形状是否正确
4. **RoPE 模式**: 检查 RoPE 模式检测是否正确
5. **数值精度**: 比较中间结果（添加 debug 输出）

### 快速验证
如果想快速验证代码逻辑（不关心数值精度），可以：
```python
# 修改 use_pretrained=False 使用小型随机模型
# 修改 num_layers=1 进一步加速
# 修改 batch_size=2 减少内存使用
```

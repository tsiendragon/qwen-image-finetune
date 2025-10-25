# 测试数据加载系统

本目录包含测试数据自动下载和加载的完整解决方案。

## 📁 文件结构

```
tests/
├── conftest.py                           # test_resources fixture 定义
├── resources_config.yaml                 # 测试资源配置
├── resources/                            # 本地缓存目录（自动下载，gitignore）
├── utils/
│   ├── test_resources.py                 # HuggingFace Hub 下载逻辑
│   └── data_loader.py                    # 数据加载工具函数 ⭐ 新增
└── examples/
    └── test_data_loading_examples.py     # 使用示例 ⭐ 新增
```

## 🎯 核心概念

### test_resources Fixture

**定义位置**: `tests/conftest.py`

**作用**: 自动从 HuggingFace Hub 下载测试数据并返回本地缓存目录路径

**类型**: `pathlib.Path`

**用法**:
```python
def test_example(test_resources):
    # test_resources 是 Path 对象: tests/resources/
    data_file = test_resources / "path/to/data.pt"
```

### 数据加载工具

**位置**: `tests/utils/data_loader.py` ⭐ 新创建

**核心函数**:

| 函数 | 用途 | 示例 |
|-----|------|------|
| `load_torch_file` | 加载单个文件 | `load_torch_file(test_resources, "path/to/file.pt")` |
| `load_torch_directory` | 批量加载目录 | `load_torch_directory(test_resources, "path/to/dir")` |
| `load_flux_transformer_input` | Flux Transformer 输入 | `load_flux_transformer_input(test_resources)` |
| `load_flux_training_sample` | Flux 训练样本 | `load_flux_training_sample(test_resources, "sample1")` |
| `load_flux_sampling_embeddings` | Flux 采样嵌入 | `load_flux_sampling_embeddings(test_resources)` |
| `prepare_test_data_for_device` | 移动数据到设备 | `prepare_test_data_for_device(data, device, dtype)` |

## 🚀 快速使用

### 方式 1: 使用便捷函数（推荐）

```python
import pytest
from tests.utils.data_loader import load_flux_transformer_input

def test_example(test_resources):
    # 使用预设函数加载数据
    data = load_flux_transformer_input(test_resources)

    # 直接使用数据
    latent_ids = data["latent_ids"]
    prompt_embeds = data["prompt_embeds"]
```

### 方式 2: 通用加载函数

```python
from tests.utils.data_loader import load_torch_file, load_torch_directory

def test_custom(test_resources):
    # 加载单个文件
    single_file = load_torch_file(
        test_resources,
        "flux_training/face_segmentation/sample1/sample_noise.pt"
    )

    # 批量加载目录
    all_files = load_torch_directory(
        test_resources,
        "flux_training/face_segmentation/sample1"
    )
```

### 方式 3: 直接路径构建

```python
def test_manual(test_resources):
    # 手动构建路径
    data_path = test_resources / "path" / "to" / "data.pt"
    data = torch.load(data_path, map_location="cpu")
```

## 📚 已更新的测试文件

### 1. test_flux_transform_custom.py ✅

**修改内容**:
```python
# 之前
input_path = test_resources / "flux_models" / "transformer" / "input" / "flux_input.pth"
data = torch.load(input_path, map_location="cpu")

# 现在
from tests.utils.data_loader import load_flux_transformer_input
data = load_flux_transformer_input(test_resources)
```

### 2. test_flux_loss.py ✅

**修改内容**:
```python
# 之前
def load_sample_data(sample_dir: Path):
    data = {
        "control_ids": torch.load(sample_dir / "sample_control_ids.pt", ...),
        "noise": torch.load(sample_dir / "sample_noise.pt", ...),
        # ... 逐个加载13个文件
    }
    return data

# 现在
from tests.utils.data_loader import load_flux_training_sample

@pytest.fixture
def sample_data_1(test_resources):
    return load_flux_training_sample(test_resources, "sample1")

@pytest.fixture
def sample_data_2(test_resources):
    return load_flux_training_sample(test_resources, "sample2")
```

### 3. test_sampling.py ✅

**修改内容**: 添加了文档字符串，说明如何使用数据加载工具

## 📖 文档

### Documentation

1. **`docs/guide/testing-data-loading.md`** ⭐
   - Complete testing data loading guide
   - Detailed usage of all functions
   - Multiple practical examples
   - FAQ section

2. **`docs/guide/test-resources-architecture.md`** ⭐
   - test_resources architecture explanation
   - Architecture and workflow diagrams
   - Configuration file documentation
   - Troubleshooting guide

3. **`tests/examples/test_data_loading_examples.py`** ⭐
   - 8个完整的使用示例
   - 可运行的测试代码
   - 涵盖所有主要使用场景

### 更新的文档

4. **`tests/conftest.py`**
   - 添加了数据加载工具的使用说明
   - 引用了详细文档链接

## 🔑 关键优势

### 之前
```python
# 每个测试文件都要写类似的代码
def load_sample_data(sample_dir: Path):
    data = {}
    data["control_ids"] = torch.load(sample_dir / "sample_control_ids.pt", ...)
    data["noise"] = torch.load(sample_dir / "sample_noise.pt", ...)
    # ... 重复代码
    return data
```

### 现在
```python
# 一行代码搞定
from tests.utils.data_loader import load_flux_training_sample
data = load_flux_training_sample(test_resources, "sample1")
```

### 优势
- ✅ **减少重复代码**: 统一的数据加载接口
- ✅ **类型安全**: 统一的错误处理和类型检查
- ✅ **易于维护**: 修改一处，所有测试受益
- ✅ **自动文档**: 函数有详细的 docstring
- ✅ **灵活性**: 支持通用和预设两种方式

## 🎓 学习路径

### 新手入门
1. 阅读 `docs/test-resources-explained.md` 了解 test_resources
2. 查看 `tests/examples/test_data_loading_examples.py` 运行示例
3. 参考已更新的测试文件学习实际用法

### 深入使用
1. 阅读 `docs/testing-data-loading.md` 学习所有函数
2. 查看 `tests/utils/data_loader.py` 了解实现细节
3. 根据需要自定义数据加载函数

## 📝 使用规范

### 推荐做法

✅ 使用预设函数加载常见数据
```python
data = load_flux_transformer_input(test_resources)
```

✅ 使用通用函数加载自定义数据
```python
data = load_torch_file(test_resources, "custom/path.pt")
```

✅ 使用 prepare_test_data_for_device 移动数据
```python
data = prepare_test_data_for_device(data, device="cuda", dtype=torch.bfloat16)
```

### 不推荐做法

❌ 在每个测试中重复写加载逻辑
```python
# 不要这样做
data = {}
for file in files:
    data[key] = torch.load(...)
```

❌ 硬编码绝对路径
```python
# 不要这样做
data = torch.load("/absolute/path/to/data.pt")
```

## 🔧 维护指南

### 添加新的测试数据

1. 上传数据到 HuggingFace Hub: `TsienDragon/qwen-image-finetune-test-resources`
2. 更新 `tests/resources_config.yaml` 添加资源定义
3. 如果是常用数据，可以在 `tests/utils/data_loader.py` 添加便捷函数

### 添加新的便捷函数

```python
# 在 tests/utils/data_loader.py 中添加

def load_your_custom_data(
    test_resources: Path,
    map_location: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """加载你的自定义数据（预设路径）"""
    return load_torch_file(
        test_resources,
        "your/custom/path/data.pt",
        map_location=map_location,
    )
```

## 📞 Getting Help

- View documentation: `docs/guide/testing-data-loading.md`
- View architecture: `docs/guide/test-resources-architecture.md`
- Check updated test files for actual usage examples

## 🎉 总结

本次更新创建了一个完整的测试数据加载系统，包括:

1. ✅ **通用数据加载工具** (`tests/utils/data_loader.py`)
2. ✅ **预设便捷函数** (针对常见测试场景)
3. ✅ **完整文档** (使用指南 + 原理详解)
4. ✅ **实用示例** (8个可运行的示例)
5. ✅ **测试文件更新** (3个测试文件已迁移)

现在你可以在任何测试中轻松使用统一的数据加载接口！🚀

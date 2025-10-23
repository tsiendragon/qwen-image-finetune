# 测试资源管理指南

本文档说明如何管理和使用存储在 HuggingFace Hub 上的测试资源。

## 概述

为了避免在 Git 仓库中存储大型测试文件（约 50MB），我们将测试资源托管在 HuggingFace Hub 上，并在运行测试时自动下载。

## 目录结构

测试资源按照以下结构组织：

```
test_resources_organized/
├── flux_models/              # Flux 模型相关测试数据
│   └── transformer/
│       └── input/            # Transformer 输入测试数据
├── flux_training/            # 训练流程测试数据
│   └── face_segmentation/    # 人脸分割训练样本
│       ├── sample1/          # 训练样本 1
│       └── sample2/          # 训练样本 2
├── flux_sampling/            # 采样和生成测试数据
│   └── embeddings/           # 嵌入向量
└── reference_outputs/        # 参考输出图像
    └── images/
```

## 配置文件

`tests/resources_config.yaml` 定义了：

1. **HuggingFace 仓库配置**：仓库 ID、类型、版本
2. **资源组定义**：每个资源组包含哪些文件
3. **测试依赖**：每个测试文件需要哪些资源组
4. **路径映射**：旧路径到新路径的映射（用于兼容性）

## 使用方法

### 运行测试（自动下载）

测试资源会在运行测试时自动下载：

```bash
# 运行所有测试（自动下载所需资源）
pytest tests/

# 运行特定测试（只下载该测试需要的资源）
pytest tests/src/models/test_flux_transform_custom.py
```

### 手动下载资源

```python
from tests.utils import ensure_test_resources, download_resource_groups

# 下载所有资源
ensure_test_resources(download_all=True)

# 下载特定资源组
download_resource_groups(['flux_input', 'flux_sampling'])

# 为特定测试下载资源
ensure_test_resources(test_file='tests/e2e/test_flux_sampling.py')
```

### 跳过自动下载

在 CI 环境中，如果已经缓存了资源，可以跳过下载：

```bash
export SKIP_DOWNLOAD_TEST_RESOURCES=1
pytest tests/
```

## 上传新资源

### 1. 组织资源文件

将新资源放入 `test_resources_organized/` 的适当目录：

```bash
# 例如：添加新的训练样本
mkdir -p test_resources_organized/flux_training/face_segmentation/sample3/
cp your_new_files/* test_resources_organized/flux_training/face_segmentation/sample3/
```

### 2. 更新配置文件

编辑 `tests/resources_config.yaml`：

```yaml
resource_groups:
  # 添加新的资源组
  flux_training_sample3:
    description: "Flux 训练流程测试样本3"
    files:
      - "flux_training/face_segmentation/sample3/file1.pt"
      - "flux_training/face_segmentation/sample3/file2.pt"
    size: "10MB"
    used_by:
      - "tests/e2e/test_new_feature.py"

test_dependencies:
  # 添加测试依赖
  "tests/e2e/test_new_feature.py":
    - flux_training_sample3
```

### 3. 上传到 HuggingFace

```bash
# 设置 HuggingFace token
export HF_TOKEN=your_huggingface_token

# 运行上传脚本
python scripts/upload_test_resources.py

# 或者指定参数
python scripts/upload_test_resources.py \
    --token YOUR_TOKEN \
    --resources-dir test_resources_organized \
    --repo-id TsienDragon/qwen-image-finetune-test-resources
```

### 4. 验证

```bash
# 删除本地资源
rm -rf tests/resources/

# 运行测试验证下载
pytest tests/e2e/test_new_feature.py -v
```

## 资源组说明

| 资源组 | 大小 | 用途 | 使用的测试 |
|--------|------|------|------------|
| `flux_input` | 19MB | Flux transformer 输入测试 | `test_flux_transform_custom.py` |
| `flux_training_sample1` | 11MB | 训练流程测试样本1 | `test_transformer_consistency.py`, `test_flux_loss.py` |
| `flux_training_sample2` | 12MB | 训练流程测试样本2 | `test_transformer_consistency.py`, `test_flux_loss.py` |
| `flux_sampling` | 7.3MB | 采样和生成测试 | `test_flux_sampling.py` |
| `reference_outputs` | 688KB | 参考输出图像 | `test_flux_sampling.py` |

## 按需下载的优势

1. **减少 Git 仓库大小**：不再在 Git 中存储大型二进制文件
2. **加快克隆速度**：新开发者克隆仓库更快
3. **按需下载**：只下载运行特定测试需要的资源
4. **易于更新**：更新测试资源不会污染 Git 历史
5. **版本控制**：可以通过 HuggingFace 的版本控制管理资源

## CI/CD 配置

在 GitHub Actions 或其他 CI 中，可以缓存下载的资源：

```yaml
- name: Cache test resources
  uses: actions/cache@v3
  with:
    path: tests/resources
    key: test-resources-${{ hashFiles('tests/resources_config.yaml') }}

- name: Run tests
  run: pytest tests/
  env:
    SKIP_DOWNLOAD_TEST_RESOURCES: ${{ steps.cache.outputs.cache-hit }}
```

## 故障排查

### 下载失败

如果自动下载失败，可以手动下载：

```python
from tests.utils import download_all_resources
download_all_resources(force_download=True)
```

### 使用本地资源

如果无法访问 HuggingFace，可以：

1. 从其他地方获取 `test_resources_organized/` 目录
2. 复制到 `tests/resources/`
3. 设置环境变量 `SKIP_DOWNLOAD_TEST_RESOURCES=1`

### 清理缓存

```bash
# 删除本地资源
rm -rf tests/resources/

# 重新下载
pytest tests/ --cache-clear
```

## 最佳实践

1. **保持资源组小而专注**：每个资源组应该只包含相关的文件
2. **文档化资源用途**：在配置中添加清晰的描述
3. **定期清理**：删除不再使用的旧资源
4. **版本标记**：重大变更时更新 `revision` 字段
5. **测试覆盖**：确保每个资源组至少被一个测试使用

## 相关文件

- `tests/resources_config.yaml` - 资源配置文件
- `tests/utils/test_resources.py` - 资源下载工具
- `tests/conftest.py` - pytest 配置（自动下载）
- `scripts/upload_test_resources.py` - 上传脚本
- `test_resources_organized/` - 组织化的资源目录（用于上传）
- `tests/resources/` - 本地资源缓存（自动生成，已在 .gitignore 中）

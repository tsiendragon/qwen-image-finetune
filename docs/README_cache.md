# 嵌入缓存系统

本项目实现了一个高效的嵌入缓存系统，用于加速 Qwen Image 模型的训练过程。通过预计算和缓存嵌入，可以显著减少训练时的 GPU 计算负载，提高训练效率。

## 功能特性

- **多类型缓存支持**: 支持 `pixel_latent`、`control_latent`、`prompt_embed`、`prompt_embeds_mask` 四种类型的嵌入缓存
- **智能哈希机制**: 基于文件路径、修改时间和内容生成唯一哈希值，确保缓存一致性
- **透明加载**: 自动检测缓存可用性，无缓存时自动回退到实时计算
- **灵活配置**: 通过配置文件轻松启用/禁用缓存功能
- **批量预计算**: 支持批量预计算数据集嵌入

## 目录结构

缓存系统会在指定的缓存目录下创建以下子目录：

```
cache_dir/
├── pixel_latent/      # 像素图像潜在向量缓存
├── control_latent/    # 控制图像潜在向量缓存
├── prompt_embed/      # 文本提示嵌入缓存
└── prompt_embeds_mask/ # 文本提示掩码缓存
```

## 配置使用

### 1. 更新配置文件

在您的训练配置文件中添加缓存设置：

```yaml
data:
  class_path: "src.data.dataset.ImageDataset"
  init_args:
    dataset_path: "/path/to/your/dataset"
    image_size: [832, 576]
    cache_dir: "/path/to/cache/directory"  # 缓存目录
    use_cache: true                        # 启用缓存
```

### 2. 预计算嵌入（推荐）

在开始训练前，预计算所有嵌入：

```bash
# 预计算整个数据集
python precompute_embeddings.py --config configs/qwen_image_edit_config.yaml

# 预计算指定范围
python precompute_embeddings.py --config configs/qwen_image_edit_config.yaml --start_idx 0 --end_idx 1000

# 测试缓存加载
python precompute_embeddings.py
```

### 3. 开始训练

正常启动训练，系统会自动使用缓存：

```bash
python src/train.py --config configs/qwen_image_edit_config.yaml
```

## 性能优势

### 训练加速

- **减少 GPU 计算**: 避免重复计算 VAE 编码和文本嵌入
- **提高吞吐量**: 缓存命中时，单个批次处理时间可减少 50-70%
- **内存优化**: 减少 GPU 内存占用，支持更大的批量大小

### 多 Epoch 训练

- **显著加速**: 第二个 epoch 开始，训练速度大幅提升
- **一致性保证**: 缓存确保每个 epoch 使用相同的嵌入
- **磁盘空间**: 典型数据集缓存大小约为原始数据的 10-20%

## API 参考

### EmbeddingCacheManager

```python
from src.data.cache_manager import EmbeddingCacheManager

# 创建缓存管理器
cache_manager = EmbeddingCacheManager("/path/to/cache")

# 保存缓存
cache_manager.save_cache("pixel_latent", file_hash, tensor_data)

# 加载缓存
cached_data = cache_manager.load_cache("pixel_latent", file_hash)

# 检查缓存状态
exists = cache_manager.cache_exists("pixel_latent", file_hash)

# 获取统计信息
stats = cache_manager.get_cache_stats()
```

### 数据集使用

```python
from src.data.dataset import ImageDataset

# 创建支持缓存的数据集
config = {
    'dataset_path': '/path/to/dataset',
    'cache_dir': '/path/to/cache',
    'use_cache': True
}
dataset = ImageDataset(config)

# 获取样本（自动使用缓存）
sample = dataset[0]
if sample['cached']:
    print("使用缓存数据")
else:
    print("实时计算数据")
```

## 最佳实践

### 1. 缓存目录选择
- 使用快速 SSD 存储
- 确保有足够的磁盘空间
- 考虑使用专用的缓存分区

### 2. 预计算策略
```bash
# 分批预计算大数据集
python precompute_embeddings.py --config config.yaml --start_idx 0 --end_idx 5000
python precompute_embeddings.py --config config.yaml --start_idx 5000 --end_idx 10000
```

### 3. 缓存管理
```bash
# 检查缓存状态
python -c "
from src.data.dataset import loader
dataloader = loader('/path/to/dataset', cache_dir='/path/to/cache')
print(dataloader.dataset.get_cache_stats())
"
```

### 4. 清理缓存
```python
# 清理特定类型缓存
cache_manager.clear_cache('pixel_latent')

# 清理所有缓存
cache_manager.clear_cache()
```

## 故障排除

### 缓存未命中
- 检查文件路径是否正确
- 确认文件未被修改
- 验证缓存目录权限

### 内存不足
- 减少批量大小
- 检查缓存文件是否过大
- 考虑分批预计算

### 性能问题
- 确保缓存目录在快速存储上
- 检查磁盘 I/O 性能
- 考虑使用 SSD 存储

## 技术实现

系统核心组件：

1. **EmbeddingCacheManager**: 负责缓存的存储和检索
2. **ImageDataset**: 支持缓存的数据集类
3. **EmbeddingCacheHook**: 用于批量预计算的钩子类
4. **Trainer**: 更新的训练器，支持缓存和非缓存数据

缓存文件采用 PyTorch 的 `.pt` 格式，确保高效的序列化和反序列化。

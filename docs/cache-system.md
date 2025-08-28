# Embedding Cache System

The Qwen Image Finetune project implements an efficient embedding cache system to accelerate the training process. By pre-computing and caching embeddings, it significantly reduces GPU computation load during training and improves training efficiency.

## Features

- **Multi-type Cache Support**: Supports caching for `pixel_latent`, `control_latent`, `prompt_embed`, and `prompt_embeds_mask`
- **Smart Hashing**: Generates unique hash values based on file path, modification time, and content for cache consistency
- **Transparent Loading**: Automatically detects cache availability and falls back to real-time computation when cache is unavailable
- **Flexible Configuration**: Easy enable/disable cache functionality through configuration files
- **Batch Pre-computation**: Supports batch pre-computing dataset embeddings

## Directory Structure

The cache system creates the following subdirectories under the specified cache directory:

```
cache_dir/
├── pixel_latent/      # Pixel image latent vector cache
├── control_latent/    # Control image latent vector cache
├── prompt_embed/      # Text prompt embedding cache
└── prompt_embeds_mask/ # Text prompt mask cache
```

## Configuration Usage

### 1. Update Configuration File

Add cache settings to your training configuration file:

```yaml
data:
  class_path: "src.data.dataset.ImageDataset"
  init_args:
    dataset_path: "/path/to/your/dataset"
    image_size: [832, 576]
    cache_dir: "/path/to/cache/directory"  # Cache directory
    use_cache: true                        # Enable cache
```

### 2. Pre-compute Embeddings (Recommended)

Pre-compute all embeddings before starting training:

```bash
# Pre-compute entire dataset
python precompute_embeddings.py --config configs/qwen_image_edit_config.yaml

# Pre-compute specified range
python precompute_embeddings.py --config configs/qwen_image_edit_config.yaml --start_idx 0 --end_idx 1000

# Test cache loading
python precompute_embeddings.py
```

### 3. Start Training

Launch training normally - the system will automatically use cache:

```bash
python src/train.py --config configs/qwen_image_edit_config.yaml
```

## Performance Benefits

### Training Acceleration

- **Reduced GPU Computation**: Avoids repeated VAE encoding and text embedding computation
- **Improved Throughput**: Cache hits can reduce single batch processing time by 50-70%
- **Memory Optimization**: Reduces GPU memory usage, supporting larger batch sizes

### Multi-Epoch Training

- **Significant Acceleration**: Starting from the second epoch, training speed increases dramatically
- **Consistency Guarantee**: Cache ensures each epoch uses identical embeddings
- **Disk Space**: Typical dataset cache size is approximately 10-20% of original data

## API Reference

### EmbeddingCacheManager

```python
from src.data.cache_manager import EmbeddingCacheManager

# Create cache manager
cache_manager = EmbeddingCacheManager("/path/to/cache")

# Save cache
cache_manager.save_cache("pixel_latent", file_hash, tensor_data)

# Load cache
cached_data = cache_manager.load_cache("pixel_latent", file_hash)

# Check cache status
exists = cache_manager.cache_exists("pixel_latent", file_hash)

# Get statistics
stats = cache_manager.get_cache_stats()
```

### Dataset Usage

```python
from src.data.dataset import ImageDataset

# Create cache-enabled dataset
config = {
    'dataset_path': '/path/to/dataset',
    'cache_dir': '/path/to/cache',
    'use_cache': True
}
dataset = ImageDataset(config)

# Get sample (automatically uses cache)
sample = dataset[0]
if sample['cached']:
    print("Using cached data")
else:
    print("Computing data in real-time")
```

## Best Practices

### 1. Cache Directory Selection
- Use fast SSD storage
- Ensure sufficient disk space
- Consider using dedicated cache partition

### 2. Pre-computation Strategy
```bash
# Batch pre-compute large datasets
python precompute_embeddings.py --config config.yaml --start_idx 0 --end_idx 5000
python precompute_embeddings.py --config config.yaml --start_idx 5000 --end_idx 10000
```

### 3. Cache Management
```bash
# Check cache status
python -c "
from src.data.dataset import loader
dataloader = loader('/path/to/dataset', cache_dir='/path/to/cache')
print(dataloader.dataset.get_cache_stats())
"
```

### 4. Cache Cleanup
```python
# Clean specific cache type
cache_manager.clear_cache('pixel_latent')

# Clean all cache
cache_manager.clear_cache()
```

## Troubleshooting

### Cache Miss
- Check if file path is correct
- Confirm file hasn't been modified
- Verify cache directory permissions

### Out of Memory
- Reduce batch size
- Check if cache files are too large
- Consider batch pre-computation

### Performance Issues
- Ensure cache directory is on fast storage
- Check disk I/O performance
- Consider using SSD storage

## Technical Implementation

Core system components:

1. **EmbeddingCacheManager**: Responsible for cache storage and retrieval
2. **ImageDataset**: Cache-enabled dataset class
3. **EmbeddingCacheHook**: Hook class for batch pre-computation
4. **Trainer**: Updated trainer supporting both cached and non-cached data

Cache files use PyTorch's `.pt` format, ensuring efficient serialization and deserialization.

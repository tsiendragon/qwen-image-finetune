"""
Shared pytest fixtures and configuration for all tests.

## 常用 Fixtures

- test_resources: 自动从 HuggingFace Hub 下载的测试资源目录
- cuda_available: 检查 CUDA 是否可用
- tmp_cache_dir: 临时缓存目录
- sample_image: 测试用 RGB 图像
- sample_grayscale_image: 测试用灰度图像

## 测试数据加载

推荐使用 `tests.utils.data_loader` 模块中的函数来加载测试数据：

```python
from tests.utils.data_loader import (
    load_flux_transformer_input,
    load_flux_training_sample,
    load_flux_sampling_embeddings,
    load_torch_file,
    load_torch_directory,
    prepare_test_data_for_device,
)

def test_example(test_resources):
    # 方式1: 使用便捷函数
    data = load_flux_transformer_input(test_resources)

    # 方式2: 加载自定义路径
    data = load_torch_file(test_resources, "path/to/file.pt")

    # 方式3: 批量加载目录
    data = load_torch_directory(test_resources, "path/to/dir")
```

Detailed documentation: `docs/guide/testing-data-loading.md`
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import os
import sys
import warnings
from os.path import abspath, dirname, join

# Tests for Qwen Image Finetune project
# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)
warnings.simplefilter(action="ignore", category=FutureWarning)


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """提供临时缓存目录用于测试"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def sample_image():
    """提供测试用 RGB 图像 (512x512)"""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image():
    """提供测试用灰度图像 (512x512)"""
    return np.random.randint(0, 255, (512, 512), dtype=np.uint8)


@pytest.fixture
def mock_processor_config():
    """提供 mock ImageProcessor 配置"""
    from qflux.data.config import ImageProcessorInitArgs
    return ImageProcessorInitArgs(
        target_size=(512, 512),
        process_type="resize"
    )


@pytest.fixture
def mock_multi_resolution_config():
    """提供 mock 多分辨率配置"""
    from qflux.data.config import ImageProcessorInitArgs
    return ImageProcessorInitArgs(
        multi_resolutions=["512*512", "640*640", "768*512", "832*576"],
        max_aspect_ratio=3.0
    )


@pytest.fixture(scope="session")
def cuda_available():
    """检查 CUDA 是否可用"""
    return torch.cuda.is_available()


@pytest.fixture(scope="session", autouse=True)
def test_resources():
    """
    自动确保测试资源从 HuggingFace Hub 下载并可用。

    数据源:
        - HuggingFace Repository: TsienDragon/qwen-image-finetune-test-resources
        - Repository Type: dataset
        - 配置文件: tests/resources_config.yaml

    行为:
        1. 首次运行时从 HuggingFace Hub 下载测试数据
        2. 后续运行使用本地缓存 (tests/resources/)
        3. 如果设置环境变量 SKIP_DOWNLOAD_TEST_RESOURCES=1，则跳过下载检查

    这在 CI 环境中很有用，可以使用预缓存的资源。
    """
    # 允许通过环境变量跳过下载（例如在 CI 中使用缓存）
    if os.environ.get("SKIP_DOWNLOAD_TEST_RESOURCES") == "1":
        resources_dir = Path(__file__).parent / "resources"
        print("\n⚠️  Skipping HuggingFace download check (SKIP_DOWNLOAD_TEST_RESOURCES=1)")
        print(f"📁 Using local resources: {resources_dir}")
        return resources_dir

    try:
        from tests.utils.test_resources import ensure_test_resources
        print("\n🔍 Checking test resources from HuggingFace Hub...")
        resources_dir = ensure_test_resources()
        print(f"✅ Test resources ready at: {resources_dir}")
        return resources_dir
    except Exception as e:
        # 如果下载失败，尝试使用本地资源（如果存在）
        resources_dir = Path(__file__).parent / "resources"
        if resources_dir.exists():
            print(f"\n⚠️  Failed to download from HuggingFace, using local cache: {resources_dir}")
            return resources_dir
        raise RuntimeError(
            f"Failed to download test resources from HuggingFace and no local resources found: {e}"
        )

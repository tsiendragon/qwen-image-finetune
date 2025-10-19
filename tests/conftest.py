"""
Shared pytest fixtures and configuration for all tests.
"""
import pytest
import torch
import numpy as np
from pathlib import Path


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
    from src.data.config import ImageProcessorInitArgs
    return ImageProcessorInitArgs(
        target_size=(512, 512),
        process_type="resize"
    )


@pytest.fixture
def mock_multi_resolution_config():
    """提供 mock 多分辨率配置"""
    from src.data.config import ImageProcessorInitArgs
    return ImageProcessorInitArgs(
        multi_resolutions=["512*512", "640*640", "768*512", "832*576"],
        max_aspect_ratio=3.0
    )


@pytest.fixture(scope="session")
def cuda_available():
    """检查 CUDA 是否可用"""
    return torch.cuda.is_available()


import pytest
import torch
import numpy as np
import PIL.Image
from qflux.utils.images import (
    resize_bhw,
    make_image_shape_devisible,
    make_image_devisible,
    calculate_dimensions,
    calculate_best_resolution,
    image_adjust_best_resolution,
)


class TestResizeBHW:
    def test_basic_resize(self):
        x = torch.randn(4, 64, 64)  # [B, H, W]
        result = resize_bhw(x, 128, 128)
        assert result.shape == (4, 128, 128)

    def test_downscale(self):
        x = torch.randn(2, 256, 256)
        result = resize_bhw(x, 64, 64)
        assert result.shape == (2, 64, 64)

    def test_different_aspect_ratio(self):
        x = torch.randn(3, 100, 200)
        result = resize_bhw(x, 50, 150)
        assert result.shape == (3, 50, 150)

    @pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
    def test_different_modes(self, mode):
        x = torch.randn(2, 64, 64)
        result = resize_bhw(x, 128, 128, mode=mode)
        assert result.shape == (2, 128, 128)


class TestMakeImageShapeDevisible:
    def test_already_divisible(self):
        w, h = make_image_shape_devisible(512, 512, vae_scale_factor=8)
        assert w == 512
        assert h == 512

    def test_round_down(self):
        # 513 should round down to 512 (divisible by 16)
        w, h = make_image_shape_devisible(513, 517, vae_scale_factor=8)
        assert w % 16 == 0
        assert h % 16 == 0
        assert w <= 513
        assert h <= 517

    def test_different_vae_scale_factor(self):
        w, h = make_image_shape_devisible(100, 100, vae_scale_factor=4)
        assert w % 8 == 0  # vae_scale_factor * 2
        assert h % 8 == 0


class TestMakeImageDevisible:
    def test_tensor_input(self):
        image = torch.randn(1, 3, 513, 517)
        result = make_image_devisible(image, vae_scale_factor=8)
        assert isinstance(result, torch.Tensor)
        assert result.shape[2] % 16 == 0
        assert result.shape[3] % 16 == 0

    def test_pil_image_input(self):
        image = PIL.Image.new("RGB", (513, 517))
        result = make_image_devisible(image, vae_scale_factor=8)
        assert isinstance(result, PIL.Image.Image)
        w, h = result.size
        assert w % 16 == 0
        assert h % 16 == 0

    def test_numpy_array_input(self):
        image = np.random.rand(517, 513, 3).astype(np.float32)
        result = make_image_devisible(image, vae_scale_factor=8)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] % 16 == 0
        assert result.shape[1] % 16 == 0


class TestCalculateDimensions:
    def test_square_aspect_ratio(self):
        target_area = 1024 * 1024
        ratio = 1.0
        w, h, _ = calculate_dimensions(target_area, ratio)
        assert w == h
        assert w % 32 == 0
        assert h % 32 == 0

    def test_wide_aspect_ratio(self):
        target_area = 1024 * 1024
        ratio = 2.0  # width:height = 2:1
        w, h, _ = calculate_dimensions(target_area, ratio)
        assert w > h
        assert w % 32 == 0
        assert h % 32 == 0

    def test_tall_aspect_ratio(self):
        target_area = 1024 * 1024
        ratio = 0.5  # width:height = 1:2
        w, h, _ = calculate_dimensions(target_area, ratio)
        assert w < h
        assert w % 32 == 0
        assert h % 32 == 0


class TestCalculateBestResolution:
    def test_square_image(self):
        w, h = calculate_best_resolution(512, 512)
        assert w % 32 == 0
        assert h % 32 == 0

    def test_wide_image(self):
        w, h = calculate_best_resolution(1024, 512)
        assert w > h
        assert w % 32 == 0
        assert h % 32 == 0

    def test_custom_best_resolution(self):
        w, h = calculate_best_resolution(800, 600, best_resolution=512 * 512)
        assert w % 32 == 0
        assert h % 32 == 0


class TestImageAdjustBestResolution:
    def test_tensor_input(self):
        image = torch.randn(1, 3, 600, 800)
        result = image_adjust_best_resolution(image)
        assert isinstance(result, torch.Tensor)
        assert result.shape[2] % 32 == 0
        assert result.shape[3] % 32 == 0

    def test_pil_image_input(self):
        image = PIL.Image.new("RGB", (800, 600))
        result = image_adjust_best_resolution(image)
        assert isinstance(result, PIL.Image.Image)
        w, h = result.size
        assert w % 32 == 0
        assert h % 32 == 0

    def test_numpy_array_input(self):
        image = np.random.rand(600, 800, 3).astype(np.uint8)
        result = image_adjust_best_resolution(image)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] % 32 == 0
        assert result.shape[1] % 32 == 0

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported image type"):
            image_adjust_best_resolution([1, 2, 3])

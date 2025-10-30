"""Test multi-resolution per-image-type configuration"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from qflux.data.config import ImageProcessorInitArgs
from qflux.data.preprocess import ImageProcessor


class TestMultiResolutionPerImageType:
    """Test advanced multi-resolution config with per-image-type candidates"""

    def test_simple_format_backward_compatible(self):
        """Test that simple list format still works (backward compatible)"""
        config = ImageProcessorInitArgs(
            process_type="fixed_pixels",
            multi_resolutions=["1024*1024", "512*512"],
        )
        processor = ImageProcessor(config)

        assert processor.multi_res_mode == "simple"
        assert processor.multi_res_target == [1048576, 262144]
        assert processor.multi_res_controls == [[1048576, 262144]]

    def test_advanced_format_target_only(self):
        """Test advanced format with only target specified"""
        config = ImageProcessorInitArgs(
            process_type="fixed_pixels",
            multi_resolutions={
                "target": ["1024*1024", "1536*1536"],
                "controls": [["512*512"]],
            },
        )
        processor = ImageProcessor(config)

        assert processor.multi_res_mode == "advanced"
        assert processor.multi_res_target == [1048576, 2359296]
        assert processor.multi_res_controls == [[262144]]

    def test_advanced_format_separate_controls(self):
        """Test advanced format with separate configs for each control"""
        config = ImageProcessorInitArgs(
            process_type="fixed_pixels",
            multi_resolutions={
                "target": ["1024*1024", "2048*2048"],
                "controls": [
                    ["512*512", "768*768"],  # First control
                    ["256*256", "384*384"],  # Second control
                    ["512*512"],  # Third control
                ],
            },
        )
        processor = ImageProcessor(config)

        assert processor.multi_res_mode == "advanced"
        assert processor.multi_res_target == [1048576, 4194304]
        assert len(processor.multi_res_controls) == 3
        assert processor.multi_res_controls[0] == [262144, 589824]
        assert processor.multi_res_controls[1] == [65536, 147456]
        assert processor.multi_res_controls[2] == [262144]

    def test_select_pixels_candidate_simple_mode(self):
        """Test pixel selection in simple mode"""
        config = ImageProcessorInitArgs(
            process_type="fixed_pixels",
            multi_resolutions=["512*512", "1024*1024", "768*768"],
        )
        processor = ImageProcessor(config)

        # Image 800x600 (area=480000) should select 768*768 (589824)
        best_pixels = processor._select_pixels_candidate(800, 600)
        assert best_pixels == 589824  # 768*768

    def test_select_pixels_candidate_advanced_mode(self):
        """Test pixel selection with specific candidates"""
        config = ImageProcessorInitArgs(
            process_type="fixed_pixels",
            multi_resolutions={
                "target": ["1024*1024", "1536*1536"],
                "controls": [["256*256", "512*512"]],
            },
        )
        processor = ImageProcessor(config)

        # Target: 800x600 should prefer 1024*1024
        best_target = processor._select_pixels_candidate(
            800, 600, candidates=processor.multi_res_target
        )
        assert best_target == 1048576  # 1024*1024

        # Control: 400x300 (area=120000) should prefer 256*256 (65536)
        # Relative errors: |65536-120000|/120000=0.454, |262144-120000|/120000=1.185
        best_control = processor._select_pixels_candidate(
            400, 300, candidates=processor.multi_res_controls[0]
        )
        assert best_control == 65536  # 256*256 (closer relative error to 120000)

    @pytest.mark.integration
    def test_preprocess_with_advanced_config(self):
        """Test preprocessing with advanced multi-resolution config"""
        config = ImageProcessorInitArgs(
            process_type="fixed_pixels",
            multi_resolutions={
                "target": ["512*512"],
                "controls": [["256*256"], ["128*128"]],
            },
            max_aspect_ratio=3.0,
        )
        processor = ImageProcessor(config)

        # Mock calculate_best_resolution to avoid import issues
        with patch("qflux.utils.images.calculate_best_resolution") as mock_calc:
            mock_calc.side_effect = lambda w, h, area: (int((area**0.5)), int((area**0.5)))

            # Create test data
            data = {
                "image": np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8),
                "control": np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8),
                "controls": [
                    np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8),
                ],
            }

            processed = processor.preprocess(data)

            # Check that images were processed
            assert "image" in processed
            assert "control" in processed
            assert "controls" in processed
            assert len(processed["controls"]) == 1

            # Check shapes are tensors
            assert processed["image"].dim() == 3  # [C, H, W]
            assert processed["control"].dim() == 3
            assert processed["controls"][0].dim() == 3

    @pytest.mark.integration
    def test_fallback_to_last_control(self):
        """Test that extra controls fallback to last config"""
        config = ImageProcessorInitArgs(
            process_type="fixed_pixels",
            multi_resolutions={
                "target": ["512*512"],
                "controls": [
                    ["256*256"],  # First control
                    ["128*128"],  # Second control
                    # No third control specified
                ],
            },
        )
        processor = ImageProcessor(config)

        # Mock calculate_best_resolution to avoid import issues
        with patch("qflux.utils.images.calculate_best_resolution") as mock_calc:
            mock_calc.side_effect = lambda w, h, area: (int((area**0.5)), int((area**0.5)))

            # Create data with 3 additional controls (4 controls total)
            data = {
                "image": np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8),
                "control": np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),
                "controls": [
                    np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                    np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                    np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
                ],
            }

            # Should not raise error, extra controls use last config
            processed = processor.preprocess(data)
            assert len(processed["controls"]) == 3

    def test_validation_positive_values(self):
        """Test that negative values are rejected"""
        with pytest.raises(ValueError, match="must be positive"):
            ImageProcessorInitArgs(
                process_type="fixed_pixels",
                multi_resolutions=["512*512", -100],
            )

    def test_validation_empty_list(self):
        """Test that empty list is rejected"""
        with pytest.raises(ValueError, match="must not be empty"):
            ImageProcessorInitArgs(
                process_type="fixed_pixels",
                multi_resolutions=[],
            )

    def test_validation_invalid_type(self):
        """Test that invalid types are rejected"""
        with pytest.raises(ValueError):
            ImageProcessorInitArgs(
                process_type="fixed_pixels",
                multi_resolutions="invalid",  # Should be list or dict
            )

    def test_string_expression_parsing(self):
        """Test that string expressions are correctly parsed"""
        config = ImageProcessorInitArgs(
            process_type="fixed_pixels",
            multi_resolutions={
                "target": ["1024*1024", "512 * 512", 262144],
                "controls": [["256*256"]],
            },
        )
        processor = ImageProcessor(config)

        assert 1048576 in processor.multi_res_target  # 1024*1024
        assert 262144 in processor.multi_res_target  # 512*512 and direct int
        assert processor.multi_res_controls[0] == [65536]  # 256*256

    @pytest.mark.integration
    def test_mask_follows_target_resolution(self):
        """Test that mask uses same candidates as target image"""
        config = ImageProcessorInitArgs(
            process_type="fixed_pixels",
            multi_resolutions={
                "target": ["512*512"],
                "controls": [["256*256"]],
            },
        )
        processor = ImageProcessor(config)

        # Mock calculate_best_resolution to avoid import issues
        with patch("qflux.utils.images.calculate_best_resolution") as mock_calc:
            mock_calc.side_effect = lambda w, h, area: (int((area**0.5)), int((area**0.5)))

            data = {
                "image": np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8),
                "mask": np.random.randint(0, 255, (400, 400), dtype=np.uint8),
                "control": np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8),
            }

            processed = processor.preprocess(data)

            # Mask should have similar resolution to image (both use target candidates)
            assert processed["mask"].shape[0] == processed["image"].shape[1]  # Height
            assert processed["mask"].shape[1] == processed["image"].shape[2]  # Width

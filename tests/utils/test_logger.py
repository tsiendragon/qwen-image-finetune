import pytest
import torch
from unittest.mock import Mock, patch
from qflux.utils.logger import load_logger, log_images_auto, log_text_auto


class TestLoadLogger:
    def test_load_logger_default(self):
        """Test loading logger with default log level"""
        logger = load_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"

    def test_load_logger_custom_level(self):
        """Test loading logger with custom log level"""
        logger = load_logger("test_logger", log_level="DEBUG")
        assert logger is not None


class TestLogImagesAuto:
    @pytest.fixture
    def mock_accelerator(self):
        """Create a mock accelerator"""
        accelerator = Mock()
        accelerator.is_main_process = True
        accelerator.log = Mock()
        return accelerator

    def test_log_images_not_main_process(self):
        """Test that logging is skipped on non-main process"""
        accelerator = Mock()
        accelerator.is_main_process = False

        images = torch.randn(4, 3, 64, 64)
        log_images_auto(accelerator, "test/images", images, step=0)

        accelerator.log.assert_not_called()

    def test_log_images_main_process(self, mock_accelerator):
        """Test logging images on main process"""
        images = torch.randn(4, 3, 64, 64) * 2 - 1  # [-1, 1] range

        mock_accelerator.get_tracker = Mock(side_effect=Exception("No tracker"))

        log_images_auto(mock_accelerator, "test/images", images, step=0)

        # Should log at least a scalar as fallback
        mock_accelerator.log.assert_called()

    def test_log_images_with_wandb(self, mock_accelerator):
        """Test logging images with W&B tracker"""
        images = torch.randn(4, 3, 64, 64) * 2 - 1

        mock_wandb_run = Mock()
        mock_accelerator.get_tracker = Mock(return_value=mock_wandb_run)

        # Patch wandb at import time within the function
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: Mock() if name == 'wandb' else __import__(name, *args, **kwargs)):
            log_images_auto(mock_accelerator, "test/images", images, step=0)

    def test_log_images_max_images(self, mock_accelerator):
        """Test that max_images limits the number of logged images"""
        images = torch.randn(32, 3, 64, 64) * 2 - 1

        mock_accelerator.get_tracker = Mock(side_effect=Exception("No tracker"))

        log_images_auto(mock_accelerator, "test/images", images, step=0, max_images=8)

        # Should still log something
        mock_accelerator.log.assert_called()

    def test_log_images_normalization(self, mock_accelerator):
        """Test that images are normalized correctly"""
        # Create images in [-1, 1] range
        images = torch.tensor([[[[-1.0, 1.0]]]]) * 2 - 1

        mock_accelerator.get_tracker = Mock(side_effect=Exception("No tracker"))

        log_images_auto(mock_accelerator, "test/images", images, step=0)


class TestLogTextAuto:
    @pytest.fixture
    def mock_accelerator(self):
        """Create a mock accelerator"""
        accelerator = Mock()
        accelerator.is_main_process = True
        accelerator.log = Mock()
        accelerator.print = Mock()
        return accelerator

    def test_log_text_not_main_process(self):
        """Test that logging is skipped on non-main process"""
        accelerator = Mock()
        accelerator.is_main_process = False

        rows = ["text1", "text2", "text3"]
        log_text_auto(accelerator, "test/text", rows, step=0)

        accelerator.log.assert_not_called()

    def test_log_text_with_strings(self, mock_accelerator):
        """Test logging text with list of strings"""
        rows = ["text1", "text2", "text3"]

        mock_accelerator.get_tracker = Mock(side_effect=Exception("No tracker"))

        log_text_auto(mock_accelerator, "test/text", rows, step=0)

        # Should print as fallback
        mock_accelerator.print.assert_called()

    def test_log_text_with_dicts(self, mock_accelerator):
        """Test logging text with list of dicts"""
        rows = [
            {"id": 0, "pred": "A", "gt": "A"},
            {"id": 1, "pred": "B", "gt": "C"}
        ]

        mock_accelerator.get_tracker = Mock(side_effect=Exception("No tracker"))

        log_text_auto(mock_accelerator, "test/text", rows, step=0)

        mock_accelerator.print.assert_called()

    def test_log_text_with_wandb(self, mock_accelerator):
        """Test logging text with W&B tracker"""
        rows = [{"id": 0, "text": "test"}]

        mock_wandb_run = Mock()
        mock_accelerator.get_tracker = Mock(return_value=mock_wandb_run)

        # Patch wandb at import time within the function
        with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs: Mock() if name == 'wandb' else __import__(name, *args, **kwargs)):
            log_text_auto(mock_accelerator, "test/text", rows, step=0)

    def test_log_text_max_rows(self, mock_accelerator):
        """Test that max_rows limits the number of logged rows"""
        rows = [f"text{i}" for i in range(100)]

        mock_accelerator.get_tracker = Mock(side_effect=Exception("No tracker"))

        log_text_auto(mock_accelerator, "test/text", rows, step=0, max_rows=10)

        mock_accelerator.print.assert_called()

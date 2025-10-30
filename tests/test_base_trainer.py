"""
Tests for BaseTrainer abstract class.
"""

import pytest
import torch
from unittest.mock import Mock
from qflux.base_trainer import BaseTrainer


class MockTrainer(BaseTrainer):
    """Mock implementation of BaseTrainer for testing."""

    def load_model(self, **kwargs):
        self.model_loaded = True

    def cache(self, train_dataloader):
        self.cache_completed = True

    def fit(self, train_dataloader):
        self.training_completed = True

    def predict(self, *args, **kwargs):
        return "mock_prediction"

    def set_model_devices(self, mode="train"):
        self.devices_set = mode

    def encode_prompt(self, *args, **kwargs):
        return torch.randn(1, 10, 512), torch.ones(1, 10)


class TestBaseTrainer:
    """Test cases for BaseTrainer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.data.batch_size = 2
        self.mock_config.cache.use_cache = True
        self.mock_config.cache.cache_dir = "/tmp/cache"

        # Create trainer instance
        self.trainer = MockTrainer(self.mock_config)

    def test_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.config == self.mock_config
        assert self.trainer.batch_size == 2
        assert self.trainer.use_cache is True
        assert self.trainer.cache_dir == "/tmp/cache"
        assert self.trainer.weight_dtype == torch.bfloat16
        assert self.trainer.global_step == 0

    def test_abstract_methods_implemented(self):
        """Test that all abstract methods are implemented."""
        # These should not raise NotImplementedError
        self.trainer.load_model()
        assert self.trainer.model_loaded

        mock_dataloader = Mock()
        self.trainer.cache(mock_dataloader)
        assert self.trainer.cache_completed

        self.trainer.fit(mock_dataloader)
        assert self.trainer.training_completed

        result = self.trainer.predict("test")
        assert result == "mock_prediction"

        self.trainer.set_model_devices("predict")
        assert self.trainer.devices_set == "predict"

        embeds, mask = self.trainer.encode_prompt("test prompt")
        assert embeds.shape == (1, 10, 512)
        assert mask.shape == (1, 10)

    def test_get_model_type(self):
        """Test model type retrieval."""
        # Test default case - need to patch getattr to simulate missing attribute
        from unittest.mock import patch

        with patch('qflux.base_trainer.getattr') as mock_getattr:
            mock_getattr.return_value = "unknown"
            result = self.trainer.get_model_type()
            assert result == "unknown"
            mock_getattr.assert_called_once_with(self.mock_config.model, 'model_type', 'unknown')

        # Test with model_type in config - directly test the getattr behavior
        with patch('qflux.base_trainer.getattr') as mock_getattr:
            mock_getattr.return_value = "flux_kontext"
            result = self.trainer.get_model_type()
            assert result == "flux_kontext"

    def test_get_precision_info(self):
        """Test precision information retrieval."""
        # Set up mock config attributes
        self.mock_config.train.mixed_precision = "bf16"
        self.mock_config.model.quantize = True

        precision_info = self.trainer.get_precision_info()

        assert "weight_dtype" in precision_info
        assert precision_info["mixed_precision"] == "bf16"
        assert precision_info["quantize"] is True

    def test_log_model_info(self, caplog):
        """Test model information logging."""
        # Set up the logger to capture logs at the correct level
        import logging
        caplog.set_level(logging.INFO, logger='qflux.base_trainer')

        # Mock the get_model_type and get_precision_info methods to return expected values
        from unittest.mock import patch

        with patch.object(self.trainer, 'get_model_type', return_value="test_model"), \
             patch.object(self.trainer, 'get_precision_info', return_value={'test': 'info'}):

            self.trainer.log_model_info()

            # Check that log messages were generated
            log_text = caplog.text
            assert "Model Type: test_model" in log_text
            assert "Batch Size: 2" in log_text
            assert "Use Cache: True" in log_text


class TestAbstractEnforcement:
    """Test that BaseTrainer properly enforces abstract methods."""

    def test_cannot_instantiate_base_trainer_directly(self):
        """Test that BaseTrainer cannot be instantiated directly."""
        mock_config = Mock()
        mock_config.data.batch_size = 1
        mock_config.cache.use_cache = False
        mock_config.cache.cache_dir = "/tmp"

        with pytest.raises(TypeError):
            BaseTrainer(mock_config)

    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations fail."""

        class IncompleteTrainer(BaseTrainer):
            def load_model(self, **kwargs):
                pass
            # Missing other abstract methods

        mock_config = Mock()
        mock_config.data.batch_size = 1
        mock_config.cache.use_cache = False
        mock_config.cache.cache_dir = "/tmp"

        with pytest.raises(TypeError):
            IncompleteTrainer(mock_config)


if __name__ == "__main__":
    pytest.main([__file__])

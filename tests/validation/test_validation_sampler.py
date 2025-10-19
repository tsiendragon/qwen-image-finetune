"""
Tests for ValidationSampler

This module tests:
1. Sampler initialization and configuration
2. Sample generation workflow
3. Embedding caching functionality
4. Error handling and edge cases
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from src.validation.validation_sampler import ValidationSampler
from src.data.config import ValidationDataConfig, DataConfig


class TestValidationSamplerInit:
    """Test ValidationSampler initialization"""

    def test_initialization_minimal_config(self):
        """Test initialization with minimal config"""
        # Mock accelerator
        mock_accelerator = Mock()
        mock_accelerator.device = torch.device("cpu")

        # Create minimal config
        config = ValidationDataConfig(
            enabled=True,
            batch_size=1,
            dataset_path="dummy_path"
        )

        sampler = ValidationSampler(
            config=config,
            accelerator=mock_accelerator
        )

        assert sampler.config == config
        assert sampler.accelerator == mock_accelerator
        assert sampler.weight_dtype == torch.bfloat16  # Default
        assert sampler.embeddings_cached is False

    def test_initialization_with_custom_dtype(self):
        """Test initialization with custom weight dtype"""
        mock_accelerator = Mock()
        mock_accelerator.device = torch.device("cpu")

        config = ValidationDataConfig(
            enabled=True,
            batch_size=1,
            dataset_path="dummy_path"
        )

        sampler = ValidationSampler(
            config=config,
            accelerator=mock_accelerator,
            weight_dtype=torch.float32
        )

        assert sampler.weight_dtype == torch.float32

    def test_initialization_with_data_config(self):
        """Test initialization with optional data_config"""
        mock_accelerator = Mock()
        mock_accelerator.device = torch.device("cpu")

        config = ValidationDataConfig(
            enabled=True,
            batch_size=1,
            dataset_path="dummy_path"
        )

        data_config = Mock(spec=DataConfig)

        sampler = ValidationSampler(
            config=config,
            accelerator=mock_accelerator,
            data_config=data_config
        )

        assert sampler.data_config == data_config


class TestValidationSamplerCaching:
    """Test embedding caching functionality"""

    def test_initial_cache_state(self):
        """Test that cache starts empty"""
        mock_accelerator = Mock()
        mock_accelerator.device = torch.device("cpu")

        config = ValidationDataConfig(
            enabled=True,
            batch_size=1,
            dataset_path="dummy_path"
        )

        sampler = ValidationSampler(
            config=config,
            accelerator=mock_accelerator
        )

        assert len(sampler.cached_embeddings) == 0
        assert sampler.embeddings_cached is False

    def test_cache_embeddings_structure(self):
        """Test structure of cached embeddings"""
        mock_accelerator = Mock()
        mock_accelerator.device = torch.device("cpu")

        config = ValidationDataConfig(
            enabled=True,
            batch_size=2,
            dataset_path="dummy_path"
        )

        sampler = ValidationSampler(
            config=config,
            accelerator=mock_accelerator
        )

        # Simulate adding cached embeddings
        sample_embedding = {
            'prompt_embeds': torch.randn(1, 128, 768),
            'pooled_prompt_embeds': torch.randn(1, 768),
        }
        sampler.cached_embeddings.append(sample_embedding)

        assert len(sampler.cached_embeddings) == 1
        assert 'prompt_embeds' in sampler.cached_embeddings[0]


class TestValidationSamplerDataset:
    """Test dataset handling"""

    @pytest.mark.integration
    def test_validation_dataset_property(self):
        """Test validation_dataset property"""
        mock_accelerator = Mock()
        mock_accelerator.device = torch.device("cpu")

        config = ValidationDataConfig(
            enabled=True,
            batch_size=1,
            dataset_path="dummy_path"
        )

        sampler = ValidationSampler(
            config=config,
            accelerator=mock_accelerator
        )

        # Initially should be None
        assert sampler.validation_dataset is None


class TestValidationSamplerConfig:
    """Test internal configuration"""

    def test_internal_config_defaults(self):
        """Test that internal config has sensible defaults"""
        mock_accelerator = Mock()
        mock_accelerator.device = torch.device("cpu")

        config = ValidationDataConfig(
            enabled=True,
            batch_size=1,
            dataset_path="dummy_path"
        )

        sampler = ValidationSampler(
            config=config,
            accelerator=mock_accelerator
        )

        # Check internal config defaults
        assert hasattr(sampler, '_internal_config')
        assert 'max_image_size' in sampler._internal_config
        assert sampler._internal_config['max_image_size'] == 512
        assert sampler._internal_config['log_prefix'] == "validation"
        assert sampler._internal_config['move_vae_to_cpu_after'] is True
        assert sampler._internal_config['skip_on_error'] is True


class TestValidationSamplerEdgeCases:
    """Test edge cases and error handling"""

    def test_disabled_validation(self):
        """Test behavior when validation is disabled"""
        mock_accelerator = Mock()
        mock_accelerator.device = torch.device("cpu")

        config = ValidationDataConfig(
            enabled=False,
            batch_size=1,
            dataset_path="dummy_path"
        )

        sampler = ValidationSampler(
            config=config,
            accelerator=mock_accelerator
        )

        # Sampler should still initialize
        assert sampler.config.enabled is False

    def test_with_cpu_device(self):
        """Test sampler works with CPU device"""
        mock_accelerator = Mock()
        mock_accelerator.device = torch.device("cpu")

        config = ValidationDataConfig(
            enabled=True,
            batch_size=1,
            dataset_path="dummy_path"
        )

        sampler = ValidationSampler(
            config=config,
            accelerator=mock_accelerator,
            weight_dtype=torch.float32  # CPU-friendly dtype
        )

        assert sampler.weight_dtype == torch.float32


@pytest.mark.e2e
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
class TestValidationSamplerE2E:
    """End-to-end tests requiring GPU and real models"""

    def test_sample_generation_e2e(self):
        """Test end-to-end sample generation workflow"""
        # This would require real model, dataset, etc.
        # Placeholder for future implementation
        pytest.skip("需要完整环境和真实模型")

    def test_cache_and_generate_e2e(self):
        """Test caching embeddings and generating samples"""
        # This would test the full workflow
        # Placeholder for future implementation
        pytest.skip("需要完整环境和真实模型")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])


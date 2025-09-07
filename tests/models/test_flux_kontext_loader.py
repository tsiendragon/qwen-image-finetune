"""
Tests for Flux Kontext model loader.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from src.models.flux_kontext_loader import (
    load_flux_kontext_vae,
    load_flux_kontext_clip,
    load_flux_kontext_t5,
    load_flux_kontext_transformer,
    load_flux_kontext_tokenizers,
    load_flux_kontext_scheduler,
    validate_flux_kontext_components
)


class TestFluxKontextLoader:
    """Test cases for Flux Kontext model loading functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_path = "test-model-path"
        self.weight_dtype = torch.bfloat16
        self.device_map = "cpu"

    @patch('src.models.flux_kontext_loader.FluxKontextPipeline')
    def test_load_flux_kontext_vae(self, mock_pipeline_class):
        """Test VAE loading."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_vae = Mock()
        mock_vae.to = Mock()
        mock_pipeline.vae = mock_vae
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Test loading
        result = load_flux_kontext_vae(self.model_path, self.weight_dtype, self.device_map)

        # Verify pipeline was created correctly
        mock_pipeline_class.from_pretrained.assert_called_once_with(
            self.model_path,
            torch_dtype=self.weight_dtype,
            use_safetensors=True,
            device_map="cpu"
        )

        # Verify VAE was moved to correct device
        mock_vae.to.assert_called_once_with(self.device_map)

        # Verify result
        assert result == mock_vae

    @patch('src.models.flux_kontext_loader.FluxKontextPipeline')
    def test_load_flux_kontext_clip(self, mock_pipeline_class):
        """Test CLIP encoder loading."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_text_encoder = Mock()
        mock_text_encoder.to = Mock()
        mock_pipeline.text_encoder = mock_text_encoder
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Test loading
        result = load_flux_kontext_clip(self.model_path, self.weight_dtype, self.device_map)

        # Verify result
        assert result == mock_text_encoder
        mock_text_encoder.to.assert_called_once_with(self.device_map)

    @patch('src.models.flux_kontext_loader.FluxKontextPipeline')
    def test_load_flux_kontext_t5(self, mock_pipeline_class):
        """Test T5 encoder loading."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_text_encoder_2 = Mock()
        mock_text_encoder_2.to = Mock()
        mock_pipeline.text_encoder_2 = mock_text_encoder_2
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Test loading
        result = load_flux_kontext_t5(self.model_path, self.weight_dtype, self.device_map)

        # Verify result
        assert result == mock_text_encoder_2
        mock_text_encoder_2.to.assert_called_once_with(self.device_map)

    @patch('src.models.flux_kontext_loader.FluxKontextPipeline')
    def test_load_flux_kontext_transformer(self, mock_pipeline_class):
        """Test transformer loading."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_transformer = Mock()
        mock_transformer.to = Mock()
        mock_pipeline.transformer = mock_transformer
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Test loading
        result = load_flux_kontext_transformer(self.model_path, self.weight_dtype, self.device_map)

        # Verify result
        assert result == mock_transformer
        mock_transformer.to.assert_called_once_with(self.device_map)

    @patch('src.models.flux_kontext_loader.FluxKontextPipeline')
    def test_load_flux_kontext_tokenizers(self, mock_pipeline_class):
        """Test tokenizers loading."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer_2 = Mock()
        mock_pipeline.tokenizer = mock_tokenizer
        mock_pipeline.tokenizer_2 = mock_tokenizer_2
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Test loading
        tokenizer, tokenizer_2 = load_flux_kontext_tokenizers(self.model_path)

        # Verify results
        assert tokenizer == mock_tokenizer
        assert tokenizer_2 == mock_tokenizer_2

    @patch('src.models.flux_kontext_loader.FluxKontextPipeline')
    def test_load_flux_kontext_scheduler(self, mock_pipeline_class):
        """Test scheduler loading."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_scheduler = Mock()
        mock_pipeline.scheduler = mock_scheduler
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Test loading
        result = load_flux_kontext_scheduler(self.model_path)

        # Verify result
        assert result == mock_scheduler

    @patch('src.models.flux_kontext_loader.FluxKontextPipeline')
    def test_loading_error_handling(self, mock_pipeline_class):
        """Test error handling during loading."""
        # Setup mock to raise exception
        mock_pipeline_class.from_pretrained.side_effect = Exception("Loading failed")

        # Test that exception is raised
        with pytest.raises(Exception, match="Loading failed"):
            load_flux_kontext_vae(self.model_path, self.weight_dtype, self.device_map)

    @patch('src.models.flux_kontext_loader.torch.cuda.empty_cache')
    @patch('src.models.flux_kontext_loader.FluxKontextPipeline')
    def test_memory_cleanup(self, mock_pipeline_class, mock_empty_cache):
        """Test that memory is cleaned up after loading."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_vae = Mock()
        mock_vae.to = Mock()
        mock_pipeline.vae = mock_vae
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline

        # Test loading
        load_flux_kontext_vae(self.model_path, self.weight_dtype, self.device_map)

        # Verify memory cleanup was called
        mock_empty_cache.assert_called()


class TestComponentValidation:
    """Test component validation functionality."""

    @patch('src.models.flux_kontext_loader.FluxKontextPipeline')
    @patch('src.models.flux_kontext_loader.torch.allclose')
    def test_validate_components_success(self, mock_allclose, mock_pipeline_class):
        """Test successful component validation."""
        # Setup mock components
        mock_vae = Mock()
        mock_text_encoder = Mock()
        mock_text_encoder_2 = Mock()
        mock_transformer = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer_2 = Mock()
        mock_scheduler = Mock()

        # Setup mock state dicts
        mock_state_dict = {"param1": torch.randn(10, 10)}
        mock_vae.state_dict.return_value = mock_state_dict
        mock_text_encoder.state_dict.return_value = mock_state_dict
        mock_text_encoder_2.state_dict.return_value = mock_state_dict
        mock_transformer.state_dict.return_value = mock_state_dict

        # Setup mock parameters
        mock_param = Mock()
        mock_param.numel.return_value = 100
        mock_param.element_size.return_value = 4

        for component in [mock_vae, mock_text_encoder, mock_text_encoder_2, mock_transformer]:
            component.parameters.return_value = [mock_param]

        # Setup mock reference pipeline
        mock_ref_pipeline = Mock()
        mock_ref_pipeline.vae = mock_vae
        mock_ref_pipeline.text_encoder = mock_text_encoder
        mock_ref_pipeline.text_encoder_2 = mock_text_encoder_2
        mock_ref_pipeline.transformer = mock_transformer
        mock_pipeline_class.from_pretrained.return_value = mock_ref_pipeline

        # Setup torch.allclose to return True
        mock_allclose.return_value = True

        # Test validation
        result = validate_flux_kontext_components(
            "test-model",
            mock_vae, mock_text_encoder, mock_text_encoder_2, mock_transformer,
            mock_tokenizer, mock_tokenizer_2, mock_scheduler
        )

        # Verify results
        assert "vae" in result
        assert "text_encoder" in result
        assert "text_encoder_2" in result
        assert "transformer" in result

        for component_result in result.values():
            assert "parameters_match" in component_result
            assert "num_parameters" in component_result
            assert "memory_mb" in component_result

    @patch('src.models.flux_kontext_loader.FluxKontextPipeline')
    def test_validate_components_error(self, mock_pipeline_class):
        """Test validation error handling."""
        # Setup mock to raise exception
        mock_pipeline_class.from_pretrained.side_effect = Exception("Validation failed")

        # Test validation
        result = validate_flux_kontext_components(
            "test-model",
            Mock(), Mock(), Mock(), Mock(),
            Mock(), Mock(), Mock()
        )

        # Verify error is captured
        assert "error" in result
        assert "Validation failed" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__])

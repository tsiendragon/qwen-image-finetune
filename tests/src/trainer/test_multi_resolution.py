"""
Tests for multi-resolution training functionality
"""
import pytest
import torch
import logging

from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMultiResolutionTrainer:
    """Test suite for multi-resolution training components"""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    @pytest.fixture
    def trainer_config(self):
        """Load a test configuration without multi_resolutions"""
        config_path = "tests/test_configs/test_example_fluxkontext_fp16.yaml"
        return load_config_from_yaml(config_path)

    @pytest.fixture
    def trainer_config_multi_res(self):
        """Load a test configuration WITH multi_resolutions"""
        config_path = "tests/test_configs/test_example_fluxkontext_fp16_multiresolution.yaml"
        return load_config_from_yaml(config_path)

    @pytest.fixture
    def trainer(self, trainer_config_multi_res):
        """Create a trainer instance with multi-resolution support"""
        return FluxKontextLoraTrainer(trainer_config_multi_res)

    # ==================== Test _convert_img_shapes_to_latent ====================

    def test_convert_img_shapes_to_latent_basic(self, trainer):
        """Test basic conversion from pixel space to latent space"""
        img_shapes_original = [
            (3, 512, 512),
            (3, 640, 640),
            (3, 768, 512),
        ]

        img_shapes_latent = trainer._convert_img_shapes_to_latent(
            img_shapes_original,
            vae_scale_factor=8,
            packing_factor=2
        )

        # Check output
        assert len(img_shapes_latent) == 3
        assert img_shapes_latent[0] == (1, 32, 32)   # 512 // 16 = 32
        assert img_shapes_latent[1] == (1, 40, 40)   # 640 // 16 = 40
        assert img_shapes_latent[2] == (1, 48, 32)   # 768 // 16 = 48, 512 // 16 = 32

    def test_convert_img_shapes_to_latent_empty(self, trainer):
        """Test conversion with empty list"""
        result = trainer._convert_img_shapes_to_latent([])
        assert result == []

    def test_convert_img_shapes_to_latent_different_channels(self, trainer):
        """Test conversion with different channel counts"""
        img_shapes_original = [
            (3, 512, 512),  # RGB
            (1, 512, 512),  # Grayscale
            (4, 512, 512),  # RGBA
        ]

        img_shapes_latent = trainer._convert_img_shapes_to_latent(img_shapes_original)

        # All should become channel=1 in latent space
        assert all(shape[0] == 1 for shape in img_shapes_latent)
        assert img_shapes_latent[0] == (1, 32, 32)
        assert img_shapes_latent[1] == (1, 32, 32)
        assert img_shapes_latent[2] == (1, 32, 32)

    def test_convert_img_shapes_to_latent_custom_scale(self, trainer):
        """Test conversion with custom vae_scale_factor"""
        img_shapes_original = [(3, 256, 256)]

        # Test with different scale factors
        result_8 = trainer._convert_img_shapes_to_latent(
            img_shapes_original, vae_scale_factor=8, packing_factor=1
        )
        result_16 = trainer._convert_img_shapes_to_latent(
            img_shapes_original, vae_scale_factor=8, packing_factor=2
        )

        assert result_8[0] == (1, 32, 32)   # 256 // 8 = 32
        assert result_16[0] == (1, 16, 16)  # 256 // 16 = 16

    def test_convert_img_shapes_to_latent_invalid_shape(self, trainer):
        """Test conversion with invalid shape format"""
        img_shapes_invalid = [(512, 512)]  # Missing channel dimension

        with pytest.raises(ValueError, match="Expected shape tuple"):
            trainer._convert_img_shapes_to_latent(img_shapes_invalid)

    # ==================== Test _should_use_multi_resolution_mode ====================

    def test_should_use_multi_resolution_mode_not_configured(self, trainer_config):
        """Should return False when multi_resolutions not configured"""
        # Create config without multi_resolutions
        config = trainer_config
        config.data.init_args.processor.init_args.multi_resolutions = None
        trainer = FluxKontextLoraTrainer(config)

        batch = {
            'img_shapes': [
                [(3, 512, 512), (3, 512, 512)],
                [(3, 640, 640), (3, 640, 640)],
            ]
        }
        # Even with different shapes, should return False without config
        assert trainer._should_use_multi_resolution_mode(batch) is False

    def test_should_use_multi_resolution_mode_single_sample(self, trainer):
        """Single sample should return False even with multi_resolutions configured"""
        batch = {
            'img_shapes': [
                [(3, 512, 512), (3, 512, 512)],  # Only 1 sample
            ]
        }
        assert trainer._should_use_multi_resolution_mode(batch) is False

    def test_should_use_multi_resolution_mode_identical_resolutions(self, trainer):
        """All samples with same resolution should return False"""
        batch = {
            'img_shapes': [
                [(3, 512, 512), (3, 512, 512)],  # Sample 1
                [(3, 512, 512), (3, 512, 512)],  # Sample 2
                [(3, 512, 512), (3, 512, 512)],  # Sample 3
                [(3, 512, 512), (3, 512, 512)],  # Sample 4
            ]
        }
        assert trainer._should_use_multi_resolution_mode(batch) is False

    def test_should_use_multi_resolution_mode_different_target_resolutions(self, trainer):
        """Samples with different target resolutions should return True"""
        batch = {
            'img_shapes': [
                [(3, 512, 512), (3, 512, 512)],  # Sample 1: 512x512
                [(3, 640, 640), (3, 640, 640)],  # Sample 2: 640x640 (different)
                [(3, 768, 512), (3, 768, 512)],  # Sample 3: 768x512 (different)
            ]
        }
        assert trainer._should_use_multi_resolution_mode(batch) is True

    def test_should_use_multi_resolution_mode_different_control_resolutions(self, trainer):
        """Samples with different control resolutions should return True"""
        batch = {
            'img_shapes': [
                [(3, 512, 512), (3, 512, 512), (3, 512, 512)],  # Sample 1: all same
                [(3, 512, 512), (3, 640, 640), (3, 512, 512)],  # Sample 2: control different
            ]
        }
        assert trainer._should_use_multi_resolution_mode(batch) is True

    def test_should_use_multi_resolution_mode_real_format(self, trainer):
        """Test with real img_shapes format from dataset"""
        batch = {
            'img_shapes': [
                [(3, 384, 672), (3, 384, 672), (3, 512, 512)],  # Sample 1
                [(3, 384, 672), (3, 384, 672), (3, 512, 512)],  # Sample 2
            ]
        }
        # All identical, should return False
        assert trainer._should_use_multi_resolution_mode(batch) is False

        # Now with different resolutions
        batch_different = {
            'img_shapes': [
                [(3, 384, 672), (3, 384, 672), (3, 512, 512)],  # Sample 1
                [(3, 512, 512), (3, 512, 512), (3, 640, 640)],  # Sample 2 (different)
            ]
        }
        assert trainer._should_use_multi_resolution_mode(batch_different) is True

    def test_should_use_multi_resolution_mode_tensor_input(self, trainer):
        """Test with tensor img_shapes (should convert to list)"""
        batch = {
            'img_shapes': torch.tensor([
                [[3, 512, 512], [3, 512, 512]],
                [[3, 640, 640], [3, 640, 640]],
            ])
        }
        assert trainer._should_use_multi_resolution_mode(batch) is True

    # ==================== Test _pad_latents_for_multi_res ====================

    def test_pad_latents_basic_functionality(self, trainer):
        """Test basic padding functionality"""
        latents = [
            torch.randn(100, 64),  # Sample 1: 100 tokens
            torch.randn(150, 64),  # Sample 2: 150 tokens
            torch.randn(120, 64),  # Sample 3: 120 tokens
        ]
        max_seq_len = 150

        padded, mask = trainer._pad_latents_for_multi_res(latents, max_seq_len)

        # Check output shapes
        assert padded.shape == (3, 150, 64), f"Expected shape (3, 150, 64), got {padded.shape}"
        assert mask.shape == (3, 150), f"Expected mask shape (3, 150), got {mask.shape}"

        # Check mask correctness
        assert mask[0, :100].all(), "First sample mask should be all True for first 100 tokens"
        assert not mask[0, 100:].any(), "First sample mask should be all False for padding"
        assert mask[1].all(), "Second sample mask should be all True (no padding)"
        assert mask[2, :120].all(), "Third sample mask should be all True for first 120 tokens"
        assert not mask[2, 120:].any(), "Third sample mask should be all False for padding"

    def test_pad_latents_infers_device_and_dtype(self, trainer):
        """Test that device and dtype are automatically inferred from input latents"""
        # Test with float32 on CPU
        latents_cpu_float32 = [
            torch.randn(100, 64, dtype=torch.float32),
            torch.randn(120, 64, dtype=torch.float32),
        ]
        padded, mask = trainer._pad_latents_for_multi_res(latents_cpu_float32, 150)

        assert padded.device == torch.device('cpu'), "Should use CPU device from input"
        assert padded.dtype == torch.float32, "Should use float32 dtype from input"
        assert mask.device == torch.device('cpu'), "Mask should be on same device"

        # Test with bfloat16 (if available)
        if torch.cuda.is_available():
            device_cuda = torch.device('cuda:0')
            latents_cuda_bf16 = [
                torch.randn(100, 64, dtype=torch.bfloat16, device=device_cuda),
                torch.randn(120, 64, dtype=torch.bfloat16, device=device_cuda),
            ]
            padded, mask = trainer._pad_latents_for_multi_res(latents_cuda_bf16, 150)

            assert padded.device == device_cuda, "Should use CUDA device from input"
            assert padded.dtype == torch.bfloat16, "Should use bfloat16 dtype from input"
            assert mask.device == device_cuda, "Mask should be on same device"

    def test_pad_latents_empty_list_error(self, trainer):
        """Should raise error for empty latent list"""
        with pytest.raises(ValueError, match="Cannot pad empty latent list"):
            trainer._pad_latents_for_multi_res([], 100)

    def test_pad_latents_invalid_dimensions_error(self, trainer):
        """Should raise error for invalid tensor dimensions"""
        latents = [
            torch.randn(100, 64),
            torch.randn(100, 64, 1),  # 3D tensor - invalid
        ]
        with pytest.raises(ValueError, match="Expected 2D latent tensor"):
            trainer._pad_latents_for_multi_res(latents, 150)

    def test_pad_latents_channel_mismatch_error(self, trainer):
        """Should raise error for channel mismatch"""
        latents = [
            torch.randn(100, 64),
            torch.randn(100, 32),  # Different channel count
        ]
        with pytest.raises(ValueError, match="Channel mismatch"):
            trainer._pad_latents_for_multi_res(latents, 150)

    def test_pad_latents_exceeds_max_seq_len_error(self, trainer):
        """Should raise error when latent exceeds max_seq_len"""
        latents = [
            torch.randn(100, 64),
            torch.randn(200, 64),  # Exceeds max_seq_len
        ]
        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            trainer._pad_latents_for_multi_res(latents, 150)

    def test_pad_latents_preserves_values(self, trainer):
        """Test that original values are preserved after padding"""
        original = torch.randn(100, 64, dtype=torch.float32)
        latents = [original]
        max_seq_len = 150

        padded, mask = trainer._pad_latents_for_multi_res(latents, max_seq_len)

        # Check that non-padded values match original
        torch.testing.assert_close(
            padded[0, :100],
            original,
            msg="Original values should be preserved"
        )

        # Check that padded values are zero
        assert (padded[0, 100:] == 0).all(), "Padded values should be zero"

    def test_pad_latents_mixed_device_conversion(self, trainer):
        """Test that latents on different devices are converted to match the first one"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Create latents on different devices
        device_cuda = torch.device('cuda:0')
        latents = [
            torch.randn(100, 64, device=device_cuda),  # First on CUDA
            torch.randn(120, 64),  # Second on CPU
        ]

        padded, mask = trainer._pad_latents_for_multi_res(latents, 150)

        # All should be on CUDA (from first latent)
        assert padded.device == device_cuda, "Should convert all to first latent's device"
        assert mask.device == device_cuda, "Mask should be on same device"

    # ==================== Test _compute_loss_multi_resolution ====================

    def test_compute_loss_multi_resolution_mse(self, trainer, device):
        """Test MSE loss computation with masking"""
        batch_size, seq_len, channels = 4, 100, 64

        model_pred = torch.randn(batch_size, seq_len, channels, device=device)
        target = torch.randn(batch_size, seq_len, channels, device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # Mask out some tokens
        attention_mask[0, 50:] = False  # Sample 0: 50 valid tokens
        attention_mask[1, 80:] = False  # Sample 1: 80 valid tokens
        attention_mask[2, 70:] = False  # Sample 2: 70 valid tokens
        # Sample 3: all valid (100 tokens)

        loss = trainer._compute_loss_multi_resolution(
            model_pred, target, attention_mask
        )

        # Loss should be a scalar
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

        # Verify loss only computed on valid tokens
        # Total valid elements = (50 + 80 + 70 + 100) * 64 = 19200
        expected_valid_elements = (50 + 80 + 70 + 100) * channels
        assert expected_valid_elements == 19200

    def test_compute_loss_multi_resolution_all_masked(self, trainer, device):
        """Test loss computation when all tokens are masked"""
        batch_size, seq_len, channels = 2, 50, 64

        model_pred = torch.randn(batch_size, seq_len, channels, device=device)
        target = torch.randn(batch_size, seq_len, channels, device=device)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        loss = trainer._compute_loss_multi_resolution(
            model_pred, target, attention_mask
        )

        # Should return zero loss with warning
        assert loss.item() == 0.0, "Loss should be zero when all tokens are masked"

    def test_compute_loss_multi_resolution_shape_mismatch_error(self, trainer, device):
        """Should raise error for shape mismatch"""
        model_pred = torch.randn(4, 100, 64, device=device)
        target = torch.randn(4, 100, 32, device=device)  # Different channels
        attention_mask = torch.ones(4, 100, dtype=torch.bool, device=device)

        with pytest.raises(ValueError, match="Shape mismatch"):
            trainer._compute_loss_multi_resolution(
                model_pred, target, attention_mask
            )

    def test_compute_loss_multi_resolution_mask_shape_error(self, trainer, device):
        """Should raise error for incompatible attention mask shape"""
        model_pred = torch.randn(4, 100, 64, device=device)
        target = torch.randn(4, 100, 64, device=device)
        attention_mask = torch.ones(4, 50, dtype=torch.bool, device=device)  # Wrong seq_len

        with pytest.raises(ValueError, match="Attention mask shape"):
            trainer._compute_loss_multi_resolution(
                model_pred, target, attention_mask
            )


# ==================== Integration Tests ====================

class TestMultiResolutionIntegration:
    """Integration tests combining multiple components"""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    @pytest.fixture
    def trainer_config(self):
        config_path = "tests/test_configs/test_example_fluxkontext_fp16.yaml"
        return load_config_from_yaml(config_path)

    @pytest.fixture
    def trainer(self, trainer_config):
        return FluxKontextLoraTrainer(trainer_config)

    def test_multi_resolution_pipeline(self, trainer, device, dtype):
        """Test the complete multi-resolution pipeline: pad -> compute loss"""
        # Step 1: Create latents with different resolutions
        latents = [
            torch.randn(100, 64),  # 100 tokens
            torch.randn(150, 64),  # 150 tokens
            torch.randn(120, 64),  # 120 tokens
        ]
        max_seq_len = 150

        # Step 2: Pad latents
        padded_latents, attention_mask = trainer._pad_latents_for_multi_res(
            latents, max_seq_len
        )

        # Step 3: Simulate model predictions and targets
        model_pred = padded_latents + torch.randn_like(padded_latents) * 0.1
        target = padded_latents

        # Step 4: Compute loss
        loss = trainer._compute_loss_multi_resolution(
            model_pred, target, attention_mask
        )

        # Verify loss is reasonable
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

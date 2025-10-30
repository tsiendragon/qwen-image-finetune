"""
Tests for FluxKontextLoraTrainer.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from qflux.flux_kontext_trainer import FluxKontextLoraTrainer


class TestFluxKontextLoraTrainer:
    """Test cases for FluxKontextLoraTrainer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create comprehensive mock config
        self.mock_config = Mock()

        # Data config
        self.mock_config.data.batch_size = 2
        self.mock_config.data.init_args.prompt_image_dropout_rate = 0.1

        # Cache config
        self.mock_config.cache.use_cache = True
        self.mock_config.cache.cache_dir = "/tmp/cache"
        self.mock_config.cache.vae_encoder_device = "cuda:0"
        self.mock_config.cache.text_encoder_device = "cuda:1"
        self.mock_config.cache.get.return_value = "cuda:1"

        # Model config
        self.mock_config.model.pretrained_model_name_or_path = "test-model"
        self.mock_config.model.quantize = False
        self.mock_config.model.lora.r = 16
        self.mock_config.model.lora.lora_alpha = 16
        self.mock_config.model.lora.init_lora_weights = "gaussian"
        self.mock_config.model.lora.target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        # Training config
        self.mock_config.train.gradient_accumulation_steps = 4
        self.mock_config.train.max_train_steps = 1000
        self.mock_config.train.num_epochs = 10
        self.mock_config.train.checkpointing_steps = 100
        self.mock_config.train.max_grad_norm = 1.0
        self.mock_config.train.mixed_precision = "bf16"
        self.mock_config.train.gradient_checkpointing = True

        # Logging config
        self.mock_config.logging.output_dir = "/tmp/output"
        self.mock_config.logging.report_to = "tensorboard"
        self.mock_config.logging.tracker_project_name = "test_project"

        # Optimizer config
        self.mock_config.optimizer.init_args = {
            "lr": 0.0001,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "eps": 1e-8
        }

        # LR scheduler config
        self.mock_config.lr_scheduler.scheduler_type = "cosine"
        self.mock_config.lr_scheduler.warmup_steps = 100

        # Predict config
        self.mock_config.predict.devices = {
            "vae": "cuda:0",
            "text_encoder": "cuda:1",
            "text_encoder_2": "cuda:1",
            "transformer": "cuda:0"
        }

    @patch('qflux.flux_kontext_trainer.check_cache_exists')
    def test_initialization(self, mock_check_cache):
        """Test trainer initialization."""
        mock_check_cache.return_value = True

        trainer = FluxKontextLoraTrainer(self.mock_config)

        # Test basic attributes
        assert trainer.config == self.mock_config
        assert trainer.batch_size == 2
        assert trainer.use_cache == True
        assert trainer.cache_dir == "/tmp/cache"
        assert trainer.quantize == False
        assert trainer.prompt_image_dropout_rate == 0.1

        # Test component attributes are initialized to None
        assert trainer.vae is None
        assert trainer.text_encoder is None
        assert trainer.text_encoder_2 is None
        assert trainer.transformer is None
        assert trainer.tokenizer is None
        assert trainer.tokenizer_2 is None
        assert trainer.scheduler is None

    @patch('qflux.flux_kontext_trainer.load_flux_kontext_scheduler')
    @patch('qflux.flux_kontext_trainer.load_flux_kontext_tokenizers')
    @patch('qflux.flux_kontext_trainer.load_flux_kontext_transformer')
    @patch('qflux.flux_kontext_trainer.load_flux_kontext_t5')
    @patch('qflux.x.flux_kontext_trainer.load_flux_kontext_clip')
    @patch('qflux.x.flux_kontext_trainer.load_flux_kontext_vae')
    @patch('qflux.flux_kontext_trainer.check_cache_exists')
    def test_load_model(self, mock_check_cache, mock_load_vae, mock_load_clip,
                       mock_load_t5, mock_load_transformer, mock_load_tokenizers,
                       mock_load_scheduler):
        """Test model loading."""
        mock_check_cache.return_value = False

        # Setup mock returns
        mock_vae = Mock()
        mock_vae.config.block_out_channels = [128, 256, 512, 512]
        mock_vae.config.latent_channels = 16
        mock_load_vae.return_value = mock_vae

        mock_text_encoder = Mock()
        mock_load_clip.return_value = mock_text_encoder

        mock_text_encoder_2 = Mock()
        mock_load_t5.return_value = mock_text_encoder_2

        mock_transformer = Mock()
        mock_load_transformer.return_value = mock_transformer

        mock_tokenizer = Mock()
        mock_tokenizer_2 = Mock()
        mock_load_tokenizers.return_value = (mock_tokenizer, mock_tokenizer_2)

        mock_scheduler = Mock()
        mock_load_scheduler.return_value = mock_scheduler

        # Create trainer and load model
        trainer = FluxKontextLoraTrainer(self.mock_config)
        trainer.load_model()

        # Verify all components were loaded
        assert trainer.vae == mock_vae
        assert trainer.text_encoder == mock_text_encoder
        assert trainer.text_encoder_2 == mock_text_encoder_2
        assert trainer.transformer == mock_transformer
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.tokenizer_2 == mock_tokenizer_2
        assert trainer.scheduler == mock_scheduler

        # Verify VAE scale factor was set correctly
        assert trainer.vae_scale_factor == 8  # 2^(4-1) = 8
        assert trainer.vae_z_dim == 16

        # Verify models were set to non-trainable
        mock_text_encoder.requires_grad_.assert_called_with(False)
        mock_text_encoder_2.requires_grad_.assert_called_with(False)
        mock_vae.requires_grad_.assert_called_with(False)
        mock_transformer.requires_grad_.assert_called_with(False)

    @patch('qflux.x.flux_kontext_trainer.check_cache_exists')
    def test_encode_clip_prompt(self, mock_check_cache):
        """Test CLIP prompt encoding."""
        mock_check_cache.return_value = False
        trainer = FluxKontextLoraTrainer(self.mock_config)

        # Setup mock tokenizer and text encoder
        mock_tokenizer = Mock()
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.attention_mask = torch.ones(1, 10)
        mock_tokenizer.return_value = mock_inputs
        trainer.tokenizer = mock_tokenizer

        mock_text_encoder = Mock()
        mock_text_encoder.device = "cuda:0"
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 10, 512)
        mock_text_encoder.return_value = mock_outputs
        trainer.text_encoder = mock_text_encoder

        # Test encoding
        embeddings, attention_mask = trainer.encode_clip_prompt("test prompt")

        # Verify tokenizer was called correctly
        mock_tokenizer.assert_called_once_with(
            ["test prompt"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Verify text encoder was called
        mock_text_encoder.assert_called_once()

        # Verify outputs
        assert embeddings.shape == (1, 10, 512)
        assert attention_mask.shape == (1, 10)

    @patch('qflux.x.x.flux_kontext_trainer.check_cache_exists')
    def test_encode_t5_prompt(self, mock_check_cache):
        """Test T5 prompt encoding."""
        mock_check_cache.return_value = False
        trainer = FluxKontextLoraTrainer(self.mock_config)

        # Setup mock tokenizer and text encoder
        mock_tokenizer_2 = Mock()
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.attention_mask = torch.ones(1, 20)
        mock_tokenizer_2.return_value = mock_inputs
        trainer.tokenizer_2 = mock_tokenizer_2

        mock_text_encoder_2 = Mock()
        mock_text_encoder_2.device = "cuda:0"
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 20, 1024)
        mock_text_encoder_2.return_value = mock_outputs
        trainer.text_encoder_2 = mock_text_encoder_2

        # Test encoding
        embeddings, attention_mask = trainer.encode_t5_prompt("test prompt")

        # Verify tokenizer was called correctly
        mock_tokenizer_2.assert_called_once_with(
            ["test prompt"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Verify text encoder was called
        mock_text_encoder_2.assert_called_once()

        # Verify outputs
        assert embeddings.shape == (1, 20, 1024)
        assert attention_mask.shape == (1, 20)

    @patch('qflux.x.x.x.x.flux_kontext_trainer.check_cache_exists')
    def test_combine_text_embeddings(self, mock_check_cache):
        """Test text embedding combination."""
        mock_check_cache.return_value = False
        trainer = FluxKontextLoraTrainer(self.mock_config)

        # Create test embeddings
        clip_embeds = torch.randn(2, 10, 512)
        t5_embeds = torch.randn(2, 20, 1024)
        clip_mask = torch.ones(2, 10)
        t5_mask = torch.ones(2, 20)

        # Test combination
        combined_embeds, combined_mask = trainer.combine_text_embeddings(
            clip_embeds, t5_embeds, clip_mask, t5_mask
        )

        # Verify concatenation
        assert combined_embeds.shape == (2, 10, 1536)  # 512 + 1024
        assert combined_mask.shape == (2, 20)  # Uses longer mask

    @patch('qflux.x.x.x.x.x.flux_kontext_trainer.check_cache_exists')
    def test_preprocess_image_for_vae(self, mock_check_cache):
        """Test image preprocessing for VAE."""
        mock_check_cache.return_value = False
        trainer = FluxKontextLoraTrainer(self.mock_config)

        # Create test PIL image
        test_image = Image.new('RGB', (64, 64), color='red')

        # Test preprocessing
        result = trainer._preprocess_image_for_vae(test_image)

        # Verify output shape and range
        assert result.shape == (3, 64, 64)  # CHW format
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    @patch('qflux.x.x.flux_kontext_trainer.check_cache_exists')
    def test_encode_vae_image(self, mock_check_cache):
        """Test VAE image encoding."""
        mock_check_cache.return_value = False
        trainer = FluxKontextLoraTrainer(self.mock_config)

        # Setup mock VAE
        mock_vae = Mock()
        mock_latent_dist = Mock()
        mock_latent_dist.sample.return_value = torch.randn(1, 16, 8, 8)
        mock_encode_result = Mock()
        mock_encode_result.latent_dist = mock_latent_dist
        mock_vae.encode.return_value = mock_encode_result
        mock_vae.config.scaling_factor = 0.18215
        trainer.vae = mock_vae

        # Test encoding
        test_image = torch.randn(1, 3, 64, 64)
        result = trainer._encode_vae_image(test_image)

        # Verify VAE was called
        mock_vae.encode.assert_called_once()

        # Verify result shape
        assert result.shape == (1, 16, 8, 8)

    @patch('qflux.flux_kontext_trainer.check_cache_exists')
    def test_predict_placeholder(self, mock_check_cache):
        """Test predict method (placeholder implementation)."""
        mock_check_cache.return_value = False
        trainer = FluxKontextLoraTrainer(self.mock_config)

        # Create test inputs
        test_image = Image.new('RGB', (64, 64), color='blue')
        test_prompt = "test prompt"

        # Test prediction
        result = trainer.predict(test_image, test_prompt)

        # Verify placeholder behavior
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64, 3)

    @patch('qflux.x.flux_kontext_trainer.check_cache_exists')
    def test_set_model_devices_train_mode(self, mock_check_cache):
        """Test model device allocation for training mode."""
        mock_check_cache.return_value = False
        trainer = FluxKontextLoraTrainer(self.mock_config)
        trainer.use_cache = False

        # Setup mock accelerator and models
        trainer.accelerator = Mock()
        trainer.accelerator.device = "cuda:0"

        trainer.vae = Mock()
        trainer.text_encoder = Mock()
        trainer.text_encoder_2 = Mock()
        trainer.transformer = Mock()

        # Test device allocation
        trainer.set_model_devices(mode="train")

        # Verify all models were moved to accelerator device
        trainer.vae.to.assert_called_with("cuda:0")
        trainer.text_encoder.to.assert_called_with("cuda:0")
        trainer.text_encoder_2.to.assert_called_with("cuda:0")
        trainer.transformer.to.assert_called_with("cuda:0")

    @patch('qflux.flux_kontext_trainer.check_cache_exists')
    def test_set_model_devices_predict_mode(self, mock_check_cache):
        """Test model device allocation for prediction mode."""
        mock_check_cache.return_value = False
        trainer = FluxKontextLoraTrainer(self.mock_config)

        # Setup mock models
        trainer.vae = Mock()
        trainer.text_encoder = Mock()
        trainer.text_encoder_2 = Mock()
        trainer.transformer = Mock()

        # Test device allocation
        trainer.set_model_devices(mode="predict")

        # Verify models were moved to configured devices
        trainer.vae.to.assert_called_with("cuda:0")
        trainer.text_encoder.to.assert_called_with("cuda:1")
        trainer.text_encoder_2.to.assert_called_with("cuda:1")
        trainer.transformer.to.assert_called_with("cuda:0")


class TestTrainingStep:
    """Test training step functionality."""

    def setup_method(self):
        """Set up test fixtures for training step tests."""
        self.mock_config = Mock()
        self.mock_config.data.batch_size = 1
        self.mock_config.cache.use_cache = False
        self.mock_config.cache.cache_dir = "/tmp"
        self.mock_config.model.quantize = False

        # Add required attributes
        self.mock_config.data.init_args.prompt_image_dropout_rate = 0.1
        self.mock_config.model.lora.r = 16
        self.mock_config.model.lora.lora_alpha = 16
        self.mock_config.model.lora.init_lora_weights = "gaussian"
        self.mock_config.model.lora.target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    @patch('qflux.flux_kontext_trainer.check_cache_exists')
    def test_training_step_cached(self, mock_check_cache):
        """Test training step with cached data."""
        mock_check_cache.return_value = False
        trainer = FluxKontextLoraTrainer(self.mock_config)

        # Setup mock accelerator
        trainer.accelerator = Mock()
        trainer.accelerator.device = "cuda:0"
        trainer.weight_dtype = torch.bfloat16

        # Create mock batch with cached data
        batch = {
            'clip_prompt_embed': torch.randn(1, 10, 512),
            't5_prompt_embed': torch.randn(1, 20, 1024),
            'pixel_latent': torch.randn(1, 16, 8, 8),
            'control_latent': torch.randn(1, 16, 8, 8),
            'clip_prompt_mask': torch.ones(1, 10),
            't5_prompt_mask': torch.ones(1, 20)
        }

        # Mock the _training_step_cached method
        trainer._training_step_cached = Mock(return_value=torch.tensor(0.5))

        # Test training step
        loss = trainer.training_step(batch)

        # Verify cached training step was called
        trainer._training_step_cached.assert_called_once_with(batch)
        assert loss == torch.tensor(0.5)

    @patch('qflux.flux_kontext_trainer.check_cache_exists')
    def test_training_step_compute(self, mock_check_cache):
        """Test training step with computation (no cache)."""
        mock_check_cache.return_value = False
        trainer = FluxKontextLoraTrainer(self.mock_config)

        # Create mock batch without cached data
        batch = {
            'image': [torch.randn(3, 64, 64)],
            'control': [torch.randn(3, 64, 64)],
            'prompt': ["test prompt"]
        }

        # Mock the _training_step_compute method
        trainer._training_step_compute = Mock(return_value=torch.tensor(0.3))

        # Test training step
        loss = trainer.training_step(batch)

        # Verify compute training step was called
        trainer._training_step_compute.assert_called_once_with(batch)
        assert loss == torch.tensor(0.3)


if __name__ == "__main__":
    pytest.main([__file__])

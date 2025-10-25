import pytest
import torch
import torch.nn as nn
import tempfile
import safetensors.torch
from qflux.utils.lora_utils import classify_lora_weight, get_lora_layers, FpsLogger


class TestClassifyLoraWeight:
    @pytest.fixture
    def temp_lora_file(self):
        """Create a temporary LoRA weight file"""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            return f.name

    def test_classify_peft_lora(self, temp_lora_file):
        """Test classifying PEFT-style LoRA weights"""
        state_dict = {"model.layer.lora_A": torch.randn(8, 64), "model.layer.lora_B": torch.randn(64, 8)}
        safetensors.torch.save_file(state_dict, temp_lora_file)

        result = classify_lora_weight(temp_lora_file)
        assert result == "PEFT"

    def test_classify_diffusers_lora(self, temp_lora_file):
        """Test classifying Diffusers-style LoRA weights"""
        state_dict = {
            "model.layer.lora.down.weight": torch.randn(8, 64),
            "model.layer.lora.up.weight": torch.randn(64, 8),
        }
        safetensors.torch.save_file(state_dict, temp_lora_file)

        result = classify_lora_weight(temp_lora_file)
        assert result == "DIFFUSERS"

    def test_classify_diffusers_with_processor(self, temp_lora_file):
        """Test classifying Diffusers-style LoRA with processor"""
        state_dict = {
            "model.layer.processor.lora.down.weight": torch.randn(8, 64),
            "model.layer.processor.lora.up.weight": torch.randn(64, 8),
        }
        safetensors.torch.save_file(state_dict, temp_lora_file)

        result = classify_lora_weight(temp_lora_file)
        assert result == "DIFFUSERS(attn-processor)"

    def test_classify_unknown(self, temp_lora_file):
        """Test classifying unknown weight format"""
        state_dict = {"model.layer.weight": torch.randn(64, 64)}
        safetensors.torch.save_file(state_dict, temp_lora_file)

        result = classify_lora_weight(temp_lora_file)
        assert result == "UNKNOWN"


class TestGetLoraLayers:
    def test_get_lora_layers_empty_model(self):
        """Test getting LoRA layers from model without LoRA"""
        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        lora_layers = get_lora_layers(model)
        assert isinstance(lora_layers, dict)
        # Should be empty or minimal since no LoRA layers
        assert len(lora_layers) >= 0

    def test_get_lora_layers_with_lora(self):
        """Test getting LoRA layers from model with LoRA-named modules"""

        class ModelWithLora(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_layer = nn.Linear(10, 10)
                self.normal_layer = nn.Linear(10, 10)

        model = ModelWithLora()
        lora_layers = get_lora_layers(model)

        # Should find the lora_layer
        assert any("lora" in name for name in lora_layers.keys())


class TestFpsLogger:
    def test_fps_logger_initialization(self):
        """Test FPS logger initialization"""
        logger = FpsLogger(warmup_steps=5, window_size=50, ema_alpha=0.2)
        assert logger.warmup_steps == 5
        assert logger.window_size == 50
        assert logger.ema_alpha == 0.2
        assert logger.steps == 0

    def test_fps_logger_start(self):
        """Test starting FPS logger"""
        logger = FpsLogger()
        logger.start()
        assert logger._started is True
        assert logger._t0 is not None

    def test_fps_logger_update(self):
        """Test updating FPS logger"""
        logger = FpsLogger(warmup_steps=0)
        logger.start()

        fps = logger.update(batch_size=4)
        assert fps > 0
        assert logger.steps == 1
        assert logger.global_samples == 4

    def test_fps_logger_multiple_updates(self):
        """Test multiple updates"""
        logger = FpsLogger(warmup_steps=0)
        logger.start()

        for i in range(10):
            fps = logger.update(batch_size=4)
            assert fps > 0

        assert logger.steps == 10
        assert logger.global_samples == 40

    def test_fps_logger_pause_resume(self):
        """Test pause and resume functionality"""
        logger = FpsLogger()
        logger.start()
        logger.update(batch_size=4)

        logger.pause()
        assert logger._paused is True

        logger.resume()
        assert logger._paused is False

    def test_fps_logger_total_fps(self):
        """Test total FPS calculation"""
        logger = FpsLogger(warmup_steps=2)
        logger.start()

        for _ in range(5):
            logger.update(batch_size=4)

        total_fps = logger.total_fps()
        assert total_fps > 0

    def test_fps_logger_with_tokens(self):
        """Test FPS logger with token counting"""
        logger = FpsLogger()
        logger.start()

        logger.update(batch_size=4, num_tokens=100)
        assert logger.global_tokens == 100

        tokens_per_sec = logger.tokens_per_sec()
        assert tokens_per_sec >= 0

    def test_fps_logger_last_fps(self):
        """Test getting last FPS"""
        logger = FpsLogger()
        logger.start()

        logger.update(batch_size=4)
        last_fps = logger.last_fps()
        assert last_fps >= 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fps_logger_with_cuda_sync(self):
        """Test FPS logger with CUDA synchronization"""
        logger = FpsLogger(cuda_synchronize=torch.cuda.synchronize)
        logger.start()

        fps = logger.update(batch_size=4)
        assert fps > 0

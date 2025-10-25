"""
测试采样工具函数。

本模块演示如何使用 tests/utils/data_loader.py 中的通用数据加载函数。
对于需要从 HuggingFace Hub 加载测试数据的测试，可以使用以下函数：
- load_torch_file: 加载单个 PyTorch 文件
- load_torch_directory: 批量加载目录中的所有文件
- load_flux_training_sample: 加载 Flux 训练样本（便捷函数）
- load_flux_transformer_input: 加载 Flux Transformer 输入（便捷函数）
- load_flux_sampling_embeddings: 加载 Flux 采样嵌入（便捷函数）

示例用法见各测试函数的文档字符串。
"""
import pytest
import torch
from unittest.mock import Mock
from qflux.utils.sampling import calculate_shift, retrieve_timesteps


class TestCalculateShift:
    def test_base_seq_len(self):
        """Test shift calculation at base sequence length"""
        shift = calculate_shift(image_seq_len=256, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
        assert shift == pytest.approx(0.5, abs=1e-6)

    def test_max_seq_len(self):
        """Test shift calculation at max sequence length"""
        shift = calculate_shift(image_seq_len=4096, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15)
        assert shift == pytest.approx(1.15, abs=1e-6)

    def test_mid_seq_len(self):
        """Test shift calculation at mid sequence length"""
        shift = calculate_shift(
            image_seq_len=2176, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15  # (256 + 4096) / 2
        )
        # Should be approximately in the middle
        assert 0.5 < shift < 1.15

    def test_linear_interpolation(self):
        """Test that shift increases linearly with sequence length"""
        shifts = []
        for seq_len in [256, 1000, 2000, 3000, 4096]:
            shift = calculate_shift(seq_len, 256, 4096, 0.5, 1.15)
            shifts.append(shift)

        # Check that shifts are monotonically increasing
        assert all(shifts[i] <= shifts[i + 1] for i in range(len(shifts) - 1))


class TestRetrieveTimesteps:
    def test_with_num_inference_steps(self):
        """Test retrieve timesteps with num_inference_steps"""
        scheduler = Mock()
        scheduler.timesteps = torch.tensor([1000, 900, 800, 700, 600])

        timesteps, num_steps = retrieve_timesteps(scheduler, num_inference_steps=5, device="cpu")

        scheduler.set_timesteps.assert_called_once()
        assert len(timesteps) == 5
        assert num_steps == 5

    def test_with_custom_timesteps(self):
        """Test retrieve timesteps with custom timesteps"""
        scheduler = Mock()
        custom_timesteps = [1000, 800, 600, 400, 200]
        scheduler.timesteps = torch.tensor(custom_timesteps)
        scheduler.set_timesteps = Mock()

        # Add timesteps parameter to signature
        import inspect

        scheduler.set_timesteps.__signature__ = inspect.Signature(
            [
                inspect.Parameter("timesteps", inspect.Parameter.KEYWORD_ONLY),
                inspect.Parameter("device", inspect.Parameter.KEYWORD_ONLY),
            ]
        )

        timesteps, num_steps = retrieve_timesteps(scheduler, timesteps=custom_timesteps, device="cpu")

        assert num_steps == 5

    def test_timesteps_and_sigmas_error(self):
        """Test that passing both timesteps and sigmas raises error"""
        scheduler = Mock()

        with pytest.raises(ValueError, match="Only one of"):
            retrieve_timesteps(scheduler, timesteps=[1000, 800, 600], sigmas=[1.0, 0.5, 0.0])

    def test_scheduler_without_timesteps_support(self):
        """Test error when scheduler doesn't support custom timesteps"""
        scheduler = Mock()
        scheduler.set_timesteps = Mock()

        # Mock signature without timesteps parameter
        import inspect

        scheduler.set_timesteps.__signature__ = inspect.Signature(
            [inspect.Parameter("num_inference_steps", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        )

        with pytest.raises(ValueError, match="does not support custom"):
            retrieve_timesteps(scheduler, timesteps=[1000, 800, 600], device="cpu")

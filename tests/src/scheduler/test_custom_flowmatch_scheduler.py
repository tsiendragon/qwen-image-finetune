"""
Tests for CustomFlowMatchEulerDiscreteScheduler

This module tests:
1. Shift calculation for different image sequence lengths
2. Timestep generation and weighting
3. Scheduler initialization and configuration
4. Error handling for invalid inputs
"""

import pytest
import torch
import numpy as np
from qflux.scheduler.custom_flowmatch_scheduler import (
    CustomFlowMatchEulerDiscreteScheduler,
    calculate_shift,
    scheduler_config
)


class TestCalculateShift:
    """Test shift calculation function"""

    def test_calculate_shift_at_base_seq_len(self):
        """Test shift calculation at base sequence length"""
        base_seq_len = 256
        base_shift = 0.5
        max_seq_len = 4096
        max_shift = 1.16

        result = calculate_shift(
            base_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            base_shift=base_shift,
            max_shift=max_shift
        )

        assert np.isclose(result, base_shift, atol=1e-6), \
            f"Expected {base_shift}, got {result}"

    def test_calculate_shift_at_max_seq_len(self):
        """Test shift calculation at maximum sequence length"""
        base_seq_len = 256
        base_shift = 0.5
        max_seq_len = 4096
        max_shift = 1.16

        result = calculate_shift(
            max_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            base_shift=base_shift,
            max_shift=max_shift
        )

        assert np.isclose(result, max_shift, atol=1e-6), \
            f"Expected {max_shift}, got {result}"

    def test_calculate_shift_linear_interpolation(self):
        """Test that shift scales linearly between base and max"""
        base_seq_len = 256
        base_shift = 0.5
        max_seq_len = 4096
        max_shift = 1.16

        # Test at midpoint
        mid_seq_len = (base_seq_len + max_seq_len) // 2
        result = calculate_shift(
            mid_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            base_shift=base_shift,
            max_shift=max_shift
        )

        # Should be approximately at midpoint of shift range
        expected_mid = (base_shift + max_shift) / 2
        assert abs(result - expected_mid) < 0.1, \
            f"Expected ~{expected_mid}, got {result}"

    @pytest.mark.parametrize("image_seq_len", [
        128,   # Below base
        256,   # At base
        512,   # Between base and max
        1024,  # Between base and max
        2048,  # Between base and max
        4096,  # At max
        8192,  # Above max
    ])
    def test_calculate_shift_various_seq_lens(self, image_seq_len):
        """Test shift calculation for various sequence lengths"""
        base_seq_len = 256
        base_shift = 0.5
        max_seq_len = 4096
        max_shift = 1.16

        result = calculate_shift(
            image_seq_len,
            base_seq_len=base_seq_len,
            max_seq_len=max_seq_len,
            base_shift=base_shift,
            max_shift=max_shift
        )

        # Result should be finite
        assert np.isfinite(result), f"Result should be finite, got {result}"

        # Result should be positive
        assert result > 0, f"Result should be positive, got {result}"


class TestCustomFlowMatchScheduler:
    """Test custom flow match scheduler"""

    @pytest.mark.integration
    def test_initialization(self):
        """Test scheduler initialization with default config"""
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

        assert scheduler.init_noise_sigma == 1.0
        assert scheduler.timestep_type == "linear"
        assert hasattr(scheduler, 'bsmntw_weights')

    @pytest.mark.integration
    def test_initialization_custom_config(self):
        """Test scheduler initialization with custom config"""
        custom_config = scheduler_config.copy()
        custom_config['shift'] = 2.0
        custom_config['num_train_timesteps'] = 500

        scheduler = CustomFlowMatchEulerDiscreteScheduler(**custom_config)

        assert scheduler.config.shift == 2.0
        assert scheduler.config.num_train_timesteps == 500

    @pytest.mark.integration
    def test_timestep_generation(self):
        """Test timestep generation"""
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

        batch_size = 4
        device = torch.device("cpu")

        # Test timestep generation (if method exists)
        if hasattr(scheduler, 'sample_timesteps'):
            timesteps = scheduler.sample_timesteps(batch_size, device)

            assert timesteps.shape == (batch_size,)
            assert timesteps.device == device
            assert torch.all(timesteps >= 0)
            assert torch.all(timesteps < scheduler.config.num_train_timesteps)

    @pytest.mark.integration
    def test_weighting_scheme(self):
        """Test that weighting scheme is properly initialized"""
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

        # Check bsmntw_weights exists and has correct shape
        assert hasattr(scheduler, 'bsmntw_weights')
        weights = scheduler.bsmntw_weights

        assert len(weights) == scheduler_config['num_train_timesteps']
        assert torch.all(torch.isfinite(weights))
        assert torch.all(weights >= 0)

        # Weights should be normalized (mean should be close to 1)
        assert abs(weights.mean().item() - 1.0) < 0.1

    @pytest.mark.integration
    def test_get_weighting(self):
        """Test getting weighting for specific timesteps"""
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

        # Test if get_weighting method exists
        if hasattr(scheduler, 'get_weighting'):
            timesteps = torch.tensor([0, 250, 500, 750, 999])
            weights = scheduler.get_weighting(timesteps)

            assert weights.shape == timesteps.shape
            assert torch.all(torch.isfinite(weights))
            assert torch.all(weights > 0)


class TestSchedulerIntegration:
    """Integration tests with actual usage patterns"""

    @pytest.mark.integration
    def test_scheduler_with_diffusion_pipeline(self):
        """Test scheduler can be used in a typical diffusion setup"""
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

        # Simulate typical usage
        num_inference_steps = 20
        scheduler.set_timesteps(num_inference_steps)

        assert len(scheduler.timesteps) == num_inference_steps
        assert torch.all(scheduler.timesteps[:-1] > scheduler.timesteps[1:]), \
            "Timesteps should be in descending order"

    @pytest.mark.integration
    def test_dynamic_shifting(self):
        """Test dynamic shifting based on image sequence length"""
        config_with_shift = scheduler_config.copy()
        config_with_shift['use_dynamic_shifting'] = True

        scheduler = CustomFlowMatchEulerDiscreteScheduler(**config_with_shift)

        # Verify dynamic shifting is enabled
        assert scheduler.config.use_dynamic_shifting is True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

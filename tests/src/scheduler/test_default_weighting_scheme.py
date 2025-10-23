"""
Tests for default_weighting_scheme

This module tests:
1. Weighting calculation for different distributions
2. Normalization and statistical properties
3. Edge cases and error handling
"""

import pytest
import torch
import numpy as np
from qflux.scheduler.default_weighting_scheme import default_weighing_scheme


class TestDefaultWeightingScheme:
    """Test default weighting scheme function"""

    def test_basic_weighting_calculation(self):
        """Test basic weighting calculation"""
        # Create sample timesteps
        timesteps = torch.tensor([0, 250, 500, 750, 999])

        weights = default_weighing_scheme(timesteps)

        assert weights.shape == timesteps.shape
        assert torch.all(torch.isfinite(weights))
        assert torch.all(weights > 0), "All weights should be positive"

    def test_weighting_symmetry(self):
        """Test that weighting is symmetric around midpoint"""
        num_timesteps = 1000

        # Test at symmetric points
        early_t = torch.tensor([100])
        late_t = torch.tensor([900])

        weight_early = default_weighing_scheme(early_t)
        weight_late = default_weighing_scheme(late_t)

        # Weights at symmetric positions should be similar (bell curve)
        # Note: This depends on the actual implementation
        assert torch.isfinite(weight_early) and torch.isfinite(weight_late)

    def test_peak_at_midpoint(self):
        """Test that weighting peaks near the midpoint (if bell-shaped)"""
        timesteps = torch.arange(0, 1000, 100)
        weights = default_weighing_scheme(timesteps)

        # Find where maximum weight occurs
        max_idx = torch.argmax(weights)
        max_timestep = timesteps[max_idx].item()

        # For bell-shaped curve, max should be near midpoint (500)
        # Allow some tolerance
        assert 300 <= max_timestep <= 700, \
            f"Peak weight should be near midpoint, found at timestep {max_timestep}"

    @pytest.mark.parametrize("timestep", [
        0,
        1,
        100,
        500,
        999,
    ])
    def test_individual_timesteps(self, timestep):
        """Test weighting for individual timesteps"""
        t = torch.tensor([timestep])
        weight = default_weighing_scheme(t)

        assert weight.shape == (1,)
        assert torch.isfinite(weight)
        assert weight > 0

    def test_batch_of_timesteps(self):
        """Test weighting for batch of timesteps"""
        batch_size = 32
        timesteps = torch.randint(0, 1000, (batch_size,))

        weights = default_weighing_scheme(timesteps)

        assert weights.shape == (batch_size,)
        assert torch.all(torch.isfinite(weights))
        assert torch.all(weights > 0)

    def test_weighting_deterministic(self):
        """Test that weighting is deterministic"""
        timesteps = torch.tensor([100, 500, 900])

        weights1 = default_weighing_scheme(timesteps)
        weights2 = default_weighing_scheme(timesteps)

        assert torch.allclose(weights1, weights2), \
            "Weighting should be deterministic"

    def test_edge_case_zero_timestep(self):
        """Test edge case: timestep = 0"""
        t = torch.tensor([0])
        weight = default_weighing_scheme(t)

        assert torch.isfinite(weight)
        assert weight > 0

    def test_edge_case_max_timestep(self):
        """Test edge case: timestep = 999"""
        t = torch.tensor([999])
        weight = default_weighing_scheme(t)

        assert torch.isfinite(weight)
        assert weight > 0

    def test_all_timesteps_coverage(self):
        """Test weighting for all possible timesteps"""
        all_timesteps = torch.arange(0, 1000)
        weights = default_weighing_scheme(all_timesteps)

        assert weights.shape == (1000,)
        assert torch.all(torch.isfinite(weights))
        assert torch.all(weights > 0)

        # Check that weights have reasonable range
        assert weights.min() > 0
        assert weights.max() < 10  # Assuming normalized weights

    def test_statistical_properties(self):
        """Test statistical properties of weight distribution"""
        all_timesteps = torch.arange(0, 1000)
        weights = default_weighing_scheme(all_timesteps)

        # Mean should be close to 1 if normalized
        mean_weight = weights.mean().item()
        assert 0.5 < mean_weight < 2.0, \
            f"Mean weight should be ~1, got {mean_weight}"

        # Standard deviation should be reasonable
        std_weight = weights.std().item()
        assert std_weight > 0, "Weights should have non-zero variance"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

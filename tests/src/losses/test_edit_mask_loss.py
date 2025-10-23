"""
Tests for MaskEditLoss

This module tests:
1. MaskEditLoss with different reduction modes
2. Foreground/background weighting
3. Element-wise weighting support
4. Gradient flow
"""

import pytest
import torch
from qflux.losses import MaskEditLoss


class TestMaskEditLoss:
    """Test suite for MaskEditLoss"""

    def test_basic_forward(self):
        """Test basic forward pass with default reduction"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        mask = (torch.rand(B, seq) > 0.5).float()
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        loss = loss_fn(pred, target, edit_mask=mask)

        assert loss.dim() == 0, "Loss should be scalar with reduction='mean'"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_reduction_none(self):
        """Test reduction='none' returns element-wise losses"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        mask = (torch.rand(B, seq) > 0.5).float()
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        loss_unreduced = loss_fn(pred, target, edit_mask=mask, reduction="none")

        # Should return element-wise losses
        assert loss_unreduced.shape == (B, seq, C), \
            f"Expected shape {(B, seq, C)}, got {loss_unreduced.shape}"

        # Manually verify computation
        element_loss = (pred - target) ** 2
        weight_mask = mask.unsqueeze(-1) * 2.0 + (1 - mask.unsqueeze(-1)) * 1.0
        expected = element_loss * weight_mask

        assert torch.allclose(loss_unreduced, expected, atol=1e-5), \
            "Loss computation doesn't match expected values"

    def test_reduction_sum(self):
        """Test reduction='sum' returns sum of losses"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        mask = (torch.rand(B, seq) > 0.5).float()
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        loss_sum = loss_fn(pred, target, edit_mask=mask, reduction="sum")
        loss_none = loss_fn(pred, target, edit_mask=mask, reduction="none")

        assert loss_sum.dim() == 0, "Sum reduction should return scalar"
        assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-5), \
            "Sum reduction should equal sum of unreduced losses"

    def test_reduction_mean(self):
        """Test reduction='mean' returns mean of losses"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        mask = (torch.rand(B, seq) > 0.5).float()
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        loss_mean = loss_fn(pred, target, edit_mask=mask, reduction="mean")

        assert loss_mean.dim() == 0, "Mean reduction should return scalar"
        assert torch.isfinite(loss_mean), "Mean loss should be finite"

    def test_foreground_background_weighting(self):
        """Test that foreground and background have different weights"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        # All foreground
        mask_fg = torch.ones(B, seq)
        loss_fg = loss_fn(pred, target, edit_mask=mask_fg, reduction="sum")

        # All background
        mask_bg = torch.zeros(B, seq)
        loss_bg = loss_fn(pred, target, edit_mask=mask_bg, reduction="sum")

        # Foreground should have 2x loss
        ratio = loss_fg.item() / loss_bg.item()
        assert 1.5 < ratio < 2.5, \
            f"Foreground/background ratio {ratio:.2f} should be close to 2.0"

    def test_with_weighting(self):
        """Test element-wise weighting parameter"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        mask = torch.ones(B, seq)
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        # Apply 2x weighting to first half
        weighting = torch.ones(B, seq, 1)
        weighting[:, :5] = 2.0

        loss_weighted = loss_fn(pred, target, weighting=weighting, edit_mask=mask, reduction="none")
        loss_unweighted = loss_fn(pred, target, weighting=None, edit_mask=mask, reduction="none")

        # First half should have 2x loss
        assert torch.allclose(
            loss_weighted[:, :5], loss_unweighted[:, :5] * 2.0, atol=1e-5
        ), "Weighted region should have 2x loss"

    def test_gradient_flow(self):
        """Test that gradients flow correctly"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        mask = torch.ones(B, seq)
        pred = torch.randn(B, seq, C, requires_grad=True)
        target = torch.randn(B, seq, C)

        loss = loss_fn(pred, target, edit_mask=mask)
        loss.backward()

        assert pred.grad is not None, "Gradients should exist"
        assert pred.grad.abs().sum() > 0, "Gradients should be non-zero"
        assert pred.grad.shape == pred.shape, "Gradient shape should match input"

    def test_error_invalid_reduction(self):
        """Test error for invalid reduction mode"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        mask = torch.ones(B, seq)
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        with pytest.raises(ValueError, match="Invalid reduction mode"):
            loss_fn(pred, target, edit_mask=mask, reduction="invalid")

    def test_different_weight_values(self):
        """Test with various weight configurations"""
        B, seq, C = 2, 10, 64
        mask = torch.ones(B, seq)
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        # Test different weight combinations
        weight_configs = [
            (1.0, 1.0),  # Equal weights
            (2.0, 1.0),  # 2x foreground
            (3.0, 1.0),  # 3x foreground
            (1.0, 0.5),  # 0.5x background
        ]

        for fg_weight, bg_weight in weight_configs:
            loss_fn = MaskEditLoss(forground_weight=fg_weight, background_weight=bg_weight)
            loss = loss_fn(pred, target, edit_mask=mask)

            assert torch.isfinite(loss), \
                f"Loss should be finite for weights ({fg_weight}, {bg_weight})"
            assert loss.item() >= 0, \
                f"Loss should be non-negative for weights ({fg_weight}, {bg_weight})"

    def test_zero_mask(self):
        """Test with all-zero mask (all background)"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        mask = torch.zeros(B, seq)
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        loss = loss_fn(pred, target, edit_mask=mask)

        assert torch.isfinite(loss), "Loss should be finite with zero mask"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_one_mask(self):
        """Test with all-one mask (all foreground)"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        mask = torch.ones(B, seq)
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        loss = loss_fn(pred, target, edit_mask=mask)

        assert torch.isfinite(loss), "Loss should be finite with one mask"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_none_mask(self):
        """Test with None mask (should default to all foreground)"""
        loss_fn = MaskEditLoss(forground_weight=2.0, background_weight=1.0)

        B, seq, C = 2, 10, 64
        pred = torch.randn(B, seq, C)
        target = torch.randn(B, seq, C)

        # With None mask
        loss_none = loss_fn(pred, target, edit_mask=None)

        # With all-ones mask (should be equivalent)
        mask_ones = torch.ones(B, seq)
        loss_ones = loss_fn(pred, target, edit_mask=mask_ones)

        assert torch.allclose(loss_none, loss_ones, atol=1e-5), \
            "None mask should be equivalent to all-ones mask"


if __name__ == "__main__":
    # Run tests
    test_suite = TestMaskEditLoss()

    print("Running: test_basic_forward...")
    test_suite.test_basic_forward()
    print("✓ Passed\n")

    print("Running: test_reduction_none...")
    test_suite.test_reduction_none()
    print("✓ Passed\n")

    print("Running: test_foreground_background_weighting...")
    test_suite.test_foreground_background_weighting()
    print("✓ Passed\n")

    print("Running: test_with_weighting...")
    test_suite.test_with_weighting()
    print("✓ Passed\n")

    print("Running: test_gradient_flow...")
    test_suite.test_gradient_flow()
    print("✓ Passed\n")

    print("All tests passed! ✓")

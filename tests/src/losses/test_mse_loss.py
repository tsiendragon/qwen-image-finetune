"""
Tests for MseLoss

This module tests:
1. MseLoss with different reduction modes
2. Element-wise weighting support
3. Gradient flow
4. Compatibility with extra kwargs
"""

import pytest
import torch
from qflux.losses import MseLoss


class TestMseLoss:
    """Test suite for MseLoss"""

    def test_basic_forward(self):
        """Test basic forward pass with default reduction"""
        loss_fn = MseLoss(reduction='mean')

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        loss = loss_fn(pred, target)

        assert loss.dim() == 0, "Loss should be scalar with reduction='mean'"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_reduction_none(self):
        """Test reduction='none' returns element-wise losses"""
        loss_fn = MseLoss(reduction='none')

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        loss_unreduced = loss_fn(pred, target)

        # Should return element-wise losses
        assert loss_unreduced.shape == (B, T, C), \
            f"Expected shape {(B, T, C)}, got {loss_unreduced.shape}"

        # Manually verify computation
        expected = (pred - target) ** 2

        assert torch.allclose(loss_unreduced, expected, atol=1e-5), \
            "Loss computation doesn't match expected values"

    def test_reduction_sum(self):
        """Test reduction='sum' returns sum of losses"""
        loss_fn = MseLoss(reduction='sum')

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        loss_sum = loss_fn(pred, target)
        loss_none = MseLoss(reduction='none')(pred, target)

        assert loss_sum.dim() == 0, "Sum reduction should return scalar"
        assert torch.allclose(loss_sum, loss_none.sum(), atol=1e-5), \
            "Sum reduction should equal sum of unreduced losses"

    def test_reduction_mean(self):
        """Test reduction='mean' returns mean of losses"""
        loss_fn = MseLoss(reduction='mean')

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        loss_mean = loss_fn(pred, target)

        assert loss_mean.dim() == 0, "Mean reduction should return scalar"
        assert torch.isfinite(loss_mean), "Mean loss should be finite"

    def test_with_weighting(self):
        """Test element-wise weighting parameter"""
        loss_fn = MseLoss(reduction='none')

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        # Apply 2x weighting to first half
        weighting = torch.ones(B, T, 1)
        weighting[:, :5] = 2.0

        loss_weighted = loss_fn(pred, target, weighting=weighting)
        loss_unweighted = loss_fn(pred, target, weighting=None)

        # First half should have 2x loss
        assert torch.allclose(
            loss_weighted[:, :5], loss_unweighted[:, :5] * 2.0, atol=1e-5
        ), "Weighted region should have 2x loss"

    def test_gradient_flow(self):
        """Test that gradients flow correctly"""
        loss_fn = MseLoss(reduction='mean')

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C, requires_grad=True)
        target = torch.randn(B, T, C)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None, "Gradients should exist"
        assert pred.grad.abs().sum() > 0, "Gradients should be non-zero"
        assert pred.grad.shape == pred.shape, "Gradient shape should match input"

    def test_error_invalid_reduction(self):
        """Test error for invalid reduction mode"""
        with pytest.raises(ValueError, match="Invalid reduction"):
            MseLoss(reduction='invalid')

    def test_error_shape_mismatch(self):
        """Test error for shape mismatch"""
        loss_fn = MseLoss(reduction='mean')

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T + 1, C)  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(pred, target)

    def test_kwargs_compatibility(self):
        """Test that extra kwargs are accepted and ignored"""
        loss_fn = MseLoss(reduction='mean')

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        # Pass extra kwargs that should be ignored
        loss = loss_fn(
            pred,
            target,
            attention_mask=torch.ones(B, T),
            edit_mask=torch.ones(B, T),
            some_random_param="ignored"
        )

        assert torch.isfinite(loss), "Loss should be finite even with extra kwargs"

    def test_extra_repr(self):
        """Test string representation"""
        loss_fn = MseLoss(reduction='mean')
        repr_str = loss_fn.extra_repr()
        assert "reduction='mean'" in repr_str

        loss_fn_sum = MseLoss(reduction='sum')
        repr_str_sum = loss_fn_sum.extra_repr()
        assert "reduction='sum'" in repr_str_sum

    def test_weighting_mean_reduction(self):
        """Test weighting with mean reduction"""
        loss_fn = MseLoss(reduction='mean')

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        # Apply weighting
        weighting = torch.ones(B, T, 1)
        weighting[:, :5] = 2.0

        loss_weighted = loss_fn(pred, target, weighting=weighting)
        loss_unweighted = loss_fn(pred, target)

        # Weighted loss should be different (larger due to 2x on first half)
        assert loss_weighted.item() > loss_unweighted.item(), \
            "Weighted loss should be larger when some weights > 1"

    def test_compatibility_with_torch_mse(self):
        """Test that results match torch.nn.functional.mse_loss for basic case"""
        loss_fn = MseLoss(reduction='mean')

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        our_loss = loss_fn(pred, target)
        torch_loss = torch.nn.functional.mse_loss(pred, target, reduction='mean')

        assert torch.allclose(our_loss, torch_loss, atol=1e-6), \
            "Our MseLoss should match torch.nn.functional.mse_loss for basic case"


if __name__ == "__main__":
    # Run tests
    test_suite = TestMseLoss()

    print("Running: test_basic_forward...")
    test_suite.test_basic_forward()
    print("✓ Passed\n")

    print("Running: test_reduction_none...")
    test_suite.test_reduction_none()
    print("✓ Passed\n")

    print("Running: test_with_weighting...")
    test_suite.test_with_weighting()
    print("✓ Passed\n")

    print("Running: test_gradient_flow...")
    test_suite.test_gradient_flow()
    print("✓ Passed\n")

    print("Running: test_kwargs_compatibility...")
    test_suite.test_kwargs_compatibility()
    print("✓ Passed\n")

    print("All tests passed! ✓")

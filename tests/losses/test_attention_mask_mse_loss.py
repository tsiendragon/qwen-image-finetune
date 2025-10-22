"""
Tests for AttentionMaskMseLoss

This module tests the channel-invariant token loss implementation,
verifying that:
1. Loss scale is independent of channel dimension
2. Token-level averaging works correctly
3. Attention masking filters padding properly
4. Edit masking applies foreground/background weights
5. Gradients flow correctly
"""

import pytest
import torch
from qflux.losses import AttentionMaskMseLoss


class TestAttentionMaskMseLoss:
    """Test suite for AttentionMaskMseLoss"""

    def test_basic_forward(self):
        """Test basic forward pass"""
        loss_fn = AttentionMaskMseLoss()

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        attention_mask = torch.ones(B, T, dtype=torch.bool)

        loss = loss_fn(pred, target, attention_mask=attention_mask)

        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_channel_invariance(self):
        """Test that loss scale is independent of channel dimension"""
        B, T = 2, 10
        attention_mask = torch.ones(B, T, dtype=torch.bool)

        # Test with different channel dimensions
        for C in [16, 32, 64, 128]:
            pred = torch.randn(B, T, C)
            target = torch.randn(B, T, C)

            loss_fn = AttentionMaskMseLoss()
            loss = loss_fn(pred, target, attention_mask=attention_mask)

            # Loss should be similar scale regardless of C
            # (exact value varies due to random data, but scale should be consistent)
            assert 0 < loss.item() < 10, \
                f"Loss {loss.item()} out of expected range for C={C}"

    def test_channel_invariance_same_data(self):
        """Test that doubling channels doubles sum but not mean loss"""
        B, T, C = 2, 10, 32
        attention_mask = torch.ones(B, T, dtype=torch.bool)

        # Create data
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        # Double the channels by concatenating
        pred_2x = torch.cat([pred, pred], dim=2)
        target_2x = torch.cat([target, target], dim=2)

        loss_fn = AttentionMaskMseLoss()
        loss_1x = loss_fn(pred, target, attention_mask=attention_mask)
        loss_2x = loss_fn(pred_2x, target_2x, attention_mask=attention_mask)

        # With channel averaging, doubling channels shouldn't change loss
        assert torch.allclose(loss_1x, loss_2x, atol=1e-5), \
            f"Loss should be channel-invariant: {loss_1x.item()} vs {loss_2x.item()}"

    def test_attention_mask_filtering(self):
        """Test that padding tokens don't contribute to loss"""
        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        # Mask: first sample has 8 valid tokens, second has 6
        attention_mask = torch.zeros(B, T, dtype=torch.bool)
        attention_mask[0, :8] = True
        attention_mask[1, :6] = True

        # Use weight=1.0 to avoid edit weighting (since no edit_mask provided)
        loss_fn = AttentionMaskMseLoss(foreground_weight=1.0, background_weight=1.0)
        loss = loss_fn(pred, target, attention_mask=attention_mask)

        # Manually compute expected loss
        # Token loss for sample 0
        token_loss_0 = ((pred[0, :8] - target[0, :8]) ** 2).mean(dim=1)  # [8]
        # Token loss for sample 1
        token_loss_1 = ((pred[1, :6] - target[1, :6]) ** 2).mean(dim=1)  # [6]

        expected = (token_loss_0.sum() + token_loss_1.sum()) / 14  # 8 + 6 = 14 valid tokens

        assert torch.allclose(loss, expected, atol=1e-5), \
            f"Loss mismatch: got {loss.item()}, expected {expected.item()}"

    def test_edit_mask_weighting(self):
        """Test that edit mask applies different weights to foreground/background"""
        loss_fn = AttentionMaskMseLoss(
            foreground_weight=2.0,
            background_weight=1.0
        )

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        attention_mask = torch.ones(B, T, dtype=torch.bool)

        # All foreground
        edit_mask_fg = torch.ones(B, T)
        loss_fg = loss_fn(pred, target, attention_mask=attention_mask, edit_mask=edit_mask_fg)

        # All background
        edit_mask_bg = torch.zeros(B, T)
        loss_bg = loss_fn(pred, target, attention_mask=attention_mask, edit_mask=edit_mask_bg)

        # Foreground loss should be higher (2x weight vs 1x weight)
        assert loss_fg.item() > loss_bg.item(), \
            "Foreground loss should be higher due to 2x weighting"

        # Ratio should be approximately 2.0
        ratio = loss_fg.item() / loss_bg.item()
        assert 1.5 < ratio < 2.5, \
            f"Weight ratio {ratio:.2f} should be close to 2.0"

    def test_edit_mask_none_defaults_to_foreground(self):
        """Test that edit_mask=None treats all tokens as foreground"""
        loss_fn = AttentionMaskMseLoss(
            foreground_weight=2.0,
            background_weight=1.0
        )

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        attention_mask = torch.ones(B, T, dtype=torch.bool)

        # edit_mask=None
        loss_none = loss_fn(pred, target, attention_mask=attention_mask, edit_mask=None)

        # Explicit all-foreground mask
        edit_mask_fg = torch.ones(B, T)
        loss_fg = loss_fn(pred, target, attention_mask=attention_mask, edit_mask=edit_mask_fg)

        assert torch.allclose(loss_none, loss_fg, atol=1e-5), \
            "edit_mask=None should be equivalent to all-foreground mask"

    def test_weighting_parameter(self):
        """Test optional element-wise weighting parameter"""
        loss_fn = AttentionMaskMseLoss()

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        attention_mask = torch.ones(B, T, dtype=torch.bool)

        # Apply weighting: first half gets 2x weight
        weighting = torch.ones(B, T, 1)
        weighting[:, :5] = 2.0

        loss_weighted = loss_fn(pred, target, weighting=weighting, attention_mask=attention_mask)
        loss_unweighted = loss_fn(pred, target, attention_mask=attention_mask)

        # Weighted loss should be different
        assert not torch.allclose(loss_weighted, loss_unweighted, atol=1e-3), \
            "Weighted and unweighted losses should differ"

    def test_reduction_modes(self):
        """Test different reduction modes"""
        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        attention_mask = torch.ones(B, T, dtype=torch.bool)

        # Test 'mean'
        loss_fn_mean = AttentionMaskMseLoss(reduction='mean')
        loss_mean = loss_fn_mean(pred, target, attention_mask=attention_mask)
        assert loss_mean.dim() == 0, "Mean reduction should return scalar"

        # Test 'sum'
        loss_fn_sum = AttentionMaskMseLoss(reduction='sum')
        loss_sum = loss_fn_sum(pred, target, attention_mask=attention_mask)
        assert loss_sum.dim() == 0, "Sum reduction should return scalar"

        # Test 'none'
        loss_fn_none = AttentionMaskMseLoss(reduction='none')
        loss_none = loss_fn_none(pred, target, attention_mask=attention_mask)
        assert loss_none.shape == (B, T), "None reduction should return per-token losses"

        # Verify relationship: mean * num_tokens ≈ sum
        num_tokens = attention_mask.sum().item()
        assert torch.allclose(loss_mean * num_tokens, loss_sum, atol=1e-5), \
            "mean * num_tokens should equal sum"

        # Verify: sum of per-token losses should equal sum reduction
        assert torch.allclose(loss_none.sum(), loss_sum, atol=1e-5), \
            "Sum of per-token losses should equal sum reduction"

    def test_gradient_flow(self):
        """Test that gradients flow correctly"""
        loss_fn = AttentionMaskMseLoss()

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C, requires_grad=True)
        target = torch.randn(B, T, C)

        attention_mask = torch.zeros(B, T, dtype=torch.bool)
        attention_mask[0, :8] = True
        attention_mask[1, :6] = True

        loss = loss_fn(pred, target, attention_mask=attention_mask)
        loss.backward()

        # Valid tokens should have gradients
        assert pred.grad[0, :8].abs().sum() > 0, "Valid tokens should have gradients"
        assert pred.grad[1, :6].abs().sum() > 0, "Valid tokens should have gradients"

        # Padding tokens should have zero gradients
        assert torch.allclose(pred.grad[0, 8:], torch.zeros_like(pred.grad[0, 8:]), atol=1e-6), \
            "Padded tokens should have zero gradients"
        assert torch.allclose(pred.grad[1, 6:], torch.zeros_like(pred.grad[1, 6:]), atol=1e-6), \
            "Padded tokens should have zero gradients"

    def test_zero_valid_tokens(self):
        """Test behavior when all tokens are padded"""
        loss_fn = AttentionMaskMseLoss()

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        attention_mask = torch.zeros(B, T, dtype=torch.bool)  # All padded

        loss = loss_fn(pred, target, attention_mask=attention_mask)

        assert loss.item() == 0.0, "Loss should be zero when all tokens are padded"

    def test_error_shape_mismatch(self):
        """Test errors for shape mismatches"""
        loss_fn = AttentionMaskMseLoss()

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T + 1, C)  # Wrong shape
        attention_mask = torch.ones(B, T, dtype=torch.bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(pred, target, attention_mask=attention_mask)

    def test_error_attention_mask_shape(self):
        """Test error for wrong attention mask shape"""
        loss_fn = AttentionMaskMseLoss()

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        attention_mask = torch.ones(B, T + 1, dtype=torch.bool)  # Wrong shape

        with pytest.raises(ValueError, match="attention_mask shape"):
            loss_fn(pred, target, attention_mask=attention_mask)

    def test_error_edit_mask_shape(self):
        """Test error for wrong edit mask shape"""
        loss_fn = AttentionMaskMseLoss()

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)
        attention_mask = torch.ones(B, T, dtype=torch.bool)
        edit_mask = torch.ones(B, T + 1)  # Wrong shape

        with pytest.raises(ValueError, match="edit_mask shape"):
            loss_fn(pred, target, attention_mask=attention_mask, edit_mask=edit_mask)

    def test_none_attention_mask(self):
        """Test with None attention_mask (should default to all valid)"""
        loss_fn = AttentionMaskMseLoss(foreground_weight=1.0, background_weight=1.0)

        B, T, C = 2, 10, 64
        pred = torch.randn(B, T, C)
        target = torch.randn(B, T, C)

        # With None attention_mask
        loss_none = loss_fn(pred, target, attention_mask=None)

        # With all-ones attention_mask (should be equivalent)
        attention_mask_ones = torch.ones(B, T, dtype=torch.bool)
        loss_ones = loss_fn(pred, target, attention_mask=attention_mask_ones)

        assert torch.allclose(loss_none, loss_ones, atol=1e-5), \
            "None attention_mask should be equivalent to all-ones mask"

    def test_error_invalid_reduction(self):
        """Test error for invalid reduction mode"""
        with pytest.raises(ValueError, match="Invalid reduction"):
            AttentionMaskMseLoss(reduction='invalid')

    def test_extra_repr(self):
        """Test string representation"""
        loss_fn = AttentionMaskMseLoss(
            foreground_weight=2.0,
            background_weight=1.0,
            eps=1e-12,
            reduction='mean'
        )

        repr_str = loss_fn.extra_repr()
        assert 'foreground_weight=2.0' in repr_str
        assert 'background_weight=1.0' in repr_str
        assert 'eps=1e-12' in repr_str
        assert "reduction='mean'" in repr_str


if __name__ == "__main__":
    # Run tests
    test_suite = TestAttentionMaskMseLoss()

    print("Running: test_basic_forward...")
    test_suite.test_basic_forward()
    print("✓ Passed\n")

    print("Running: test_channel_invariance...")
    test_suite.test_channel_invariance()
    print("✓ Passed\n")

    print("Running: test_channel_invariance_same_data...")
    test_suite.test_channel_invariance_same_data()
    print("✓ Passed\n")

    print("Running: test_attention_mask_filtering...")
    test_suite.test_attention_mask_filtering()
    print("✓ Passed\n")

    print("Running: test_edit_mask_weighting...")
    test_suite.test_edit_mask_weighting()
    print("✓ Passed\n")

    print("Running: test_gradient_flow...")
    test_suite.test_gradient_flow()
    print("✓ Passed\n")

    print("All tests passed! ✓")

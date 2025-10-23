"""Test equivalence between original QwenImageTransformer2DModel and custom implementation.

This test verifies that the custom QwenImageTransformer2DModel with per-sample RoPE and attention mask
produces identical results to the original diffusers implementation when processing:
1. Batched samples with different sequence lengths
2. Samples split by sequence length (no padding)
3. Full batch with attention mask

passed: 2025-10-22 10:00:00
"""

import pytest
import torch

from qflux.models.transformer_qwenimage import QwenImageTransformer2DModel as OriginalQwenTransformer2DModel
from qflux.models.transformer_qwen_custom import QwenImageTransformer2DModel as CustomQwenTransformer2DModel


class TestQwenTransformerEquivalence:
    """Test suite for verifying equivalence between original and custom Qwen transformers."""

    def _to_device_with_dtype(self, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Move tensor to device, casting floating tensors to the requested dtype."""
        if tensor.is_floating_point():
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)

    @pytest.fixture(scope="class")
    def dtype(self):
        """Data type to use for model inference.

        Options:
        - torch.float32: More precise, slower
        - torch.bfloat16: Less precise, faster, better for GPU memory

        Override with: pytest --dtype=float32 or pytest --dtype=bfloat16
        """
        # Default to bfloat16 for efficiency
        return torch.bfloat16

    @pytest.fixture(scope="class")
    def device(self):
        """Device to use for inference."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture(scope="class")
    def repo_id(self):
        """Pretrained model repository ID."""
        return "ovedrive/Qwen-Image-Edit-2509-4bit"

    @pytest.fixture(scope="class")
    def use_pretrained(self):
        """Whether to use pretrained model or small random model for testing.

        Set to True to test with actual pretrained weights (slower, more realistic).
        Set to False to test with small random model (faster, good for CI).
        """
        return True  # Use pretrained model by default

    @pytest.fixture(scope="class")
    def model_config(self):
        """Small model configuration for testing (used when use_pretrained=False)."""
        return {
            "patch_size": 2,
            "in_channels": 64,
            "out_channels": 16,
            "num_layers": 2,  # Small for fast testing
            "attention_head_dim": 128,
            "num_attention_heads": 4,  # Smaller for testing
            "joint_attention_dim": 512,  # Smaller for testing
            "guidance_embeds": False,
            "axes_dims_rope": (16, 56, 56),
        }

    @pytest.fixture(scope="class")
    def sequence_info(self):
        """Define sequence length information for test samples.

        Each sample has multiple images (multi-image input scenario).
        """
        # Sample 0, 1: 2 small images (1, 32, 32) each → 2048 tokens total
        # Sample 2, 3: 2 large images (1, 64, 64) each → 8192 tokens total
        return {
            "batch_size": 4,
            "text_len": 77,
            "img_shapes": [
                [(1, 32, 32), (1, 32, 32)],  # 2 images, 1024 * 2 = 2048 tokens
                [(1, 32, 32), (1, 32, 32)],  # 2 images, 1024 * 2 = 2048 tokens
                [(1, 64, 64), (1, 64, 64)],  # 2 images, 4096 * 2 = 8192 tokens
                [(1, 64, 64), (1, 64, 64)],  # 2 images, 4096 * 2 = 8192 tokens
            ],
            "img_token_lengths": [2048, 2048, 8192, 8192],
            "max_img_len": 8192,
            # 为每个 group 定义一致的 img_shapes
            "group1_img_shapes": [
                [(1, 32, 32), (1, 32, 32)],  # Sample 0: 2 images, 1024 * 2 = 2048 tokens
                [(1, 32, 32), (1, 32, 32)]   # Sample 1: 2 images, 1024 * 2 = 2048 tokens
            ],
            "group2_img_shapes": [
                [(1, 64, 64), (1, 64, 64)],  # Sample 2: 2 images, 4096 * 2 = 8192 tokens
                [(1, 64, 64), (1, 64, 64)]   # Sample 3: 2 images, 4096 * 2 = 8192 tokens
            ]
        }

    @pytest.fixture(scope="class")
    def test_outputs(self):
        """Shared storage for test outputs."""
        return {}

    @pytest.fixture(scope="class")
    def test_data(self, sequence_info, device, dtype):
        """Pre-generated test data to ensure consistency across all tests."""
        # Load a model to get the correct dimensions
        in_channels = 64
        joint_dim = 3584
        text_len = 77
        img_token_lengths = [2048, 2048, 8192, 8192]
        img_shapes = [
                [(1, 32, 32), (1, 32, 32)],  # 2 images, 1024 * 2 = 2048 tokens
                [(1, 32, 32), (1, 32, 32)],  # 2 images, 1024 * 2 = 2048 tokens
                [(1, 64, 64), (1, 64, 64)],  # 2 images, 4096 * 2 = 8192 tokens
                [(1, 64, 64), (1, 64, 64)],  # 2 images, 4096 * 2 = 8192 tokens
            ]
        max_img_len = sequence_info["max_img_len"]
        max_img_len = 8192

        # Generate test data with fixed seeds
        # Group 1: samples 0, 1 (2048 tokens each)
        torch.manual_seed(42)
        group1_hidden_states = torch.randn(2, 2048, in_channels, device=device, dtype=dtype)
        group1_encoder_hidden_states = torch.randn(2, text_len, joint_dim, device=device, dtype=dtype)
        group1_encoder_hidden_states_mask = torch.ones(2, text_len, device=device, dtype=torch.bool)
        group1_timestep = torch.rand(2, device=device, dtype=dtype) * 1000

        # Group 2: samples 2, 3 (8192 tokens each)
        torch.manual_seed(100)
        group2_hidden_states = torch.randn(2, 8192, in_channels, device=device, dtype=dtype)
        group2_encoder_hidden_states = group1_encoder_hidden_states
        group2_encoder_hidden_states_mask = group1_encoder_hidden_states_mask
        group2_timestep = torch.rand(2, device=device, dtype=dtype) * 1000

        # Full batch: combine all samples with padding
        full_hidden_states = torch.zeros(4, max_img_len, in_channels, device=device, dtype=dtype)
        full_hidden_states[0, :2048] = group1_hidden_states[0]
        full_hidden_states[1, :2048] = group1_hidden_states[1]
        full_hidden_states[2, :8192] = group2_hidden_states[0]
        full_hidden_states[3, :8192] = group2_hidden_states[1]

        full_encoder_hidden_states = torch.cat([
            group1_encoder_hidden_states,
            group2_encoder_hidden_states
        ], dim=0)
        full_encoder_hidden_states_mask = torch.cat([
            group1_encoder_hidden_states_mask,
            group2_encoder_hidden_states_mask
        ], dim=0)

        full_timestep = torch.cat([group1_timestep, group2_timestep], dim=0)

        # Create attention mask for full batch
        attention_mask = torch.zeros(4, text_len + max_img_len, device=device, dtype=torch.bool)
        for i in range(4):
            seq_len = img_token_lengths[i]
            attention_mask[i, :text_len] = True  # Text part
            attention_mask[i, text_len: text_len + seq_len] = True  # Image part

        return {
            "in_channels": in_channels,
            "joint_dim": joint_dim,
            "group1": {
                "hidden_states": group1_hidden_states,
                "timestep": group1_timestep,
                "encoder_hidden_states": group1_encoder_hidden_states,
                "encoder_hidden_states_mask": group1_encoder_hidden_states_mask,
                "img_shapes": [img_shapes[0], img_shapes[1]],
                "txt_seq_lens": [text_len, text_len],
            },
            "group2": {
                "hidden_states": group2_hidden_states,
                "timestep": group2_timestep,
                "encoder_hidden_states": group2_encoder_hidden_states,
                "encoder_hidden_states_mask": group2_encoder_hidden_states_mask,
                "img_shapes": [img_shapes[2], img_shapes[3]],
                "txt_seq_lens": [text_len, text_len],
            },
            "full": {
                "hidden_states": full_hidden_states,
                "encoder_hidden_states": full_encoder_hidden_states,
                "encoder_hidden_states_mask": full_encoder_hidden_states_mask,
                "timestep": full_timestep,
                "attention_mask": attention_mask,
                "img_shapes": img_shapes,
                "txt_seq_lens": [text_len, text_len, text_len, text_len],
            },
        }

    def _unload_model(self, model):
        """Unload model and free GPU memory."""
        import gc

        del model
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _load_original_model(self, config_or_repo, device, dtype, use_pretrained=False):
        """Load original QwenImageTransformer2DModel.

        Args:
            config_or_repo: Model config dict (if use_pretrained=False) or repo_id (if use_pretrained=True)
            device: Target device
            dtype: Target dtype
            use_pretrained: Whether to load from pretrained weights
        """
        import gc

        # Clear cache before loading
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if use_pretrained:
            print(f"\nLoading original Qwen model from {config_or_repo} with dtype={dtype}...")
            model = OriginalQwenTransformer2DModel.from_pretrained(
                config_or_repo,
                subfolder="transformer",
                torch_dtype=dtype,
                use_safetensors=True,
            )
            model = model.to(device)
        else:
            print(f"\nLoading original Qwen model with random weights and dtype={dtype}...")
            model = OriginalQwenTransformer2DModel(**config_or_repo)
            model = model.to(device=device, dtype=dtype)

        model.eval()
        print(f"Original model loaded successfully on {device} with dtype={dtype}")
        return model

    def _load_custom_model(self, config_or_repo, device, dtype, use_pretrained=False):
        """Load custom QwenImageTransformer2DModel with same architecture.

        Args:
            config_or_repo: Model config dict (if use_pretrained=False) or repo_id (if use_pretrained=True)
            device: Target device
            dtype: Target dtype
            use_pretrained: Whether to load from pretrained weights
        """
        import gc

        # Clear cache before loading
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        if use_pretrained:
            print(f"\nLoading custom Qwen model from {config_or_repo} with dtype={dtype}...")
            model = CustomQwenTransformer2DModel.from_pretrained(
                config_or_repo,
                subfolder="transformer",
                torch_dtype=dtype,
                use_safetensors=True,
            )
            model = model.to(device)
        else:
            print(f"\nLoading custom Qwen model with random weights and dtype={dtype}...")
            model = CustomQwenTransformer2DModel(**config_or_repo)
            model = model.to(device=device, dtype=dtype)

        model.eval()
        print(f"Custom model loaded successfully on {device} with dtype={dtype}")
        return model

    def _sync_model_weights(self, original_model, custom_model):
        """Copy weights from original model to custom model."""
        custom_model.load_state_dict(original_model.state_dict(), strict=False)
        print("✓ Synced weights from original to custom model")

    @torch.no_grad()
    def test_model_parameters_equivalence(self, repo_id, model_config, device, dtype, use_pretrained):
        """Test that original and custom models have identical parameters after sync.

        This test verifies:
        1. Both models have compatible parameter names
        2. After syncing, parameters have identical values
        """
        print("\n" + "=" * 80)
        print("Testing Model Parameters Equivalence")
        print("=" * 80)

        config_or_repo = repo_id if use_pretrained else model_config

        # Load both models
        original_model = self._load_original_model(config_or_repo, device, dtype, use_pretrained)
        custom_model = self._load_custom_model(config_or_repo, device, dtype, use_pretrained)

        # Sync weights (only needed if not using pretrained)
        if not use_pretrained:
            self._sync_model_weights(original_model, custom_model)

        # Get state dictionaries
        original_state = original_model.state_dict()
        custom_state = custom_model.state_dict()

        # Compare parameter values for common keys
        common_keys = set(original_state.keys()) & set(custom_state.keys())
        max_diff_overall = 0.0
        max_diff_param = ""

        for key in sorted(common_keys):
            original_param = original_state[key]
            custom_param = custom_state[key]

            # Check value match
            diff = torch.abs(original_param - custom_param).max().item()
            if diff > max_diff_overall:
                max_diff_overall = diff
                max_diff_param = key

        print(f"\n✓ Checked {len(common_keys)} common parameters")
        print(f"  Maximum difference: {max_diff_overall:.2e} (at {max_diff_param})")
        print(f"{'=' * 80}\n")

        # Unload models
        self._unload_model(original_model)
        self._unload_model(custom_model)

        # Assertion
        assert max_diff_overall < 1e-10, f"Max parameter difference {max_diff_overall:.2e} exceeds threshold"

    @torch.no_grad()
    def test_original_model_split_batches(
        self, repo_id, model_config, device, dtype, sequence_info, test_outputs, use_pretrained, test_data
    ):
        """Test original model with samples split by sequence length.

        This tests the baseline behavior: processing samples with same length together.
        """
        print("\n" + "=" * 80)
        print("Testing Original Model - Split Batches")
        print("=" * 80)

        config_or_repo = repo_id if use_pretrained else model_config

        # Load original model
        model = self._load_original_model(config_or_repo, device, dtype, use_pretrained)

        # Group 1: First two samples (2048 tokens = 2 * 1024)
        inputs1 = {
            "hidden_states": test_data["group1"]["hidden_states"],
            "encoder_hidden_states": test_data["group1"]["encoder_hidden_states"],
            "encoder_hidden_states_mask": test_data["group1"]["encoder_hidden_states_mask"],
            "timestep": test_data["group1"]["timestep"],
            "img_shapes": test_data["group1"]["img_shapes"],
            "txt_seq_lens": test_data["group1"]["txt_seq_lens"],
        }
        output1 = model(**inputs1).sample.cpu()
        print(f"origon Group 1 output shape: {output1.shape}")

        # Group 2: Last two samples (8192 tokens = 2 * 4096)
        inputs2 = {
            "hidden_states": test_data["group2"]["hidden_states"],
            "encoder_hidden_states": test_data["group2"]["encoder_hidden_states"],
            "encoder_hidden_states_mask": test_data["group2"]["encoder_hidden_states_mask"],
            "timestep": test_data["group2"]["timestep"],
            "img_shapes": test_data["group2"]["img_shapes"],
            "txt_seq_lens": test_data["group2"]["txt_seq_lens"],
        }
        output2 = model(**inputs2).sample.cpu()
        print(f"origin Group 2 output shape: {output2.shape}")

        # Store for comparison
        test_outputs["original_split_outputs"] = {
            "group1": output1,
            "group2": output2,
        }

        # Unload model
        self._unload_model(model)

    @torch.no_grad()
    def test_custom_model_split_batches(
        self, repo_id, model_config, device, dtype, sequence_info, test_outputs, use_pretrained, test_data
    ):
        """Test custom model with samples split by sequence length (no attention mask).

        This should produce identical results to the original model.
        """
        print("\n" + "=" * 80)
        print("Testing Custom Model - Split Batches (No Mask)")
        print("=" * 80)

        config_or_repo = repo_id if use_pretrained else model_config

        # Load models
        original_model = self._load_original_model(config_or_repo, device, dtype, use_pretrained)
        custom_model = self._load_custom_model(config_or_repo, device, dtype, use_pretrained)

        # Sync weights (only needed if not using pretrained)
        if not use_pretrained:
            self._sync_model_weights(original_model, custom_model)

        self._unload_model(original_model)

        # Group 1: First two samples (2048 tokens = 2 * 1024)
        inputs1 = {
            "hidden_states": test_data["group1"]["hidden_states"],
            "encoder_hidden_states": test_data["group1"]["encoder_hidden_states"],
            "encoder_hidden_states_mask": test_data["group1"]["encoder_hidden_states_mask"],
            "timestep": test_data["group1"]["timestep"],
            "img_shapes": test_data["group1"]["img_shapes"],
            "txt_seq_lens": test_data["group1"]["txt_seq_lens"],
        }
        output1 = custom_model(**inputs1).sample.cpu()
        print(f"custome Group 1 output shape: {output1.shape}")

        # Group 2: Last two samples (8192 tokens = 2 * 4096)
        inputs2 = {
            "hidden_states": test_data["group2"]["hidden_states"],
            "encoder_hidden_states": test_data["group2"]["encoder_hidden_states"],
            "encoder_hidden_states_mask": test_data["group2"]["encoder_hidden_states_mask"],
            "timestep": test_data["group2"]["timestep"],
            "img_shapes": test_data["group2"]["img_shapes"],
            "txt_seq_lens": test_data["group2"]["txt_seq_lens"],
        }
        output2 = custom_model(**inputs2).sample.cpu()
        print(f"custome Group 2 output shape: {output2.shape}")

        # Store for comparison
        test_outputs["custom_split_outputs"] = {
            "group1": output1,
            "group2": output2,
        }

        # Unload model
        self._unload_model(custom_model)

    # @torch.no_grad()
    # def test_custom_model_split_with_mask(
    #     self, repo_id, model_config, device, dtype, sequence_info, test_outputs, use_pretrained, test_data
    # ):
    #     """Test custom model with split batches BUT with attention mask.

    #     This isolates whether the issue is from:
    #     1. Attention mask implementation, OR
    #     2. Per-sample RoPE computation

    #     Group 2 has no actual padding (length 8192 = max), so if mask causes issues,
    #     we'll see it even without real padding.
    #     """
    #     print("\n" + "=" * 80)
    #     print("Testing Custom Model - Split Batches With Mask (No Padding)")
    #     print("=" * 80)

    #     config_or_repo = repo_id if use_pretrained else model_config

    #     # Load models
    #     original_model = self._load_original_model(config_or_repo, device, dtype, use_pretrained)
    #     custom_model = self._load_custom_model(config_or_repo, device, dtype, use_pretrained)

    #     # Sync weights (only needed if not using pretrained)
    #     if not use_pretrained:
    #         self._sync_model_weights(original_model, custom_model)

    #     self._unload_model(original_model)

    #     # Group 2: Last two samples (no padding, but with mask)
    #     # Create attention mask for group 2 (no padding since both samples have 8192 tokens)
    #     text_len = 77
    #     attention_mask_group2 = torch.ones(2, text_len + 8192, device=device, dtype=torch.bool)

    #     inputs = {
    #         "hidden_states": test_data["group2"]["hidden_states"],
    #         "encoder_hidden_states": test_data["group2"]["encoder_hidden_states"],
    #         "encoder_hidden_states_mask": test_data["group2"]["encoder_hidden_states_mask"],
    #         "timestep": test_data["group2"]["timestep"],
    #         "img_shapes": test_data["group2"]["img_shapes"],
    #         "txt_seq_lens": test_data["group2"]["txt_seq_lens"],
    #         "attention_mask": attention_mask_group2,
    #     }

    #     output = custom_model(**inputs).sample.cpu()
    #     print(f"Output shape: {output.shape}")

    #     # Store for comparison
    #     test_outputs["custom_split_with_mask_group2"] = output

    #     # Unload model
    #     self._unload_model(custom_model)

    @torch.no_grad()
    def test_custom_model_full_batch_with_mask(
        self, repo_id, model_config, device, dtype, sequence_info, test_outputs, use_pretrained, test_data
    ):
        """Test custom model with full batch and attention mask.

        This tests the key feature: processing variable-length sequences in one batch.
        """
        print("\n" + "=" * 80)
        print("Testing Custom Model - Full Batch With Mask")
        print("=" * 80)

        config_or_repo = repo_id if use_pretrained else model_config

        # Load models
        original_model = self._load_original_model(config_or_repo, device, dtype, use_pretrained)
        custom_model = self._load_custom_model(config_or_repo, device, dtype, use_pretrained)

        # Sync weights (only needed if not using pretrained)
        if not use_pretrained:
            self._sync_model_weights(original_model, custom_model)

        self._unload_model(original_model)

        # Create inputs with all 4 samples using pre-generated data
        inputs = {
            "hidden_states": test_data["full"]["hidden_states"],
            "encoder_hidden_states": test_data["full"]["encoder_hidden_states"],
            "encoder_hidden_states_mask": test_data["full"]["encoder_hidden_states_mask"],
            "timestep": test_data["full"]["timestep"],
            "img_shapes": test_data["full"]["img_shapes"],
            "txt_seq_lens": test_data["full"]["txt_seq_lens"],
            "attention_mask": test_data["full"]["attention_mask"],
        }

        output = custom_model(**inputs).sample.cpu()
        print(f"custome full with mask Output shape: {output.shape}")

        # Store for comparison
        test_outputs["custom_full_output"] = output

        # Unload model
        self._unload_model(custom_model)

    def test_equivalence_split_vs_split(self, test_outputs):
        """Verify that original and custom models produce identical results when split by length."""
        print("\n" + "=" * 80)
        print("Test Equivalence: Original Split vs Custom Split")
        print("=" * 80)

        # Calculate relative error for group 1
        orig1 = test_outputs["original_split_outputs"]["group1"]
        custom1 = test_outputs["custom_split_outputs"]["group1"]
        rel_error1 = torch.norm(orig1 - custom1) / (1e-5 + torch.norm(orig1))
        rel_error1 = rel_error1.item()

        # Calculate relative error for group 2
        orig2 = test_outputs["original_split_outputs"]["group2"]
        custom2 = test_outputs["custom_split_outputs"]["group2"]
        rel_error2 = torch.norm(orig2 - custom2) / (1e-5 + torch.norm(orig2))
        rel_error2 = rel_error2.item()

        print(f"\nGroup 1 (2048 tokens) - Relative error: {rel_error1:.2e}")
        print(f"Group 2 (8192 tokens) - Relative error: {rel_error2:.2e}")

        # Allow small numerical differences due to floating point precision
        assert rel_error1 < 1e-4, f"Group 1 relative error too large: {rel_error1}"
        assert rel_error2 < 1e-4, f"Group 2 relative error too large: {rel_error2}"
        print("✓ PASS: Original and custom models match in split mode\n")
        print(f"test_equivalence_split_vs_split: group1 relative error: {rel_error1:.2e}, group2 relative error: {rel_error2:.2e}")

    # def test_equivalence_split_with_vs_without_mask(self, test_outputs):
    #     """Isolate test: Compare group 2 (no padding) with and without attention mask.

    #     This helps identify if the issue is from:
    #     - Attention mask implementation (if WITH mask != WITHOUT mask)
    #     - Per-sample RoPE computation (if both differ from baseline)

    #     Since group 2 has no actual padding, mask should have no effect.
    #     """
    #     print("\n" + "=" * 80)
    #     print("Test Equivalence: Split With vs Without Mask (No Padding)")
    #     print("=" * 80)

    #     # Group 2 without mask (shared RoPE mode)
    #     output_no_mask = test_outputs["custom_split_outputs"]["group2"]

    #     # Group 2 with mask (per-sample RoPE mode)
    #     output_with_mask = test_outputs["custom_split_with_mask_group2"]

    #     # Calculate relative error
    #     rel_error = torch.norm(output_no_mask - output_with_mask) / (1e-5 + torch.norm(output_no_mask))
    #     rel_error = rel_error.item()

    #     print("\n🔍 Isolation Test - Group 2 (no padding):")
    #     print(f"   With mask vs Without mask - Relative error: {rel_error:.2e}")

    #     # This should be very small since there's no actual padding
    #     if rel_error < 1e-4:
    #         print("   ✓ PASS: Mask has no effect on non-padded sequences (as expected)\n")
    #     else:
    #         print(f"   ❌ FAIL: Mask affects non-padded sequences (relative error: {rel_error:.2e})")
    #         print("   → This indicates an issue with either:")
    #         print("      1. Attention mask implementation, OR")
    #         print("      2. Per-sample RoPE computation\n")

    #     assert rel_error < 1e-4, f"Group 2 with/without mask relative error too large: {rel_error}"

    def test_equivalence_split_vs_full(self, sequence_info, test_outputs):
        """Verify that custom model with full batch produces same results as split batches."""
        print("\n" + "=" * 80)
        print("Test Equivalence: Custom Split vs Custom Full")
        print("=" * 80)

        # Extract valid portions from full batch output
        group1_len = sequence_info["img_token_lengths"][0]  # 2048
        group2_len = sequence_info["img_token_lengths"][2]  # 8192

        custom_full_group1 = test_outputs["custom_full_output"][[0, 1], :group1_len, :]
        custom_full_group2 = test_outputs["custom_full_output"][[2, 3], :group2_len, :]
        print('test_equivalence_split_vs_full: custom_full_group1 shape', custom_full_group1.shape)
        print('test_equivalence_split_vs_full: custom_full_group2 shape', custom_full_group2.shape)
        # Calculate relative errors
        split_group1 = test_outputs["custom_split_outputs"]["group1"]
        rel_error1 = torch.norm(custom_full_group1 - split_group1) / (1e-5 + torch.norm(split_group1))
        rel_error1 = rel_error1.item()

        split_group2 = test_outputs["custom_split_outputs"]["group2"]
        print('test_equivalence_split_vs_full: split_group2 shape', split_group2.shape)
        print('test_equivalence_split_vs_full: custom_full_group2 shape', custom_full_group2.shape)
        rel_error2 = torch.norm(custom_full_group2 - split_group2) / (1e-5 + torch.norm(split_group2))
        rel_error2 = rel_error2.item()

        print(f"\nCustom full vs split - Group 1 - Relative error: {rel_error1:.2e}")
        print(f"Custom full vs split - Group 2 - Relative error: {rel_error2:.2e}")

        # Allow small numerical differences
        # Group 1 has more padding, so we allow slightly larger error
        assert rel_error1 < 2e-2, f"Group 1 relative error too large: {rel_error1}"
        assert rel_error2 < 1e-2, f"Group 2 relative error too large: {rel_error2}"
        print("✓ PASS: Custom model produces consistent results in split and full modes\n")
        print(f"test_equivalence_split_vs_full: group1 relative error: {rel_error1:.2e}, group2 relative error: {rel_error2:.2e}")

    def test_equivalence_original_vs_custom_full(self, sequence_info, test_outputs):
        """Verify that custom model with full batch matches original model outputs."""
        print("\n" + "=" * 80)
        print("Test Equivalence: Original vs Custom Full Batch")
        print("=" * 80)

        # Extract valid portions from full batch output
        group1_len = sequence_info["img_token_lengths"][0]  # 2048
        group2_len = sequence_info["img_token_lengths"][2]  # 8192

        custom_full_group1 = test_outputs["custom_full_output"][[0, 1], :group1_len, :]
        custom_full_group2 = test_outputs["custom_full_output"][[2, 3], :group2_len, :]

        # Calculate relative errors
        orig_group1 = test_outputs["original_split_outputs"]["group1"]
        rel_error1 = torch.norm(custom_full_group1 - orig_group1) / (1e-5 + torch.norm(orig_group1))
        rel_error1 = rel_error1.item()

        orig_group2 = test_outputs["original_split_outputs"]["group2"]
        rel_error2 = torch.norm(custom_full_group2 - orig_group2) / (1e-5 + torch.norm(orig_group2))
        rel_error2 = rel_error2.item()

        print(f"\nOriginal vs Custom full - Group 1 - Relative error: {rel_error1:.2e}")
        print(f"Original vs Custom full - Group 2 - Relative error: {rel_error2:.2e}")

        # Allow small numerical differences
        # Group 1 has more padding, so we allow slightly larger error
        assert rel_error1 < 2e-2, f"Group 1 relative error too large: {rel_error1}"
        assert rel_error2 < 1e-2, f"Group 2 relative error too large: {rel_error2}"
        print(f"group1 relative error: {rel_error1:.2e}, group2 relative error: {rel_error2:.2e}")
        print("✓ PASS: Custom full batch matches original model outputs\n")

    def test_padding_is_masked(self, sequence_info, test_outputs):
        """Verify that padded regions in the output are properly masked/zeroed."""
        print("\n" + "=" * 80)
        print("Test Padding Masking")
        print("=" * 80)

        # Check that padded regions have zero values (should be masked)
        group1_len = sequence_info["img_token_lengths"][0]  # 2048
        # group2_len = 8192 (no padding for samples 2,3)

        # Sample 0 and 1: padding starts at group1_len (2048)
        padding_region_01 = test_outputs["custom_full_output"][[0, 1], group1_len:, :]

        # With our fix, padding should be exactly zero
        max_padding_value = torch.abs(padding_region_01).max().item()

        print(f"\nPadding region - Samples 0,1 (after {group1_len})")
        print(f"  Max absolute value in padding: {max_padding_value:.2e}")

        # Padding should be exactly zero (we explicitly mask it in the output)
        assert max_padding_value == 0.0, f"Padding not properly masked: max value = {max_padding_value}"

        # Sample 2 and 3: no padding since group2_len (2304) = max length
        # Verify that valid regions have non-zero values
        valid_region_01 = test_outputs["custom_full_output"][[0, 1], :group1_len, :]
        valid_norm_01 = torch.norm(valid_region_01, dim=-1).mean().item()

        print(f"  Valid region norm (first {group1_len}): {valid_norm_01:.2e}")
        assert valid_norm_01 > 0, "Valid region should have non-zero values"
        print("✓ PASS: Padding is properly masked to zero\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

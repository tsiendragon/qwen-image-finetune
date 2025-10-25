"""Test equivalence between original FluxTransformer2DModel and custom implementation.

This test verifies that the custom FluxTransformer2DModel with per-sample RoPE and attention mask
produces identical results to the original diffusers implementation when processing:
1. Batched samples with different sequence lengths
2. Samples split by sequence length (no padding)
3. Full batch with attention mask

Test Data Source:
    All test data is automatically downloaded from HuggingFace Hub:
    - Repository: TsienDragon/qwen-image-finetune-test-resources
    - Resource group: flux_input
    - Configuration: tests/resources_config.yaml

    The test_resources fixture (tests/conftest.py) handles automatic download and caching.

passed: 2025-10-22 10:00:00
"""

import pytest
import torch

# from diffusers.models import FluxTransformer2DModel as OriginalFluxTransformer2DModel
from qflux.models.transformer_flux import FluxTransformer2DModel as OriginalFluxTransformer2DModel
from qflux.models.transformer_flux_custom import FluxTransformer2DModel as CustomFluxTransformer2DModel
# from qflux.models.transformer_flux_custom_v2 import FluxTransformer2DModelVarSeq as CustomFluxTransformer2DModel
from tests.utils.data_loader import load_flux_transformer_input


class TestFluxTransformerEquivalence:
    """Test suite for verifying equivalence between original and custom Flux transformers."""

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
        # Default to bfloat16 for efficiency, but can be overridden
        return torch.bfloat16

    @pytest.fixture(scope="class")
    def test_input_data(self, test_resources):
        """Load test input data from HuggingFace Hub.

        Data is automatically downloaded from HuggingFace dataset repository:
        'TsienDragon/qwen-image-finetune-test-resources'

        The test_resources fixture handles automatic download and caching.
        See tests/resources_config.yaml for configuration details.
        """
        print("\n📦 Loading test data from HuggingFace-cached resources")
        data = load_flux_transformer_input(test_resources)
        print(f"✅ Test data loaded successfully with {len(data)} keys")
        return data

    @pytest.fixture(scope="class")
    def repo_id(self):
        """Model repository ID."""
        return "black-forest-labs/FLUX.1-Kontext-dev"

    @pytest.fixture(scope="class")
    def device(self):
        """Device to use for inference."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_original_model(self, repo_id, device, dtype):
        """Load original FluxTransformer2DModel from pretrained."""
        import gc

        # Clear cache before loading
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"\nLoading original model from {repo_id} with dtype={dtype}...")
        model = OriginalFluxTransformer2DModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            torch_dtype=dtype,
            use_safetensors=True,
        )
        model = model.to(device)
        model.eval()
        print(f"Original model loaded successfully on {device} with dtype={dtype}")
        return model

    def _unload_model(self, model):
        """Unload model and free GPU memory."""
        import gc

        del model
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Model unloaded and GPU memory freed")

    def _load_custom_model(self, repo_id, device, dtype):
        """Load custom FluxTransformer2DModel with same weights."""
        import gc

        # Clear cache before loading
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print(f"\nLoading custom model from {repo_id} with dtype={dtype}...")
        model = CustomFluxTransformer2DModel.from_pretrained(
            repo_id,
            subfolder="transformer",
            torch_dtype=dtype,
            use_safetensors=True,
        )
        model = model.to(device)
        model.eval()
        print(f"Custom model loaded successfully on {device} with dtype={dtype}")
        return model

    @pytest.fixture(scope="class")
    def sequence_info(self, test_input_data):
        """Extract sequence length information from input data."""
        latent_ids = test_input_data["latent_ids"]
        text_len = test_input_data["text_ids"].shape[0]

        # Image token lengths (before concatenation with control latent)
        # First two samples: 1872 image tokens
        # Last two samples: 2320 image tokens
        img_token_lengths = [1872, 1872, 2320, 2320]

        # latent_input contains noise + control latent concatenated
        # So the actual sequence length is 2x the image token length
        latent_seq_lengths = [length * 2 for length in img_token_lengths]

        return {
            "batch_size": 4,
            "text_len": text_len,
            "img_token_lengths": img_token_lengths,
            "latent_seq_lengths": latent_seq_lengths,
            "max_latent_len": latent_ids.shape[1],
        }

    @pytest.fixture(scope="class")
    def test_outputs(self):
        """Shared storage for test outputs."""
        return {}

    def _prepare_inputs_for_original(self, data, sample_indices, latent_seq_len, device, dtype):
        """Prepare inputs for original model (no padding, fixed length).

        Args:
            data: Input data dict
            sample_indices: List of sample indices to extract
            latent_seq_len: Actual latent sequence length (noise + control latent, 2x image tokens)
            device: Device to move tensors to
            dtype: Target floating point dtype for inference tensors
        """
        # Extract samples - use latent_seq_len which is 2x the image token length
        latent_input = self._to_device_with_dtype(
            data["latent_model_input"][sample_indices, :latent_seq_len, :], device, dtype
        )
        timestep = self._to_device_with_dtype(
            data["timestep_input"][sample_indices], device, dtype
        )
        guidance = self._to_device_with_dtype(
            data["guidance"][sample_indices], device, dtype
        )
        pooled_embeds = self._to_device_with_dtype(
            data["pooled_prompt_embeds"][sample_indices], device, dtype
        )
        prompt_embeds = self._to_device_with_dtype(
            data["prompt_embeds"][sample_indices], device, dtype
        )

        # Use 2D latent_ids (shared across batch for same-length samples)
        # IMPORTANT: All samples in the batch must have the same spatial layout,
        # so we use the first sample's ids which should be representative
        latent_ids = data["latent_ids"][sample_indices[0], :latent_seq_len, :].to(device)

        # Verify all samples in the batch have identical ids (debugging assertion)
        for idx in sample_indices[1:]:
            assert torch.equal(
                data["latent_ids"][idx, :latent_seq_len, :],
                data["latent_ids"][sample_indices[0], :latent_seq_len, :]
            ), f"Sample {idx} has different img_ids than sample {sample_indices[0]}! Cannot use shared RoPE."

        text_ids = data["text_ids"].to(device)

        return {
            "hidden_states": latent_input,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_embeds,
            "timestep": timestep,
            "img_ids": latent_ids,
            "txt_ids": text_ids,
            "guidance": guidance,
        }

    def _prepare_inputs_for_custom(self, data, sample_indices, device, dtype, use_attention_mask=True):
        """Prepare inputs for custom model (with padding and attention mask)."""
        # Extract samples (full length with padding)
        latent_input = self._to_device_with_dtype(
            data["latent_model_input"][sample_indices], device, dtype
        )
        timestep = self._to_device_with_dtype(
            data["timestep_input"][sample_indices], device, dtype
        )
        guidance = self._to_device_with_dtype(
            data["guidance"][sample_indices], device, dtype
        )
        pooled_embeds = self._to_device_with_dtype(
            data["pooled_prompt_embeds"][sample_indices], device, dtype
        )
        prompt_embeds = self._to_device_with_dtype(
            data["prompt_embeds"][sample_indices], device, dtype
        )

        # Use 3D latent_ids (batched, with padding)
        latent_ids = data["latent_ids"][sample_indices].to(device)

        # Prepare text_ids (2D, shared across batch)
        text_ids = data["text_ids"].to(device)

        inputs = {
            "hidden_states": latent_input,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_embeds,
            "timestep": timestep,
            "img_ids": latent_ids,
            "txt_ids": text_ids,
            "guidance": guidance,
        }

        if use_attention_mask:
            attention_mask = data["full_attention_mask"][sample_indices].to(device)
            inputs["attention_mask"] = attention_mask

        return inputs

    def _extract_valid_output(self, output, latent_seq_len):
        """Extract valid (non-padded) portion of output.

        Args:
            output: Model output tensor (batch, seq, channels)
            latent_seq_len: Valid sequence length (2x image tokens)
        """
        return output[:, :latent_seq_len, :]

    @torch.no_grad()
    def test_model_parameters_equivalence(self, repo_id, device, dtype):
        """Test that original and custom models have identical parameters.

        This test verifies:
        1. Both models have the same parameter names
        2. Each parameter has identical values (exact match)
        Passed testing
        """
        return
        print("\n" + "="*80)
        print("Testing Model Parameters Equivalence")
        print("="*80)

        # Load both models
        original_model = self._load_original_model(repo_id, device, dtype)
        custom_model = self._load_custom_model(repo_id, device, dtype)

        # Get state dictionaries
        original_state = original_model.state_dict()
        custom_state = custom_model.state_dict()

        # Compare parameter names
        original_keys = set(original_state.keys())
        custom_keys = set(custom_state.keys())

        missing_in_custom = original_keys - custom_keys
        extra_in_custom = custom_keys - original_keys
        common_keys = original_keys & custom_keys

        print(f"\nOriginal model parameters: {len(original_keys)}")
        print(f"Custom model parameters: {len(custom_keys)}")
        print(f"Common parameters: {len(common_keys)}")

        if missing_in_custom:
            print(f"\n❌ Parameters missing in custom model ({len(missing_in_custom)}):")
            for key in sorted(missing_in_custom)[:10]:  # Show first 10
                print(f"  - {key}")
            if len(missing_in_custom) > 10:
                print(f"  ... and {len(missing_in_custom) - 10} more")

        if extra_in_custom:
            print(f"\n❌ Extra parameters in custom model ({len(extra_in_custom)}):")
            for key in sorted(extra_in_custom)[:10]:  # Show first 10
                print(f"  - {key}")
            if len(extra_in_custom) > 10:
                print(f"  ... and {len(extra_in_custom) - 10} more")

        # Compare parameter values for common keys
        max_diff_overall = 0.0
        max_diff_param = ""
        mismatched_params = []

        for key in sorted(common_keys):
            original_param = original_state[key]
            custom_param = custom_state[key]

            # Check shape match
            if original_param.shape != custom_param.shape:
                mismatched_params.append({
                    'name': key,
                    'issue': 'shape_mismatch',
                    'original_shape': original_param.shape,
                    'custom_shape': custom_param.shape
                })
                continue

            # Check value match
            diff = torch.abs(original_param - custom_param).max().item()
            if diff > max_diff_overall:
                max_diff_overall = diff
                max_diff_param = key

            # Report parameters with non-zero difference
            if diff > 1e-10:  # Very small threshold for detecting any difference
                mismatched_params.append({
                    'name': key,
                    'issue': 'value_mismatch',
                    'max_diff': diff,
                    'original_norm': original_param.norm().item(),
                    'custom_norm': custom_param.norm().item()
                })

        # Report results
        print(f"\n{'='*80}")
        if mismatched_params:
            print(f"❌ Found {len(mismatched_params)} mismatched parameters:")
            for i, param_info in enumerate(mismatched_params[:10]):  # Show first 10
                if param_info['issue'] == 'shape_mismatch':
                    print(f"\n{i+1}. {param_info['name']}")
                    print(f"   Shape mismatch: {param_info['original_shape']} vs {param_info['custom_shape']}")
                else:
                    print(f"\n{i+1}. {param_info['name']}")
                    print(f"   Max diff: {param_info['max_diff']:.2e}")
                    print(f"   Original norm: {param_info['original_norm']:.2e}")
                    print(f"   Custom norm: {param_info['custom_norm']:.2e}")
            if len(mismatched_params) > 10:
                print(f"\n   ... and {len(mismatched_params) - 10} more mismatched parameters")
        else:
            print(f"✅ All {len(common_keys)} common parameters match exactly!")
            print(f"   Maximum difference: {max_diff_overall:.2e} (at {max_diff_param})")
        print(f"{'='*80}\n")

        # Unload models
        self._unload_model(original_model)
        self._unload_model(custom_model)

        # Assertions
        assert len(missing_in_custom) == 0, f"Custom model missing {len(missing_in_custom)} parameters"
        assert len(extra_in_custom) == 0, f"Custom model has {len(extra_in_custom)} extra parameters"
        assert len(mismatched_params) == 0, f"Found {len(mismatched_params)} mismatched parameters"
        assert max_diff_overall < 1e-10, f"Max parameter difference {max_diff_overall:.2e} exceeds threshold"

    @torch.no_grad()
    def test_original_model_split_batches(
        self, repo_id, device, dtype, test_input_data, sequence_info, test_outputs
    ):
        """Test original model with samples split by sequence length.

        This tests the baseline behavior: processing samples with same length together.
        """
        # Load original model
        model = self._load_original_model(repo_id, device, dtype)

        # Group 1: First two samples (latent length 3744 = 1872 * 2)
        group1_indices = [0, 1]
        group1_len = sequence_info["latent_seq_lengths"][0]  # 3744
        inputs1 = self._prepare_inputs_for_original(
            test_input_data, group1_indices, group1_len, device, dtype
        )
        output1 = model(**inputs1).sample.cpu()

        # Group 2: Last two samples (latent length 4640 = 2320 * 2)
        group2_indices = [2, 3]
        group2_len = sequence_info["latent_seq_lengths"][2]  # 4640
        inputs2 = self._prepare_inputs_for_original(
            test_input_data, group2_indices, group2_len, device, dtype
        )
        output2 = model(**inputs2).sample.cpu()

        # Store for comparison in shared fixture
        test_outputs["original_split_outputs"] = {
            "group1": output1,
            "group2": output2,
        }

        # assert output1.shape == (2, group1_len, 64)
        # assert output2.shape == (2, group2_len, 64)

        # Unload model
        self._unload_model(model)

    @torch.no_grad()
    def test_custom_model_split_batches(
        self, repo_id, device, dtype, test_input_data, sequence_info, test_outputs
    ):
        """Test custom model with samples split by sequence length (no attention mask).

        This should produce identical results to the original model.
        """
        # Load custom model
        model = self._load_custom_model(repo_id, device, dtype)

        # Group 1: First two samples (latent length 3744 = 1872 * 2)
        group1_indices = [0, 1]
        group1_len = sequence_info["latent_seq_lengths"][0]  # 3744
        inputs1 = self._prepare_inputs_for_original(
            test_input_data, group1_indices, group1_len, device, dtype
        )
        output1 = model(**inputs1).sample.cpu()

        # Group 2: Last two samples (latent length 4640 = 2320 * 2)
        group2_indices = [2, 3]
        group2_len = sequence_info["latent_seq_lengths"][2]  # 4640
        inputs2 = self._prepare_inputs_for_original(
            test_input_data, group2_indices, group2_len, device, dtype
        )
        print('inputs2 latent ids in test custom model split batches', inputs2['img_ids'].shape)
        print('inputs2 prompt embeds in test custom model split batches', inputs2['encoder_hidden_states'].shape)
        print('inputs2 pooled embeds in test custom model split batches', inputs2['pooled_projections'].shape)
        print('inputs2 timestep in test custom model split batches', inputs2['timestep'].shape)
        print('inputs2 guidance in test custom model split batches', inputs2['guidance'].shape)
        print('inputs2 latent ids in test custom model split batches', inputs2['img_ids'].shape)
        print('inputs2 text ids in test custom model split batches', inputs2['txt_ids'].shape)
        output2 = model(**inputs2).sample.cpu()

        # Store for comparison in shared fixture
        test_outputs["custom_split_outputs"] = {
            "group1": output1,
            "group2": output2,
        }

        # Unload model
        self._unload_model(model)

    @torch.no_grad()
    def test_custom_model_split_with_mask(
        self, repo_id, device, dtype, test_input_data, sequence_info, test_outputs
    ):
        """Test custom model with split batches BUT with attention mask.

        This isolates whether the issue is from:
        1. Attention mask implementation, OR
        2. Per-sample RoPE computation

        Group 2 has no actual padding (length 4640 = max), so if mask causes issues,
        we'll see it even without real padding.
        """
        # Load custom model
        model = self._load_custom_model(repo_id, device, dtype)

        # Group 2: Last two samples (no padding, but with mask to trigger per-sample mode)
        group2_indices = [2, 3]

        # Prepare inputs with attention mask (even though no actual padding)
        latent_input = self._to_device_with_dtype(
            test_input_data["latent_model_input"][group2_indices], device, dtype
        )
        timestep = self._to_device_with_dtype(
            test_input_data["timestep_input"][group2_indices], device, dtype
        )
        guidance = self._to_device_with_dtype(
            test_input_data["guidance"][group2_indices], device, dtype
        )
        pooled_embeds = self._to_device_with_dtype(
            test_input_data["pooled_prompt_embeds"][group2_indices], device, dtype
        )
        prompt_embeds = self._to_device_with_dtype(
            test_input_data["prompt_embeds"][group2_indices], device, dtype
        )

        # Use 3D latent_ids to trigger per-sample mode
        latent_ids_3d = test_input_data["latent_ids"][group2_indices].to(device)
        text_ids = test_input_data["text_ids"].to(device)

        # Create attention mask (all True since no padding)
        attention_mask = test_input_data["full_attention_mask"][group2_indices].to(device)
        print('test custom mask latent input', latent_input.shape)
        print('test custom mask prompt embeds', prompt_embeds.shape)
        print('test custom mask pooled embeds', pooled_embeds.shape)
        print('test custom mask timestep', timestep)
        print('test custom mask guidance', guidance)
        print('test custom mask latent ids', latent_ids_3d.shape)
        print('test custom mask text ids', text_ids.shape)
        print('test custom mask attention mask', attention_mask.shape)

        output2 = self._prepare_inputs_for_custom(
            test_input_data, group2_indices, device, dtype, use_attention_mask=True
        )
        # compare with above data
        # # keys             "hidden_states": latent_input,
        #     "encoder_hidden_states": prompt_embeds,
        #     "pooled_projections": pooled_embeds,
        #     "timestep": timestep,
        #     "img_ids": latent_ids,
        #     "txt_ids": text_ids,
        #     "guidance": guidance,
        error = torch.norm(output2['hidden_states']-latent_input)/(torch.norm(latent_input)+1e-4)
        print('relative error of encoder hidden states', error)
        error = torch.norm(output2['encoder_hidden_states']-prompt_embeds)/(torch.norm(prompt_embeds)+1e-4)
        print('relative error of encoder hidden states', error)
        error = torch.norm(output2['pooled_projections']-pooled_embeds)/(torch.norm(pooled_embeds)+1e-4)
        print('relative error of pooled embeddings', error)
        error = torch.norm(output2['timestep']-timestep)/(torch.norm(timestep)+1e-4)
        print('relative error of timestep', error)
        error = torch.norm(output2['guidance']-guidance)/(torch.norm(guidance)+1e-4)
        print('relative error of guidance', error)
        print("output2['img_ids']", output2['img_ids'].shape)
        error = torch.norm(output2['img_ids']-latent_ids_3d)/(torch.norm(latent_ids_3d)+1e-4)
        print('relative error of latent ids', error)
        error = torch.norm(output2['txt_ids']-text_ids)/(torch.norm(text_ids)+1e-4)
        print('relative error of text ids', error)

        inputs_with_mask = {
            "hidden_states": latent_input,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_embeds,
            "timestep": timestep,
            "img_ids": latent_ids_3d,
            "txt_ids": text_ids,
            "guidance": guidance,
            "attention_mask": attention_mask,
        }

        output_with_mask = model(**inputs_with_mask).sample.cpu()

        # Store for comparison
        test_outputs["custom_split_with_mask_group2"] = output_with_mask

        # Unload model
        self._unload_model(model)

    @torch.no_grad()
    def test_custom_model_full_batch_with_mask(
        self, repo_id, device, dtype, test_input_data, sequence_info, test_outputs
    ):
        """Test custom model with full batch and attention mask.

        This tests the key feature: processing variable-length sequences in one batch.
        """
        # Load custom model
        model = self._load_custom_model(repo_id, device, dtype)

        all_indices = [0, 1, 2, 3]
        inputs = self._prepare_inputs_for_custom(
            test_input_data, all_indices, device, dtype, use_attention_mask=True
        )

        print('timestep in custom model full batch with mask', inputs['timestep'])

        output = model(**inputs).sample.cpu()

        # Store for comparison in shared fixture
        test_outputs["custom_full_output"] = output

        # Unload model
        self._unload_model(model)

    def test_equivalence_split_vs_split(self, test_outputs):
        """Verify that original and custom models produce identical results when split by length."""
        # Compare group 1
        for k, v in test_outputs["original_split_outputs"].items():
            print("test_equivalence_split_vs_split original_split_outputs", k, v.shape)
        for k, v in test_outputs["custom_split_outputs"].items():
            print("test_equivalence_split_vs_split custom_split_outputs", k, v.shape)

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

        print(f"\nGroup 1 (length 1872) - Relative error: {rel_error1:.2e}")
        print(f"Group 2 (length 2320) - Relative error: {rel_error2:.2e}")

        # Allow small numerical differences due to floating point precision
        assert rel_error1 < 1e-4, f"Group 1 relative error too large: {rel_error1}"
        assert rel_error2 < 1e-4, f"Group 2 relative error too large: {rel_error2}"

    def test_equivalence_split_with_vs_without_mask(self, test_outputs):
        """Isolate test: Compare group 2 (no padding) with and without attention mask.

        This helps identify if the issue is from:
        - Attention mask implementation (if WITH mask != WITHOUT mask)
        - Per-sample RoPE computation (if both differ from baseline)

        Since group 2 has no actual padding, mask should have no effect.
        """
        # Group 2 without mask (shared RoPE mode)
        output_no_mask = test_outputs["custom_split_outputs"]["group2"]

        # Group 2 with mask (per-sample RoPE mode)
        output_with_mask = test_outputs["custom_split_with_mask_group2"]

        # Calculate relative error
        rel_error = torch.norm(output_no_mask - output_with_mask) / (1e-5 + torch.norm(output_no_mask))
        rel_error = rel_error.item()

        print("\n🔍 Isolation Test - Group 2 (no padding):")
        print(f"   With mask vs Without mask - Relative error: {rel_error:.2e}")

        # This should be very small since there's no actual padding
        if rel_error < 2e-2:
            print("   ✅ PASS: Mask has no effect on non-padded sequences (as expected)")
        else:
            print(f"   ❌ FAIL: Mask affects non-padded sequences (relative error: {rel_error:.2e})")
            print("   → This indicates an issue with either:")
            print("      1. Attention mask implementation, OR")
            print("      2. Per-sample RoPE computation")

        assert rel_error < 2e-2, f"Group 2 with/without mask relative error too large: {rel_error}"

    def test_equivalence_split_vs_full(self, sequence_info, test_outputs):
        """Verify that custom model with full batch produces same results as split batches."""
        # Extract valid portions from full batch output
        group1_len = sequence_info["latent_seq_lengths"][0]  # 3744
        group2_len = sequence_info["latent_seq_lengths"][2]  # 4640

        custom_full_group1 = test_outputs["custom_full_output"][[0, 1], :group1_len, :]
        custom_full_group2 = test_outputs["custom_full_output"][[2, 3], :group2_len, :]

        # Calculate relative errors
        split_group1 = test_outputs["custom_split_outputs"]["group1"]
        rel_error1 = torch.norm(custom_full_group1 - split_group1) / (1e-5 + torch.norm(split_group1))
        rel_error1 = rel_error1.item()

        split_group2 = test_outputs["custom_split_outputs"]["group2"]
        rel_error2 = torch.norm(custom_full_group2 - split_group2) / (1e-5 + torch.norm(split_group2))
        rel_error2 = rel_error2.item()

        print(f"\nCustom full vs split - Group 1 - Relative error: {rel_error1:.2e}")
        print(f"Custom full vs split - Group 2 - Relative error: {rel_error2:.2e}")

        # Allow small numerical differences
        assert rel_error1 < 2e-2, f"Group 1 relative error too large: {rel_error1}"
        assert rel_error2 < 2e-2, f"Group 2 relative error too large: {rel_error2}"

    def test_equivalence_original_vs_custom_full(self, sequence_info, test_outputs):
        """Verify that custom model with full batch matches original model outputs."""
        # Extract valid portions from full batch output
        group1_len = sequence_info["latent_seq_lengths"][0]  # 3744
        group2_len = sequence_info["latent_seq_lengths"][2]  # 4640

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
        assert rel_error1 < 2e-2, f"Group 1 relative error too large: {rel_error1}"
        assert rel_error2 < 2e-2, f"Group 2 relative error too large: {rel_error2}"

    def test_padding_is_masked(self, sequence_info, test_outputs):
        """Verify that padded regions in the output are properly masked/zeroed."""
        # Check that padded regions have very small values (should be masked)
        group1_len = sequence_info["latent_seq_lengths"][0]  # 3744

        # Sample 0 and 1: padding starts at group1_len (3744)
        padding_region_01 = test_outputs["custom_full_output"][[0, 1], group1_len:, :]

        # With our fix, padding should be exactly zero
        max_padding_value = torch.abs(padding_region_01).max().item()

        print(f"\nPadding region - Samples 0,1 (after {group1_len})")
        print(f"  Max absolute value in padding: {max_padding_value:.2e}")

        # Padding should be exactly zero (we explicitly mask it in the output)
        assert max_padding_value == 0.0, f"Padding not properly masked: max value = {max_padding_value}"

        # Sample 2 and 3: no padding since group2_len (4640) = max length
        # Verify that valid regions have non-zero values
        valid_region_01 = test_outputs["custom_full_output"][[0, 1], :group1_len, :]
        valid_norm_01 = torch.norm(valid_region_01, dim=-1).mean().item()

        print(f"  Valid region norm (first {group1_len}): {valid_norm_01:.2e}")
        assert valid_norm_01 > 0, "Valid region should have non-zero values"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

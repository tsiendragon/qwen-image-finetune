"""
Test transformer consistency between custom and original implementations.

This module verifies that:
1. transformer_flux_custom.py and transformer_flux.py produce consistent results
2. Under same resolution (shared RoPE mode), predictions should be identical
3. Both models handle the same inputs correctly
"""

import pytest
import torch
import logging
from pathlib import Path
from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml

logger = logging.getLogger(__name__)


def load_sample_data(sample_dir: Path):
    """Load all sample data from a directory

    The sample data contains pre-computed tensors from a training run:
    - image_latents: Target image latents [B, T, C]
    - control_latents: Control image latents [B, T, C]
    - noise: Random noise added during training [B, T, C]
    - model_pred: Model's prediction output [B, T, C]
    - expected_loss: The loss value that should be computed
    - prompt_embeds, pooled_prompt_embeds, text_ids, control_ids: Additional embeddings
    """
    data = {
        "control_ids": torch.load(sample_dir / "sample_control_ids.pt", map_location="cpu", weights_only=True),
        "control_latents": torch.load(sample_dir / "sample_control_latents.pt", map_location="cpu", weights_only=True),
        "image_latents": torch.load(sample_dir / "sample_image_latents.pt", map_location="cpu", weights_only=True),
        "noise": torch.load(sample_dir / "sample_noise.pt", map_location="cpu", weights_only=True),
        "pooled_prompt_embeds": torch.load(
            sample_dir / "sample_pooled_prompt_embeds.pt", map_location="cpu", weights_only=True
        ),
        "prompt_embeds": torch.load(sample_dir / "sample_prompt_embeds.pt", map_location="cpu", weights_only=True),
        "text_ids": torch.load(sample_dir / "sample_text_ids.pt", map_location="cpu", weights_only=True),
        "model_pred": torch.load(sample_dir / "sample_model_pred.pt", map_location="cpu", weights_only=True),
        "expected_loss": torch.load(sample_dir / "sample_loss.pt", map_location="cpu", weights_only=True),
        "latent_model_input": torch.load(
            sample_dir / "sample_latent_model_input.pt", map_location="cpu", weights_only=True
        ),
        "t": torch.load(sample_dir / "sample_t.pt", map_location="cpu", weights_only=True),
        "latent_ids": torch.load(sample_dir / "sample_latent_ids.pt", map_location="cpu", weights_only=True),
        "guidance": torch.load(sample_dir / "sample_guidance.pt", map_location="cpu", weights_only=True),
    }
    return data


@pytest.fixture
def sample_data_1(test_resources):
    """Fixture to load first sample data"""
    return load_sample_data(test_resources / "flux_training" / "face_segmentation" / "sample1")


@pytest.fixture
def sample_data_2(test_resources):
    """Fixture to load second sample data"""
    return load_sample_data(test_resources / "flux_training" / "face_segmentation" / "sample2")


class MockAccelerator:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.is_main_process = True

def assert_tensor_close(x: torch.Tensor, y: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-5, key="tensor"):
    """Assert two tensors are close within tolerance"""
    x = x.float().detach().cpu()
    y = y.float().detach().cpu()

    assert x.shape == y.shape, f"Shape mismatch for {key}: {x.shape} != {y.shape}"

    relative_error = torch.norm(x - y) / (torch.norm(x) + torch.norm(y) + 1e-6)
    max_abs_diff = torch.abs(x - y).max()

    print(f"\n{key} comparison:")
    print(f"  Shape: {x.shape}")
    print(f"  Relative error: {relative_error:.6e}")
    print(f"  Max absolute difference: {max_abs_diff:.6e}")
    print(f"  X range: [{x.min():.6f}, {x.max():.6f}]")
    print(f"  Y range: [{y.min():.6f}, {y.max():.6f}]")

    assert relative_error < rtol or max_abs_diff < atol, (
        f"Tensors not close for {key}:\n"
        f"  Relative error: {relative_error:.6e} (threshold: {rtol})\n"
        f"  Max abs diff: {max_abs_diff:.6e} (threshold: {atol})"
    )


def create_trainer(config_path: str, device: str = "cuda:0"):
    """Helper function to create and setup a trainer"""
    config = load_config_from_yaml(config_path)
    config.model.lora.pretrained_weight = "TsienDragon/flux-kontext-face-segmentation"
    trainer = FluxKontextLoraTrainer(config)
    trainer.load_model()
    FluxKontextLoraTrainer.load_pretrain_lora_model(trainer.dit, config, config.model.lora.adapter_name)
    trainer.accelerator = MockAccelerator(device=device)
    trainer.setup_model_device_train_mode(stage="fit", cache=True)
    trainer.configure_optimizers()
    trainer.setup_criterion()
    trainer.weight_dtype = torch.bfloat16
    logger.info(f"Trainer loaded on {device}")
    return trainer


def cleanup_trainer(trainer):
    """Cleanup trainer and free GPU memory"""
    del trainer.dit
    del trainer
    torch.cuda.empty_cache()


class TestTransformerConsistency:
    """Test suite for transformer consistency between custom and original implementations

    The goal is to verify that both implementations produce the same results
    when using the same inputs and same resolution (shared RoPE mode).

    Strategy: Load two models on different GPUs simultaneously (requires 2 GPUs).
    """

    @pytest.fixture(scope="class")
    def trainer_custom(self):
        """Create trainer with custom transformer on GPU 0"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for this test")

        config_path = "tests/test_configs/test_example_fluxkontext_fp16_faceseg_multiresolution.yaml"
        trainer = create_trainer(config_path, device="cuda:0")
        yield trainer
        # Cleanup after all tests in this class
        del trainer.dit
        del trainer
        torch.cuda.empty_cache()

    @pytest.fixture(scope="class")
    def trainer_original(self):
        """Create trainer with original transformer on GPU 1"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 GPUs for this test")

        config_path = "tests/test_configs/test_example_fluxkontext_fp16.yaml"
        trainer = create_trainer(config_path, device="cuda:1")
        yield trainer
        # Cleanup after all tests in this class
        del trainer.dit
        del trainer
        torch.cuda.empty_cache()

    def test_consistency_sample1(self, trainer_custom, trainer_original, sample_data_1):
        """Test consistency for sample 1 (832x576 resolution)

        This test verifies that custom and original transformers produce
        identical predictions when given the same inputs.

        Models run on different GPUs (cuda:0 and cuda:1) simultaneously.
        """
        data = sample_data_1
        device_custom = "cuda:0"
        device_original = "cuda:1"
        dtype = torch.bfloat16

        print("\nTesting sample 1 consistency:")
        print(f"  Custom model on: {device_custom}")
        print(f"  Original model on: {device_original}")
        print(f"  latent_model_input: {data['latent_model_input'].shape}")
        print(f"  latent_ids: {data['latent_ids'].shape} (ndim={data['latent_ids'].ndim})")
        print(f"  text_ids: {data['text_ids'].shape}")

        # Run both transformers
        with torch.no_grad():
            trainer_custom.dit.eval()
            trainer_original.dit.eval()

            # Custom transformer on GPU 0
            print("\n[1/2] Running custom transformer on cuda:0...")
            latent_model_input_custom = data["latent_model_input"].to(device_custom).to(dtype)
            pooled_prompt_embeds_custom = data["pooled_prompt_embeds"].to(device_custom).to(dtype)
            prompt_embeds_custom = data["prompt_embeds"].to(device_custom).to(dtype)
            text_ids_custom = data["text_ids"].to(device_custom).to(dtype)
            latent_ids_custom = data["latent_ids"].to(device_custom).to(dtype)
            t_custom = data["t"].to(device_custom).to(dtype)
            guidance_custom = data["guidance"].to(device_custom).to(dtype)

            pred_custom = trainer_custom.dit(
                hidden_states=latent_model_input_custom,
                timestep=t_custom,
                guidance=guidance_custom,
                pooled_projections=pooled_prompt_embeds_custom,
                encoder_hidden_states=prompt_embeds_custom,
                txt_ids=text_ids_custom,
                img_ids=latent_ids_custom,  # 2D tensor for shared mode
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            print(f"  Custom prediction shape: {pred_custom.shape}")

            # Original transformer on GPU 1
            print("\n[2/2] Running original transformer on cuda:1...")
            latent_model_input_original = data["latent_model_input"].to(device_original).to(dtype)
            pooled_prompt_embeds_original = data["pooled_prompt_embeds"].to(device_original).to(dtype)
            prompt_embeds_original = data["prompt_embeds"].to(device_original).to(dtype)
            text_ids_original = data["text_ids"].to(device_original).to(dtype)
            latent_ids_original = data["latent_ids"].to(device_original).to(dtype)
            t_original = data["t"].to(device_original).to(dtype)
            guidance_original = data["guidance"].to(device_original).to(dtype)

            pred_original = trainer_original.dit(
                hidden_states=latent_model_input_original,
                timestep=t_original,
                guidance=guidance_original,
                pooled_projections=pooled_prompt_embeds_original,
                encoder_hidden_states=prompt_embeds_original,
                txt_ids=text_ids_original,
                img_ids=latent_ids_original,  # 2D tensor for shared mode
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            print(f"  Original prediction shape: {pred_original.shape}")

        # Assert - Predictions should be identical (after moving to CPU)
        print("\n[3/3] Comparing predictions...")
        assert_tensor_close(pred_custom, pred_original, rtol=1e-4, atol=1e-4, key="model_prediction_sample1")

        print("✓ Sample 1 consistency test passed!")

    def test_consistency_sample2(self, trainer_custom, trainer_original, sample_data_2):
        """Test consistency for sample 2 (different resolution)

        This test verifies that custom and original transformers produce
        identical predictions for a different resolution.

        Models run on different GPUs (cuda:0 and cuda:1) simultaneously.
        """
        data = sample_data_2
        device_custom = "cuda:0"
        device_original = "cuda:1"
        dtype = torch.bfloat16

        print("\nTesting sample 2 consistency:")
        print(f"  Custom model on: {device_custom}")
        print(f"  Original model on: {device_original}")
        print(f"  latent_model_input: {data['latent_model_input'].shape}")
        print(f"  latent_ids: {data['latent_ids'].shape} (ndim={data['latent_ids'].ndim})")
        print(f"  text_ids: {data['text_ids'].shape}")

        # Run both transformers
        with torch.no_grad():
            trainer_custom.dit.eval()
            trainer_original.dit.eval()

            # Custom transformer on GPU 0
            print("\n[1/2] Running custom transformer on cuda:0...")
            latent_model_input_custom = data["latent_model_input"].to(device_custom).to(dtype)
            pooled_prompt_embeds_custom = data["pooled_prompt_embeds"].to(device_custom).to(dtype)
            prompt_embeds_custom = data["prompt_embeds"].to(device_custom).to(dtype)
            text_ids_custom = data["text_ids"].to(device_custom).to(dtype)
            latent_ids_custom = data["latent_ids"].to(device_custom).to(dtype)
            t_custom = data["t"].to(device_custom).to(dtype)
            guidance_custom = data["guidance"].to(device_custom).to(dtype)

            pred_custom = trainer_custom.dit(
                hidden_states=latent_model_input_custom,
                timestep=t_custom,
                guidance=guidance_custom,
                pooled_projections=pooled_prompt_embeds_custom,
                encoder_hidden_states=prompt_embeds_custom,
                txt_ids=text_ids_custom,
                img_ids=latent_ids_custom,  # 2D tensor for shared mode
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            print(f"  Custom prediction shape: {pred_custom.shape}")

            # Original transformer on GPU 1
            print("\n[2/2] Running original transformer on cuda:1...")
            latent_model_input_original = data["latent_model_input"].to(device_original).to(dtype)
            pooled_prompt_embeds_original = data["pooled_prompt_embeds"].to(device_original).to(dtype)
            prompt_embeds_original = data["prompt_embeds"].to(device_original).to(dtype)
            text_ids_original = data["text_ids"].to(device_original).to(dtype)
            latent_ids_original = data["latent_ids"].to(device_original).to(dtype)
            t_original = data["t"].to(device_original).to(dtype)
            guidance_original = data["guidance"].to(device_original).to(dtype)

            pred_original = trainer_original.dit(
                hidden_states=latent_model_input_original,
                timestep=t_original,
                guidance=guidance_original,
                pooled_projections=pooled_prompt_embeds_original,
                encoder_hidden_states=prompt_embeds_original,
                txt_ids=text_ids_original,
                img_ids=latent_ids_original,  # 2D tensor for shared mode
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            print(f"  Original prediction shape: {pred_original.shape}")

        # Assert - Predictions should be identical (after moving to CPU)
        print("\n[3/3] Comparing predictions...")
        assert_tensor_close(pred_custom, pred_original, rtol=1e-4, atol=1e-4, key="model_prediction_sample2")

        print("✓ Sample 2 consistency test passed!")

    def test_model_architecture_consistency(self):
        """Test that model architectures have consistent structure

        Verify that both models have the same number of parameters
        and similar layer structure.
        """
        device = "cuda:0"

        # Load custom model and count parameters
        print("\n[1/2] Loading custom transformer...")
        config_path_custom = "tests/test_configs/test_example_fluxkontext_fp16_faceseg_multiresolution.yaml"
        trainer_custom = create_trainer(config_path_custom, device=device)
        params_custom = sum(p.numel() for p in trainer_custom.dit.parameters())
        cleanup_trainer(trainer_custom)

        # Load original model and count parameters
        print("\n[2/2] Loading original transformer...")
        config_path_original = "tests/test_configs/test_example_fluxkontext_fp16.yaml"
        trainer_original = create_trainer(config_path_original, device=device)
        params_original = sum(p.numel() for p in trainer_original.dit.parameters())
        cleanup_trainer(trainer_original)

        print(f"\nModel architecture comparison:")
        print(f"  Custom model parameters: {params_custom:,}")
        print(f"  Original model parameters: {params_original:,}")

        # The custom model might have slightly different parameters due to
        # additional rope cache, but core transformer should be the same
        # Allow small difference for additional cache structures
        assert (
            abs(params_custom - params_original) / params_original < 0.01
        ), f"Parameter count differs significantly: {params_custom} vs {params_original}"

        print("✓ Architecture consistency test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

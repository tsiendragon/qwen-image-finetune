"""
Tests comparing ZImageT2ITrainer against the official diffusers ZImagePipeline.

Validates that our trainer's implementation exactly reproduces what the
official diffusers pipeline produces for text-to-image generation:

  1. Component weights are identical (VAE, text encoder, transformer)
  2. encode_prompt() produces the same embeddings given identical prompts
  3. End-to-end inference (same fixed initial latents + pre-computed
     embeddings) produces the same output image

Tests are designed to require *no pre-saved reference files* — the
official pipeline itself is the reference.

Usage:
    pytest tests/e2e/test_zimage_t2i_vs_diffusers.py -m e2e -v
"""
import logging

import numpy as np
import pytest
import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

from qflux.data.config import load_config_from_yaml
from qflux.trainer.zimage_t2i_trainer import ZImageT2ITrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A serene mountain landscape with a clear blue lake reflecting snowcapped peaks"
HEIGHT, WIDTH = 1024, 1024
E2E_NUM_STEPS = 4
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_gpu(min_free_gb: float = 10.0) -> int | None:
    if not torch.cuda.is_available():
        return None
    best, best_free = 0, 0.0
    for i in range(torch.cuda.device_count()):
        try:
            free = torch.cuda.mem_get_info(i)[0] / 2**30
            logger.info(f"GPU {i}: {free:.1f} GB free")
            if free > best_free:
                best, best_free = i, free
        except Exception:
            pass
    if best_free < min_free_gb:
        logger.warning(f"Best GPU has only {best_free:.1f} GB free (wanted {min_free_gb})")
    return best


def assert_relative_error(x, y, rtol: float = 1e-4, key: str = "tensor") -> float:
    """Assert relative L2 error is below *rtol* and return it."""
    import PIL.Image

    def _to_tensor(v):
        if isinstance(v, PIL.Image.Image):
            v = np.array(v)
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v.astype(np.float32))
        return v.float().detach().cpu()

    x, y = _to_tensor(x), _to_tensor(y)
    assert x.shape == y.shape, f"Shape mismatch for '{key}': {x.shape} != {y.shape}"
    err = torch.norm(x - y) / (torch.norm(x) + torch.norm(y) + 1e-8)
    assert float(err) < rtol, (
        f"Relative error {err:.6f} ≥ {rtol} for '{key}'\n"
        f"  x: min={x.min():.4f}  max={x.max():.4f}\n"
        f"  y: min={y.min():.4f}  max={y.max():.4f}"
    )
    logger.info(f"[OK] '{key}': relative error = {err:.6f} (< {rtol})")
    return float(err)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def infer_device() -> str:
    gpu_id = _find_free_gpu(min_free_gb=10)
    return f"cuda:{gpu_id}" if gpu_id is not None else "cpu"


@pytest.fixture(scope="module")
def t2i_trainer(infer_device):
    """ZImageT2ITrainer loaded from the same checkpoint (no LoRA)."""
    config = load_config_from_yaml(
        "tests/test_configs/test_example_zimage_t2i_fp16.yaml"
    )
    config.model.pretrained_model_name_or_path = MODEL_ID
    config.model.lora.pretrained_weight = None
    config.predict.devices.vae = infer_device
    config.predict.devices.text_encoder = infer_device
    config.predict.devices.dit = infer_device
    trainer = ZImageT2ITrainer(config)
    trainer.setup_predict()
    return trainer


@pytest.fixture(scope="module")
def diffusers_pipeline(infer_device):
    """Official ZImagePipeline from the same checkpoint."""
    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(infer_device)
    return pipe


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestZImageT2IVsDiffusers:
    """Verify that ZImageT2ITrainer faithfully reproduces ZImagePipeline."""

    # ── Test 1: weights ──────────────────────────────────────────────────

    @pytest.mark.e2e
    def test_component_weights_match_pipeline(self, t2i_trainer, diffusers_pipeline):
        """
        Every parameter in the trainer's VAE, text_encoder, and transformer
        must be numerically identical to the corresponding pipeline component.
        """
        trainer = t2i_trainer
        pipe = diffusers_pipeline

        for m in (pipe.vae, pipe.text_encoder, pipe.transformer,
                  trainer.vae, trainer.text_encoder, trainer.dit):
            m.to("cpu")

        def _compare(pipe_module, trainer_module, label: str):
            total = match = 0
            for (n_p, p_p), (n_t, p_t) in zip(
                pipe_module.named_parameters(),
                trainer_module.named_parameters(),
            ):
                assert n_p == n_t, f"[{label}] name mismatch: {n_p} != {n_t}"
                assert p_p.shape == p_t.shape, (
                    f"[{label}] shape mismatch for '{n_p}': {p_p.shape} != {p_t.shape}"
                )
                if torch.allclose(p_p.float(), p_t.float(), rtol=1e-5, atol=1e-8):
                    match += 1
                total += 1
            logger.info(f"[{label}] {match}/{total} parameters match exactly")
            assert match == total, (
                f"[{label}] {total - match} parameter(s) differ between pipeline and trainer"
            )

        _compare(pipe.vae, trainer.vae, "VAE")
        _compare(pipe.text_encoder, trainer.text_encoder, "text_encoder")
        _compare(pipe.transformer, trainer.dit, "transformer")

    # ── Test 2: prompt embeddings ────────────────────────────────────────

    @pytest.mark.e2e
    def test_prompt_embeddings_match_pipeline(self, t2i_trainer, diffusers_pipeline):
        """
        trainer.encode_prompt() and pipeline._encode_prompt() must produce
        identical variable-length embeddings for the same prompt text.

        Z-Image returns variable-length tensors per sample; we compare the
        unpadded per-sample tensors directly.
        """
        trainer = t2i_trainer
        pipe = diffusers_pipeline
        device = next(trainer.text_encoder.parameters()).device

        with torch.inference_mode():
            # Trainer: returns (padded_embeds, mask); unpad for comparison
            t_embeds_padded, t_mask = trainer.encode_prompt(prompt=[PROMPT])
            t_embeds = t_embeds_padded[0][t_mask[0]]  # (seq_len, hidden)

        with torch.inference_mode():
            # Pipeline: returns list[Tensor] of variable length
            p_embeds_list = pipe._encode_prompt(
                prompt=[PROMPT],
                device=device,
            )
            p_embeds = p_embeds_list[0]  # (seq_len, hidden)

        assert_relative_error(t_embeds, p_embeds, rtol=1e-4, key="prompt_embeds")

    # ── Test 3: end-to-end ───────────────────────────────────────────────

    @pytest.mark.e2e
    def test_end_to_end_output_matches_pipeline(self, t2i_trainer, diffusers_pipeline):
        """
        Using pre-computed trainer embeddings and the same fixed initial
        latents injected into both runs, the final decoded images must be
        nearly identical (rtol ≤ 5%).

        By sharing prompt_embeds and noise latents between the two runs we
        isolate purely the denoising loop and VAE decode.
        """
        trainer = t2i_trainer
        pipe = diffusers_pipeline
        h, w = HEIGHT, WIDTH

        # ── Step A: prepare embeddings via trainer ──────────────────────
        batch = trainer.prepare_predict_batch_data(
            prompt=PROMPT,
            height=h,
            width=w,
            num_inference_steps=E2E_NUM_STEPS,
            guidance_scale=0.0,
        )
        embeddings = trainer.prepare_embeddings(batch, stage="predict")

        # ── Step B: generate fixed initial latents ──────────────────────
        h_lat = 2 * (h // (trainer.vae_scale_factor * 2))
        w_lat = 2 * (w // (trainer.vae_scale_factor * 2))
        gen = torch.Generator(device=trainer.dit.device).manual_seed(SEED)
        noise = torch.randn(
            (1, trainer.num_channels_latents, h_lat, w_lat),
            generator=gen,
            device=trainer.dit.device,
            dtype=torch.float32,
        )

        # ── Step C: trainer denoising ────────────────────────────────────
        embeddings["latents"] = noise.clone()
        trainer_latents = trainer.sampling_from_embeddings(embeddings)
        trainer_img_t = trainer.decode_vae_latent(trainer_latents, h, w)
        trainer_np = (
            trainer_img_t.detach().permute(0, 2, 3, 1).float().cpu().numpy()[0] * 255
        ).round().clip(0, 255).astype(np.uint8)

        # ── Step D: diffusers pipeline denoising ─────────────────────────
        # Provide pre-computed embeddings and the same noise latents
        t_padded = embeddings["prompt_embeds"].to(pipe.transformer.device)
        t_mask = embeddings["prompt_embeds_mask"].to(pipe.transformer.device)
        # Unpad to list[Tensor] as expected by pipeline
        prompt_embeds_list = [t_padded[0][t_mask[0]]]

        pipe_out = pipe(
            prompt_embeds=prompt_embeds_list,
            negative_prompt_embeds=[],
            latents=noise.clone().to(pipe.transformer.device),
            num_inference_steps=E2E_NUM_STEPS,
            height=h,
            width=w,
            guidance_scale=0.0,
            output_type="np",
        )
        pipe_np = (pipe_out.images[0] * 255).round().clip(0, 255).astype(np.uint8)

        assert_relative_error(trainer_np, pipe_np, rtol=0.05, key="output_image")

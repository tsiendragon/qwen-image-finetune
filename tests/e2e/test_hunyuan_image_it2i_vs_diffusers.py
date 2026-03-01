"""
Tests comparing HunyuanImageIT2ITrainer against the official diffusers
HunyuanImageRefinerPipeline.

Validates that our trainer faithfully reproduces the pipeline for image-to-image:

  1. Component weights are identical (VAE, text_encoder, transformer)
  2. encode_prompt() produces matching Qwen embeddings (single encoder, Llama template)
  3. End-to-end inference with shared embeddings + fixed noise + source image matches output

Key architectural differences from T2I:
  - AutoencoderKLHunyuanImageRefiner (3D causal, spatial_compression=16)
  - Single Qwen encoder (no ByT5), Llama-style template, drop_idx=36, max_len=256
  - 5D latents (B, latent_channels, 1, H//16, W//16) with token interleaving
  - Model input = cat([noisy_target, cond_latents], dim=1) — doubled channel concat

Tests use no pre-saved reference files — the official pipeline is the reference.

Usage:
    pytest tests/e2e/test_hunyuan_image_it2i_vs_diffusers.py -m e2e -v
"""
import logging

import numpy as np
import pytest
import torch
from diffusers.pipelines.hunyuan_image.pipeline_hunyuanimage_refiner import (
    HunyuanImageRefinerPipeline,
)

from qflux.data.config import load_config_from_yaml
from qflux.trainer.hunyuan_image_it2i_trainer import HunyuanImageIT2ITrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers"
PROMPT = "A serene mountain landscape with a clear blue lake reflecting snowcapped peaks"
HEIGHT, WIDTH = 1024, 1024
E2E_NUM_STEPS = 4
SEED = 42
DISTILLED_GUIDANCE_SCALE = 3.25
CONDITIONING_STRENGTH = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_gpu(min_free_gb: float = 20.0) -> int | None:
    """Find GPU with at least min_free_gb available."""
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


def _make_random_source_image(height, width, device, dtype, seed=0) -> torch.Tensor:
    """Create a random synthetic source image (B=1, 3, H, W) in [0, 1]."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    img = torch.rand(1, 3, height, width, generator=gen, dtype=dtype)
    return img.to(device)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def infer_device() -> str:
    gpu_id = _find_free_gpu(min_free_gb=20)
    return f"cuda:{gpu_id}" if gpu_id is not None else "cpu"


@pytest.fixture(scope="module")
def it2i_trainer(infer_device):
    """HunyuanImageIT2ITrainer loaded from the same checkpoint (no LoRA)."""
    config = load_config_from_yaml(
        "tests/test_configs/test_example_hunyuan_image_it2i_fp16.yaml"
    )
    config.model.pretrained_model_name_or_path = MODEL_ID
    config.model.lora.pretrained_weight = None
    config.predict.devices.vae = infer_device
    config.predict.devices.text_encoder = infer_device
    config.predict.devices.dit = infer_device
    trainer = HunyuanImageIT2ITrainer(config)
    trainer.setup_predict()
    return trainer


@pytest.fixture(scope="module")
def diffusers_pipeline(infer_device):
    """Official HunyuanImageRefinerPipeline from the same checkpoint."""
    pipe = HunyuanImageRefinerPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(infer_device)
    return pipe


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestHunyuanImageIT2IVsDiffusers:
    """Verify that HunyuanImageIT2ITrainer faithfully reproduces HunyuanImageRefinerPipeline."""

    # ── Test 1: weights ──────────────────────────────────────────────────

    @pytest.mark.e2e
    def test_component_weights_match_pipeline(self, it2i_trainer, diffusers_pipeline):
        """
        Every parameter in the trainer's VAE, text_encoder, and transformer must
        be numerically identical to the corresponding pipeline component.
        """
        trainer = it2i_trainer
        pipe = diffusers_pipeline

        modules_to_compare = [
            (pipe.vae, trainer.vae, "VAE"),
            (pipe.text_encoder, trainer.text_encoder, "text_encoder"),
            (pipe.transformer, trainer.dit, "transformer"),
        ]
        for m_pipe, m_trainer, _ in modules_to_compare:
            m_pipe.to("cpu")
            m_trainer.to("cpu")

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
                f"[{label}] {total - match} parameter(s) differ"
            )

        for pipe_m, trainer_m, label in modules_to_compare:
            _compare(pipe_m, trainer_m, label)

    # ── Test 2: prompt embeddings ────────────────────────────────────────

    @pytest.mark.e2e
    def test_prompt_embeddings_match_pipeline(self, it2i_trainer, diffusers_pipeline):
        """
        trainer.encode_prompt() must produce the same Qwen embeddings
        as pipeline.encode_prompt() for the same prompt.
        """
        trainer = it2i_trainer
        pipe = diffusers_pipeline
        device = next(trainer.text_encoder.parameters()).device

        with torch.inference_mode():
            t_pe, t_pm = trainer.encode_prompt(prompt=[PROMPT])

        with torch.inference_mode():
            p_pe, p_pm = pipe.encode_prompt(
                prompt=PROMPT,
                device=device,
                batch_size=1,
            )

        assert_relative_error(t_pe, p_pe, rtol=1e-4, key="prompt_embeds (Qwen)")
        assert_relative_error(t_pm.float(), p_pm.float(), rtol=1e-5, key="prompt_embeds_mask")

    # ── Test 3: end-to-end ───────────────────────────────────────────────

    @pytest.mark.e2e
    def test_end_to_end_output_matches_pipeline(self, it2i_trainer, diffusers_pipeline):
        """
        Using a shared synthetic source image, pre-computed trainer embeddings, and
        the same fixed initial latents injected into both runs, the final decoded
        images must be nearly identical (rtol ≤ 5%).

        Both runs use the same conditioning latents derived from the source image.
        """
        trainer = it2i_trainer
        pipe = diffusers_pipeline
        h, w = HEIGHT, WIDTH
        device = trainer.dit.device
        dtype = trainer.weight_dtype

        # ── Step A: synthetic source image ──────────────────────────────
        source_img = _make_random_source_image(h, w, device, dtype, seed=0)
        # Normalize to [-1, 1] for pipeline
        source_img_neg1_1 = source_img * 2.0 - 1.0

        # ── Step B: prepare embeddings via trainer ──────────────────────
        batch = trainer.prepare_predict_batch_data(
            prompt=PROMPT,
            image=source_img,
            height=h,
            width=w,
            num_inference_steps=E2E_NUM_STEPS,
            distilled_guidance_scale=DISTILLED_GUIDANCE_SCALE,
            strength=CONDITIONING_STRENGTH,
        )
        embeddings = trainer.prepare_embeddings(batch, stage="predict")

        # ── Step C: shared initial noise latents ────────────────────────
        h_lat = h // trainer.vae_scale_factor
        w_lat = w // trainer.vae_scale_factor
        gen = torch.Generator(device=device).manual_seed(SEED)
        noise = torch.randn(
            (1, trainer.num_channels_latents, 1, h_lat, w_lat),
            generator=gen,
            device=device,
            dtype=torch.float32,
        )

        # Shared conditioning latents (same cond_noise seed for both runs)
        gen_cond = torch.Generator(device=device).manual_seed(SEED + 1)
        cond_noise = torch.randn(
            (1, trainer.num_channels_latents, 1, h_lat, w_lat),
            generator=gen_cond,
            device=device,
            dtype=torch.float32,
        )
        source_latents = embeddings["source_latents"].to(device, dtype=dtype)
        cond_latents_shared = (
            CONDITIONING_STRENGTH * cond_noise.to(dtype)
            + (1.0 - CONDITIONING_STRENGTH) * source_latents
        )

        # ── Step D: trainer denoising ────────────────────────────────────
        embeddings["latents"] = noise.clone()
        # Inject pre-built cond_latents so both runs use identical conditioning
        embeddings["_cond_latents_override"] = cond_latents_shared.clone()
        trainer_latents = trainer.sampling_from_embeddings(embeddings)
        trainer_img = trainer.decode_vae_latent(trainer_latents, h, w)
        trainer_np = (
            trainer_img.detach().permute(0, 2, 3, 1).float().cpu().numpy()[0] * 255
        ).round().clip(0, 255).astype(np.uint8)

        # ── Step E: diffusers pipeline denoising ─────────────────────────
        dit_dev = pipe.transformer.device
        pipe_out = pipe(
            prompt_embeds=embeddings["prompt_embeds"].to(dit_dev),
            prompt_embeds_mask=embeddings["prompt_embeds_mask"].to(dit_dev),
            image=source_img_neg1_1.to(dit_dev),
            latents=noise.clone().to(dit_dev),
            num_inference_steps=E2E_NUM_STEPS,
            height=h,
            width=w,
            distilled_guidance_scale=DISTILLED_GUIDANCE_SCALE,
            output_type="np",
        )
        pipe_np = (pipe_out.images[0] * 255).round().clip(0, 255).astype(np.uint8)

        assert_relative_error(trainer_np, pipe_np, rtol=0.05, key="output_image")

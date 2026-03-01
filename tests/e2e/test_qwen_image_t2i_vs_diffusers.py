"""
Tests comparing QwenImageT2ITrainer against the official diffusers
QwenImagePipeline.

Validates that our trainer's implementation exactly reproduces what the
official diffusers pipeline produces for text-to-image generation:

  1. Component weights are identical (VAE, text encoder, transformer)
  2. encode_prompt() produces the same embeddings given identical prompts
  3. End-to-end inference (same fixed initial latents + pre-computed
     embeddings) produces the same output image

Tests are designed to require *no pre-saved reference files* — the
official pipeline itself is the reference.

Usage:
    pytest tests/e2e/test_qwen_image_t2i_vs_diffusers.py -m e2e -v
"""
import logging

import numpy as np
import pytest
import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

from qflux.data.config import load_config_from_yaml
from qflux.trainer.qwen_image_t2i_trainer import QwenImageT2ITrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen-Image"
PROMPT = "A serene mountain landscape with a clear blue lake reflecting snowcapped peaks"
HEIGHT, WIDTH = 1024, 1024
E2E_NUM_STEPS = 5
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
    """QwenImageT2ITrainer loaded from the same checkpoint (no LoRA)."""
    config = load_config_from_yaml(
        "tests/test_configs/test_example_qwen_image_t2i_fp16.yaml"
    )
    config.model.pretrained_model_name_or_path = MODEL_ID
    config.model.lora.pretrained_weight = None
    config.predict.devices.vae = infer_device
    config.predict.devices.text_encoder = infer_device
    config.predict.devices.dit = infer_device
    trainer = QwenImageT2ITrainer(config)
    trainer.setup_predict()
    return trainer


@pytest.fixture(scope="module")
def diffusers_pipeline(infer_device):
    """Official QwenImagePipeline from the same checkpoint."""
    pipe = QwenImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(infer_device)
    return pipe


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestQwenT2IVsDiffusers:
    """Verify that QwenImageT2ITrainer faithfully reproduces QwenImagePipeline."""

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
        trainer.encode_prompt() and pipeline._get_qwen_prompt_embeds() must
        produce identical (prompt_embeds, prompt_embeds_mask) for the same text.
        No image is involved — this is a pure text-to-image model.
        """
        trainer = t2i_trainer
        pipe = diffusers_pipeline
        device = next(trainer.text_encoder.parameters()).device

        with torch.inference_mode():
            t_embeds, t_mask = trainer.encode_prompt(prompt=[PROMPT])

        with torch.inference_mode():
            p_embeds, p_mask = pipe.encode_prompt(
                prompt=PROMPT,
                device=device,
            )

        assert_relative_error(t_embeds, p_embeds, rtol=1e-4, key="prompt_embeds")
        if p_mask is not None:
            assert_relative_error(t_mask, p_mask.float(), rtol=1e-5, key="prompt_embeds_mask")
        else:
            # Pipeline returns None mask when all tokens are valid
            logger.info("[OK] prompt_embeds_mask: pipeline returned None (all valid), trainer mask is all-ones")
            assert t_mask.all(), "trainer mask should be all-ones when pipeline returns None"

    # ── Test 3: end-to-end ───────────────────────────────────────────────

    @pytest.mark.e2e
    def test_end_to_end_output_matches_pipeline(self, t2i_trainer, diffusers_pipeline):
        """
        Using pre-computed trainer embeddings and the same fixed initial
        latents injected into both the trainer and the pipeline, the final
        decoded images must be nearly identical.

        By sharing prompt_embeds / prompt_embeds_mask and noise latents
        between the two runs we isolate purely the denoising loop and
        VAE decode, both of which must be compatible.
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
            guidance_scale=None,
            true_cfg_scale=1.0,
            negative_prompt="",
            weight_dtype=trainer.weight_dtype,
        )
        embeddings = trainer.prepare_embeddings(batch, stage="predict")

        # ── Step B: generate fixed initial latents ──────────────────────
        n_ch = trainer.dit.config.in_channels // 4
        h_lat = 2 * (h // (trainer.vae_scale_factor * 2))
        w_lat = 2 * (w // (trainer.vae_scale_factor * 2))
        gen = torch.Generator(device=trainer.dit.device).manual_seed(SEED)
        noise_unpacked = torch.randn(
            (1, 1, n_ch, h_lat, w_lat),
            generator=gen,
            device=trainer.dit.device,
            dtype=trainer.weight_dtype,
        )
        noise_packed = trainer._pack_latents(noise_unpacked, 1, n_ch, h_lat, w_lat)

        # ── Step C: trainer denoising ────────────────────────────────────
        embeddings["latents"] = noise_packed
        trainer_latents = trainer.sampling_from_embeddings(embeddings)
        trainer_img_t = trainer.decode_vae_latent(trainer_latents, h, w)
        trainer_np = (
            trainer_img_t.detach().permute(0, 2, 3, 1).float().cpu().numpy()[0] * 255
        ).round().clip(0, 255).astype(np.uint8)

        # ── Step D: diffusers pipeline denoising ─────────────────────────
        pipe_out = pipe(
            prompt_embeds=embeddings["prompt_embeds"].to(pipe.transformer.device),
            prompt_embeds_mask=embeddings["prompt_embeds_mask"].to(pipe.transformer.device),
            latents=noise_packed.to(pipe.transformer.device),
            num_inference_steps=E2E_NUM_STEPS,
            height=h,
            width=w,
            true_cfg_scale=1.0,
            guidance_scale=None,
            output_type="np",
        )
        pipe_np = (pipe_out.images[0] * 255).round().clip(0, 255).astype(np.uint8)

        assert_relative_error(trainer_np, pipe_np, rtol=0.05, key="output_image")

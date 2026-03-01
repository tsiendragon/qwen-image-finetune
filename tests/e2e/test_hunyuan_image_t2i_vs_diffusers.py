"""
Tests comparing HunyuanImageT2ITrainer against the official diffusers
HunyuanImagePipeline.

Validates that our trainer faithfully reproduces the pipeline for text-to-image:

  1. Component weights are identical (VAE, text_encoder, text_encoder_2, transformer)
  2. encode_prompt() produces matching Qwen + ByT5 embeddings
  3. End-to-end inference with shared embeddings + noise latents matches output

HunyuanImage uses dual text encoders:
  - Qwen2.5-VL-7B-Instruct (primary): hidden_states[-3], drop_idx=34
  - ByT5 (glyph/OCR encoder): for quoted text in prompts

Tests use no pre-saved reference files — the official pipeline is the reference.

Usage:
    pytest tests/e2e/test_hunyuan_image_t2i_vs_diffusers.py -m e2e -v
"""
import logging

import numpy as np
import pytest
import torch
from diffusers.pipelines.hunyuan_image.pipeline_hunyuanimage import HunyuanImagePipeline

from qflux.data.config import load_config_from_yaml
from qflux.trainer.hunyuan_image_t2i_trainer import HunyuanImageT2ITrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "tencent/HunyuanImage-3.0-Instruct-DistilA"
PROMPT = "A serene mountain landscape with a clear blue lake reflecting snowcapped peaks"
HEIGHT, WIDTH = 1024, 1024
E2E_NUM_STEPS = 4
SEED = 42
DISTILLED_GUIDANCE_SCALE = 3.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_gpu(min_free_gb: float = 20.0) -> int | None:
    """Find GPU with at least min_free_gb available (HunyuanImage is large)."""
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
    gpu_id = _find_free_gpu(min_free_gb=20)
    return f"cuda:{gpu_id}" if gpu_id is not None else "cpu"


@pytest.fixture(scope="module")
def t2i_trainer(infer_device):
    """HunyuanImageT2ITrainer loaded from the same checkpoint (no LoRA)."""
    config = load_config_from_yaml(
        "tests/test_configs/test_example_hunyuan_image_t2i_fp16.yaml"
    )
    config.model.pretrained_model_name_or_path = MODEL_ID
    config.model.lora.pretrained_weight = None
    config.predict.devices.vae = infer_device
    config.predict.devices.text_encoder = infer_device
    config.predict.devices.text_encoder_2 = infer_device
    config.predict.devices.dit = infer_device
    trainer = HunyuanImageT2ITrainer(config)
    trainer.setup_predict()
    return trainer


@pytest.fixture(scope="module")
def diffusers_pipeline(infer_device):
    """Official HunyuanImagePipeline from the same checkpoint."""
    pipe = HunyuanImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(infer_device)
    return pipe


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestHunyuanImageT2IVsDiffusers:
    """Verify that HunyuanImageT2ITrainer faithfully reproduces HunyuanImagePipeline."""

    # ── Test 1: weights ──────────────────────────────────────────────────

    @pytest.mark.e2e
    def test_component_weights_match_pipeline(self, t2i_trainer, diffusers_pipeline):
        """
        Every parameter in the trainer's VAE, text_encoder, text_encoder_2,
        and transformer must be numerically identical to the corresponding
        pipeline component.
        """
        trainer = t2i_trainer
        pipe = diffusers_pipeline

        modules_to_compare = [
            (pipe.vae, trainer.vae, "VAE"),
            (pipe.text_encoder, trainer.text_encoder, "text_encoder"),
            (pipe.text_encoder_2, trainer.text_encoder_2, "text_encoder_2"),
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
    def test_prompt_embeddings_match_pipeline(self, t2i_trainer, diffusers_pipeline):
        """
        trainer.encode_prompt() must produce the same Qwen + ByT5 embeddings
        as pipeline.encode_prompt() for the same prompt.
        """
        trainer = t2i_trainer
        pipe = diffusers_pipeline
        device = next(trainer.text_encoder.parameters()).device

        with torch.inference_mode():
            t_pe, t_pm, t_pe2, t_pm2 = trainer.encode_prompt(prompt=[PROMPT])

        with torch.inference_mode():
            p_pe, p_pm, p_pe2, p_pm2 = pipe.encode_prompt(
                prompt=PROMPT,
                device=device,
                batch_size=1,
            )

        assert_relative_error(t_pe, p_pe, rtol=1e-4, key="prompt_embeds (Qwen)")
        assert_relative_error(t_pm.float(), p_pm.float(), rtol=1e-5, key="prompt_embeds_mask")
        assert_relative_error(t_pe2, p_pe2, rtol=1e-4, key="prompt_embeds_2 (ByT5)")
        assert_relative_error(t_pm2.float(), p_pm2.float(), rtol=1e-5, key="prompt_embeds_mask_2")

    # ── Test 3: end-to-end ───────────────────────────────────────────────

    @pytest.mark.e2e
    def test_end_to_end_output_matches_pipeline(self, t2i_trainer, diffusers_pipeline):
        """
        Using pre-computed trainer embeddings and the same fixed initial
        latents injected into both runs, the final decoded images must be
        nearly identical (rtol ≤ 5%).

        The guider system in the pipeline is disabled (distilled model returns
        a single unconditional prediction); we skip it and call the transformer
        directly for an apples-to-apples comparison.
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
            distilled_guidance_scale=DISTILLED_GUIDANCE_SCALE,
        )
        embeddings = trainer.prepare_embeddings(batch, stage="predict")

        # ── Step B: fixed noise latents ─────────────────────────────────
        h_lat = h // trainer.vae_scale_factor
        w_lat = w // trainer.vae_scale_factor
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
        trainer_img = trainer.decode_vae_latent(trainer_latents, h, w)
        trainer_np = (
            trainer_img.detach().permute(0, 2, 3, 1).float().cpu().numpy()[0] * 255
        ).round().clip(0, 255).astype(np.uint8)

        # ── Step D: diffusers pipeline denoising ─────────────────────────
        dit_dev = pipe.transformer.device
        pipe_out = pipe(
            prompt_embeds=embeddings["prompt_embeds"].to(dit_dev),
            prompt_embeds_mask=embeddings["prompt_embeds_mask"].to(dit_dev),
            prompt_embeds_2=embeddings["prompt_embeds_2"].to(dit_dev),
            prompt_embeds_mask_2=embeddings["prompt_embeds_mask_2"].to(dit_dev),
            latents=noise.clone().to(dit_dev),
            num_inference_steps=E2E_NUM_STEPS,
            height=h,
            width=w,
            distilled_guidance_scale=DISTILLED_GUIDANCE_SCALE,
            output_type="np",
        )
        pipe_np = (pipe_out.images[0] * 255).round().clip(0, 255).astype(np.uint8)

        assert_relative_error(trainer_np, pipe_np, rtol=0.05, key="output_image")

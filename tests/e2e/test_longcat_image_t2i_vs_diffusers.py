"""
Tests comparing LongCatImageT2ITrainer against the official diffusers LongCatImagePipeline.

Validates that our trainer faithfully reproduces the pipeline for text-to-image:

  1. Component weights are identical (VAE, text_encoder, transformer)
  2. encode_prompt() produces matching embeddings (prefix/suffix template, hidden_states[-1])
  3. End-to-end inference with shared embeddings + fixed noise matches output

Key architectural details:
  - Qwen2_5_VLForConditionalGeneration text encoder (text-only mode)
  - Prefix/suffix template wrapping, hidden_states[-1], content slice extraction
  - split_quotation for char-level tokenization of quoted text
  - Packed 2×2 latents: (B, C, H, W) → (B, N, C*4)
  - img_ids with modality_id=1 and offset=TOKENIZER_MAX_LENGTH=512
  - guidance=None passed to transformer
  - t / 1000 for transformer timestep during inference

Tests use no pre-saved reference files — the official pipeline is the reference.

Usage:
    pytest tests/e2e/test_longcat_image_t2i_vs_diffusers.py -m e2e -v
"""
import logging

import numpy as np
import pytest
import torch
from diffusers.pipelines.longcat_image.pipeline_longcat_image import LongCatImagePipeline

from qflux.data.config import load_config_from_yaml
from qflux.trainer.longcat_image_t2i_trainer import LongCatImageT2ITrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "meituan-longcat/LongCat-Image"
PROMPT = "A serene mountain landscape with a clear blue lake reflecting snowcapped peaks"
HEIGHT, WIDTH = 1024, 1024
E2E_NUM_STEPS = 4
SEED = 42


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def infer_device() -> str:
    gpu_id = _find_free_gpu(min_free_gb=20)
    return f"cuda:{gpu_id}" if gpu_id is not None else "cpu"


@pytest.fixture(scope="module")
def t2i_trainer(infer_device):
    """LongCatImageT2ITrainer loaded from the same checkpoint (no LoRA)."""
    config = load_config_from_yaml(
        "tests/test_configs/test_example_longcat_image_t2i_fp16.yaml"
    )
    config.model.pretrained_model_name_or_path = MODEL_ID
    config.model.lora.pretrained_weight = None
    config.predict.devices.vae = infer_device
    config.predict.devices.text_encoder = infer_device
    config.predict.devices.dit = infer_device
    trainer = LongCatImageT2ITrainer(config)
    trainer.setup_predict()
    return trainer


@pytest.fixture(scope="module")
def diffusers_pipeline(infer_device):
    """Official LongCatImagePipeline from the same checkpoint."""
    pipe = LongCatImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(infer_device)
    return pipe


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestLongCatImageT2IVsDiffusers:
    """Verify that LongCatImageT2ITrainer faithfully reproduces LongCatImagePipeline."""

    # ── Test 1: weights ──────────────────────────────────────────────────

    @pytest.mark.e2e
    def test_component_weights_match_pipeline(self, t2i_trainer, diffusers_pipeline):
        """
        Every parameter in the trainer's VAE, text_encoder, and transformer must
        be numerically identical to the corresponding pipeline component.
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
                f"[{label}] {total - match} parameter(s) differ"
            )

        _compare(pipe.vae, trainer.vae, "VAE")
        _compare(pipe.text_encoder, trainer.text_encoder, "text_encoder")
        _compare(pipe.transformer, trainer.dit, "transformer")

    # ── Test 2: prompt embeddings ────────────────────────────────────────

    @pytest.mark.e2e
    def test_prompt_embeddings_match_pipeline(self, t2i_trainer, diffusers_pipeline):
        """
        trainer.encode_prompt() must produce the same embeddings as the pipeline
        for the same prompt. LongCat uses prefix/suffix template with hidden_states[-1].
        """
        trainer = t2i_trainer
        pipe = diffusers_pipeline
        device = next(trainer.text_encoder.parameters()).device

        with torch.inference_mode():
            t_pe, _ = trainer.encode_prompt(prompt=[PROMPT])

        with torch.inference_mode():
            p_pe, _ = pipe.encode_prompt(
                prompt=PROMPT,
                device=device,
                batch_size=1,
            )

        assert_relative_error(t_pe, p_pe, rtol=1e-4, key="prompt_embeds")

    # ── Test 3: end-to-end ───────────────────────────────────────────────

    @pytest.mark.e2e
    def test_end_to_end_output_matches_pipeline(self, t2i_trainer, diffusers_pipeline):
        """
        Using pre-computed trainer embeddings and the same fixed initial latents
        injected into both runs, the final decoded images must be nearly identical
        (rtol ≤ 5%).
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

        # ── Step B: shared initial noise latents ────────────────────────
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
        trainer_img = trainer.decode_vae_latent(trainer_latents, h, w)
        trainer_np = (
            trainer_img.detach().permute(0, 2, 3, 1).float().cpu().numpy()[0] * 255
        ).round().clip(0, 255).astype(np.uint8)

        # ── Step D: diffusers pipeline denoising ─────────────────────────
        dit_dev = pipe.transformer.device
        pipe_out = pipe(
            prompt_embeds=embeddings["prompt_embeds"].to(dit_dev),
            latents=noise.clone().to(dit_dev),
            num_inference_steps=E2E_NUM_STEPS,
            height=h,
            width=w,
            guidance_scale=0.0,
            output_type="np",
        )
        pipe_np = (pipe_out.images[0] * 255).round().clip(0, 255).astype(np.uint8)

        assert_relative_error(trainer_np, pipe_np, rtol=0.05, key="output_image")

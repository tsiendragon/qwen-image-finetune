"""
Tests comparing HiDreamI1T2ITrainer against the official diffusers HiDreamImagePipeline.

Validates that our trainer faithfully reproduces the pipeline for text-to-image:

  1. Component weights are identical (VAE, all text encoders, transformer)
  2. encode_prompt() produces matching T5 / Llama / pooled CLIP embeddings
  3. End-to-end inference with shared embeddings + fixed noise matches output

Key architectural details:
  - 4 text encoders: CLIP×2 (pooled), T5 (seq), Llama-3.1 (all hidden layers)
  - pooled_prompt_embeds = cat([clip1.text_embeds, clip2.text_embeds], dim=-1)
  - encoder_hidden_states_t5: (B, T5_MAX_LENGTH, hidden)
  - encoder_hidden_states_llama3: (num_layers, B, LLAMA_MAX_LENGTH, hidden) — stacked
  - HiDreamImageTransformer2DModel; 4D latents (B, 64, H//vae_scale, W//vae_scale)
  - Transformer output MUST be negated: noise_pred = -transformer_output
  - AutoencoderKL with shift_factor; decode: (latent/scaling_factor) + shift_factor

Tests use no pre-saved reference files — the official pipeline is the reference.

Usage:
    pytest tests/e2e/test_hidream_i1_t2i_vs_diffusers.py -m e2e -v
"""
import logging

import numpy as np
import pytest
import torch
from diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline

from qflux.data.config import load_config_from_yaml
from qflux.trainer.hidream_i1_t2i_trainer import HiDreamI1T2ITrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "HiDream-ai/HiDream-I1-Full"
PROMPT = "A majestic eagle soaring over snow-capped mountains at golden hour"
HEIGHT, WIDTH = 1024, 1024
E2E_NUM_STEPS = 4
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_gpu(min_free_gb: float = 30.0) -> int | None:
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
    gpu_id = _find_free_gpu(min_free_gb=30)
    return f"cuda:{gpu_id}" if gpu_id is not None else "cpu"


@pytest.fixture(scope="module")
def t2i_trainer(infer_device):
    config = load_config_from_yaml(
        "tests/test_configs/test_example_hidream_i1_t2i_fp16.yaml"
    )
    config.model.pretrained_model_name_or_path = MODEL_ID
    config.model.lora.pretrained_weight = None
    config.predict.devices.vae = infer_device
    config.predict.devices.text_encoder = infer_device
    config.predict.devices.dit = infer_device
    trainer = HiDreamI1T2ITrainer(config)
    trainer.setup_predict()
    return trainer


@pytest.fixture(scope="module")
def diffusers_pipeline(infer_device):
    pipe = HiDreamImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(infer_device)
    return pipe


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestHiDreamI1T2IVsDiffusers:
    """Verify that HiDreamI1T2ITrainer faithfully reproduces HiDreamImagePipeline."""

    # ── Test 1: weights ──────────────────────────────────────────────────

    @pytest.mark.e2e
    def test_component_weights_match_pipeline(self, t2i_trainer, diffusers_pipeline):
        trainer = t2i_trainer
        pipe = diffusers_pipeline

        # Move all to CPU for fair comparison
        for m in (
            pipe.vae, pipe.text_encoder, pipe.text_encoder_2,
            pipe.text_encoder_3, pipe.text_encoder_4, pipe.transformer,
            trainer.vae, trainer.text_encoder, trainer.text_encoder_2,
            trainer.text_encoder_3, trainer.text_encoder_4, trainer.dit,
        ):
            m.to("cpu")

        def _compare(pipe_module, trainer_module, label: str):
            total = match = 0
            for (n_p, p_p), (n_t, p_t) in zip(
                pipe_module.named_parameters(),
                trainer_module.named_parameters(),
            ):
                assert n_p == n_t, f"[{label}] name mismatch: {n_p} != {n_t}"
                assert p_p.shape == p_t.shape
                if torch.allclose(p_p.float(), p_t.float(), rtol=1e-5, atol=1e-8):
                    match += 1
                total += 1
            logger.info(f"[{label}] {match}/{total} parameters match exactly")
            assert match == total, f"[{label}] {total - match} parameter(s) differ"

        _compare(pipe.vae, trainer.vae, "VAE")
        _compare(pipe.text_encoder, trainer.text_encoder, "CLIP-1")
        _compare(pipe.text_encoder_2, trainer.text_encoder_2, "CLIP-2")
        _compare(pipe.text_encoder_3, trainer.text_encoder_3, "T5")
        _compare(pipe.text_encoder_4, trainer.text_encoder_4, "Llama")
        _compare(pipe.transformer, trainer.dit, "transformer")

    # ── Test 2: prompt embeddings ────────────────────────────────────────

    @pytest.mark.e2e
    def test_prompt_embeddings_match_pipeline(self, t2i_trainer, diffusers_pipeline):
        trainer = t2i_trainer
        pipe = diffusers_pipeline
        device = next(trainer.text_encoder.parameters()).device

        with torch.inference_mode():
            t_t5, t_llama, t_pooled = trainer.encode_prompt(prompt=[PROMPT])

        with torch.inference_mode():
            # HiDreamImagePipeline.encode_prompt returns
            # (prompt_embeds_t5, prompt_embeds_llama3, pooled_prompt_embeds, ...)
            p_out = pipe.encode_prompt(
                prompt=PROMPT,
                prompt_2=PROMPT,
                prompt_3=PROMPT,
                prompt_4=PROMPT,
                device=device,
            )
            p_t5, p_llama, p_pooled = p_out[0], p_out[1], p_out[2]

        assert_relative_error(t_t5, p_t5, rtol=1e-4, key="encoder_hidden_states_t5")
        # Compare first and last Llama layer
        assert_relative_error(t_llama[0], p_llama[0], rtol=1e-4, key="llama_layer[0]")
        assert_relative_error(t_llama[-1], p_llama[-1], rtol=1e-4, key="llama_layer[-1]")
        assert_relative_error(t_pooled, p_pooled, rtol=1e-4, key="pooled_prompt_embeds")

    # ── Test 3: end-to-end ───────────────────────────────────────────────

    @pytest.mark.e2e
    def test_end_to_end_output_matches_pipeline(self, t2i_trainer, diffusers_pipeline):
        trainer = t2i_trainer
        pipe = diffusers_pipeline
        h, w = HEIGHT, WIDTH

        batch = trainer.prepare_predict_batch_data(
            prompt=PROMPT, height=h, width=w,
            num_inference_steps=E2E_NUM_STEPS, guidance_scale=1.0,  # no CFG for exact match
        )
        embeddings = trainer.prepare_embeddings(batch, stage="predict")

        h_lat = h // trainer.vae_scale_factor
        w_lat = w // trainer.vae_scale_factor
        gen = torch.Generator(device=trainer.dit.device).manual_seed(SEED)
        noise = torch.randn(
            (1, trainer.num_channels_latents, h_lat, w_lat),
            generator=gen, device=trainer.dit.device, dtype=torch.float32,
        )

        embeddings["latents"] = noise.clone()
        trainer_img = trainer.sampling_from_embeddings(embeddings)
        trainer_np = (
            trainer_img.detach().permute(0, 2, 3, 1).float().cpu().numpy()[0] * 255
        ).round().clip(0, 255).astype(np.uint8)

        dit_dev = pipe.transformer.device
        pipe_out = pipe(
            prompt_embeds=embeddings["prompt_embeds_t5"].to(dit_dev),
            prompt_embeds_2=embeddings["prompt_embeds_llama3"].to(dit_dev),
            pooled_prompt_embeds=embeddings["pooled_prompt_embeds"].to(dit_dev),
            latents=noise.clone().to(dit_dev),
            num_inference_steps=E2E_NUM_STEPS,
            height=h, width=w,
            guidance_scale=1.0,
            output_type="np",
        )
        pipe_np = (pipe_out.images[0] * 255).round().clip(0, 255).astype(np.uint8)

        assert_relative_error(trainer_np, pipe_np, rtol=0.05, key="output_image")

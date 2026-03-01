"""
Tests comparing BriaFiboT2ITrainer against the official diffusers BriaFiboPipeline.

Validates that our trainer faithfully reproduces the pipeline for text-to-image:

  1. Component weights are identical (VAE, text_encoder, transformer)
  2. encode_prompt() produces matching SmolLM3 embeddings (last 2 layers cat + all layers)
  3. End-to-end inference with shared embeddings + fixed noise matches output

Key architectural details:
  - SmolLM3ForCausalLM + AutoTokenizer, max_length=256
  - encoder_hidden_states: cat(hidden_states[-1], hidden_states[-2], dim=-1) → (B, seq, 4096)
  - text_encoder_layers: list of all (B, seq, 2048) per-layer hidden states
  - attention_mask: (B, seq) → (B, 1, seq, seq) matrix via einsum
  - AutoencoderKLWan: 5D (B, C, 1, H, W); vae_scale_factor=16;
    normalize: (latent - mean) * std; encode returns .latent_dist.mean
  - 4D latents (B, C, H, W), packed to (B, H*W, C) — no 2×2 packing
  - img_ids: (H*W, 3) with [0, row, col]; txt_ids: (B, seq, 3) zeros
  - FlowMatchEulerDiscreteScheduler with dynamic shift

Tests use no pre-saved reference files — the official pipeline is the reference.

Usage:
    pytest tests/e2e/test_bria_fibo_t2i_vs_diffusers.py -m e2e -v
"""
import logging

import numpy as np
import pytest
import torch
from diffusers.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline

from qflux.data.config import load_config_from_yaml
from qflux.trainer.bria_fibo_t2i_trainer import BriaFiboT2ITrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "briaai/FIBO"
PROMPT = "A serene mountain landscape with a clear blue lake reflecting snowcapped peaks"
HEIGHT, WIDTH = 1024, 1024
E2E_NUM_STEPS = 4
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_gpu(min_free_gb: float = 20.0) -> int | None:
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
    gpu_id = _find_free_gpu(min_free_gb=20)
    return f"cuda:{gpu_id}" if gpu_id is not None else "cpu"


@pytest.fixture(scope="module")
def t2i_trainer(infer_device):
    config = load_config_from_yaml(
        "tests/test_configs/test_example_bria_fibo_t2i_fp16.yaml"
    )
    config.model.pretrained_model_name_or_path = MODEL_ID
    config.model.lora.pretrained_weight = None
    config.predict.devices.vae = infer_device
    config.predict.devices.text_encoder = infer_device
    config.predict.devices.dit = infer_device
    trainer = BriaFiboT2ITrainer(config)
    trainer.setup_predict()
    return trainer


@pytest.fixture(scope="module")
def diffusers_pipeline(infer_device):
    pipe = BriaFiboPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(infer_device)
    return pipe


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestBriaFiboT2IVsDiffusers:
    """Verify that BriaFiboT2ITrainer faithfully reproduces BriaFiboPipeline."""

    # ── Test 1: weights ──────────────────────────────────────────────────

    @pytest.mark.e2e
    def test_component_weights_match_pipeline(self, t2i_trainer, diffusers_pipeline):
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
                assert p_p.shape == p_t.shape
                if torch.allclose(p_p.float(), p_t.float(), rtol=1e-5, atol=1e-8):
                    match += 1
                total += 1
            logger.info(f"[{label}] {match}/{total} parameters match exactly")
            assert match == total, f"[{label}] {total - match} parameter(s) differ"

        _compare(pipe.vae, trainer.vae, "VAE")
        _compare(pipe.text_encoder, trainer.text_encoder, "text_encoder")
        _compare(pipe.transformer, trainer.dit, "transformer")

    # ── Test 2: prompt embeddings ────────────────────────────────────────

    @pytest.mark.e2e
    def test_prompt_embeddings_match_pipeline(self, t2i_trainer, diffusers_pipeline):
        trainer = t2i_trainer
        pipe = diffusers_pipeline
        device = next(trainer.text_encoder.parameters()).device

        with torch.inference_mode():
            t_pe, t_layers, t_mask = trainer.encode_prompt(prompt=[PROMPT])

        with torch.inference_mode():
            # Pipeline encode_prompt returns (encoder_hidden_states, text_encoder_layers, attn_mask)
            p_pe, p_layers, p_mask = pipe.encode_prompt(
                prompt=PROMPT,
                device=device,
                batch_size=1,
            )

        assert_relative_error(t_pe, p_pe, rtol=1e-4, key="encoder_hidden_states (last 2 layers cat)")
        # Compare first and last layer from text_encoder_layers
        assert_relative_error(t_layers[0], p_layers[0], rtol=1e-4, key="text_encoder_layers[0]")
        assert_relative_error(t_layers[-1], p_layers[-1], rtol=1e-4, key="text_encoder_layers[-1]")
        assert torch.equal(t_mask.cpu(), p_mask.bool().cpu()), "attention_mask mismatch"

    # ── Test 3: end-to-end ───────────────────────────────────────────────

    @pytest.mark.e2e
    def test_end_to_end_output_matches_pipeline(self, t2i_trainer, diffusers_pipeline):
        trainer = t2i_trainer
        pipe = diffusers_pipeline
        h, w = HEIGHT, WIDTH

        batch = trainer.prepare_predict_batch_data(
            prompt=PROMPT, height=h, width=w,
            num_inference_steps=E2E_NUM_STEPS, guidance_scale=3.5,
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
            prompt_embeds=embeddings["prompt_embeds"].to(dit_dev),
            prompt_attention_mask=embeddings["prompt_embeds_mask"].to(dit_dev),
            latents=noise.clone().to(dit_dev),
            num_inference_steps=E2E_NUM_STEPS,
            height=h, width=w,
            guidance_scale=3.5,
            output_type="np",
        )
        pipe_np = (pipe_out.images[0] * 255).round().clip(0, 255).astype(np.uint8)

        assert_relative_error(trainer_np, pipe_np, rtol=0.05, key="output_image")

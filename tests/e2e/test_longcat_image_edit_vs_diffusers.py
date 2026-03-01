"""
Tests comparing LongCatImageEditTrainer against the official diffusers LongCatImageEditPipeline.

Validates that our trainer faithfully reproduces the pipeline for image-to-image editing:

  1. Component weights are identical (VAE, text_encoder, transformer)
  2. encode_prompt() produces matching multimodal embeddings (VL-encoded source image + text)
  3. VAE source encoding matches pipeline's _encode_vae_image() (argmax/mode)
  4. End-to-end inference with shared source image + fixed noise matches output

Key architectural details:
  - Qwen2_5_VLForConditionalGeneration text encoder (multimodal mode)
  - Source image at half-res → Qwen2VLProcessor → expand image tokens in prefix
  - prefix_len = index of <|vision_start|> (drops system header); hidden_states[-1]
  - Prompt embedding includes: vision_start + image_tokens + vision_end + content_512_tokens
  - Source → VAE full-res encode: (latents - shift_factor) * scaling_factor
  - latent_model_input = cat([noisy_target, source_latents], dim=1) (token dim)
  - img_ids: target modality_id=1, source modality_id=2, both offset by seq_len
  - model output sliced to first image_seq_len tokens

Tests use no pre-saved reference files — the official pipeline is the reference.

Usage:
    pytest tests/e2e/test_longcat_image_edit_vs_diffusers.py -m e2e -v
"""
import logging

import numpy as np
import pytest
import torch
from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import LongCatImageEditPipeline

from qflux.data.config import load_config_from_yaml
from qflux.trainer.longcat_image_edit_trainer import LongCatImageEditTrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "meituan-longcat/LongCat-Image-Edit"
PROMPT = "Make the sky more vivid orange and purple during sunset"
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
def edit_trainer(infer_device):
    """LongCatImageEditTrainer loaded from the same checkpoint (no LoRA)."""
    config = load_config_from_yaml(
        "tests/test_configs/test_example_longcat_image_edit_fp16.yaml"
    )
    config.model.pretrained_model_name_or_path = MODEL_ID
    config.model.lora.pretrained_weight = None
    config.predict.devices.vae = infer_device
    config.predict.devices.text_encoder = infer_device
    config.predict.devices.dit = infer_device
    trainer = LongCatImageEditTrainer(config)
    trainer.setup_predict()
    return trainer


@pytest.fixture(scope="module")
def diffusers_pipeline(infer_device):
    """Official LongCatImageEditPipeline from the same checkpoint."""
    pipe = LongCatImageEditPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(infer_device)
    return pipe


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestLongCatImageEditVsDiffusers:
    """Verify that LongCatImageEditTrainer faithfully reproduces LongCatImageEditPipeline."""

    # ── Test 1: weights ──────────────────────────────────────────────────

    @pytest.mark.e2e
    def test_component_weights_match_pipeline(self, edit_trainer, diffusers_pipeline):
        """
        Every parameter in the trainer's VAE, text_encoder, and transformer must
        be numerically identical to the corresponding pipeline component.
        """
        trainer = edit_trainer
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

    # ── Test 2: VAE source encoding ──────────────────────────────────────

    @pytest.mark.e2e
    def test_vae_encoding_matches_pipeline(self, edit_trainer, diffusers_pipeline):
        """
        trainer._vae_encode_source() must produce the same latents as
        pipeline._encode_vae_image() for the same source image.

        Both use argmax (mode()) and apply (latents - shift_factor) * scaling_factor.
        """
        trainer = edit_trainer
        pipe = diffusers_pipeline
        device = next(trainer.vae.parameters()).device
        dtype = trainer.weight_dtype

        source_img = _make_random_source_image(HEIGHT, WIDTH, device, dtype, seed=0)
        source_neg1_1 = source_img * 2.0 - 1.0

        with torch.inference_mode():
            t_latents = trainer._vae_encode_source(source_neg1_1.to(device, dtype=dtype))

        with torch.inference_mode():
            p_latents = pipe._encode_vae_image(source_neg1_1.to(pipe.vae.device, dtype=dtype), generator=None)

        assert_relative_error(t_latents, p_latents, rtol=1e-4, key="vae_source_latents")

    # ── Test 3: prompt embeddings ────────────────────────────────────────

    @pytest.mark.e2e
    def test_prompt_embeddings_match_pipeline(self, edit_trainer, diffusers_pipeline):
        """
        trainer.encode_prompt() must produce the same multimodal embeddings
        as pipeline.encode_prompt() for the same prompt and source image.
        """
        trainer = edit_trainer
        pipe = diffusers_pipeline
        device = next(trainer.text_encoder.parameters()).device
        dtype = trainer.weight_dtype

        source_img = _make_random_source_image(HEIGHT, WIDTH, device, dtype, seed=0)
        # Pipeline uses half-res PIL for VL encoding
        import PIL.Image
        source_half_arr = (
            source_img[0].permute(1, 2, 0).float().cpu().numpy() * 255
        ).clip(0, 255).astype("uint8")
        source_half_pil = PIL.Image.fromarray(source_half_arr).resize(
            (WIDTH // 2, HEIGHT // 2), PIL.Image.BILINEAR
        )
        # Pipeline expects the preprocessed-for-VAE image too, but for encode_prompt
        # it only needs the prompt_image (half-res PIL)
        with torch.inference_mode():
            t_pe, _ = trainer.encode_prompt(
                prompt=[PROMPT], source_image=source_img.to(device, dtype=dtype)
            )

        with torch.inference_mode():
            p_pe, _ = pipe.encode_prompt(
                prompt=[PROMPT],
                image=source_half_pil,
                num_images_per_prompt=1,
            )

        assert_relative_error(t_pe, p_pe, rtol=1e-4, key="prompt_embeds (multimodal)")

    # ── Test 4: end-to-end ───────────────────────────────────────────────

    @pytest.mark.e2e
    def test_end_to_end_output_matches_pipeline(self, edit_trainer, diffusers_pipeline):
        """
        Using a shared source image, pre-computed trainer embeddings, and
        the same fixed initial noise latents, the final decoded images must
        be nearly identical (rtol ≤ 5%).
        """
        trainer = edit_trainer
        pipe = diffusers_pipeline
        h, w = HEIGHT, WIDTH
        device = trainer.dit.device
        dtype = trainer.weight_dtype

        # ── Step A: synthetic source image ──────────────────────────────
        source_img = _make_random_source_image(h, w, device, dtype, seed=0)

        # ── Step B: prepare embeddings via trainer ──────────────────────
        batch = trainer.prepare_predict_batch_data(
            prompt=PROMPT,
            source_image=source_img,
            height=h,
            width=w,
            num_inference_steps=E2E_NUM_STEPS,
            guidance_scale=0.0,
        )
        embeddings = trainer.prepare_embeddings(batch, stage="predict")

        # ── Step C: shared initial noise latents ────────────────────────
        h_lat = 2 * (h // (trainer.vae_scale_factor * 2))
        w_lat = 2 * (w // (trainer.vae_scale_factor * 2))
        gen = torch.Generator(device=device).manual_seed(SEED)
        noise = torch.randn(
            (1, trainer.num_channels_latents, h_lat, w_lat),
            generator=gen,
            device=device,
            dtype=torch.float32,
        )

        # ── Step D: trainer denoising ────────────────────────────────────
        embeddings["latents"] = noise.clone()
        trainer_latents = trainer.sampling_from_embeddings(embeddings)
        trainer_img = trainer.decode_vae_latent(trainer_latents, h, w)
        trainer_np = (
            trainer_img.detach().permute(0, 2, 3, 1).float().cpu().numpy()[0] * 255
        ).round().clip(0, 255).astype(np.uint8)

        # ── Step E: pipeline denoising ───────────────────────────────────
        import PIL.Image
        source_neg1_1 = source_img * 2.0 - 1.0
        source_arr = (source_img[0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype("uint8")
        source_pil = PIL.Image.fromarray(source_arr)

        dit_dev = pipe.transformer.device
        pipe_out = pipe(
            image=source_pil,
            prompt=PROMPT,
            latents=noise.clone().to(dit_dev),
            num_inference_steps=E2E_NUM_STEPS,
            guidance_scale=0.0,
            output_type="np",
        )
        pipe_np = (pipe_out.images[0] * 255).round().clip(0, 255).astype(np.uint8)

        assert_relative_error(trainer_np, pipe_np, rtol=0.05, key="output_image")

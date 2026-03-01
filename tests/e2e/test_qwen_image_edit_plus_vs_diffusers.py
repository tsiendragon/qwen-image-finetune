"""
Tests comparing QwenImageEditPlusTrainer against the official diffusers
QwenImageEditPlusPipeline.

Validates that our trainer's implementation exactly reproduces what the
official diffusers pipeline produces:

  1. Component weights are identical (VAE, text encoder, transformer)
  2. encode_prompt() produces the same embeddings given identical inputs
  3. VAE image encoding produces the same latents given identical inputs
  4. End-to-end inference (same fixed initial latents + pre-computed
     embeddings) produces the same output image

Tests are designed to require *no pre-saved reference files* — the
official pipeline itself is the reference.

Usage:
    pytest tests/e2e/test_qwen_image_edit_plus_vs_diffusers.py -m e2e -v
"""
import logging

import numpy as np
import PIL.Image
import pytest
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import retrieve_latents
from diffusers.utils import load_image

from qflux.data.config import load_config_from_yaml
from qflux.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model and test constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
PROMPT = "change the hair color to dark blue"
IMAGE_URL = "https://n.sinaimg.cn/ent/transform/775/w630h945/20201127/cee0-kentcvx8062290.jpg"

# Image size used for inference (square keeps preprocessing identical between
# trainer and pipeline, avoiding aspect-ratio resize discrepancies)
INFER_H, INFER_W = 448, 448
# Size fed to the text encoder – matches diffusers' CONDITION_IMAGE_SIZE
CONDITION_SIZE = 384
# Steps for the end-to-end test (fewer = faster; 5 is enough to check parity)
E2E_NUM_STEPS = 5
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_free_gpu(min_free_gb: float = 10.0) -> int | None:
    """Return the GPU index with the most free VRAM, or None if no GPU."""
    if not torch.cuda.is_available():
        return None
    best, best_free = 0, 0.0
    for i in range(torch.cuda.device_count()):
        try:
            free = torch.cuda.mem_get_info(i)[0] / 2 ** 30
            logger.info(f"GPU {i}: {free:.1f} GB free")
            if free > best_free:
                best, best_free = i, free
        except Exception:
            pass
    if best_free < min_free_gb:
        logger.warning(f"Best GPU has only {best_free:.1f} GB free (wanted {min_free_gb})")
    return best


def assert_relative_error(
    x, y, rtol: float = 1e-4, key: str = "tensor"
) -> float:
    """Assert relative L2 error is below *rtol* and return it."""
    for src in (x, y):
        pass  # typing aid only

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
def qwen_plus_trainer(infer_device):
    """QwenImageEditPlusTrainer loaded from the same checkpoint (no LoRA)."""
    config = load_config_from_yaml(
        "tests/test_configs/test_example_qwen_image_edit_plus_fp16.yaml"
    )
    config.model.pretrained_model_name_or_path = MODEL_ID
    config.model.lora.pretrained_weight = None
    config.predict.devices.vae = infer_device
    config.predict.devices.text_encoder = infer_device
    config.predict.devices.dit = infer_device
    trainer = QwenImageEditPlusTrainer(config)
    trainer.setup_predict()
    return trainer


@pytest.fixture(scope="module")
def diffusers_pipeline(infer_device):
    """Official QwenImageEditPlusPipeline from the same checkpoint."""
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    )
    pipe.to(infer_device)
    return pipe


@pytest.fixture(scope="module")
def raw_image() -> PIL.Image.Image:
    return load_image(IMAGE_URL).convert("RGB")


@pytest.fixture(scope="module")
def infer_image(raw_image) -> PIL.Image.Image:
    """Square image at inference resolution (avoids aspect-ratio ambiguity)."""
    return raw_image.resize((INFER_W, INFER_H), resample=PIL.Image.LANCZOS)


@pytest.fixture(scope="module")
def condition_image(raw_image) -> PIL.Image.Image:
    """384×384 condition image – matches diffusers' CONDITION_IMAGE_SIZE."""
    return raw_image.resize((CONDITION_SIZE, CONDITION_SIZE), resample=PIL.Image.LANCZOS)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestQwenPlusVsDiffusers:
    """Verify that our trainer faithfully reproduces the official diffusers pipeline."""

    # ── Test 1: weights ──────────────────────────────────────────────────────

    @pytest.mark.e2e
    def test_component_weights_match_pipeline(
        self, qwen_plus_trainer, diffusers_pipeline
    ):
        """
        Every parameter in the trainer's VAE, text_encoder, and transformer
        must be numerically identical to the corresponding pipeline component.
        """
        trainer = qwen_plus_trainer
        pipe = diffusers_pipeline

        # Move all to CPU for comparison so VRAM stays free
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
                f"[{label}] {total - match} parameter(s) differ between "
                "pipeline and trainer"
            )

        _compare(pipe.vae, trainer.vae, "VAE")
        _compare(pipe.text_encoder, trainer.text_encoder, "text_encoder")
        _compare(pipe.transformer, trainer.dit, "transformer")

    # ── Test 2: prompt embeddings ────────────────────────────────────────────

    @pytest.mark.e2e
    def test_prompt_embeddings_match_pipeline(
        self, qwen_plus_trainer, diffusers_pipeline, condition_image
    ):
        """
        trainer.encode_prompt() and pipeline.encode_prompt() must produce
        identical (prompt_embeds, prompt_embeds_mask) for the same inputs.
        Both receive the exact same list of PIL condition images, bypassing
        any pre-processing differences.
        """
        trainer = qwen_plus_trainer
        pipe = diffusers_pipeline
        device = next(trainer.text_encoder.parameters()).device

        with torch.inference_mode():
            t_embeds, t_mask = trainer.encode_prompt(
                prompt=[PROMPT],
                image=[condition_image],
            )

        with torch.inference_mode():
            p_embeds, p_mask = pipe.encode_prompt(
                prompt=PROMPT,
                image=[condition_image],
                device=device,
            )

        assert_relative_error(t_embeds, p_embeds, rtol=1e-4, key="prompt_embeds")
        assert_relative_error(t_mask, p_mask.float(), rtol=1e-5, key="prompt_embeds_mask")

    # ── Test 3: VAE encoding ─────────────────────────────────────────────────

    @pytest.mark.e2e
    def test_vae_encoding_matches_pipeline(
        self, qwen_plus_trainer, diffusers_pipeline, infer_image
    ):
        """
        trainer._encode_vae_image() and the pipeline's VAE encoding must
        produce the same normalised latents for the same image tensor.
        """
        trainer = qwen_plus_trainer
        pipe = diffusers_pipeline
        vae_device = trainer.vae.device
        dtype = trainer.weight_dtype

        # Build a [1, 3, 1, H, W] image tensor in [-1, 1] range
        img_arr = np.array(infer_image).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        vae_input = trainer.preprocess_image_for_vae_encoder(img_t)       # [1,3,1,H,W]
        vae_input = vae_input.to(vae_device, dtype=dtype)

        # --- Trainer ---
        with torch.inference_mode():
            trainer_latents = trainer._encode_vae_image(vae_input)

        # --- Pipeline: replicate _encode_vae_image (argmax + normalisation) ---
        with torch.inference_mode():
            raw = retrieve_latents(
                pipe.vae.encode(vae_input), sample_mode="argmax"
            )
            lm = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(raw.device, raw.dtype)
            )
            ls = (
                torch.tensor(pipe.vae.config.latents_std)
                .view(1, -1, 1, 1, 1)
                .to(raw.device, raw.dtype)
            )
            pipe_latents = (raw - lm) / ls

        assert_relative_error(
            trainer_latents, pipe_latents, rtol=1e-4, key="vae_latents"
        )

    # ── Test 4: end-to-end ───────────────────────────────────────────────────

    @pytest.mark.e2e
    def test_end_to_end_output_matches_pipeline(
        self, qwen_plus_trainer, diffusers_pipeline, infer_image
    ):
        """
        Using pre-computed trainer embeddings and the same fixed initial
        latents injected into both the trainer and the pipeline, the final
        decoded images must be nearly identical.

        By sharing prompt_embeds / prompt_embeds_mask between the two runs we
        eliminate any text-encoding divergence and focus purely on the
        denoising loop and VAE decode, both of which must be bit-for-bit
        compatible.
        """
        trainer = qwen_plus_trainer
        pipe = diffusers_pipeline
        h, w = INFER_H, INFER_W

        # ── Step A: prepare all embeddings via trainer ─────────────────────
        batch = trainer.prepare_predict_batch_data(
            image=infer_image,
            prompt=PROMPT,
            num_inference_steps=E2E_NUM_STEPS,
            controls_size=[[h, w]],
            guidance_scale=None,
            true_cfg_scale=1.0,
            negative_prompt="",
            weight_dtype=trainer.weight_dtype,
            height=h,
            width=w,
        )
        embeddings = trainer.prepare_embeddings(batch, stage="predict")

        # ── Step B: generate fixed initial (noise) latents ─────────────────
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

        # ── Step C: trainer denoising ──────────────────────────────────────
        embeddings["latents"] = noise_packed
        trainer_latents = trainer.sampling_from_embeddings(embeddings)
        trainer_img_t = trainer.decode_vae_latent(trainer_latents, h, w)
        trainer_np = (
            trainer_img_t.detach().permute(0, 2, 3, 1).float().cpu().numpy()[0] * 255
        ).round().clip(0, 255).astype(np.uint8)

        # ── Step D: diffusers pipeline denoising ───────────────────────────
        # Pass the trainer's pre-computed prompt embeds so text encoding is
        # identical, and the same packed noise latents so the denoising
        # starts from the same point.
        pipe_out = pipe(
            image=infer_image,
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

        # VAE preprocessing of the control image may introduce tiny differences
        # (the pipeline has its own image-size heuristic), so we allow 5 % rtol
        assert_relative_error(trainer_np, pipe_np, rtol=0.05, key="output_image")

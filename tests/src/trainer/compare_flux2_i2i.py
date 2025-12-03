"""
Compare Flux2LoraTrainer I2I with original Flux2Pipeline.

Usage:
    # Use different devices for pipeline and trainer (recommended to avoid OOM)
    python tests/src/trainer/compare_flux2_i2i.py --pipeline-device cuda:0 --trainer-device cuda:1

    # Use same device for both
    python tests/src/trainer/compare_flux2_i2i.py --pipeline-device cuda:1

    # Legacy: use --device for both (deprecated)
    python tests/src/trainer/compare_flux2_i2i.py --device cuda:1
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor

from qflux.trainer.flux2_trainer import Flux2LoraTrainer
from qflux.data.config import load_config_from_yaml


def check_gpu_memory(device: str) -> tuple[float, float]:
    """Check available GPU memory."""
    if not device.startswith("cuda"):
        return 0.0, 0.0

    device_idx = int(device.split(":")[-1]) if ":" in device else 0
    total_memory = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(device_idx) / 1024**3
    reserved_memory = torch.cuda.memory_reserved(device_idx) / 1024**3
    free_memory = total_memory - reserved_memory

    return free_memory, total_memory


def load_pipeline(model_path: str, device: str = "cuda"):
    """Load original Flux2Pipeline with optimized memory usage."""
    print(f"Loading Flux2Pipeline on {device}...")

    # Check available memory if using CUDA
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        free_mem, total_mem = check_gpu_memory(device)
        print(f"GPU memory status: {free_mem:.2f} GB free / {total_mem:.2f} GB total")

        # FLUX.2 pipeline typically needs ~20-25GB, warn if less than 30GB free
        if free_mem < 30.0:
            print(f"Warning: Only {free_mem:.2f} GB free memory. FLUX.2 pipeline may need ~20-25GB.")
            print("Consider using a different GPU or freeing up memory.")

    # Load pipeline - simple approach
    print("Loading pipeline components...")
    pipe = Flux2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    print(f"Pipeline loaded on {device}")
    if device.startswith("cuda"):
        free_mem, total_mem = check_gpu_memory(device)
        print(f"Final GPU memory: {free_mem:.2f} GB free / {total_mem:.2f} GB total")
    return pipe


def load_trainer(config_path: str, device: str = None):
    """Load Flux2LoraTrainer with optional device override."""
    print("Loading Flux2LoraTrainer...")
    config = load_config_from_yaml(config_path)

    # Override device settings if provided
    if device:
        if not hasattr(config, 'predict') or config.predict is None:
            from qflux.data.config import PredictConfig, DeviceConfig
            config.predict = PredictConfig(devices=DeviceConfig())

        if not hasattr(config.predict, 'devices') or config.predict.devices is None:
            from qflux.data.config import DeviceConfig
            config.predict.devices = DeviceConfig()

        # Set all components to the specified device
        config.predict.devices.vae = device
        config.predict.devices.text_encoder = device
        config.predict.devices.dit = device
        print(f"Trainer device overridden to: {device}")

    trainer = Flux2LoraTrainer(config)
    trainer.setup_predict()
    print("Trainer loaded")
    return trainer


def get_comparison_device(device1, device2):
    """
    Select the best device for comparison.
    Priority: GPU > CPU. If both are GPU, prefer device1.

    Args:
        device1: First device (str or torch.device, e.g., 'cuda:0')
        device2: Second device (str or torch.device, e.g., 'cuda:1')

    Returns:
        Selected device for comparison
    """
    # Convert to string for comparison
    dev1_str = str(device1)
    dev2_str = str(device2)

    # If both are GPU, prefer device1 (pipeline device)
    if dev1_str.startswith("cuda") and dev2_str.startswith("cuda"):
        return device1
    # If one is GPU, prefer GPU
    if dev1_str.startswith("cuda"):
        return device1
    if dev2_str.startswith("cuda"):
        return device2
    # Both are CPU or other
    return device1


def resize_to_target_area(image: Image.Image, target_area: int) -> Image.Image:
    """Resize image to approximately target area while maintaining aspect ratio."""
    width, height = image.size
    current_area = width * height

    if current_area <= target_area:
        return image

    # Calculate scale factor
    scale = (target_area / current_area) ** 0.5
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def preprocess_image(image: Image.Image, pipe: Flux2Pipeline):
    """Preprocess image for I2I."""
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)

    img = image.copy()
    image_width, image_height = img.size

    # Resize if too large
    if image_width * image_height > 1024 * 1024:
        img = resize_to_target_area(img, 1024 * 1024)
        image_width, image_height = img.size

    # Make dimensions divisible
    multiple_of = pipe.vae_scale_factor * 2
    image_width = (image_width // multiple_of) * multiple_of
    image_height = (image_height // multiple_of) * multiple_of

    processed = image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")

    return processed, image_width, image_height


def compare_image_encoding(pipe, trainer, processed_image, pipeline_device, trainer_device):
    """Compare image encoding between pipeline and trainer."""
    print("\n" + "=" * 60)
    print("1. IMAGE ENCODING COMPARISON (prepare_image_latents)")
    print("=" * 60)

    seed = 42

    # Pipeline
    with torch.inference_mode():
        pipe_image_latents, pipe_image_latent_ids = pipe.prepare_image_latents(
            images=[processed_image],
            batch_size=1,
            generator=torch.Generator(device=pipeline_device).manual_seed(seed),
            device=pipeline_device,
            dtype=pipe.vae.dtype,
        )

    print(f"\n[Pipeline] (device: {pipeline_device})")
    print(f"  image_latents shape: {pipe_image_latents.shape}")
    print(f"  image_latent_ids shape: {pipe_image_latent_ids.shape}")
    print(f"  image_latents mean: {pipe_image_latents.mean().item():.6f}")

    # Trainer - use trainer's VAE device
    trainer_vae_device = next(trainer.vae.parameters()).device
    with torch.inference_mode():
        trainer_image_latents, trainer_image_latent_ids = trainer.prepare_image_latents(
            images=[processed_image],
            batch_size=1,
            generator=torch.Generator(device=trainer_vae_device).manual_seed(seed),
            device=trainer_vae_device,
            dtype=pipe.vae.dtype,
        )

    print(f"\n[Trainer] (device: {trainer_vae_device})")
    print(f"  image_latents shape: {trainer_image_latents.shape}")
    print(f"  image_latent_ids shape: {trainer_image_latent_ids.shape}")
    print(f"  image_latents mean: {trainer_image_latents.mean().item():.6f}")

    # Select best device for comparison (prefer GPU)
    comparison_device = get_comparison_device(pipeline_device, trainer_vae_device)
    print(f"\n[Comparison] (comparing on device: {comparison_device})")

    # Move both tensors to comparison device
    pipe_latents_on_comp = pipe_image_latents.to(comparison_device)
    trainer_latents_on_comp = trainer_image_latents.to(comparison_device)

    shape_match = pipe_latents_on_comp.shape == trainer_latents_on_comp.shape
    print(f"  image_latents shape match: {shape_match}")

    if shape_match:
        # Ensure same dtype for comparison
        trainer_latents_typed = trainer_latents_on_comp.to(pipe_latents_on_comp.dtype)
        diff = (pipe_latents_on_comp - trainer_latents_typed).abs()
        print(f"  image_latents max diff: {diff.max().item():.6f}")
        print(f"  image_latents mean diff: {diff.mean().item():.6f}")

    # Compare latent IDs
    pipe_ids_on_comp = pipe_image_latent_ids.to(comparison_device)
    trainer_ids_on_comp = trainer_image_latent_ids.to(comparison_device)

    ids_shape_match = pipe_ids_on_comp.shape == trainer_ids_on_comp.shape
    print(f"  image_latent_ids shape match: {ids_shape_match}")

    if ids_shape_match:
        ids_match = torch.equal(pipe_ids_on_comp, trainer_ids_on_comp)
        print(f"  image_latent_ids match: {ids_match}")

    return shape_match


def compare_tensor(name: str, t1: torch.Tensor, t2: torch.Tensor, comparison_device: str = "cpu"):
    """Compare two tensors and print statistics."""
    t1_comp = t1.to(comparison_device).float()
    t2_comp = t2.to(comparison_device).float()

    print(f"\n  [{name}]")
    print(f"    Shape: pipe={t1.shape}, trainer={t2.shape}, match={t1.shape == t2.shape}")

    if t1.shape != t2.shape:
        print(f"    WARNING: Shape mismatch, cannot compare values")
        return False

    print(f"    Dtype: pipe={t1.dtype}, trainer={t2.dtype}")
    print(f"    Mean: pipe={t1_comp.mean().item():.6f}, trainer={t2_comp.mean().item():.6f}")
    print(f"    Std:  pipe={t1_comp.std().item():.6f}, trainer={t2_comp.std().item():.6f}")

    diff = (t1_comp - t2_comp).abs()
    print(f"    Diff: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

    # Check if they're close
    is_close = torch.allclose(t1_comp, t2_comp, rtol=1e-3, atol=1e-3)
    print(f"    Close (rtol=1e-3, atol=1e-3): {is_close}")

    return is_close


def compare_intermediate_steps(pipe, trainer, test_image, prompt, pipeline_device, trainer_device, seed=42):
    """Compare intermediate steps between pipeline and trainer."""
    print("\n" + "=" * 60)
    print("2. INTERMEDIATE STEPS COMPARISON")
    print("=" * 60)

    # Preprocess image to get consistent dimensions
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)
    img = test_image.copy()
    image_width, image_height = img.size

    if image_width * image_height > 1024 * 1024:
        img = resize_to_target_area(img, 1024 * 1024)
        image_width, image_height = img.size

    multiple_of = pipe.vae_scale_factor * 2
    target_width = (image_width // multiple_of) * multiple_of
    target_height = (image_height // multiple_of) * multiple_of

    print(f"\n[Configuration]")
    print(f"  Target dimensions: {target_width} x {target_height}")
    print(f"  Seed: {seed}")
    print(f"  Pipeline device: {pipeline_device}")
    print(f"  Trainer device: {trainer_device}")

    # Get comparison device
    comparison_device = get_comparison_device(pipeline_device, str(next(trainer.dit.parameters()).device))
    print(f"  Comparison device: {comparison_device}")

    # ============================================================
    # Step 1: Text Encoding
    # ============================================================
    print("\n" + "-" * 40)
    print("Step 1: TEXT ENCODING")
    print("-" * 40)

    # Pipeline text encoding - Flux2Pipeline.encode_prompt returns (prompt_embeds, text_ids)
    with torch.inference_mode():
        pipe_prompt_embeds, pipe_text_ids = pipe.encode_prompt(
            prompt=prompt,
            device=pipeline_device,
            max_sequence_length=512,
        )

    # Trainer text encoding
    with torch.inference_mode():
        trainer_prompt_embeds, trainer_text_ids = trainer.encode_prompt(
            prompt=[prompt],
            max_sequence_length=512,
        )

    print(f"\n[Pipeline Text Encoding]")
    print(f"  prompt_embeds: {pipe_prompt_embeds.shape}, device={pipe_prompt_embeds.device}")
    print(f"  text_ids: {pipe_text_ids.shape}")

    print(f"\n[Trainer Text Encoding]")
    print(f"  prompt_embeds: {trainer_prompt_embeds.shape}, device={trainer_prompt_embeds.device}")
    print(f"  text_ids: {trainer_text_ids.shape}")

    compare_tensor("prompt_embeds", pipe_prompt_embeds, trainer_prompt_embeds, comparison_device)
    compare_tensor("text_ids", pipe_text_ids, trainer_text_ids, comparison_device)

    # ============================================================
    # Step 2: Image Preprocessing
    # ============================================================
    print("\n" + "-" * 40)
    print("Step 2: IMAGE PREPROCESSING")
    print("-" * 40)

    # Preprocess for both
    processed_image = image_processor.preprocess(
        img, height=target_height, width=target_width, resize_mode="crop"
    )
    print(f"  Processed image tensor: {processed_image.shape}")

    # ============================================================
    # Step 3: Image Latent Encoding (VAE)
    # ============================================================
    print("\n" + "-" * 40)
    print("Step 3: IMAGE LATENT ENCODING (VAE)")
    print("-" * 40)

    # Pipeline image latents
    with torch.inference_mode():
        pipe_image_latents, pipe_image_ids = pipe.prepare_image_latents(
            images=[processed_image],
            batch_size=1,
            generator=torch.Generator("cpu").manual_seed(seed),
            device=pipeline_device,
            dtype=torch.bfloat16,
        )

    # Trainer image latents
    trainer_vae_device = next(trainer.vae.parameters()).device
    with torch.inference_mode():
        trainer_image_latents, trainer_image_ids = trainer.prepare_image_latents(
            images=[processed_image],
            batch_size=1,
            generator=torch.Generator("cpu").manual_seed(seed),
            device=trainer_vae_device,
            dtype=torch.bfloat16,
        )

    compare_tensor("image_latents", pipe_image_latents, trainer_image_latents, comparison_device)
    compare_tensor("image_latent_ids", pipe_image_ids, trainer_image_ids, comparison_device)

    # ============================================================
    # Step 4: Initial Noise Latents
    # ============================================================
    print("\n" + "-" * 40)
    print("Step 4: INITIAL NOISE LATENTS")
    print("-" * 40)

    # Calculate latent dimensions
    latent_height = 2 * (target_height // (pipe.vae_scale_factor * 2))
    latent_width = 2 * (target_width // (pipe.vae_scale_factor * 2))
    num_channels = pipe.transformer.config.in_channels // 4

    print(f"  Latent dimensions: {latent_height} x {latent_width}")
    print(f"  Num channels: {num_channels}")

    # Pipeline noise
    from diffusers.utils.torch_utils import randn_tensor
    pipe_shape = (1, num_channels, latent_height, latent_width)
    pipe_noise = randn_tensor(pipe_shape, generator=torch.Generator("cpu").manual_seed(seed), device="cpu", dtype=torch.bfloat16)

    # Trainer noise (same generator, same seed)
    trainer_noise = randn_tensor(pipe_shape, generator=torch.Generator("cpu").manual_seed(seed), device="cpu", dtype=torch.bfloat16)

    compare_tensor("initial_noise", pipe_noise, trainer_noise, "cpu")

    # ============================================================
    # Step 5: Scheduler Setup
    # ============================================================
    print("\n" + "-" * 40)
    print("Step 5: SCHEDULER SETUP")
    print("-" * 40)

    num_inference_steps = 20

    # Calculate mu for dynamic shifting (needed for Flux2 scheduler)
    # This is the same calculation used in Flux2Pipeline
    num_latent_pixels = latent_height * latent_width
    mu = pipe.scheduler.config.base_shift + pipe.scheduler.config.max_shift / (
        pipe.scheduler.config.base_shift + num_latent_pixels / 256 / 256
    )
    print(f"  Calculated mu for dynamic shifting: {mu:.4f}")

    # Pipeline scheduler
    try:
        pipe.scheduler.set_timesteps(num_inference_steps, device=pipeline_device, mu=mu)
    except TypeError:
        # Fallback if mu is not supported
        pipe.scheduler.set_timesteps(num_inference_steps, device=pipeline_device)
    pipe_timesteps = pipe.scheduler.timesteps
    pipe_sigmas = pipe.scheduler.sigmas if hasattr(pipe.scheduler, 'sigmas') else None

    print(f"\n[Pipeline Scheduler]")
    print(f"  Scheduler type: {type(pipe.scheduler).__name__}")
    print(f"  Timesteps: {pipe_timesteps[:5].tolist()}... (first 5)")
    if pipe_sigmas is not None:
        print(f"  Sigmas: {pipe_sigmas[:5].tolist()}... (first 5)")

    # Trainer scheduler
    try:
        trainer.sampling_scheduler.set_timesteps(num_inference_steps, device=str(next(trainer.dit.parameters()).device), mu=mu)
    except TypeError:
        trainer.sampling_scheduler.set_timesteps(num_inference_steps, device=str(next(trainer.dit.parameters()).device))
    trainer_timesteps = trainer.sampling_scheduler.timesteps
    trainer_sigmas = trainer.sampling_scheduler.sigmas if hasattr(trainer.sampling_scheduler, 'sigmas') else None

    print(f"\n[Trainer Scheduler]")
    print(f"  Scheduler type: {type(trainer.sampling_scheduler).__name__}")
    print(f"  Timesteps: {trainer_timesteps[:5].tolist()}... (first 5)")
    if trainer_sigmas is not None:
        print(f"  Sigmas: {trainer_sigmas[:5].tolist()}... (first 5)")

    # Compare timesteps
    compare_tensor("timesteps", pipe_timesteps.float(), trainer_timesteps.float(), "cpu")

    # ============================================================
    # Step 6: Compare Preprocessed Condition Images
    # ============================================================
    print("\n" + "-" * 40)
    print("Step 6: CONDITION IMAGE PREPROCESSING")
    print("-" * 40)

    # Pipeline preprocessing
    pipe_processed = image_processor.preprocess(
        img, height=target_height, width=target_width, resize_mode="crop"
    )
    print(f"\n[Pipeline Image Preprocessing]")
    print(f"  Input image size: {img.size}")
    print(f"  Processed tensor shape: {pipe_processed.shape}")
    print(f"  Processed tensor range: [{pipe_processed.min():.4f}, {pipe_processed.max():.4f}]")
    print(f"  Processed tensor mean: {pipe_processed.mean():.6f}")

    # Trainer preprocessing (using VaeImageProcessor - matches predict flow)
    trainer_processed = trainer.image_processor.preprocess(
        img, height=target_height, width=target_width, resize_mode="crop"
    )
    print(f"\n[Trainer Image Preprocessing]")
    print(f"  Input image size: {img.size}")
    print(f"  Processed tensor shape: {trainer_processed.shape}")
    print(f"  Processed tensor range: [{trainer_processed.min():.4f}, {trainer_processed.max():.4f}]")
    print(f"  Processed tensor mean: {trainer_processed.mean():.6f}")

    # Compare
    if pipe_processed.shape == trainer_processed.shape:
        compare_tensor("preprocessed_image", pipe_processed, trainer_processed, "cpu")
    else:
        print(f"\n  ⚠️  Shape mismatch: pipe={pipe_processed.shape}, trainer={trainer_processed.shape}")

    # ============================================================
    # SUMMARY: Key Findings
    # ============================================================
    print("\n" + "=" * 40)
    print("INTERMEDIATE COMPARISON SUMMARY")
    print("=" * 40)

    return True


def compare_full_i2i(pipe, trainer, test_image, prompt, pipeline_device, trainer_device, seed=42):
    """Compare full I2I generation."""
    print("\n" + "=" * 60)
    print("3. FULL I2I GENERATION COMPARISON")
    print("=" * 60)

    # Preprocess image to get consistent dimensions
    image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)
    img = test_image.copy()
    image_width, image_height = img.size

    # Resize if too large (same logic as preprocess_image)
    if image_width * image_height > 1024 * 1024:
        img = resize_to_target_area(img, 1024 * 1024)
        image_width, image_height = img.size

    # Make dimensions divisible by vae_scale_factor * 2
    multiple_of = pipe.vae_scale_factor * 2
    target_width = (image_width // multiple_of) * multiple_of
    target_height = (image_height // multiple_of) * multiple_of

    print(f"\n[Using consistent dimensions: {target_width} x {target_height}]")
    print(f"[Using seed: {seed}]")

    # Pipeline I2I - use CPU generator for reproducibility across devices
    print(f"\n[Generating with Pipeline...] (device: {pipeline_device})")
    # Create generator on CPU for consistent random state
    generator_pipe = torch.Generator("cpu").manual_seed(seed)
    with torch.inference_mode():
        pipe_output = pipe(
            image=test_image,
            prompt=prompt,
            height=target_height,
            width=target_width,
            num_inference_steps=20,
            guidance_scale=3.5,
            generator=generator_pipe,
            output_type="pil",
        )
        pipe_image = pipe_output.images[0]
    print(f"Pipeline output size: {pipe_image.size}")

    # Trainer I2I - get the actual device used by trainer's DIT
    trainer_dit_device = next(trainer.dit.parameters()).device
    print(f"\n[Generating with Trainer...] (device: {trainer_dit_device})")
    # Create generator on CPU for consistent random state
    generator_trainer = torch.Generator("cpu").manual_seed(seed)
    trainer_output = trainer.predict(
        image=test_image,
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=3.5,
        true_cfg_scale=1.0,
        negative_prompt="",
        weight_dtype=torch.bfloat16,
        height=target_height,
        width=target_width,
        output_type='pil',
        generator=generator_trainer,
    )

    if isinstance(trainer_output, list):
        trainer_image = trainer_output[0]
    else:
        trainer_image = trainer_output
    print(f"Trainer output size: {trainer_image.size}")

    # Compare pixel differences
    print("\n[Pixel Difference Statistics]")
    pipe_arr = np.array(pipe_image.resize((512, 512)))
    trainer_arr = np.array(trainer_image.resize((512, 512)))

    diff = np.abs(pipe_arr.astype(float) - trainer_arr.astype(float))
    print(f"  Max pixel diff: {diff.max():.2f}")
    print(f"  Mean pixel diff: {diff.mean():.2f}")
    print(f"  Std pixel diff: {diff.std():.2f}")

    return pipe_image, trainer_image


def save_comparison(pipe_image, trainer_image, output_path="i2i_comparison.png"):
    """Save side-by-side comparison."""
    w1, h1 = pipe_image.size
    w2, h2 = trainer_image.size

    max_h = max(h1, h2)
    max_w = max(w1, w2)

    canvas = Image.new('RGB', (max_w * 2 + 20, max_h + 40), color='white')
    canvas.paste(pipe_image, (0, 30))
    canvas.paste(trainer_image, (max_w + 20, 30))

    canvas.save(output_path)
    print(f"\nComparison saved to {output_path}")


def validate_device(device: str) -> bool:
    """Validate if device is available."""
    if device.startswith("cuda"):
        device_id = device.split(":")[-1] if ":" in device else "0"
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if device_id.isdigit():
                device_idx = int(device_id)
                if device_idx >= num_gpus:
                    print(f"Warning: Device {device} not available. Only {num_gpus} GPU(s) available.")
                    print(f"Available devices: {[f'cuda:{i}' for i in range(num_gpus)]}")
                    return False
                print(f"Device {device} (GPU {device_id}): {torch.cuda.get_device_properties(device_idx).total_memory / 1024**3:.2f} GB")
            else:
                print(f"Device {device} (GPU 0): {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("Warning: CUDA is not available. Falling back to CPU.")
            return device == "cpu"
    return True


def get_devices():
    """Get CUDA devices for pipeline and trainer from command line or environment."""
    parser = argparse.ArgumentParser(description="Compare Flux2LoraTrainer I2I with Flux2Pipeline")
    parser.add_argument(
        "--pipeline-device",
        type=str,
        default=None,
        help="CUDA device for Pipeline (e.g., 'cuda:0', 'cuda:1'). Default: 'cuda' or from CUDA_VISIBLE_DEVICES"
    )
    parser.add_argument(
        "--trainer-device",
        type=str,
        default=None,
        help="CUDA device for Trainer (e.g., 'cuda:1', 'cuda:2'). Default: same as pipeline-device"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device for both Pipeline and Trainer (deprecated, use --pipeline-device and --trainer-device instead)"
    )
    args = parser.parse_args()

    # Handle deprecated --device argument
    if args.device:
        print("Warning: --device is deprecated. Use --pipeline-device and --trainer-device instead.")
        pipeline_device = args.device
        trainer_device = args.device
    else:
        # Get pipeline device
        if args.pipeline_device:
            pipeline_device = args.pipeline_device
        else:
            # Check CUDA_VISIBLE_DEVICES environment variable
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                pipeline_device = "cuda:0"
                print(f"Using pipeline device from CUDA_VISIBLE_DEVICES={cuda_visible} -> {pipeline_device}")
            else:
                pipeline_device = "cuda"
                print(f"Using default pipeline device: {pipeline_device}")

        # Get trainer device (default to pipeline device if not specified)
        if args.trainer_device:
            trainer_device = args.trainer_device
        else:
            trainer_device = pipeline_device
            print(f"Using trainer device: {trainer_device} (same as pipeline)")

    # Validate devices
    if not validate_device(pipeline_device):
        return None, None
    if not validate_device(trainer_device):
        return None, None

    print(f"\nDevice Configuration:")
    print(f"  Pipeline: {pipeline_device}")
    print(f"  Trainer:  {trainer_device}")

    return pipeline_device, trainer_device


def main():
    # Get device configuration
    pipeline_device, trainer_device = get_devices()
    if pipeline_device is None or trainer_device is None:
        print("Error: Invalid device configuration. Exiting.")
        return

    # Configuration
    model_path = "diffusers/FLUX.2-dev-bnb-4bit"
    config_path = "configs/flux2_i2i_fp4_config.yaml"

    # Load test image
    test_image_url = "https://www.ripponmedicalservices.co.uk/images/easyblog_articles/89/b2ap3_large_ee72093c-3c01-433a-8d25-701cca06c975.jpg"
    print(f"Loading test image from {test_image_url}")
    test_image = load_image(test_image_url)
    print(f"Test image size: {test_image.size}")

    # I2I prompt
    i2i_prompt = "Turn her hair into flowing blue flames"

    # Load models
    print(f"\nLoading Pipeline on {pipeline_device}...")
    pipe = load_pipeline(model_path, pipeline_device)

    print(f"\nLoading Trainer on {trainer_device}...")
    trainer = load_trainer(config_path, trainer_device)

    # Preprocess image
    processed_image, width, height = preprocess_image(test_image, pipe)
    print(f"Processed image: {width} x {height}")

    # Compare image encoding (Step 1)
    encoding_match = compare_image_encoding(pipe, trainer, processed_image, pipeline_device, trainer_device)

    # Compare intermediate steps (Step 2)
    compare_intermediate_steps(pipe, trainer, test_image, i2i_prompt, pipeline_device, trainer_device)

    # Compare full I2I generation (Step 3)
    pipe_image, trainer_image = compare_full_i2i(pipe, trainer, test_image, i2i_prompt, pipeline_device, trainer_device)

    # Save comparison
    save_comparison(pipe_image, trainer_image)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()


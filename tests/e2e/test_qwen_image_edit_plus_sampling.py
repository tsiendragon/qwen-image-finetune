"""
Tests for Qwen Image Edit Plus Sampling

This module tests the qwen image edit plus sampling pipeline, verifying that:
1. The model can handle multiple input images
2. Sampling produces valid output images for multi-image editing tasks
3. The pipeline handles different image sizes and configurations
4. Intermediate embeddings match expected reference values
"""
import pytest
import torch
import logging
import numpy as np
import PIL.Image
import os
import random
from diffusers.utils import load_image
from qflux.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from qflux.data.config import load_config_from_yaml

logger = logging.getLogger(__name__)


def find_free_gpu(min_free_memory_gb=10):
    """
    Find a GPU with the most free memory.

    Args:
        min_free_memory_gb: Minimum free memory in GB required

    Returns:
        int: GPU device ID with most free memory, or 0 if no GPU meets requirements
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return None

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.warning("No GPU found, using CPU")
        return None

    max_free_memory = 0
    best_gpu = 0

    for gpu_id in range(num_gpus):
        try:
            torch.cuda.set_device(gpu_id)
            free_memory = torch.cuda.mem_get_info(gpu_id)[0] / (1024 ** 3)  # Convert to GB
            total_memory = torch.cuda.mem_get_info(gpu_id)[1] / (1024 ** 3)
            logger.info(f"GPU {gpu_id}: {free_memory:.2f}GB free / {total_memory:.2f}GB total")

            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = gpu_id
        except Exception as e:
            logger.warning(f"Error checking GPU {gpu_id}: {e}")
            continue

    if max_free_memory < min_free_memory_gb:
        logger.warning(
            f"No GPU with {min_free_memory_gb}GB free memory found. "
            f"Best GPU {best_gpu} has {max_free_memory:.2f}GB free"
        )

    logger.info(f"Selected GPU {best_gpu} with {max_free_memory:.2f}GB free memory")
    return best_gpu


def assert_ralative_error_tensor(x: torch.Tensor, y: torch.Tensor, rtol: float = 1e-5, key='tensor'):
    """Assert relative error between two tensors is below threshold"""
    if isinstance(x, PIL.Image.Image):
        x = np.array(x)
    if isinstance(y, PIL.Image.Image):
        y = np.array(y)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    x = x.float().detach().cpu()
    y = y.float().detach().cpu()
    assert x.shape == y.shape, f"Shape mismatch for {key}: {x.shape} != {y.shape}"
    relative_error = torch.norm(x - y) / (torch.norm(x) + torch.norm(y) + 1e-6)
    assert relative_error < rtol, (
        f"Relative error {relative_error} is greater than {rtol} for {key}\n"
        f"x min: {x.min()}, y min: {y.min()}\n"
        f"x max: {x.max()}, y max: {y.max()}"
    )
    logger.info(f"Shape for {key}: {x.shape}, {y.shape}")
    logger.info(f"Relative error {relative_error} is less than {rtol} for {key}")
    return relative_error


@pytest.fixture
def qwen_plus_trainer():
    """Fixture to create a QwenImageEditPlusTrainer instance with auto GPU selection"""
    config_path = "tests/test_configs/test_example_qwen_image_edit_plus_fp4.yaml"
    config = load_config_from_yaml(config_path)
    config.model.pretrained_model_name_or_path = 'Qwen/Qwen-Image-Edit-2509'
    # Do not load LoRA weights
    config.model.lora.pretrained_weight = None
    config.data.init_args.processor.init_args.process_type = "resize"
    config.data.init_args.processor.init_args.resize_mode = "bilinear"

    # Auto-select GPU with most free memory
    gpu_id = find_free_gpu(min_free_memory_gb=10)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
        logger.info(f"Using device: {device}")
        # Update config to use selected GPU
        config.predict.devices.vae = device
        config.predict.devices.text_encoder = device
        config.predict.devices.dit = device
    else:
        logger.warning("No suitable GPU found, using default config devices")

    trainer = QwenImageEditPlusTrainer(config)
    return trainer


@pytest.fixture(
    params=[torch.bfloat16],
    ids=["bfloat16"]
)
def weight_dtype(request):
    """Fixture to parametrize weight dtype: bfloat16"""
    return request.param


@pytest.fixture
def sample_images():
    """Fixture to provide sample input images"""
    img1_url = "https://thumbs.dreamstime.com/b/cartoon-city-street-vector-mayfair-district-london-town-panorama-illustration-horizontal-banner-advertising-empry-343357781.jpg"
    img2_url = 'https://pic.616pic.com/ys_bnew_img/00/48/11/TOuMQUisPT.jpg'
    image1 = load_image(img1_url).convert('RGB')
    image2 = load_image(img2_url).convert('RGB')

    return [image1, image2]


@pytest.fixture
def reference_embeddings(test_resources):
    """Fixture to load reference embeddings from resources"""
    data_path = test_resources / "qwen_plus_sampling" / "embeddings" / "qwen_plus_edit_plus_sampling_rst.pt"
    embeddings = torch.load(data_path, map_location="cpu")
    return embeddings


class TestQwenImageEditPlusSampling:
    """Test suite for Qwen Image Edit Plus sampling pipeline"""

    @pytest.mark.e2e
    def test_predict_generates_valid_image_multi_input(
        self, qwen_plus_trainer, sample_images, reference_embeddings, weight_dtype
    ):
        """Test that predict generates a valid output image with multiple input images"""
        # Arrange
        # 1) 固定随机种子（Python / NumPy / PyTorch CPU+GPU）
        prompt = "put the second character in the second image to the first image, where he walked in the street himself"
        logger.info(f"Testing with weight_dtype: {weight_dtype}")
        logger.info(f"Prompt: {prompt}")
        assert reference_embeddings['prompt'] == prompt, (
            f"Prompt should be {prompt}, but got {reference_embeddings['prompt']}"
        )

        num_inference_steps = 20
        true_cfg_scale = 4.5
        guidance_scale = 1.0
        negative_prompt = " "

        # Act: Setup trainer
        qwen_plus_trainer.weight_dtype = weight_dtype
        qwen_plus_trainer.setup_predict()

        # Prepare batch data with multiple images
        batch = qwen_plus_trainer.prepare_predict_batch_data(
            image=sample_images,  # List of images
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            controls_size=[[448, 800], [592, 592]],  # Auto-detect from images
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            weight_dtype=weight_dtype,
            height=448,  # Auto-detect
            width=800,  # Auto-detect
        )

        logger.info("Batch data prepared")

        # Prepare embeddings
        embeddings = qwen_plus_trainer.prepare_embeddings(batch, stage="predict", debug=True)

        target_height = embeddings["height"]
        target_width = embeddings["width"]
        logger.info(f"Target size: {target_height}x{target_width}")

        # Verify key embeddings exist
        assert "prompt_embeds" in embeddings, "Should have prompt_embeds"
        assert "control_latents" in embeddings, "Should have control_latents"
        assert "img_shapes" in embeddings, "Should have img_shapes"

        logger.info(f"Embeddings keys: {embeddings.keys()}")

        logger.info("Comparing img_shapes with reference...")
        img_shapes_ref = torch.Tensor(reference_embeddings['img_shapes'])
        img_shapes = torch.Tensor(embeddings['img_shapes'])
        # img_shapes = img_shapes / torch.tensor([[[3, 16, 16]]], device=img_shapes.device)
        print('img_shapes', img_shapes, 'img_shapes_ref', img_shapes_ref)
        assert_ralative_error_tensor(
            img_shapes,
            img_shapes_ref,
            key="img_shapes",
            rtol=1e-5
        )
        logger.info("Comparing condition_images with reference...")
        for i in range(len(embeddings["condition_images"])):
            print(
                f'condition_images {i} shape', embeddings["condition_images"][i].size,
                f'reference_condition_images {i} shape', reference_embeddings["condition_images"][i].size
            )
            assert_ralative_error_tensor(
                embeddings["condition_images"][i],
                reference_embeddings["condition_images"][i],
                key=f"condition_images_{i}",
                rtol=0.01,
            )

        # Compare with reference embeddings if available
        prompt_embeds_model_inputs = embeddings['prompt_embeds_model_inputs']
        prompt_embeds_model_inputs_ref = reference_embeddings['prompt_embeds_model_inputs']
        assert_ralative_error_tensor(
            prompt_embeds_model_inputs.input_ids,
            prompt_embeds_model_inputs_ref.input_ids,
            key="prompt_embeds_model_inputs.input_ids",
            rtol=0.01,
        )
        assert_ralative_error_tensor(
            prompt_embeds_model_inputs.attention_mask,
            prompt_embeds_model_inputs_ref.attention_mask,
            key="prompt_embeds_model_inputs.attention_mask",
            rtol=0.01,
        )
        assert_ralative_error_tensor(
            prompt_embeds_model_inputs.pixel_values,
            prompt_embeds_model_inputs_ref.pixel_values,
            key="prompt_embeds_model_inputs.pixel_values",
            rtol=0.01,
        )

        assert_ralative_error_tensor(
            prompt_embeds_model_inputs.image_grid_thw,
            prompt_embeds_model_inputs_ref.image_grid_thw,
            key="prompt_embeds_model_inputs.image_grid_thw",
            rtol=0.01,
        )
        model_inputs = {
            "input_ids": prompt_embeds_model_inputs.input_ids,
            "attention_mask": prompt_embeds_model_inputs.attention_mask,
            "pixel_values": prompt_embeds_model_inputs.pixel_values,
            "image_grid_thw": prompt_embeds_model_inputs.image_grid_thw,
        }

        # error 0.005462461616843939 for prompt_embeds_model_inputs.pixel_values
        # other is 0
        assert_ralative_error_tensor(
            embeddings["prompt_hidden_states"],
            reference_embeddings["prompt_hidden_states"],
            key="prompt_hidden_states",
            rtol=0.3,
        )
        assert_ralative_error_tensor(
            embeddings["negative_prompt_hidden_states"],
            reference_embeddings["negative_prompt_hidden_states"],
            key="negative_prompt_hidden_states",
            rtol=0.3,
        )

        assert_ralative_error_tensor(
            embeddings["prompt_embeds_mask"],
            reference_embeddings["prompt_embeds_mask"],
            key="prompt_embeds_mask",
            rtol=0.001,
        )
        logger.info("Comparing prompt_embeds with reference...")
        assert_ralative_error_tensor(
            embeddings["prompt_embeds"],
            reference_embeddings["prompt_embeds"],
            key="prompt_embeds",
            rtol=0.5,
        )

        logger.info("Comparing negative_prompt_embeds_mask with reference...")
        assert_ralative_error_tensor(
            embeddings["negative_prompt_embeds_mask"],
            reference_embeddings["negative_prompt_embeds_mask"],
            key="negative_prompt_embeds_mask",
            rtol=0.01,
        )
        logger.info("Comparing negative_prompt_embeds with reference...")
        assert_ralative_error_tensor(
            embeddings["negative_prompt_embeds"],
            reference_embeddings["negative_prompt_embeds"],
            key="negative_prompt_embeds",
            rtol=0.5,
        )

        logger.info("Compare vae image latents with reference...")
        vae_images1 = embeddings['control']
        vae_images2 = embeddings['control_1']
        ref_vae_images1 = reference_embeddings['vae_images'][0]
        ref_vae_images2 = reference_embeddings['vae_images'][1]
        assert_ralative_error_tensor(
            vae_images1,
            ref_vae_images1,
            key="vae_images1",
            rtol=0.02,
        )
        assert_ralative_error_tensor(
            vae_images2,
            ref_vae_images2,
            key="vae_images2",
            rtol=0.02,
        )

        logger.info("Compare control_latents with reference...")
        control_latents = embeddings['control_latents']
        ref_control_latents = reference_embeddings['image_latents']
        assert_ralative_error_tensor(
            control_latents,
            ref_control_latents,
            key="control_latents",
            rtol=0.04,
        )

        # Check guidance scale settings
        assert embeddings['true_cfg_scale'] == true_cfg_scale, (
            f"true_cfg_scale should be {true_cfg_scale}, but got {embeddings['true_cfg_scale']}"
        )
        assert embeddings.get('guidance') == guidance_scale, (
            f"guidance should be {guidance_scale}, but got {embeddings.get('guidance')}"
        )

        # Use reference latents if available, otherwise generate random
        if 'latents' in reference_embeddings:
            logger.info("Using reference latents for sampling")
            embeddings['latents'] = reference_embeddings['latents'].to(qwen_plus_trainer.dit.device)

        # Sample from embeddings
        logger.info("Starting sampling...")
        latents = qwen_plus_trainer.sampling_from_embeddings(embeddings)
        logger.info(f"Sampling complete. Latents shape: {latents.shape}")

        # Compare final latents with reference if available
        if 'final_latent' in reference_embeddings:
            logger.info("Comparing final latents with reference...")
            assert_ralative_error_tensor(
                latents,
                reference_embeddings["final_latent"],
                key="final_latent",
                rtol=0.1
            )

        # Decode VAE latent to image
        image = qwen_plus_trainer.decode_vae_latent(latents, target_height, target_width)

        # Assert: Validate output image
        assert image is not None, "Generated image should not be None"
        assert image.shape[0] == 1, "Batch size should be 1"
        assert image.shape[1] == 3, "Image should have 3 channels (RGB)"
        assert image.shape[2] == target_height, f"Image height should be {target_height}"
        assert image.shape[3] == target_width, f"Image width should be {target_width}"

        # Convert to numpy for comparison
        image_np = image.detach().permute(0, 2, 3, 1).float().cpu().numpy()
        image_np = image_np[0]
        image_np = (image_np * 255).round().astype("uint8")
        ref_image = reference_embeddings["image_out"]
        assert_ralative_error_tensor(
            image_np,
            ref_image,
            key="image_out",
            rtol=0.05,
        )

        logger.info(f"Output image range: [{image_np.min()}, {image_np.max()}]")
        logger.info("Test completed successfully!")


        # Cleanup
        del embeddings
        del latents
        torch.cuda.empty_cache()

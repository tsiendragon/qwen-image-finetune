"""
Tests for Qwen Image Edit Sampling

This module tests the qwen image edit sampling pipeline, verifying that:
1. The model can load pretrained LoRA weights
2. Sampling produces valid output images
3. The pipeline handles different image sizes and configurations
4. Intermediate embeddings match expected reference values
5. Pipeline weights match trainer component weights
"""
import pytest
import torch
import logging
import cv2
import PIL.Image
import numpy as np
from diffusers.utils import load_image
from diffusers import QwenImageEditPipeline
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
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


def assert_ralative_error_tensor(x: torch.Tensor, y: torch.Tensor, rtol: float = 1e-5, key='control'):
    x = x.float().detach().cpu()
    y = y.float().detach().cpu()
    assert x.shape == y.shape, f"Shape mismatch for {key}: {x.shape} != {y.shape}"
    relative_error = torch.norm(x - y) / (torch.norm(x) + torch.norm(y) + 1e-6)
    assert relative_error < rtol, (
        f"Relative error {relative_error} is greater than {rtol} for {key}"
        f"x min: {x.min()}, y min: {y.min()}"
        f"x max: {x.max()}, y max: {y.max()}"
    )
    print(f'shape for {key}', x.shape, y.shape)
    print(f"Relative error {relative_error} is less than {rtol} for {key}")
    return relative_error


@pytest.fixture
def qwen_trainer():
    """Fixture to create a QwenImageEditTrainer instance with auto GPU selection"""
    config_path = "tests/test_configs/test_example_qwen_image_edit_fp16.yaml"
    config = load_config_from_yaml(config_path)
    config.model.lora.pretrained_weight = "TsienDragon/qwen-image-edit-lora-face-segmentation"
    config.data.init_args.processor.init_args.process_type = "resize"
    config.data.init_args.processor.init_args.resize_mode = "bilinear"
    config.data.init_args.processor.init_args.target_size = [1248, 832]
    config.data.init_args.processor.init_args.controls_size = [[1248, 832]]

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

    trainer = QwenImageEditTrainer(config)
    return trainer


@pytest.fixture(
    params=[torch.bfloat16],
    ids=["bfloat16"]
)
def weight_dtype(request):
    """Fixture to parametrize weight dtype: bfloat16"""
    return request.param


@pytest.fixture
def sample_image():
    """Fixture to provide a sample input image"""
    image_url = "https://n.sinaimg.cn/ent/transform/775/w630h945/20201127/cee0-kentcvx8062290.jpg"
    return load_image(image_url).convert('RGB')


@pytest.fixture
def reference_embeddings(test_resources):
    """Fixture to load reference embeddings from resources"""
    data_path = test_resources / "qwen_sampling" / "embeddings" / "sampling_dict.pth"
    embeddings = torch.load(data_path, map_location="cpu")
    return embeddings


class TestQwenImageEditSampling:
    """Test suite for Qwen Image Edit sampling pipeline"""

    @pytest.mark.e2e
    def test_pipeline_weights_match_trainer_components(self, qwen_trainer):
        """Test that pipeline loaded weights match trainer component weights"""
        # Arrange: Load models and weights
        qwen_trainer.setup_predict()
        logger.info("Loading pipeline for weight comparison...")

        # Create pipeline with same configuration as trainer
        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
        )
        # Keep pipeline on CPU for weight comparison
        pipe.to("cpu")

        # Load LoRA weights (same as trainer)
        lora_path = "TsienDragon/qwen-image-edit-lora-face-segmentation"
        pipe.load_lora_weights(lora_path, adapter_name="lora_edit")
        logger.info(f"Loaded LoRA weights from {lora_path}")

        # Move trainer components to CPU for comparison
        qwen_trainer.vae.to("cpu")
        qwen_trainer.text_encoder.to("cpu")
        qwen_trainer.dit.to("cpu")

        # Act & Assert: Compare VAE weights
        logger.info("Comparing VAE weights...")
        vae_match_count = 0
        vae_total_count = 0
        for (name_pipe, param_pipe), (name_trainer, param_trainer) in zip(
            pipe.vae.named_parameters(), qwen_trainer.vae.named_parameters()
        ):
            assert name_pipe == name_trainer, f"VAE parameter name mismatch: {name_pipe} != {name_trainer}"
            assert param_pipe.shape == param_trainer.shape, (
                f"VAE parameter shape mismatch for {name_pipe}: "
                f"{param_pipe.shape} != {param_trainer.shape}"
            )
            if torch.allclose(param_pipe, param_trainer, rtol=1e-5, atol=1e-8):
                vae_match_count += 1
            vae_total_count += 1

        logger.info(f"VAE: {vae_match_count}/{vae_total_count} parameters match")
        assert vae_match_count == vae_total_count, (
            f"VAE weights mismatch: only {vae_match_count}/{vae_total_count} parameters match"
        )

        # Act & Assert: Compare text_encoder weights
        logger.info("Comparing text_encoder weights...")
        text_encoder_match_count = 0
        text_encoder_total_count = 0
        for (name_pipe, param_pipe), (name_trainer, param_trainer) in zip(
            pipe.text_encoder.named_parameters(), qwen_trainer.text_encoder.named_parameters()
        ):
            assert name_pipe == name_trainer, (
                f"text_encoder parameter name mismatch: {name_pipe} != {name_trainer}"
            )
            assert param_pipe.shape == param_trainer.shape, (
                f"text_encoder parameter shape mismatch for {name_pipe}: "
                f"{param_pipe.shape} != {param_trainer.shape}"
            )
            if torch.allclose(param_pipe, param_trainer, rtol=1e-5, atol=1e-8):
                text_encoder_match_count += 1
            text_encoder_total_count += 1

        logger.info(
            f"text_encoder: {text_encoder_match_count}/{text_encoder_total_count} parameters match"
        )
        assert text_encoder_match_count == text_encoder_total_count, (
            f"text_encoder weights mismatch: only "
            f"{text_encoder_match_count}/{text_encoder_total_count} parameters match"
        )

        # Act & Assert: Compare transformer (dit) weights
        logger.info("Comparing transformer (dit) weights...")
        dit_match_count = 0
        dit_total_count = 0
        for (name_pipe, param_pipe), (name_trainer, param_trainer) in zip(
            pipe.transformer.named_parameters(), qwen_trainer.dit.named_parameters()
        ):
            assert name_pipe == name_trainer, (
                f"transformer parameter name mismatch: {name_pipe} != {name_trainer}"
            )
            assert param_pipe.shape == param_trainer.shape, (
                f"transformer parameter shape mismatch for {name_pipe}: "
                f"{param_pipe.shape} != {param_trainer.shape}"
            )
            if torch.allclose(param_pipe, param_trainer, rtol=1e-5, atol=1e-8):
                dit_match_count += 1
            dit_total_count += 1

        logger.info(f"transformer: {dit_match_count}/{dit_total_count} parameters match")
        assert dit_match_count == dit_total_count, (
            f"transformer weights mismatch: only {dit_match_count}/{dit_total_count} parameters match"
        )

        logger.info("All component weights match successfully!")

        # Cleanup
        del pipe
        torch.cuda.empty_cache()

    @pytest.mark.e2e
    def test_predict_generates_valid_image(
        self, qwen_trainer, sample_image, reference_embeddings, weight_dtype
    ):
        """Test that predict generates a valid output image"""
        # Arrange
        prompt = "change the image from the face to the face segmentation mask"
        logger.info(f"Testing with weight_dtype: {weight_dtype}")

        num_inference_steps = 20
        height = 1248
        width = 832
        sample_image = sample_image.resize((width, height), resample=PIL.Image.LANCZOS)
        assert reference_embeddings['prompt'] == prompt, (
            f"Prompt should be {prompt}, but got {reference_embeddings['prompt']}"
        )

        # Act
        qwen_trainer.weight_dtype = weight_dtype
        qwen_trainer.setup_predict()
        batch = qwen_trainer.prepare_predict_batch_data(
            image=sample_image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            controls_size=[[height, width]],
            guidance_scale=None,
            true_cfg_scale=1.0,
            negative_prompt="",
            weight_dtype=weight_dtype,
            height=height,
            width=width,
        )
        print('prompt', prompt)
        embeddings = qwen_trainer.prepare_embeddings(batch, stage="predict")

        target_height = embeddings["height"]
        target_width = embeddings["width"]
        assert target_height == height, f"Target height should be {height}"
        assert target_width == width, f"Target width should be {width}"

        # Verify embeddings match reference
        # verify prompt control
        # processed_image = embeddings['control']
        processed_image = embeddings['prompt_control']  # ([1, 3, 1248, 832])
        processed_image = processed_image[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
        processed_image = torch.from_numpy(np.array(PIL.Image.fromarray(processed_image))).float()

        print('processed_image min', processed_image.min(), 'processed_image max', processed_image.max())
        print('processed_image', processed_image.shape)
        # processed_image = (processed_image*0.5 + 0.5) * 255.0
        reference_processed_image = torch.from_numpy(np.array(reference_embeddings['prompt_image'])).float()

        assert_ralative_error_tensor(
            processed_image,
            reference_processed_image,
            key="prompt_image",
            rtol=0.01
        )

        control_image = embeddings['control']
        control_image_ref = reference_embeddings['control_image']  # [(3, 1, 1248, 832)]
        control_image_ref = torch.from_numpy(control_image_ref).unsqueeze(0)  # [1,3,1,1248,832]
        print(
            'control_image shape', control_image.shape,
            'control_image_ref shape', control_image_ref.shape
        )
        assert control_image.shape == control_image_ref.shape, (
            f"Control image shape should be {control_image_ref.shape}"
        )
        assert_ralative_error_tensor(
            control_image,
            control_image_ref,
            key="control_image",
            rtol=0.01
        )

        assert_ralative_error_tensor(
            embeddings["control_latents"],
            reference_embeddings["image_latents"],
            key="control_latents",
            rtol=0.05
        )

        assert_ralative_error_tensor(
            embeddings["prompt_embeds_mask"],
            reference_embeddings["prompt_embeds_mask"],
            key="prompt_embeds_mask"
        )

        img_shapes_ref = reference_embeddings['img_shapes']
        img_shapes = embeddings['img_shapes']
        img_shapes_ref = torch.Tensor(img_shapes_ref)
        img_shapes = torch.Tensor(img_shapes)
        assert_ralative_error_tensor(
            img_shapes,
            img_shapes_ref,
            key="img_shapes",
            rtol=1e-5
        )
        assert embeddings['true_cfg_scale'] == 1.0, (
            f"true_cfg_scale should be 1.0, but got {embeddings['true_cfg_scale']}"
        )
        assert embeddings['guidance'] is None, (
            f"guidance should be None, but got {embeddings['guidance']}"
        )
        assert_ralative_error_tensor(
            embeddings["prompt_embeds"],
            reference_embeddings["prompt_embeds"],
            key="prompt_embeds",
            rtol=0.2,
        )
        embeddings['latents'] = reference_embeddings['latents'].to(qwen_trainer.dit.device)
        latents = qwen_trainer.sampling_from_embeddings(embeddings)
        print('latents shape', latents.shape)
        assert_ralative_error_tensor(
            latents,
            reference_embeddings["final_latent"],
            key="final_latent",
            rtol=0.05
        )
        image = qwen_trainer.decode_vae_latent(latents, target_height, target_width)
        # image = qwen_trainer.decode_vae_latent(reference_embeddings["final_latent"], target_height, target_width)

        # Assert
        assert image is not None, "Generated image should not be None"
        assert image.shape[0] == 1, "Batch size should be 1"
        assert image.shape[1] == 3, "Image should have 3 channels (RGB)"
        assert image.shape[2] == height, f"Image height should be {height}"
        assert image.shape[3] == width, f"Image width should be {width}"

        # Convert to PIL for optional saving
        image_np = image.detach().permute(0, 2, 3, 1).float().cpu().numpy()
        image_np = image_np[0]
        image_np = (image_np * 255).round().astype("uint8")

        ref_image = reference_embeddings["image_out"]
        assert ref_image.shape == image_np.shape, f"Reference image shape should be {image_np.shape}"
        print('image_np min', image_np.min(), 'image_np max', image_np.max())
        print('ref_image min', ref_image.min(), 'ref_image max', ref_image.max())
        image_np = image_np.astype(np.float32)
        ref_image = ref_image.astype(np.float32)
        rel_error = np.linalg.norm(image_np - ref_image) / (
            np.linalg.norm(image_np) + np.linalg.norm(ref_image) + 1e-6
        )
        # output_path = 'test_qwen_sampling_output.png'
        # cv2.imwrite(output_path, image_np[..., ::-1])
        # print('save image to test_qwen_sampling_output.png')
        assert rel_error < 0.02, f"Relative error {rel_error} is greater than 0.02"
        print(f"Relative error {rel_error} is less than 0.02")

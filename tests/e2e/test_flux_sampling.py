"""
Tests for Flux Kontext Sampling

This module tests the flux kontext sampling pipeline, verifying that:
1. The model can load pretrained LoRA weights
2. Sampling produces valid output images
3. The pipeline handles different image sizes and configurations
4. Intermediate embeddings match expected reference values
"""
import pytest
import torch
import logging
import cv2
import PIL.Image
import numpy as np
from diffusers.utils import load_image
from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml
logger = logging.getLogger(__name__)


def assert_ralative_error_tensor(x: torch.Tensor, y: torch.Tensor, rtol: float = 1e-5, key='control'):
    x = x.float().detach().cpu()
    y = y.float().detach().cpu()
    assert x.shape == y.shape, f"Shape mismatch for {key}: {x.shape} != {y.shape}"
    relative_error = torch.norm(x - y) / (torch.norm(x) + torch.norm(y)+1e-6)
    assert relative_error < rtol, (
        f"Relative error {relative_error} is greater than {rtol} for {key}"
        f"x min: {x.min()}, y min: {y.min()}"
        f"x max: {x.max()}, y max: {y.max()}"
    )
    print('shape for {key}', x.shape, y.shape)
    print(f"Relative error {relative_error} is less than {rtol} for {key}")
    return relative_error


@pytest.fixture
def flux_trainer():
    """Fixture to create a FluxKontextLoraTrainer instance"""
    config_path = "tests/test_configs/test_example_fluxkontext_fp16.yaml"
    config = load_config_from_yaml(config_path)
    config.model.lora.pretrained_weight = "TsienDragon/flux-kontext-face-segmentation"
    config.data.init_args.processor.init_args.process_type = "resize"
    config.data.init_args.processor.init_args.resize_mode = "bilinear"
    config.data.init_args.processor.init_args.target_size = [944, 624]
    config.data.init_args.processor.init_args.controls_size = [[944, 624]]
    trainer = FluxKontextLoraTrainer(config)
    return trainer


@pytest.fixture
def sample_image():
    """Fixture to provide a sample input image"""
    image_url = "https://n.sinaimg.cn/ent/transform/775/w630h945/20201127/cee0-kentcvx8062290.jpg"
    return load_image(image_url)


@pytest.fixture
def reference_embeddings(test_resources):
    """Fixture to load reference embeddings from resources"""
    data_path = test_resources / "flux_sampling" / "embeddings" / "sampling_dict.pth"
    embeddings = torch.load(data_path, map_location="cpu")
    # embeddings_dir = test_resources / "flux_sampling" / "embeddings"
    # images_dir = test_resources / "reference_outputs" / "images"
    # embeddings = {
    #     "control": torch.load(embeddings_dir / "sample_control_image.pt", map_location=torch.device("cpu"), weights_only=True),
    #     "latents": torch.load(embeddings_dir / "sample_latents.pt", map_location=torch.device("cpu"), weights_only=True),
    #     "latent_ids": torch.load(embeddings_dir / "sample_latent_ids.pt", map_location=torch.device("cpu"), weights_only=True),
    #     "control_latents": torch.load(embeddings_dir / "sample_control_latents.pt", map_location=torch.device("cpu"), weights_only=True),
    #     "control_ids": torch.load(embeddings_dir / "sample_control_ids.pt", map_location=torch.device("cpu"), weights_only=True),
    #     "text_ids": torch.load(embeddings_dir / "sample_text_ids.pt", map_location=torch.device("cpu"), weights_only=True),
    #     "prompt_embeds": torch.load(embeddings_dir / "sample_prompt_embeds.pt", map_location=torch.device("cpu"), weights_only=True),
    #     "pooled_prompt_embeds": torch.load(embeddings_dir / "sample_pooled_prompt_embeds.pt", map_location=torch.device("cpu"), weights_only=True),
    #     "output_image": cv2.imread(str(images_dir / "test_flux_kontext_output.png"))[..., ::-1],
    # }
    return embeddings


class TestFluxSampling:
    """Test suite for Flux Kontext sampling pipeline"""

    @pytest.mark.e2e
    def test_predict_generates_valid_image(self, flux_trainer, sample_image, reference_embeddings):
        """Test that predict generates a valid output image"""
        # Arrange
        prompt = "change the image from the face to the face segmentation mask"
        num_inference_steps = 20
        height = 944
        width = 624
        sample_image = sample_image.resize((width, height), resample=PIL.Image.LANCZOS)

        # Act
        flux_trainer.setup_predict()
        batch = flux_trainer.prepare_predict_batch_data(
            image=sample_image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            controls_size=[[height, width]],
            guidance_scale=3.5,
            true_cfg_scale=1.0,
            negative_prompt="",
            weight_dtype=torch.bfloat16,
            height=height,
            width=width,
        )
        print('prompt', prompt)
        embeddings = flux_trainer.prepare_embeddings(batch, stage="predict")
        target_height = embeddings["height"]
        target_width = embeddings["width"]
        assert target_height == height, f"Target height should be {height}"
        assert target_width == width, f"Target width should be {width}"
        # Note: Uncomment below to save control embeddings for debugging
        # torch.save(embeddings["control"], test_resources / "flux_sampling" / "embeddings" / "aaa_sample_control_image.pt")
        assert_ralative_error_tensor(embeddings["pooled_prompt_embeds"], reference_embeddings["pooled_prompt_embeds"], key="pooled_prompt_embeds")
        assert_ralative_error_tensor(embeddings["prompt_embeds"], reference_embeddings["prompt_embeds"], key="prompt_embeds")
        assert_ralative_error_tensor(embeddings["text_ids"], reference_embeddings["text_ids"], key="text_ids")
        assert_ralative_error_tensor(embeddings["control_ids"], reference_embeddings["image_ids"], key="control_ids")
        assert_ralative_error_tensor(embeddings["control"], reference_embeddings["control_image"], key="control", rtol=0.005)
        assert_ralative_error_tensor(embeddings["control_latents"], reference_embeddings["image_latents"], key="control_latents", rtol=0.05)

        embeddings['latents'] = reference_embeddings['latents'].to(flux_trainer.dit.device)
        embeddings['latent_ids'] = reference_embeddings['latent_ids'].to(flux_trainer.dit.device)
        embeddings['guidance'] = reference_embeddings['guidance'].item()
        embeddings['control_latents'] = reference_embeddings['image_latents']

        latents = flux_trainer.sampling_from_embeddings(embeddings)
        print('latents shape', latents.shape)
        assert_ralative_error_tensor(latents, reference_embeddings["final_latents"], key="final_latent", rtol=0.05)
        image = flux_trainer.decode_vae_latent(latents, target_height, target_width)

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

        ref_image = reference_embeddings["image"]
        assert ref_image.shape == image_np.shape, f"Reference image shape should be {image_np.shape}"
        print('image_np min', image_np.min(), 'image_np max', image_np.max())
        print('ref_image min', ref_image.min(), 'ref_image max', ref_image.max())
        rel_error = np.linalg.norm(image_np - ref_image) / (np.linalg.norm(image_np) + np.linalg.norm(ref_image)+1e-6)
        print(f"Relative error {rel_error} is less than 1e-4")
        # cv2.imwrite('test_sampling_output.png', image_np[..., ::-1])
        # print('save image to test_sampling_output.png')
        # Note: Uncomment below to save output image for debugging
        # cv2.imwrite(str(test_resources / "reference_outputs" / "images" / "test_flux_kontext_output_unit_test.png"), image_np[..., ::-1])
        assert rel_error < 0.01, f"Relative error {rel_error} is greater than 0.01"

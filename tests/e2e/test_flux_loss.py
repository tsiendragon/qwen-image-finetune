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
from pathlib import Path
from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml
from qflux.utils.tools import pad_latents_for_multi_res

logger = logging.getLogger(__name__)

RESOURCES_DIR = Path(__file__).parent.parent / "resources"


def load_sample_data(sample_dir: Path):
    """Load all sample data from a directory

    The sample data contains pre-computed tensors from a training run:
    - image_latents: Target image latents [B, T, C]
    - control_latents: Control image latents [B, T, C]
    - noise: Random noise added during training [B, T, C]
    - model_pred: Model's prediction output [B, T, C]
    - expected_loss: The loss value that should be computed
    - prompt_embeds, pooled_prompt_embeds, text_ids, control_ids: Additional embeddings
    """
    data = {
        "control_ids": torch.load(sample_dir / "sample_control_ids.pt", map_location="cpu", weights_only=True),
        "control_latents": torch.load(sample_dir / "sample_control_latents.pt", map_location="cpu", weights_only=True),
        "image_latents": torch.load(sample_dir / "sample_image_latents.pt", map_location="cpu", weights_only=True),
        "noise": torch.load(sample_dir / "sample_noise.pt", map_location="cpu", weights_only=True),
        "pooled_prompt_embeds": torch.load(sample_dir / "sample_pooled_prompt_embeds.pt", map_location="cpu", weights_only=True),
        "prompt_embeds": torch.load(sample_dir / "sample_prompt_embeds.pt", map_location="cpu", weights_only=True),
        "text_ids": torch.load(sample_dir / "sample_text_ids.pt", map_location="cpu", weights_only=True),
        "model_pred": torch.load(sample_dir / "sample_model_pred.pt", map_location="cpu", weights_only=True),
        "expected_loss": torch.load(sample_dir / "sample_loss.pt", map_location="cpu", weights_only=True),
        "latent_model_input": torch.load(sample_dir / "sample_latent_model_input.pt", map_location="cpu", weights_only=True),
        "t": torch.load(sample_dir / "sample_t.pt", map_location="cpu", weights_only=True),
        "latent_ids": torch.load(sample_dir / "sample_latent_ids.pt", map_location="cpu", weights_only=True),
        "guidance": torch.load(sample_dir / "sample_guidance.pt", map_location="cpu", weights_only=True),
    }
    return data


@pytest.fixture
def sample_data_1():
    """Fixture to load first sample data"""
    return load_sample_data(RESOURCES_DIR / "flux_training_face_seg_sample1")


@pytest.fixture
def sample_data_2():
    """Fixture to load second sample data"""
    return load_sample_data(RESOURCES_DIR / "flux_training_face_seg_sample2")


class MockAccelerator:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.is_main_process = True


@pytest.fixture(scope="class")
def mock_trainer():
    """Create a mock trainer with minimal setup for loss computation"""
    config_path = "tests/test_configs/test_example_fluxkontext_fp16_faceseg_multiresolution.yaml"
    # config_path = "tests/test_configs/test_example_fluxkontext_fp16.yaml"
    print(f"Loading config from {config_path}")
    config = load_config_from_yaml(config_path)

    config.model.lora.pretrained_weight = "TsienDragon/flux-kontext-face-segmentation"
    trainer = FluxKontextLoraTrainer(config)
    trainer.load_model()
    FluxKontextLoraTrainer.load_pretrain_lora_model(trainer.dit, config, config.model.lora.adapter_name)
    trainer.accelerator = MockAccelerator(device="cuda:0")
    trainer.setup_model_device_train_mode(stage="fit", cache=True)
    trainer.configure_optimizers()
    trainer.setup_criterion()
    trainer.weight_dtype = torch.bfloat16
    print('trainer.accelerator.device', trainer.accelerator.device)
    return trainer


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


class TestFluxKontextLossComputation:
    """Test suite for Flux Kontext loss computation

    These tests verify that the loss computation produces expected results
    given pre-computed model predictions and targets.

    The test approach:
    1. Load pre-saved model_pred, image_latents, and noise from training run
    2. Compute target = noise - image_latents (flow matching target)
    3. Call forward_loss(model_pred, target) to get loss
    4. Compare with expected_loss from the training run
    """

    def test_forward_loss_sample1(self, mock_trainer, sample_data_1):
        return
        """Test forward_loss computation with sample 1

        This test verifies that given the same model_pred and target,
        the loss function produces the same loss value.
        """
        # Arrange
        data = sample_data_1
        device = mock_trainer.accelerator.device
        dtype = mock_trainer.weight_dtype

        # 640 928
        # 928, 640
        # 832, 576
        height, width = 832, 576

        # Load pre-computed tensors
        # model_pred = data["model_pred"].to(device).to(dtype)
        image_latents = data["image_latents"].to(device).to(dtype)
        text_ids = data["text_ids"].to(device).to(dtype)
        control_latents = data["control_latents"].to(device).to(dtype)
        control_ids = data["control_ids"].to(device).to(dtype)
        pooled_prompt_embeds = data["pooled_prompt_embeds"].to(device).to(dtype)
        prompt_embeds = data["prompt_embeds"].to(device).to(dtype)
        noise = data["noise"].to(device).to(dtype)
        model_pred = data["model_pred"].to(device).to(dtype)

        expected_loss = data["expected_loss"].to(device)
        time_steps = torch.Tensor([0.7109, 0.1611]).to(device).to(dtype)
        latent_model_input = data["latent_model_input"].to(device).to(dtype)
        t = data["t"].to(device).to(dtype)
        latent_ids = data["latent_ids"].to(device).to(dtype)
        guidance = data["guidance"].to(device).to(dtype)
        print(t, guidance)

        # Compute target using flow matching formula
        # target = noise - image_latents
        embeddings = {
            "image_latents": image_latents,
            "text_ids": text_ids,
            "control_latents": control_latents,
            "control_ids": control_ids,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_embeds": prompt_embeds,
            "image": torch.rand(2, 3, height, width),
            "noise": noise,
            "timestep": time_steps,
        }

        # Act
        with torch.no_grad():
            mock_trainer.dit.eval()
            model_pred_inference, model_pred_loss = mock_trainer._compute_loss_shared_mode(
                embeddings,
                return_pred=True
            )
            model_pred2 = mock_trainer.dit(
                hidden_states=latent_model_input,
                timestep=t,
                guidance=guidance,  # must pass to guidance for FluxKontextDev
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            model_pred2 = model_pred2[:, :image_latents.shape[1]]

        assert abs(model_pred_loss - expected_loss) < 1e-2, f"Loss mismatch for sample 1: {model_pred_loss} != {expected_loss}"
        print(f"Loss matched for sample 1: {model_pred_loss} ~= {expected_loss}")
        assert_ralative_error_tensor(model_pred_inference, model_pred, rtol=1e-1, key='model_pred_inference')
        assert_ralative_error_tensor(model_pred2, model_pred_inference, rtol=1e-1, key='model_pred2')
        print(f"Model pred matched for sample 1: {model_pred.shape} ~= {model_pred.shape}")
        target = noise - image_latents
        # calculate mse loss
        loss2 = mock_trainer.forward_loss(model_pred, target)
        print(f"Loss2: {loss2}")
        mse_loss = torch.nn.functional.mse_loss(model_pred, target)
        assert abs(mse_loss - expected_loss) < 1e-4, f"MSE loss mismatch for sample 1: {mse_loss} != {expected_loss}"
        print(f"MSE loss matched for sample 1: {mse_loss} ~= {expected_loss}")

    def test_forward_loss_sample2(self, mock_trainer, sample_data_2):
        return
        """Test forward_loss computation with sample 2

        This test verifies that given the same model_pred and target,
        the loss function produces the same loss value.
        """
        # Arrange
        data = sample_data_2
        device = mock_trainer.accelerator.device
        dtype = mock_trainer.weight_dtype

        height, width = 928, 640

        # Load pre-computed tensors
        # model_pred = data["model_pred"].to(device).to(dtype)
        image_latents = data["image_latents"].to(device).to(dtype)
        text_ids = data["text_ids"].to(device).to(dtype)
        control_latents = data["control_latents"].to(device).to(dtype)
        control_ids = data["control_ids"].to(device).to(dtype)
        pooled_prompt_embeds = data["pooled_prompt_embeds"].to(device).to(dtype)
        prompt_embeds = data["prompt_embeds"].to(device).to(dtype)
        noise = data["noise"].to(device).to(dtype)
        model_pred = data["model_pred"].to(device).to(dtype)

        expected_loss = data["expected_loss"].to(device)
        time_steps = torch.Tensor([0.7109, 0.1611]).to(device).to(dtype)
        latent_model_input = data["latent_model_input"].to(device).to(dtype)
        t = data["t"].to(device).to(dtype)
        latent_ids = data["latent_ids"].to(device).to(dtype)
        guidance = data["guidance"].to(device).to(dtype)
        print(t, guidance)

        # Compute target using flow matching formula
        # target = noise - image_latents
        embeddings = {
            "image_latents": image_latents,
            "text_ids": text_ids,
            "control_latents": control_latents,
            "control_ids": control_ids,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_embeds": prompt_embeds,
            "image": torch.rand(2, 3, height, width),
            "noise": noise,
            "timestep": time_steps,
        }

        # Act
       # Act
        with torch.no_grad():
            mock_trainer.dit.eval()
            model_pred_inference, model_pred_loss = mock_trainer._compute_loss_shared_mode(
                embeddings,
                return_pred=True
            )
            model_pred2 = mock_trainer.dit(
                hidden_states=latent_model_input,
                timestep=t,
                guidance=guidance,  # must pass to guidance for FluxKontextDev
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            model_pred2 = model_pred2[:, :image_latents.shape[1]]

        assert abs(model_pred_loss - expected_loss) < 1e-2, f"Loss mismatch for sample 1: {model_pred_loss} != {expected_loss}"
        print(f"Loss matched for sample 1: {model_pred_loss} ~= {expected_loss}, error {model_pred_loss - expected_loss}")
        assert_ralative_error_tensor(model_pred_inference, model_pred, rtol=1e-1, key='model_pred_inference')
        assert_ralative_error_tensor(model_pred2, model_pred_inference, rtol=1e-1, key='model_pred2')
        print(f"Model pred matched for sample 1: {model_pred.shape} ~= {model_pred.shape}")
        target = noise - image_latents
        # calculate mse loss
        loss2 = mock_trainer.forward_loss(model_pred, target)
        print(f"Loss2: {loss2}")
        mse_loss = torch.nn.functional.mse_loss(model_pred, target)
        assert abs(mse_loss - expected_loss) < 1e-4, f"MSE loss mismatch for sample 1: {mse_loss} != {expected_loss}"
        print(f"MSE loss matched for sample 1: {mse_loss} ~= {expected_loss}, error {mse_loss - expected_loss}")

    def test_compute_loss_multi_resolution_mode(self, mock_trainer, sample_data_1, sample_data_2):
        device = mock_trainer.accelerator.device
        dtype = mock_trainer.weight_dtype

        model_preds = []
        model_losses = []
        data = sample_data_2
        height, width = 928, 640
        # Load pre-computed tensors
        # model_pred = data["model_pred"].to(device).to(dtype)
        image_latents = data["image_latents"].to(device).to(dtype)
        text_ids = data["text_ids"].to(device).to(dtype)
        control_latents = data["control_latents"].to(device).to(dtype)
        control_ids = data["control_ids"].to(device).to(dtype)
        pooled_prompt_embeds = data["pooled_prompt_embeds"].to(device).to(dtype)
        prompt_embeds = data["prompt_embeds"].to(device).to(dtype)
        noise = data["noise"].to(device).to(dtype)
        time_steps = torch.Tensor([0.7109, 0.1611]).to(device).to(dtype)

        # Compute target using flow matching formula
        # target = noise - image_latents
        embeddings = {
            "image_latents": image_latents,
            "text_ids": text_ids,
            "control_latents": control_latents,
            "control_ids": control_ids,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_embeds": prompt_embeds,
            "image": torch.rand(2, 3, height, width),
            "noise": noise,
            "timestep": time_steps,
        }

        # Act
        # Act
        with torch.no_grad():
            mock_trainer.dit.eval()
            model_pred_inference, model_pred_loss = mock_trainer._compute_loss_shared_mode(
                embeddings,
                return_pred=True
            )
        model_pred_inference = [xi for xi in model_pred_inference]
        model_preds += model_pred_inference
        model_losses.append(model_pred_loss.item())

        data = sample_data_1
        height, width = 832, 576
        image_latents = data["image_latents"].to(device).to(dtype)
        text_ids = data["text_ids"].to(device).to(dtype)
        control_latents = data["control_latents"].to(device).to(dtype)
        control_ids = data["control_ids"].to(device).to(dtype)
        pooled_prompt_embeds = data["pooled_prompt_embeds"].to(device).to(dtype)
        prompt_embeds = data["prompt_embeds"].to(device).to(dtype)
        noise = data["noise"].to(device).to(dtype)
        time_steps = torch.Tensor([0.7109, 0.1611]).to(device).to(dtype)
        embeddings = {
            "image_latents": image_latents,
            "text_ids": text_ids,
            "control_latents": control_latents,
            "control_ids": control_ids,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_embeds": prompt_embeds,
            "image": torch.rand(2, 3, height, width),
            "noise": noise,
            "timestep": time_steps,
        }
        print('image_latents', image_latents.shape, 'noise', noise.shape, 'control_latents', control_latents.shape)

        # Act
        with torch.no_grad():
            mock_trainer.dit.eval()
            model_pred_inference, model_pred_loss = mock_trainer._compute_loss_shared_mode(
                embeddings,
                return_pred=True
            )
        model_pred_inference = [xi for xi in model_pred_inference]
        for x in model_pred_inference:
            print('model_pred_inference shape', x.shape)
        model_preds += model_pred_inference
        model_losses.append(model_pred_loss.item())

        img_shapes = [
            [(3, 832, 576), (3, 832, 576)],
            [(3, 832, 576), (3, 832, 576)],
            [(3, 928, 640), (3, 928, 640)],
            [(3, 928, 640), (3, 928, 640)],
        ]

        print(sample_data_1['text_ids'].shape, sample_data_1['control_latents'].shape)
        control_latents = [xi for xi in sample_data_1['control_latents']]
        control_latents += [xi for xi in sample_data_2['control_latents']]
        for xi in control_latents:
            print('control_latents shape', xi.shape)
        control_latents, _ = pad_latents_for_multi_res(control_latents, max_seq_len=None)
        print('control_latents', control_latents[0].shape, control_latents[1].shape)
        print('control_latents', control_latents.shape)

        image_latents = [xi for xi in sample_data_1['image_latents']]
        image_latents += [xi for xi in sample_data_2['image_latents']]
        image_latents, _ = pad_latents_for_multi_res(image_latents, max_seq_len=None)
        print('image_latents', image_latents.shape, image_latents.shape)

        pooled_prompt_embeds = [xi for xi in sample_data_1['pooled_prompt_embeds']]
        pooled_prompt_embeds += [xi for xi in sample_data_2['pooled_prompt_embeds']]
        pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
        print('pooled_prompt_embeds', pooled_prompt_embeds.shape)

        prompt_embeds = [xi for xi in sample_data_1['prompt_embeds']]
        prompt_embeds += [xi for xi in sample_data_2['prompt_embeds']]
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        print('prompt_embeds', prompt_embeds.shape)

        noise = [xi for xi in sample_data_1['noise']]
        noise += [xi for xi in sample_data_2['noise']]
        for i, xi in enumerate(noise):
            print('noise shape', i, xi.shape)

        # time_steps = [xi for xi in sample_data_1['t']]
        # time_steps += [xi for xi in sample_data_2['t']]
        print('timestep', sample_data_1['t'], sample_data_2['t'])
        time_steps = torch.Tensor([0.7109, 0.1611, 0.7109, 0.1611]).to(device).to(dtype)
        time_steps = time_steps.reshape(4, 1)

        embeddings = {
            "text_ids": sample_data_1['text_ids'],
            "control_latents": control_latents,
            "image_latents": image_latents,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_embeds": prompt_embeds,
            "img_shapes": img_shapes,
            "noise": noise,
            "timestep": time_steps,
        }

        with torch.inference_mode():
            mock_trainer.dit.eval()
            model_pred_inference, loss,  = mock_trainer._compute_loss_multi_resolution_mode(embeddings, return_pred=True)

        print('model_pred', model_pred_inference.shape, model_pred_inference.shape)
        # model pred shape [4, 2320, 64]
        assert model_pred_inference.shape[0] == 4, f"Model pred shape mismatch: {model_pred_inference.shape[0]} != 4"
        print('loss', loss.item(), 'expected loss', model_losses)
        for i in range(4):
            x =  model_preds[i]
            y = model_pred_inference[i]
            print('lenx', len(x), len(y))
            print('model_pred', i, x.shape, y.shape)
            assert_ralative_error_tensor(x, y[:x.shape[0]], rtol=1e-2, key='model_pred')

        print('model_losses', model_losses)
        model_losses = sum(model_losses)/2
        assert abs(loss.item() - model_losses) < 1e-4, f"Loss mismatch: {loss.item()} != {model_losses}"
        print('Loss matched: {loss.item()} ~= {model_losses}')

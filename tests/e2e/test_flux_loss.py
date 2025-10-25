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
from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml
from qflux.utils.tools import pad_latents_for_multi_res
from tests.utils.data_loader import load_flux_training_sample

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_data_1(test_resources):
    """Fixture to load first sample data"""
    return load_flux_training_sample(test_resources, "sample1")


@pytest.fixture
def sample_data_2(test_resources):
    """Fixture to load second sample data"""
    return load_flux_training_sample(test_resources, "sample2")


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
    print('criterion', trainer.criterion)
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
    print(f'shape for {key}: {x.shape}, {y.shape}')
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
        image_latents = data["sample_image_latents"].to(device).to(dtype)
        text_ids = data["sample_text_ids"].to(device).to(dtype)
        control_latents = data["sample_control_latents"].to(device).to(dtype)
        control_ids = data["sample_control_ids"].to(device).to(dtype)
        pooled_prompt_embeds = data["sample_pooled_prompt_embeds"].to(device).to(dtype)
        prompt_embeds = data["sample_prompt_embeds"].to(device).to(dtype)
        noise = data["sample_noise"].to(device).to(dtype)
        model_pred = data["sample_model_pred"].to(device).to(dtype)

        expected_loss = data["sample_loss"].to(device)
        time_steps = torch.Tensor([0.7109, 0.1611]).to(device).to(dtype)
        latent_model_input = data["sample_latent_model_input"].to(device).to(dtype)
        t = data["sample_t"].to(device).to(dtype)
        latent_ids = data["sample_latent_ids"].to(device).to(dtype)
        guidance = data["sample_guidance"].to(device).to(dtype)
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
                embeddings, return_pred=True
            )
            model_pred2 = mock_trainer.dit(
                hidden_states=latent_model_input,
                timestep=t,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            model_pred2 = model_pred2[:, :image_latents.shape[1]]

        assert abs(model_pred_loss - expected_loss) < 1e-2, (
            f"Loss mismatch for sample 1: {model_pred_loss} != {expected_loss}"
        )
        print(f"Loss matched for sample 1: {model_pred_loss} ~= {expected_loss}")
        assert_ralative_error_tensor(
            model_pred_inference, model_pred, rtol=1e-1, key='model_pred_inference'
        )
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
        image_latents = data["sample_image_latents"].to(device).to(dtype)
        text_ids = data["sample_text_ids"].to(device).to(dtype)
        control_latents = data["sample_control_latents"].to(device).to(dtype)
        control_ids = data["sample_control_ids"].to(device).to(dtype)
        pooled_prompt_embeds = data["sample_pooled_prompt_embeds"].to(device).to(dtype)
        prompt_embeds = data["sample_prompt_embeds"].to(device).to(dtype)
        noise = data["sample_noise"].to(device).to(dtype)
        model_pred = data["sample_model_pred"].to(device).to(dtype)

        expected_loss = data["sample_loss"].to(device)
        time_steps = torch.Tensor([0.7109, 0.1611]).to(device).to(dtype)
        latent_model_input = data["sample_latent_model_input"].to(device).to(dtype)
        t = data["sample_t"].to(device).to(dtype)
        latent_ids = data["sample_latent_ids"].to(device).to(dtype)
        guidance = data["sample_guidance"].to(device).to(dtype)
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
                embeddings, return_pred=True
            )
            model_pred2 = mock_trainer.dit(
                hidden_states=latent_model_input,
                timestep=t,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]
            model_pred2 = model_pred2[:, :image_latents.shape[1]]

        assert abs(model_pred_loss - expected_loss) < 1e-2, (
            f"Loss mismatch for sample 1: {model_pred_loss} != {expected_loss}"
        )
        print(
            f"Loss matched for sample 1: {model_pred_loss} ~= {expected_loss}, "
            f"error {model_pred_loss - expected_loss}"
        )
        assert_ralative_error_tensor(
            model_pred_inference, model_pred, rtol=1e-1, key='model_pred_inference'
        )
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

        data = sample_data_1
        height, width = 832, 576
        image_latents = data["sample_image_latents"].to(device).to(dtype)
        text_ids = data["sample_text_ids"].to(device).to(dtype)
        control_latents = data["sample_control_latents"].to(device).to(dtype)
        control_ids = data["sample_control_ids"].to(device).to(dtype)
        pooled_prompt_embeds = data["sample_pooled_prompt_embeds"].to(device).to(dtype)
        prompt_embeds = data["sample_prompt_embeds"].to(device).to(dtype)
        noise = data["sample_noise"].to(device).to(dtype)
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

        data = sample_data_2
        height, width = 928, 640
        # Load pre-computed tensors
        image_latents = data["sample_image_latents"].to(device).to(dtype)
        text_ids = data["sample_text_ids"].to(device).to(dtype)
        control_latents = data["sample_control_latents"].to(device).to(dtype)
        control_ids = data["sample_control_ids"].to(device).to(dtype)
        pooled_prompt_embeds = data["sample_pooled_prompt_embeds"].to(device).to(dtype)
        prompt_embeds = data["sample_prompt_embeds"].to(device).to(dtype)
        noise = data["sample_noise"].to(device).to(dtype)
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

        img_shapes = [
            [(3, 832, 576), (3, 832, 576)],
            [(3, 832, 576), (3, 832, 576)],
            [(3, 928, 640), (3, 928, 640)],
            [(3, 928, 640), (3, 928, 640)],
        ]

        print(sample_data_1['sample_text_ids'].shape, sample_data_1['sample_control_latents'].shape)
        control_latents = [xi for xi in sample_data_1['sample_control_latents']]
        control_latents += [xi for xi in sample_data_2['sample_control_latents']]
        for xi in control_latents:
            print('control_latents shape', xi.shape)
        control_latents, _ = pad_latents_for_multi_res(control_latents, max_seq_len=None)
        print('control_latents', control_latents[0].shape, control_latents[1].shape)
        print('control_latents', control_latents.shape)

        image_latents = [xi for xi in sample_data_1['sample_image_latents']]
        image_latents += [xi for xi in sample_data_2['sample_image_latents']]
        image_latents, _ = pad_latents_for_multi_res(image_latents, max_seq_len=None)
        print('image_latents', image_latents.shape, image_latents.shape)

        pooled_prompt_embeds = [xi for xi in sample_data_1['sample_pooled_prompt_embeds']]
        pooled_prompt_embeds += [xi for xi in sample_data_2['sample_pooled_prompt_embeds']]
        pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
        print('pooled_prompt_embeds', pooled_prompt_embeds.shape)

        prompt_embeds = [xi for xi in sample_data_1['sample_prompt_embeds']]
        prompt_embeds += [xi for xi in sample_data_2['sample_prompt_embeds']]
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        print('prompt_embeds', prompt_embeds.shape)

        noise = [xi for xi in sample_data_1['sample_noise']]
        noise += [xi for xi in sample_data_2['sample_noise']]
        for i, xi in enumerate(noise):
            print('noise shape', i, xi.shape)
        print('timestep', sample_data_1['sample_t'], sample_data_2['sample_t'])
        time_steps = torch.Tensor([0.7109, 0.1611, 0.7109, 0.1611]).to(device).to(dtype)
        time_steps = time_steps.reshape(4, 1)

        embeddings = {
            "text_ids": sample_data_1['sample_text_ids'],
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
            loss, model_pred_inference_dict = mock_trainer._compute_loss_multi_resolution_mode(
                embeddings, return_pred=True
            )
        model_pred_inference = model_pred_inference_dict['model_pred']

        # Debug: 手动计算 loss 来验证
        print('\n=== Debug Multi-Resolution Loss ===')
        print('model_pred shape:', model_pred_inference.shape)

        # 手动计算每个样本的 loss
        manual_token_losses = []
        manual_valid_tokens = 0

        for i in range(4):
            if i < 2:
                seq_len = 1872
                noise_i = noise[i][:seq_len]
                image_latents_i = image_latents[i, :seq_len, :]
            else:
                seq_len = 2320
                noise_i = noise[i][:seq_len]
                image_latents_i = image_latents[i, :seq_len, :]

            pred_i = model_pred_inference[i, :seq_len, :].float()
            target_i = (noise_i - image_latents_i).float()

            # 计算 MSE
            token_loss_i = ((pred_i.float().cpu() - target_i.float().cpu()) ** 2).mean(dim=-1)  # [seq_len]
            manual_token_losses.append(token_loss_i.sum().item())
            manual_valid_tokens += seq_len

            print(f'Sample {i}: valid_tokens={seq_len}, token_loss_sum={token_loss_i.sum().item():.4f}')

        manual_loss = sum(manual_token_losses) / manual_valid_tokens
        print('\nManual calculation:')
        print(f'  Total valid tokens: {manual_valid_tokens}')
        print(f'  Total token loss sum: {sum(manual_token_losses):.4f}')
        print(f'  Manual loss: {manual_loss:.6f}')
        print(f'  Trainer loss: {loss.item():.6f}')
        print(f'  Difference: {abs(manual_loss - loss.item()):.6f}')
        print('===================================\n')

        print('model_pred', model_pred_inference.shape, model_pred_inference.shape)
        # model pred shape [4, 2320, 64]
        assert model_pred_inference.shape[0] == 4, f"Model pred shape mismatch: {model_pred_inference.shape[0]} != 4"
        print('loss', loss.item(), 'expected loss', model_losses)
        for i in range(4):
            x = model_preds[i]
            y = model_pred_inference[i]
            print('lenx', len(x), len(y))
            print('model_pred', i, x.shape, y.shape)
            assert_ralative_error_tensor(x, y[:x.shape[0]], rtol=0.01, key='model_pred')

        print('model_losses', model_losses)
        model_losses = sum(model_losses)/2
        assert abs(loss.item() - model_losses) < 0.001, f"Loss mismatch: {loss.item()} != {model_losses}"
        print(f'Loss matched: {loss.item()} ~= {model_losses}')

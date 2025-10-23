"""
Test padding equivalence between original and custom FluxTransformer2DModel.
passed: 2025-10-22 10:00:00
"""
import torch
from qflux.models.transformer_flux_custom import FluxTransformer2DModel


def _prepare_latent_image_ids(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
    return ids.view(-1, 3)


def _build_model() -> FluxTransformer2DModel:
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=64,
        out_channels=64,
        num_layers=1,
        num_single_layers=0,
        attention_head_dim=64,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=16,
        guidance_embeds=False,
        axes_dims_rope=(8, 28, 28),
    )


def test_padding_equivalence_single_sample():
    torch.manual_seed(0)
    device = torch.device("cpu")

    model = _build_model().to(device)
    model.eval()

    batch_size = 1
    seq_txt = 4
    height, width = 2, 4  # seq_img = 8
    seq_img = height * width
    pad_tokens = 4

    hidden_states = torch.randn(batch_size, seq_img, 64, device=device)
    encoder_hidden_states = torch.randn(batch_size, seq_txt, 32, device=device)
    pooled = torch.randn(batch_size, 16, device=device)
    timestep = torch.randint(0, 1000, (batch_size,), device=device)
    img_ids = _prepare_latent_image_ids(height, width, device=device, dtype=torch.float32)
    txt_ids = torch.stack(
        [
            torch.tensor([0.0, float(i), 0.0], device=device)
            for i in range(seq_txt)
        ],
        dim=0,
    )

    # Baseline without padding
    with torch.no_grad():
        out_base = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample

    # Add padding tokens at the tail
    hidden_states_padded = torch.cat(
        [hidden_states, torch.zeros(batch_size, pad_tokens, 64, device=device)], dim=1
    )
    extra_ids = torch.zeros(pad_tokens, 3, device=device)
    extra_ids[:, 1] = height + torch.arange(pad_tokens, device=device)
    img_ids_padded = torch.cat([img_ids, extra_ids], dim=0)

    mask = torch.ones(batch_size, seq_txt + seq_img + pad_tokens, device=device, dtype=torch.bool)
    mask[:, seq_txt + seq_img :] = False

    with torch.no_grad():
        out_padded = model(
            hidden_states=hidden_states_padded,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids_padded,
            txt_ids=txt_ids,
            attention_mask=mask,
        ).sample

    assert torch.allclose(out_base, out_padded[:, :seq_img], atol=1e-5)

"""
Flux Kontext Model Loading Module
Provides functions to load individual components of Flux Kontext models.
"""

import logging

import torch
from diffusers import FluxKontextPipeline


logger = logging.getLogger(__name__)


def load_flux_kontext_vae(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda",
    use_pipeline: bool = False,
):
    """
    Load Flux Kontext VAE component.
    Args:
        model_path: Path to the pretrained model
        weight_dtype: Weight data type for the model
        device_map: Device to load the model on
    Returns:
        VAE model component
    """
    logger.info(f"Loading Flux Kontext VAE from {model_path}")

    if not use_pipeline:
        from diffusers.models import AutoencoderKL

        return AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=weight_dtype,
            device_map=device_map,
            use_safetensors=True,
        )
    else:
        pipe = FluxKontextPipeline.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )

        vae = pipe.vae
        vae.to(device_map)

        # Clean up pipeline reference
        del pipe
        torch.cuda.empty_cache()

        logger.info(f"Successfully loaded Flux Kontext VAE with dtype {weight_dtype}")
        return vae


def load_flux_kontext_clip(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda",
    use_pipeline: bool = False,
):
    """
    Load Flux Kontext CLIP text encoder component.

    Args:
        model_path: Path to the pretrained model
        weight_dtype: Weight data type for the model
        device_map: Device to load the model on

    Returns:
        CLIP text encoder model component
    """
    logger.info(f"Loading Flux Kontext CLIP encoder from {model_path}")
    # ---- Text encoders (Transformers) ----
    from transformers import CLIPTextModel

    if not use_pipeline:
        txt = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=weight_dtype, device_map=device_map
        )
        return txt
    else:
        # Load the full pipeline temporarily to extract CLIP encoder
        pipe = FluxKontextPipeline.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )

        text_encoder = pipe.text_encoder
        text_encoder.to(device_map)

        # Clean up pipeline reference
        del pipe
        torch.cuda.empty_cache()

        logger.info(f"Successfully loaded Flux Kontext CLIP encoder with dtype {weight_dtype}")
        return text_encoder


def load_flux_kontext_t5(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cpu",
    use_pipeline: bool = False,
):
    """
    Load Flux Kontext T5 text encoder component.

    Args:
        model_path: Path to the pretrained model
        weight_dtype: Weight data type for the model
        device_map: Device to load the model on

    Returns:
        T5 text encoder model component
    """
    logger.info(f"Loading Flux Kontext T5 encoder from {model_path}")

    if not use_pipeline:
        from transformers import T5EncoderModel

        return T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=weight_dtype, device_map=device_map
        )
    else:
        # Load the full pipeline temporarily to extract T5 encoder
        pipe = FluxKontextPipeline.from_pretrained(model_path, torch_dtype=weight_dtype, use_safetensors=True)

        text_encoder_2 = pipe.text_encoder_2
        text_encoder_2.to(device_map)

        # Clean up pipeline reference
        del pipe
        torch.cuda.empty_cache()

        logger.info(f"Successfully loaded Flux Kontext T5 encoder with dtype {weight_dtype}")
        return text_encoder_2


def load_flux_kontext_transformer(
    repo: str = "black-forest-labs/FLUX.1-Kontext-dev",
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = "cpu",  # or "auto"
    variant: str | None = None,  # e.g. "fp16", "bf16", or custom
    use_pipeline: bool = False,
    use_multi_resolution: bool = False,
):
    """
    加载 FluxKontext 的条件 Transformer (MMDiT) 主干。
    如果 use_multi_resolution, 则从自定义的transformer_flux_custom.py中加载, 支持多种分辨率的训练
    """
    if use_multi_resolution:
        from qflux.models.transformer_flux_custom import FluxTransformer2DModel
    else:
        from diffusers.models import FluxTransformer2DModel  # type: ignore

    # from qflux.models.transformer_flux import FluxTransformer2DModel
    # from diffusers.models import FluxTransformer2DModel

    if not use_pipeline:
        return FluxTransformer2DModel.from_pretrained(
            repo,
            subfolder="transformer",
            torch_dtype=weight_dtype,
            device_map=device_map,
            use_safetensors=True,
            variant=variant,
        )
    else:
        pipe = FluxKontextPipeline.from_pretrained(repo, torch_dtype=weight_dtype, use_safetensors=True)

        transformer = pipe.transformer
        transformer.to(device_map)
        del pipe
        torch.cuda.empty_cache()
    return transformer


def load_dreamomni2_transformer(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = "cpu",
    use_pipeline: bool = False,
):
    """
    Load DreamOmni2 transformer.
    """
    edit_lora = "TsienDragon/DreamOmni2"
    transformer = load_flux_kontext_transformer(model_path, weight_dtype, device_map, use_pipeline=use_pipeline)
    transformer.load_lora_adapter(edit_lora, adapter_name="edit")
    transformer.fuse_lora("edit")
    transformer.delete_adapters("edit")
    return transformer


def load_flux_kontext_tokenizers(model_path: str, use_pipeline: bool = False):
    """
    Load Flux Kontext tokenizers (CLIP and T5).

    Args:
        model_path: Path to the pretrained model

    Returns:
        Tuple of (clip_tokenizer, t5_tokenizer)
    """
    logger.info(f"Loading Flux Kontext tokenizers from {model_path}")

    if not use_pipeline:
        from transformers import CLIPTokenizer, T5TokenizerFast

        tok_clip = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        tok_t5 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2")
        return tok_clip, tok_t5
    else:
        # Load the full pipeline temporarily to extract tokenizers
        pipe = FluxKontextPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )

        tokenizer = pipe.tokenizer  # CLIP tokenizer
        tokenizer_2 = pipe.tokenizer_2  # T5 tokenizer

        # Clean up pipeline reference
        del pipe
        torch.cuda.empty_cache()

        logger.info("Successfully loaded Flux Kontext tokenizers")
        return tokenizer, tokenizer_2


def load_flux_kontext_scheduler(model_path: str):
    """
    Load Flux Kontext scheduler.

    Args:
        model_path: Path to the pretrained model

    Returns:
        Scheduler component
    """
    logger.info(f"Loading Flux Kontext scheduler from {model_path}")

    # Load the full pipeline temporarily to extract scheduler
    pipe = FluxKontextPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    scheduler = pipe.scheduler

    # Clean up pipeline reference
    del pipe
    torch.cuda.empty_cache()

    logger.info("Successfully loaded Flux Kontext scheduler")
    return scheduler


if __name__ == "__main__":
    tokenizer1, tokenizer2 = load_flux_kontext_tokenizers("black-forest-labs/FLUX.1-Kontext-dev", use_pipeline=False)
    print(tokenizer1, tokenizer2, type(tokenizer1), type(tokenizer2))
    vae = load_flux_kontext_vae("black-forest-labs/FLUX.1-Kontext-dev", use_pipeline=False)
    print("vae", type(vae))

    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    ckpt = "black-forest-labs/FLUX.1-Kontext-dev"
    tok_clip = CLIPTokenizer.from_pretrained(ckpt, subfolder="tokenizer")
    tok_t5 = T5TokenizerFast.from_pretrained(ckpt, subfolder="tokenizer_2")

    enc_clip = CLIPTextModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    enc_t5 = T5EncoderModel.from_pretrained(ckpt, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    print("type", type(tok_clip), type(tok_t5), type(enc_clip), type(enc_t5))

    t5 = load_flux_kontext_t5("black-forest-labs/FLUX.1-Kontext-dev", use_pipeline=False)
    print("t5", type(t5))
    clip = load_flux_kontext_clip("black-forest-labs/FLUX.1-Kontext-dev", use_pipeline=False)
    print("clip", type(clip))
    transformer = load_flux_kontext_transformer("black-forest-labs/FLUX.1-Kontext-dev", use_pipeline=False)
    print("transformer", type(transformer))
    vae = load_flux_kontext_vae("black-forest-labs/FLUX.1-Kontext-dev", use_pipeline=False)
    print("vae", type(vae))
    tokenizer1, tokenizer2 = load_flux_kontext_tokenizers("black-forest-labs/FLUX.1-Kontext-dev", use_pipeline=False)
    print("tokenizer1", type(tokenizer1), type(tokenizer2))
    scheduler = load_flux_kontext_scheduler("black-forest-labs/FLUX.1-Kontext-dev")
    print("scheduler", type(scheduler))

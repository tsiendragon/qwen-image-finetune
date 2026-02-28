"""
Flux2 Klein Model Loading Module
Provides functions to load individual components of Flux2 Klein models.
"""

import logging

import torch
from diffusers import Flux2KleinPipeline


logger = logging.getLogger(__name__)


def load_flux2_klein_vae(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda",
    use_pipeline: bool = False,
):
    """
    Load Flux2 Klein VAE component (AutoencoderKLFlux2).

    Args:
        model_path: Path or repo id of the pretrained model.
        weight_dtype: Weight data type for the model.
        device_map: Device mapping for loading.
        use_pipeline: If True, load via Flux2KleinPipeline and extract component.
    """
    logger.info(f"Loading Flux2 Klein VAE from {model_path}")

    if not use_pipeline:
        from diffusers.models import AutoencoderKLFlux2

        return AutoencoderKLFlux2.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=weight_dtype,
            device_map=device_map,
            use_safetensors=True,
        )
    else:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        vae = pipe.vae
        vae.to(device_map)

        del pipe
        torch.cuda.empty_cache()

        logger.info(f"Successfully loaded Flux2 Klein VAE with dtype {weight_dtype}")
        return vae


def load_flux2_klein_text_encoder(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda",
    use_pipeline: bool = False,
):
    """
    Load Flux2 Klein Qwen3 text encoder component.

    Args:
        model_path: Path or repo id of the pretrained model.
        weight_dtype: Weight data type for the model.
        device_map: Device mapping for loading.
        use_pipeline: If True, load via Flux2KleinPipeline and extract component.
    """
    logger.info(f"Loading Flux2 Klein Qwen3 text encoder from {model_path}")

    if not use_pipeline:
        from transformers import Qwen3ForCausalLM

        return Qwen3ForCausalLM.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            device_map=device_map,
        )
    else:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        text_encoder = pipe.text_encoder
        text_encoder.to(device_map)

        del pipe
        torch.cuda.empty_cache()

        logger.info(f"Successfully loaded Flux2 Klein Qwen3 text encoder with dtype {weight_dtype}")
        return text_encoder


def load_flux2_klein_transformer(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = "cpu",
    use_pipeline: bool = False,
):
    """
    Load Flux2 Klein conditional Transformer (Flux2Transformer2DModel).
    """
    from diffusers.models import Flux2Transformer2DModel

    logger.info(f"Loading Flux2 Klein transformer from {model_path}")

    if not use_pipeline:
        return Flux2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
            device_map=device_map,
            use_safetensors=True,
        )
    else:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
            use_safetensors=True,
        )
        transformer = pipe.transformer
        transformer.to(device_map)

        del pipe
        torch.cuda.empty_cache()

        logger.info(f"Successfully loaded Flux2 Klein transformer with dtype {weight_dtype}")
        return transformer


def load_flux2_klein_tokenizer(model_path: str, use_pipeline: bool = False):
    """
    Load Flux2 Klein tokenizer (Qwen2TokenizerFast).

    Args:
        model_path: Path or repo id of the pretrained model.
        use_pipeline: If True, load via Flux2KleinPipeline and extract component.
    """
    logger.info(f"Loading Flux2 Klein tokenizer from {model_path}")

    if not use_pipeline:
        from transformers import Qwen2TokenizerFast

        tok = Qwen2TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
        return tok
    else:
        pipe = Flux2KleinPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        tokenizer = pipe.tokenizer

        del pipe
        torch.cuda.empty_cache()

        logger.info("Successfully loaded Flux2 Klein tokenizer")
        return tokenizer


def load_flux2_klein_scheduler(model_path: str):
    """
    Load Flux2 Klein scheduler (FlowMatchEulerDiscreteScheduler).

    Args:
        model_path: Path or repo id of the pretrained model.
    """
    logger.info(f"Loading Flux2 Klein scheduler from {model_path}")

    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    scheduler = pipe.scheduler

    del pipe
    torch.cuda.empty_cache()

    logger.info("Successfully loaded Flux2 Klein scheduler")
    return scheduler


if __name__ == "__main__":
    # Simple manual smoke test for local debugging
    ckpt = "black-forest-labs/FLUX.2-klein-base-9B"
    tok = load_flux2_klein_tokenizer(ckpt, use_pipeline=False)
    print("tokenizer", type(tok))
    vae = load_flux2_klein_vae(ckpt, use_pipeline=False)
    print("vae", type(vae))
    text_encoder = load_flux2_klein_text_encoder(ckpt, use_pipeline=False)
    print("text_encoder", type(text_encoder))
    transformer = load_flux2_klein_transformer(ckpt, use_pipeline=False)
    print("transformer", type(transformer))
    scheduler = load_flux2_klein_scheduler(ckpt)
    print("scheduler", type(scheduler))


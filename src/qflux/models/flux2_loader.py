"""
FLUX.2 Model Loading Module
Provides functions to load individual components of FLUX.2 models.

FLUX.2 uses different components compared to FluxKontext:
- Single text encoder: Mistral3ForConditionalGeneration (VLM)
- Tokenizer: PixtralProcessor (AutoProcessor)
- VAE: AutoencoderKLFlux2 (with batch_norm)
- Transformer: Flux2Transformer2DModel
"""

import logging

import torch


logger = logging.getLogger(__name__)


def load_flux2_vae(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cpu",
):
    """
    Load FLUX.2 VAE component with batch_norm architecture.

    Args:
        model_path: Path to the pretrained model (e.g., 'black-forest-labs/FLUX.2-dev')
        weight_dtype: Weight data type for the model
        device_map: Device to load the model on

    Returns:
        AutoencoderKLFlux2 model component
    """
    logger.info(f"Loading FLUX.2 VAE from {model_path}")

    from diffusers.models import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        device_map=device_map,
        use_safetensors=True,
    )

    logger.info(f"Successfully loaded FLUX.2 VAE with dtype {weight_dtype}")
    return vae


def load_flux2_text_encoder(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cpu",
):
    """
    Load FLUX.2 text encoder (Mistral3ForConditionalGeneration).

    FLUX.2 uses a single VLM (Mistral3) as text encoder, unlike FluxKontext
    which uses CLIP + T5 dual encoders.

    Args:
        model_path: Path to the pretrained model
        weight_dtype: Weight data type for the model
        device_map: Device to load the model on

    Returns:
        Mistral3ForConditionalGeneration model component
    """
    logger.info(f"Loading FLUX.2 text encoder (Mistral3) from {model_path}")

    from transformers import Mistral3ForConditionalGeneration

    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        device_map=device_map,
        use_safetensors=True,
    )

    logger.info(f"Successfully loaded FLUX.2 text encoder with dtype {weight_dtype}")
    return text_encoder


def load_flux2_tokenizer(model_path: str):
    """
    Load FLUX.2 tokenizer (PixtralProcessor via AutoProcessor).

    Args:
        model_path: Path to the pretrained model

    Returns:
        AutoProcessor (PixtralProcessor) for text/image processing
    """
    logger.info(f"Loading FLUX.2 tokenizer from {model_path}")

    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(
        model_path,
        subfolder="tokenizer",
    )

    logger.info("Successfully loaded FLUX.2 tokenizer")
    return tokenizer


def load_flux2_transformer(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = "cpu",
    variant: str | None = None,
):
    """
    Load FLUX.2 Transformer (Flux2Transformer2DModel).

    Args:
        model_path: Path to the pretrained model
        weight_dtype: Weight data type for the model
        device_map: Device to load the model on
        variant: Model variant (e.g., 'fp16', 'bf16')

    Returns:
        Flux2Transformer2DModel
    """
    logger.info(f"Loading FLUX.2 Transformer from {model_path}")

    from diffusers.models import Flux2Transformer2DModel

    transformer = Flux2Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        device_map=device_map,
        use_safetensors=True,
        variant=variant,
    )

    logger.info(f"Successfully loaded FLUX.2 Transformer with dtype {weight_dtype}")
    return transformer


def load_flux2_scheduler(model_path: str):
    """
    Load FLUX.2 scheduler (FlowMatchEulerDiscreteScheduler).

    Args:
        model_path: Path to the pretrained model

    Returns:
        FlowMatchEulerDiscreteScheduler
    """
    logger.info(f"Loading FLUX.2 scheduler from {model_path}")

    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
    )

    logger.info("Successfully loaded FLUX.2 scheduler")
    return scheduler


def load_flux2_pipeline(
    model_path: str,
    weight_dtype: torch.dtype = torch.bfloat16,
):
    """
    Load complete FLUX.2 pipeline (for reference/validation).

    Args:
        model_path: Path to the pretrained model
        weight_dtype: Weight data type for the model

    Returns:
        Flux2Pipeline
    """
    logger.info(f"Loading complete FLUX.2 Pipeline from {model_path}")

    from diffusers import Flux2Pipeline

    pipe = Flux2Pipeline.from_pretrained(
        model_path,
        torch_dtype=weight_dtype,
        use_safetensors=True,
    )

    logger.info("Successfully loaded FLUX.2 Pipeline")
    return pipe


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """
    FLUX.2 specific mu calculation for timestep scheduling.

    This function computes the empirical mu value used in FLUX.2's
    timestep shift calculation, which differs from FluxKontext's calculate_shift().

    Args:
        image_seq_len: Length of image sequence (height * width after packing)
        num_steps: Number of inference steps

    Returns:
        Computed mu value for scheduler
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


if __name__ == "__main__":
    # Test loading components
    model_path = "black-forest-labs/FLUX.2-dev"

    # Test mu calculation
    mu = compute_empirical_mu(1024, 50)
    print(f"Computed mu for seq_len=1024, steps=50: {mu}")

    mu = compute_empirical_mu(4096, 50)
    print(f"Computed mu for seq_len=4096, steps=50: {mu}")

    mu = compute_empirical_mu(5000, 50)
    print(f"Computed mu for seq_len=5000, steps=50: {mu}")


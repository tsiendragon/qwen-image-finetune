#!/usr/bin/env python3
"""
Test sampling script for multi-resolution hair segmentation with FluxKontext
Based on test_example_fluxkontext_multiresolution.yaml config
"""
import torch
import os
import sys
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_config(config_path: str, lora_weight: str = None, device: str = 'cuda:0'):
    """Load and setup configuration"""
    config = load_config_from_yaml(config_path)

    # Override predict devices
    config.predict.devices.vae = "cuda:0"
    config.predict.devices.text_encoder = "cuda:1"
    config.predict.devices.text_encoder_2 = "cuda:1"
    config.predict.devices.dit = "cuda:2"

    # Set LoRA weight if provided
    if lora_weight and os.path.exists(lora_weight):
        config.model.lora.pretrained_weight = lora_weight
        logger.info(f"Using LoRA weight: {lora_weight}")
    else:
        config.model.lora.pretrained_weight = None
        logger.info("No LoRA weight provided, using base model")

    return config


def load_test_dataset(dataset_name: str = "TsienDragon/figaro_hair_segmentation_1k", split: str = "test"):
    """Load test dataset from HuggingFace"""
    logger.info(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"Loaded {len(dataset)} test samples")
    return dataset


def process_sample(
    trainer,
    prompt_image: Image.Image,
    prompt: str,
    num_inference_steps: int = 20,
    cfg_scale: float = 1.0,
):
    """Process a single sample and generate output"""
    # width, height = prompt_image.size
    height, width = [832, 576]
    # controls_size = trainer.config.data.init_args.processor.init_args.controls_size

    # Run prediction - only needs prompt_image
    result = trainer.predict(
        prompt_image=prompt_image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        controls_size=None,
        true_cfg_scale=cfg_scale,
        negative_prompt="",
        weight_dtype=torch.bfloat16,
        # height=height,
        # width=width,
        output_type='pil',
        use_native_size=True,
    )

    # Handle result - can be list or single image
    if isinstance(result, list):
        return result[0]
    return result


def save_outputs(
    save_folder: str,
    idx: int,
    prompt_image: Image.Image,
    target_image: Image.Image,
    generated_image: Image.Image,
    prompt: str,
    basename: str = None,
):
    """Save all outputs to organized folders"""
    if basename is None:
        basename = f"sample_{idx:04d}.png"

    # Create output directories
    folders = ['images', 'prompts']
    for folder in folders:
        os.makedirs(os.path.join(save_folder, folder), exist_ok=True)

    # Save images with different suffixes
    base_name = basename.replace('.png', '').replace('.jpg', '')
    prompt_image.save(os.path.join(save_folder,  f"{base_name}_input.png"))
    target_image.save(os.path.join(save_folder,  f"{base_name}_target.png"))
    generated_image.save(os.path.join(save_folder,  f"{base_name}_generated.png"))


def main():
    parser = argparse.ArgumentParser(description="Test sampling script for multi-resolution hair segmentation")
    parser.add_argument(
        "--config",
        type=str,
        default="tests/test_configs/test_example_fluxkontext_multiresolution.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--lora-weight",
        type=str,
        default=None,
        help="Path to LoRA weight file (optional)"
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="/tmp/flux_kontext_sampling_multiresolution",
        help="Output folder for generated images"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="TsienDragon/figaro_hair_segmentation_1k",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (train/test/validation)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process (None = all)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="512x512",
        help="Target resolution (format: WIDTHxHEIGHT)"
    )

    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    target_resolution = (height, width)

    logger.info("=" * 80)
    logger.info("Starting multi-resolution sampling test")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Dataset: {args.dataset_name} ({args.split})")
    logger.info(f"Output folder: {args.save_folder}")
    logger.info(f"Target resolution: {target_resolution}")
    logger.info(f"Inference steps: {args.steps}")
    logger.info(f"CFG scale: {args.cfg_scale}")
    logger.info(f"Device: {args.device}")

    # Setup configuration
    config_path = os.path.join(project_root, args.config)
    config = setup_config(config_path, args.lora_weight, args.device)

    config.data.init_args.processor.init_args.multi_resolutions = None
    config.data.init_args.processor.init_args.process_type = 'resize'
    # config.data.init_args.processor.init_args.target_size = [512, 512]  #

    # Create trainer
    logger.info("Initializing trainer...")
    trainer = FluxKontextLoraTrainer(config)

    # Load test dataset
    dataset = load_test_dataset(args.dataset_name, args.split)

    # Determine number of samples to process
    num_samples = len(dataset) if args.num_samples is None else min(args.num_samples, len(dataset))
    logger.info(f"Processing {num_samples} samples...")

    # Process samples
    os.makedirs(args.save_folder, exist_ok=True)

    for i in tqdm(range(num_samples), desc="Generating samples"):
        sample = dataset[i]

        # Extract data from sample
        # Dataset fields: 'id', 'control_images', 'control_mask', 'target_image', 'prompt'

        # control_images is the list of input images, take the first one as prompt_image
        control_images = sample['control_images']
        if isinstance(control_images, list) and len(control_images) > 0:
            prompt_image = control_images[0].convert('RGB')
        else:
            prompt_image = control_images.convert('RGB')


        # target_image is the ground truth output
        target_image = sample['target_image'].convert('RGB')

        # Get prompt
        prompt = sample.get('prompt', "Generate hair segmentation mask from the image")

        # prompt = 'Generate hair segmentation mask from the image'
        print('prompt', prompt)
        print('target_image', target_image.size)
        print('prompt_image', prompt_image.size)

        # Resize images to target resolution while maintaining aspect ratio
        img_width, img_height = prompt_image.size
        max_width = 1024
        if img_width > max_width:
            scale = max_width / img_width
            new_width = max_width
            new_height = int(img_height * scale)
        else:
            new_width = img_width
            new_height = img_height
        new_width = (new_width // 32) * 32
        new_height = (new_height // 32) * 32
        prompt_image = prompt_image.resize((new_width, new_height))
        target_image = target_image.resize((new_width, new_height))


        # Generate image (only needs prompt_image and prompt)
        generated_image = process_sample(
            trainer=trainer,
            prompt_image=prompt_image,
            prompt=prompt,
            num_inference_steps=args.steps,
            cfg_scale=args.cfg_scale,
        )

        # Save outputs
        basename = f"sample_{i:04d}.png"
        save_outputs(
            save_folder=args.save_folder,
            idx=i,
            prompt_image=prompt_image,
            target_image=target_image,
            generated_image=generated_image,
            prompt=prompt,
            basename=basename,
        )

    logger.info("=" * 80)
    logger.info(f"Sampling complete! Results saved to: {args.save_folder}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

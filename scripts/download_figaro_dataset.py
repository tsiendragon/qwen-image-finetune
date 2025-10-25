"""
Download and organize Figaro Hair Segmentation dataset from Hugging Face Hub.

This script downloads the Allison/figaro_hair_segmentation_1000 dataset
and organizes it according to the structure required by the training pipeline.

Usage:
    python script/download_figaro_dataset.py --output_dir /path/to/output [--hf_token TOKEN]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from qflux.utils.huggingface import upload_editing_dataset  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories(base_dir: Path) -> tuple:
    """Create directory structure for train and test splits."""
    splits = []

    for split_name in ['train', 'test']:
        split_dir = base_dir / split_name
        control_dir = split_dir / 'control_images'
        training_dir = split_dir / 'training_images'

        control_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True, exist_ok=True)

        splits.append({
            'name': split_name,
            'control_dir': control_dir,
            'training_dir': training_dir
        })

    return splits


def generate_prompt(idx: int, split: str = 'train') -> str:
    """Generate a descriptive prompt for hair segmentation task."""
    # Vary the prompts for better generalization
    prompts = [
        "Generate hair segmentation mask from the image",
        "Create binary mask for hair region",
        "Segment the hair in this photo",
        "Extract hair segmentation from the image",
        "Produce hair mask for the given image",
    ]
    return prompts[idx % len(prompts)]


def save_sample(
    sample: dict,
    idx: int,
    split_dirs: dict,
    sample_id: Optional[str] = None
) -> bool:
    """
    Save a single sample to the organized directory structure.

    Args:
        sample: Dataset sample with 'image' (control photo) and 'label' (target mask) fields
        idx: Sample index
        split_dirs: Dictionary with 'control_dir' and 'training_dir' paths
        sample_id: Optional custom sample ID

    Returns:
        True if sample was saved successfully, False otherwise
    """
    try:
        # Generate sample ID
        if sample_id is None:
            sample_id = f"sample_{idx:05d}"

        # Get image (control) and label (target mask)
        control_image = sample['image']  # Photo is the control
        target_mask = sample.get('label', None)  # Segmentation mask is the target

        if control_image is None:
            logger.warning(f"Skipping sample {idx}: missing control image")
            return False

        if target_mask is None:
            logger.warning(f"Skipping sample {idx}: missing target label")
            return False

        # Convert to PIL Image if needed
        if not isinstance(control_image, Image.Image):
            logger.warning(f"Sample {idx}: control image is not PIL.Image, skipping")
            return False

        if not isinstance(target_mask, Image.Image):
            logger.warning(f"Sample {idx}: target label is not PIL.Image, skipping")
            return False

        # Save control image (the photo)
        control_path = split_dirs['control_dir'] / f"{sample_id}.jpg"
        control_image.convert('RGB').save(control_path, 'JPEG', quality=95)

        # Save target image (the segmentation mask)
        target_path = split_dirs['training_dir'] / f"{sample_id}.png"
        # Convert mask to grayscale or RGB based on original
        if target_mask.mode in ['L', '1']:
            target_mask.convert('L').save(target_path, 'PNG')
        else:
            target_mask.convert('RGB').save(target_path, 'PNG')

        # Generate and save prompt
        prompt = generate_prompt(idx)
        prompt_path = split_dirs['training_dir'] / f"{sample_id}.txt"
        prompt_path.write_text(prompt, encoding='utf-8')

        return True

    except Exception as e:
        logger.error(f"Error saving sample {idx}: {e}")
        return False


def download_and_organize(
    output_dir: str,
    hf_token: Optional[str] = None,
    repo_id: str = "Allison/figaro_hair_segmentation_1000",
    upload_to_hub: bool = False,
    upload_repo_id: Optional[str] = None,
    upload_private: bool = False
) -> None:
    """
    Download Figaro dataset and organize it into the required structure.

    Args:
        output_dir: Output directory path
        hf_token: Optional HuggingFace token for private datasets
        repo_id: HuggingFace dataset repository ID
        upload_to_hub: Whether to upload organized dataset to HuggingFace Hub
        upload_repo_id: Repository ID for upload (e.g., 'TsienDragon/figaro_hair_segmentation_1k')
        upload_private: Whether to make the uploaded dataset private
    """
    output_path = Path(output_dir)
    logger.info(f"Output directory: {output_path}")

    # Setup directories
    logger.info("Creating directory structure...")
    splits = setup_directories(output_path)
    split_dict = {s['name']: s for s in splits}

    # Download dataset
    logger.info(f"Downloading dataset from {repo_id}...")
    try:
        # Set token if provided
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token

        # Load dataset
        dataset = load_dataset(repo_id, trust_remote_code=True)
        logger.info(f"Dataset loaded: {dataset}")

        # Check available splits
        available_splits = list(dataset.keys())
        logger.info(f"Available splits: {available_splits}")

    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

    # Process each split
    for split_name in available_splits:
        if split_name not in split_dict:
            # Map validation to test if needed
            if split_name == 'validation':
                target_split = 'test'
            else:
                logger.warning(f"Unknown split '{split_name}', using as 'train'")
                target_split = 'train'
        else:
            target_split = split_name

        split_data = dataset[split_name]
        split_dirs = split_dict[target_split]

        logger.info(f"Processing {split_name} split ({len(split_data)} samples) -> {target_split}/")

        success_count = 0
        for idx, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            if save_sample(sample, idx, split_dirs):
                success_count += 1

        logger.info(f"Successfully saved {success_count}/{len(split_data)} samples for {split_name}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Dataset organization complete!")
    logger.info("="*60)
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info("\nDirectory structure:")
    for split in splits:
        logger.info(f"\n{split['name']}/")
        logger.info(f"  control_images/: {len(list(split['control_dir'].glob('*.jpg')))} photos (control)")
        logger.info(f"  training_images/: {len(list(split['training_dir'].glob('*.png')))} masks (target)")
        logger.info(f"                    {len(list(split['training_dir'].glob('*.txt')))} prompts")

    # Upload to HuggingFace Hub if requested
    if upload_to_hub:
        if not upload_repo_id:
            logger.error("Upload requested but no upload_repo_id provided!")
            return

        logger.info("\n" + "="*60)
        logger.info(f"Uploading to HuggingFace Hub: {upload_repo_id}")
        logger.info("="*60)

        try:
            upload_editing_dataset(
                root_dir=str(output_path),
                repo_id=upload_repo_id,
                private=upload_private
            )
            logger.info(f"\n✓ Successfully uploaded to: https://huggingface.co/datasets/{upload_repo_id}")
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Download and organize Figaro Hair Segmentation dataset"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory path for organized dataset'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace token (optional, for private datasets)'
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default='Allison/figaro_hair_segmentation_1000',
        help='HuggingFace dataset repository ID'
    )
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload organized dataset to HuggingFace Hub after organization'
    )
    parser.add_argument(
        '--upload_repo_id',
        type=str,
        default='TsienDragon/figaro_hair_segmentation_1k',
        help='Repository ID for uploading organized dataset (default: TsienDragon/figaro_hair_segmentation_1k)'
    )
    parser.add_argument(
        '--upload_private',
        action='store_true',
        help='Make the uploaded dataset private'
    )

    args = parser.parse_args()

    download_and_organize(
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        repo_id=args.repo_id,
        upload_to_hub=args.upload,
        upload_repo_id=args.upload_repo_id,
        upload_private=args.upload_private
    )


if __name__ == '__main__':
    main()

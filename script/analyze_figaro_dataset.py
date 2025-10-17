"""
Analyze Figaro Hair Segmentation dataset statistics.

This script downloads the Allison/figaro_hair_segmentation_1000 dataset
and analyzes the image shapes, sizes, and other statistics.

Usage:
    python script/analyze_figaro_dataset.py [--output_file stats.json]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_image_shapes(dataset, split_name: str) -> Dict:
    """Analyze image shapes for a dataset split."""
    logger.info(f"Analyzing {split_name} split ({len(dataset)} samples)...")

    control_shapes = []
    target_shapes = []
    control_aspect_ratios = []
    target_aspect_ratios = []
    control_areas = []
    target_areas = []

    skipped_samples = 0

    for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {split_name}")):
        try:
            # Analyze control image (photo)
            control_image = sample['image']
            if isinstance(control_image, Image.Image):
                width, height = control_image.size
                control_shapes.append((width, height))
                control_aspect_ratios.append(width / height)
                control_areas.append(width * height)
            else:
                logger.warning(f"Sample {idx}: control image is not PIL.Image")
                skipped_samples += 1
                continue

            # Analyze target image (mask)
            target_image = sample.get('label', None)
            if target_image is not None and isinstance(target_image, Image.Image):
                width, height = target_image.size
                target_shapes.append((width, height))
                target_aspect_ratios.append(width / height)
                target_areas.append(width * height)
            else:
                logger.warning(f"Sample {idx}: target image is not PIL.Image")
                skipped_samples += 1
                continue

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            skipped_samples += 1
            continue

    # Calculate statistics
    def calculate_stats(values, name):
        if not values:
            return {}

        values = np.array(values)
        return {
            f"{name}_count": len(values),
            f"{name}_min": float(np.min(values)),
            f"{name}_max": float(np.max(values)),
            f"{name}_mean": float(np.mean(values)),
            f"{name}_median": float(np.median(values)),
            f"{name}_std": float(np.std(values)),
            f"{name}_q25": float(np.percentile(values, 25)),
            f"{name}_q75": float(np.percentile(values, 75))
        }

    # Calculate shape statistics
    control_widths = [shape[0] for shape in control_shapes]
    control_heights = [shape[1] for shape in control_shapes]
    target_widths = [shape[0] for shape in target_shapes]
    target_heights = [shape[1] for shape in target_shapes]

    stats = {
        "split_name": split_name,
        "total_samples": len(dataset),
        "processed_samples": len(control_shapes),
        "skipped_samples": skipped_samples,
        "control_images": {
            **calculate_stats(control_widths, "width"),
            **calculate_stats(control_heights, "height"),
            **calculate_stats(control_aspect_ratios, "aspect_ratio"),
            **calculate_stats(control_areas, "area"),
            "unique_shapes": len(set(control_shapes)),
            "most_common_shape": max(set(control_shapes), key=control_shapes.count) if control_shapes else None
        },
        "target_images": {
            **calculate_stats(target_widths, "width"),
            **calculate_stats(target_heights, "height"),
            **calculate_stats(target_aspect_ratios, "aspect_ratio"),
            **calculate_stats(target_areas, "area"),
            "unique_shapes": len(set(target_shapes)),
            "most_common_shape": max(set(target_shapes), key=target_shapes.count) if target_shapes else None
        }
    }

    # Check if control and target shapes match
    shape_matches = sum(1 for c, t in zip(control_shapes, target_shapes) if c == t)
    stats["shape_consistency"] = {
        "matching_shapes": shape_matches,
        "total_pairs": len(control_shapes),
        "match_percentage": (shape_matches / len(control_shapes) * 100)
        if control_shapes else 0
    }

    return stats


def analyze_dataset(repo_id: str = "Allison/figaro_hair_segmentation_1000",
                    hf_token: Optional[str] = None) -> Dict:
    """Analyze the entire dataset."""
    logger.info(f"Loading dataset from {repo_id}...")

    try:
        # Set token if provided
        if hf_token:
            import os
            os.environ['HF_TOKEN'] = hf_token

        # Load dataset
        dataset = load_dataset(repo_id, trust_remote_code=True)
        logger.info(f"Dataset loaded: {dataset}")

        # Analyze each split
        all_stats = {}
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            all_stats[split_name] = analyze_image_shapes(split_data, split_name)

        # Calculate overall statistics
        total_samples = 0
        total_processed = 0

        for split_stats in all_stats.values():
            total_samples += split_stats["total_samples"]
            total_processed += split_stats["processed_samples"]

            # Note: For overall analysis across splits, you'd need to re-analyze
            # all samples together, which is not implemented here for efficiency

        all_stats["overall"] = {
            "total_samples": total_samples,
            "total_processed": total_processed,
            "total_skipped": total_samples - total_processed,
            "splits": list(dataset.keys())
        }

        return all_stats

    except Exception as e:
        logger.error(f"Failed to analyze dataset: {e}")
        raise


def print_summary(stats: Dict) -> None:
    """Print a human-readable summary of the statistics."""
    print("\n" + "="*80)
    print("FIGARO HAIR SEGMENTATION DATASET ANALYSIS")
    print("="*80)

    for split_name, split_stats in stats.items():
        if split_name == "overall":
            continue

        print(f"\n{split_name.upper()} SPLIT")
        print("-" * 40)
        print(f"Total samples: {split_stats['total_samples']}")
        print(f"Processed: {split_stats['processed_samples']}")
        print(f"Skipped: {split_stats['skipped_samples']}")

        # Control images (photos)
        control = split_stats['control_images']
        print("\nControl Images (Photos):")
        dims_str = (f"{control['width_min']:.0f}x{control['height_min']:.0f} to "
                    f"{control['width_max']:.0f}x{control['height_max']:.0f}")
        print(f"  Dimensions: {dims_str}")
        print(f"  Most common: {control['most_common_shape']}")
        ar_str = (f"{control['aspect_ratio_min']:.2f} to {control['aspect_ratio_max']:.2f} "
                  f"(mean: {control['aspect_ratio_mean']:.2f})")
        print(f"  Aspect ratio: {ar_str}")
        area_str = (f"{control['area_min']:.0f} to {control['area_max']:.0f} pixels "
                    f"(mean: {control['area_mean']:.0f})")
        print(f"  Area: {area_str}")
        print(f"  Unique shapes: {control['unique_shapes']}")

        # Target images (masks)
        target = split_stats['target_images']
        print("\nTarget Images (Masks):")
        dims_str = (f"{target['width_min']:.0f}x{target['height_min']:.0f} to "
                    f"{target['width_max']:.0f}x{target['height_max']:.0f}")
        print(f"  Dimensions: {dims_str}")
        print(f"  Most common: {target['most_common_shape']}")
        ar_str = (f"{target['aspect_ratio_min']:.2f} to {target['aspect_ratio_max']:.2f} "
                  f"(mean: {target['aspect_ratio_mean']:.2f})")
        print(f"  Aspect ratio: {ar_str}")
        area_str = (f"{target['area_min']:.0f} to {target['area_max']:.0f} pixels "
                    f"(mean: {target['area_mean']:.0f})")
        print(f"  Area: {area_str}")
        print(f"  Unique shapes: {target['unique_shapes']}")

        # Shape consistency
        consistency = split_stats['shape_consistency']
        print("\nShape Consistency:")
        match_str = (f"{consistency['matching_shapes']}/{consistency['total_pairs']} "
                     f"({consistency['match_percentage']:.1f}%)")
        print(f"  Matching pairs: {match_str}")

    # Overall summary
    if "overall" in stats:
        overall = stats["overall"]
        print("\nOVERALL SUMMARY")
        print("-" * 40)
        print(f"Total samples: {overall['total_samples']}")
        print(f"Successfully processed: {overall['total_processed']}")
        print(f"Skipped: {overall['total_skipped']}")
        print(f"Available splits: {', '.join(overall['splits'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Figaro Hair Segmentation dataset statistics"
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default='Allison/figaro_hair_segmentation_1000',
        help='HuggingFace dataset repository ID'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace token (optional, for private datasets)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='figaro_dataset_stats.json',
        help='Output file to save detailed statistics (JSON format)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only print summary, suppress progress bars'
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Analyze dataset
        stats = analyze_dataset(
            repo_id=args.repo_id,
            hf_token=args.hf_token
        )

        # Print summary
        print_summary(stats)

        # Save detailed statistics
        output_path = Path(args.output_file)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"\nDetailed statistics saved to: {output_path.absolute()}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

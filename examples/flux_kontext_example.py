#!/usr/bin/env python3
"""
Flux Kontext LoRA Training Example Script

This script demonstrates how to use the FluxKontextLoraTrainer for training
and inference with different configurations.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from flux_kontext_trainer import FluxKontextLoraTrainer
from data.config import load_config
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Flux Kontext LoRA Training Example")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flux_kontext_fp4.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "cache", "test", "validate"],
        default="test",
        help="Mode to run: train, cache, test, or validate"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Override dataset path from config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of training steps for testing"
    )

    args = parser.parse_args()

    print(f"🚀 Starting Flux Kontext Example - Mode: {args.mode}")
    print(f"📁 Config: {args.config}")

    # Load configuration
    try:
        config = load_config(args.config)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return

    # Override config values if provided
    if args.dataset_path:
        config.data.init_args.dataset_path = args.dataset_path
        print(f"📂 Dataset path overridden: {args.dataset_path}")

    if args.output_dir:
        config.logging.output_dir = args.output_dir
        print(f"📁 Output directory overridden: {args.output_dir}")

    # Create trainer
    try:
        trainer = FluxKontextLoraTrainer(config)
        print("✅ Trainer created successfully")
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        return

    # Execute based on mode
    if args.mode == "validate":
        validate_components(trainer)
    elif args.mode == "test":
        test_training_pipeline(trainer, config, args.steps)
    elif args.mode == "cache":
        run_caching(trainer, config)
    elif args.mode == "train":
        run_training(trainer, config)

    print("🎉 Example completed!")


def validate_components(trainer):
    """Validate that model components load correctly."""
    print("\n🔍 Validating model components...")

    try:
        # Load model components
        trainer.load_model()
        print("✅ Model components loaded successfully")

        # Check component attributes
        components = [
            ("VAE", trainer.vae),
            ("CLIP Encoder", trainer.text_encoder),
            ("T5 Encoder", trainer.text_encoder_2),
            ("Transformer", trainer.transformer),
            ("CLIP Tokenizer", trainer.tokenizer),
            ("T5 Tokenizer", trainer.tokenizer_2),
            ("Scheduler", trainer.scheduler)
        ]

        for name, component in components:
            if component is not None:
                print(f"✅ {name}: Loaded")

                # Check if it's a model with parameters
                if hasattr(component, 'parameters'):
                    param_count = sum(p.numel() for p in component.parameters())
                    print(f"   Parameters: {param_count:,}")
            else:
                print(f"❌ {name}: Not loaded")

        # Test text encoding
        print("\n🔤 Testing text encoding...")
        test_prompt = "A beautiful landscape with mountains and trees"

        if trainer.text_encoder is not None and trainer.tokenizer is not None:
            clip_embeds, clip_mask = trainer.encode_clip_prompt([test_prompt])
            print(f"✅ CLIP encoding: {clip_embeds.shape}")

        if trainer.text_encoder_2 is not None and trainer.tokenizer_2 is not None:
            t5_embeds, t5_mask = trainer.encode_t5_prompt([test_prompt])
            print(f"✅ T5 encoding: {t5_embeds.shape}")

        # Test combined encoding
        combined_embeds, combined_mask = trainer.encode_prompt([test_prompt])
        print(f"✅ Combined encoding: {combined_embeds.shape}")

        print("✅ Component validation completed successfully")

    except Exception as e:
        print(f"❌ Component validation failed: {e}")
        import traceback
        traceback.print_exc()


def test_training_pipeline(trainer, config, max_steps):
    """Test the training pipeline with minimal steps."""
    print(f"\n🏋️ Testing training pipeline ({max_steps} steps)...")

    try:
        # Override config for testing
        config.train.max_train_steps = max_steps
        config.train.num_epochs = 1
        config.data.batch_size = min(config.data.batch_size, 1)  # Use small batch for testing

        # Create a minimal dataset
        print("📦 Loading dataset...")
        from data.dataset import ImageDataset

        dataset = ImageDataset(**config.data.init_args)
        print(f"✅ Dataset loaded: {len(dataset)} samples")

        # Create dataloader with minimal batch size
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=False,  # Don't shuffle for testing
            num_workers=0   # Avoid multiprocessing issues
        )

        # Test training
        print("🚂 Starting training test...")
        trainer.fit(dataloader)

        print("✅ Training pipeline test completed successfully")

    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


def run_caching(trainer, config):
    """Run the caching process."""
    print("\n💾 Running caching process...")

    try:
        # Load dataset
        from data.dataset import ImageDataset
        dataset = ImageDataset(**config.data.init_args)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Run caching
        trainer.cache(dataloader)
        print("✅ Caching completed successfully")

    except Exception as e:
        print(f"❌ Caching failed: {e}")
        import traceback
        traceback.print_exc()


def run_training(trainer, config):
    """Run full training."""
    print("\n🏋️ Running full training...")

    try:
        # Load dataset
        from data.dataset import ImageDataset
        dataset = ImageDataset(**config.data.init_args)
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.batch_size,
            shuffle=config.data.shuffle,
            num_workers=config.data.num_workers
        )

        # Run training
        trainer.fit(dataloader)
        print("✅ Training completed successfully")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def print_system_info():
    """Print system information for debugging."""
    print("\n💻 System Information:")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name} ({props.total_memory // 1024**3} GB)")


if __name__ == "__main__":
    print_system_info()
    main()

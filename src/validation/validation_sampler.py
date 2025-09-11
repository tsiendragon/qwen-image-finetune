"""
ValidationSampler for monitoring training progress through image sampling.
Supports both FluxKontext and QwenImageEdit trainers.
"""

import random
import logging
from typing import Dict, List, Optional, Union
import torch
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


class ValidationSampler:
    """Universal validation sampler that works with any model architecture"""

    def __init__(self, config, accelerator, weight_dtype):
        self.config = config
        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.validation_dataset = None

        # Cached embeddings storage
        self.cached_embeddings = []  # List of cached embedding dictionaries
        self.embeddings_cached = False

        # Internal configuration with sensible defaults
        self._internal_config = {
            'max_image_size': 512,
            'log_prefix': 'validation',
            'move_vae_to_cpu_after': True,
            'skip_on_error': True,
            'min_samples_per_process': 1
        }

    def setup_validation_dataset(self, train_dataset=None):
        """Setup validation dataset from config or training dataset"""
        if self.config.validation_data is None:
            # Use subset of training dataset
            self.validation_dataset = self._create_subset_from_training(train_dataset)
        elif isinstance(self.config.validation_data, str):
            # Load from dataset path
            self.validation_dataset = self._load_dataset_from_path(self.config.validation_data)
        elif isinstance(self.config.validation_data, list):
            # Use control-prompt pairs
            self.validation_dataset = self._create_dataset_from_pairs(self.config.validation_data)
        else:
            raise ValueError(f"Unsupported validation_data format: {type(self.config.validation_data)}")

    def cache_embeddings(self, trainer, embeddings_config):
        """Cache embeddings for all validation samples using trainer's methods"""
        if self.embeddings_cached:
            return

        self.cached_embeddings = []

        try:
            logger.info(f"Caching embeddings for {len(self.validation_dataset)} validation samples...")

            for idx, sample in enumerate(self.validation_dataset):
                cached_sample = {'sample_idx': idx}

                # Cache VAE embeddings using trainer's encode methods
                if embeddings_config.get('cache_vae_embeddings', False):
                    if 'control_image' in sample:
                        # Use trainer's existing VAE encoding method
                        cached_sample['control_latents'] = trainer._encode_vae_image_for_validation(sample['control_image'])
                    if 'target_image' in sample:
                        cached_sample['target_latents'] = trainer._encode_vae_image_for_validation(sample['target_image'])

                # Cache text embeddings using trainer's encode methods
                if embeddings_config.get('cache_text_embeddings', False):
                    if 'prompt' in sample:
                        # Use trainer's existing prompt encoding method
                        cached_sample['text_embeddings'] = trainer._encode_prompt_for_validation(
                            sample['prompt'],
                            sample.get('control_image')
                        )

                # Store original sample data for logging
                cached_sample['original_sample'] = sample
                self.cached_embeddings.append(cached_sample)

            logger.info(f"Successfully cached embeddings for {len(self.cached_embeddings)} samples")

        except Exception as e:
            logger.error(f"Failed to cache embeddings: {e}")
            self.accelerator.print(f"Failed to cache embeddings: {e}")
            return

        self.embeddings_cached = True

    def should_run_validation(self, global_step: int) -> bool:
        """Check if validation should run at current step"""
        if self.config.validation_steps <= 0:
            return False
        return global_step % self.config.validation_steps == 0

    def run_validation_loop(self, global_step: int, trainer):
        """Main validation loop using cached embeddings and trainer's methods"""
        if not self.embeddings_cached:
            self.accelerator.print("Warning: Embeddings not cached, skipping validation")
            return

        try:
            # Sample from cached embeddings
            num_samples = min(self.config.num_samples, len(self.cached_embeddings))
            selected_indices = self._select_sample_indices(num_samples, global_step)

            logger.info(f"Running validation sampling for step {global_step}, generating {num_samples} samples")

            for idx in selected_indices:
                cached_sample = self.cached_embeddings[idx]

                # Generate sample using cached embeddings and trainer's model
                generated_latents = trainer._generate_latents_for_validation(cached_sample)

                # Decode latents to image using trainer's decode method
                generated_image = trainer._decode_latents_for_validation(generated_latents)

                # Log to tensorboard
                self._log_validation_sample(global_step, idx, cached_sample, generated_image, trainer)

            logger.info(f"Completed validation sampling for step {global_step}")

        except Exception as e:
            logger.error(f"Validation sampling failed at step {global_step}: {e}")
            if not self._internal_config['skip_on_error']:
                raise

    def _create_subset_from_training(self, train_dataset):
        """Create validation subset from training dataset"""
        if train_dataset is None:
            logger.warning("No training dataset provided for validation subset creation")
            return []

        # Use first few samples as validation set
        max_validation_samples = min(20, len(train_dataset))
        validation_samples = []

        for i in range(max_validation_samples):
            sample = train_dataset[i]
            validation_samples.append(sample)

        logger.info(f"Created validation subset with {len(validation_samples)} samples from training dataset")
        return validation_samples

    def _load_dataset_from_path(self, dataset_path: str):
        """Load validation dataset from file path"""
        import os

        if not os.path.exists(dataset_path):
            logger.error(f"Validation dataset path does not exist: {dataset_path}")
            return []

        # Simple implementation: assume directory with images
        validation_samples = []

        if os.path.isdir(dataset_path):
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

            for filename in os.listdir(dataset_path):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(dataset_path, filename)
                    # Use filename (without extension) as prompt
                    prompt = os.path.splitext(filename)[0].replace('_', ' ')

                    validation_samples.append({
                        'control_image': image_path,
                        'target_image': image_path,  # Same as control for simplicity
                        'prompt': prompt
                    })

        logger.info(f"Loaded {len(validation_samples)} samples from dataset path: {dataset_path}")
        return validation_samples

    def _create_dataset_from_pairs(self, pairs: List[Dict]):
        """Create validation dataset from control-prompt pairs"""
        validation_samples = []

        for pair in pairs:
            if 'control' in pair and 'prompt' in pair:
                validation_samples.append({
                    'control_image': pair['control'],
                    'target_image': pair.get('target', pair['control']),  # Use control as target if not specified
                    'prompt': pair['prompt']
                })

        logger.info(f"Created validation dataset with {len(validation_samples)} control-prompt pairs")
        return validation_samples

    def _select_sample_indices(self, num_samples, global_step):
        """Select sample indices for validation"""
        # Use deterministic selection based on global_step and seed for reproducible results
        random.seed(self.config.seed + global_step)

        available_indices = list(range(len(self.cached_embeddings)))
        if num_samples >= len(available_indices):
            return available_indices

        return random.sample(available_indices, num_samples)

    def _log_validation_sample(self, global_step, sample_idx, cached_sample, generated_image, trainer):
        """Log validation sample to tensorboard using trainer's logging capabilities"""
        try:
            # Log generated image
            trainer._log_image_to_tensorboard(
                f"validation/generated/step_{global_step}/sample_{sample_idx}",
                generated_image,
                global_step
            )

            # Log original images for comparison
            original_sample = cached_sample['original_sample']

            if 'control_image' in original_sample:
                control_image = original_sample['control_image']
                if isinstance(control_image, str):
                    control_image = Image.open(control_image)

                trainer._log_image_to_tensorboard(
                    f"validation/control/step_{global_step}/sample_{sample_idx}",
                    control_image,
                    global_step
                )

            if 'target_image' in original_sample:
                target_image = original_sample['target_image']
                if isinstance(target_image, str):
                    target_image = Image.open(target_image)

                trainer._log_image_to_tensorboard(
                    f"validation/target/step_{global_step}/sample_{sample_idx}",
                    target_image,
                    global_step
                )

            # Log prompt as text
            if 'prompt' in original_sample:
                trainer._log_text_to_tensorboard(
                    f"validation/prompts/step_{global_step}/sample_{sample_idx}",
                    original_sample['prompt'],
                    global_step
                )

        except Exception as e:
            logger.error(f"Failed to log validation sample {sample_idx}: {e}")
            self.accelerator.print(f"Failed to log validation sample {sample_idx}: {e}")

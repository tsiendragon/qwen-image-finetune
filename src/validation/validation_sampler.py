"""
ValidationSampler for monitoring training progress through image sampling.
Supports both FluxKontext and QwenImageEdit trainers.
"""

import random
import logging
from typing import Dict, List, Optional, Union
import torch
import numpy as np
import os
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from PIL import Image
from accelerate import Accelerator
from src.data.config import ValidationDataConfig
from src.data.config import DataConfig
import math
from src.utils.tools import sample_indices_per_rank
from src.data.dataset import ImageDataset
import cv2

logger = logging.getLogger(__name__)


class ValidationSampler:
    """Universal validation sampler that works with any model architecture"""

    def __init__(
        self,
        config: ValidationDataConfig,
        accelerator: Accelerator,
        weight_dtype: torch.dtype = torch.bfloat16,
        data_config: DataConfig | None = None,
    ):
        self.config = config
        self.data_config = data_config
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
            'samples_per_process': 1
        }
        self.get_accelerator_attribute()
        self.setup_validation_dataset()

    def get_accelerator_attribute(self):
        self.local_rank = self.accelerator.local_process_index    # 本机内 rank
        self.world_size = self.accelerator.num_processes          # 总进程数（= 全局并行数）

    def setup_validation_dataset(self, train_dataset=None):

        """Setup validation dataset from config or training dataset"""
        # only return control and prompt
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

    def _repeat_datasets(self, train_dataset: Union[Dataset, List[Dict]]):
        dataset_size = len(train_dataset)
        if dataset_size < self.world_size * self.config.num_samples:
            # repeat the dataset to match the size use concat dataset
            repeat_num = math.ceil(self.world_size * self.config.num_samples / dataset_size)
            if isinstance(train_dataset, List):
                train_dataset = train_dataset * repeat_num
            else:
                train_dataset = ConcatDataset([train_dataset] * repeat_num)
        return train_dataset

    def _create_subset_from_training(self, train_dataset):
        """Create validation subset from training dataset"""
        if train_dataset is None:
            logger.warning("No training dataset provided for validation subset creation")
            return []
        assert self.data_config is not None, "data_config is required for using training dataset"
        train_dataset = self._repeat_datasets(train_dataset)

        self.selected_indices = sample_indices_per_rank(
            self.accelerator,
            len(train_dataset),
            self.config.num_samples,
            seed=self.config.seed,
            replacement=False,
            global_shuffle=False)

        # Use first few samples as validation set
        self.selected_datasets = []  # List dict
        for i in self.selected_indices:
            data_item = train_dataset[i]
            new_data = {}
            if 'prompt' in data_item:
                new_data['prompt'] = data_item['prompt']
            if 'control' in data_item:  # suppose control is a Union[list[tensor], tensor]
                new_data['control'] = data_item['control']
            self.selected_datasets.append(data_item)

    def _load_dataset_from_path(self, dataset_path: str):
        """Load validation dataset from file path"""
        assert self.data_config is not None, "data_config is required to instantiated dataset class"
        init_args = self.data_config.init_args
        init_args['dataset_path'] = dataset_path
        dataset = ImageDataset(init_args)
        return self._create_subset_from_training(dataset)

    def _create_dataset_from_pairs(self, pairs: List[Dict]):
        """Create validation dataset from control-prompt pairs"""
        validation_samples = []  # List dict

        for pair in pairs:
            if 'control' in pair and 'prompt' in pair:
                prompt = pair['prompt']
                with open(pair['prompt'], 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                control = pair['control']
                control = cv2.imread(control)
                control = cv2.cvtColor(control, cv2.COLOR_BGR2RGB)
                control = control.transpose(2, 0, 1)
                validation_samples.append({
                    'control': control,
                    'prompt': prompt,
                })
        return self._create_subset_from_training(validation_samples)

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

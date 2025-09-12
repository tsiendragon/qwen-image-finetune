"""
ValidationSampler for monitoring training progress through image sampling.
Supports both FluxKontext and QwenImageEdit trainers.
"""
import logging
from typing import Dict, List, Union
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset
from accelerate import Accelerator
from src.data.config import ValidationDataConfig
from src.data.config import DataConfig
import math
import cv2

from src.utils.logger import log_images_auto
from src.utils.tools import sample_indices_per_rank
from src.data.dataset import ImageDataset

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
            "max_image_size": 512,
            "log_prefix": "validation",
            "move_vae_to_cpu_after": True,
            "skip_on_error": True,
            "samples_per_process": 1,
        }
        self.get_accelerator_attribute()
        self.setup_validation_dataset()

    def get_accelerator_attribute(self):
        self.local_rank = self.accelerator.local_process_index  # 本机内 rank
        self.world_size = self.accelerator.num_processes  # 总进程数（= 全局并行数）

    def setup_validation_dataset(self, train_dataset=None):
        """Setup validation dataset from config or training dataset"""
        # only return control and prompt
        if self.config.validation_data is None:
            # Use subset of training dataset
            self._create_subset_from_training(train_dataset)
        elif isinstance(self.config.validation_data, str):
            # Load from dataset path
            self._load_dataset_from_path(self.config.validation_data)
        elif isinstance(self.config.validation_data, list):
            # Use control-prompt pairs
            self._create_dataset_from_pairs(self.config.validation_data)
        else:
            raise ValueError(
                f"Unsupported validation_data format: {type(self.config.validation_data)}"
            )

    def _repeat_datasets(self, train_dataset: Union[Dataset, List[Dict]]):
        dataset_size = len(train_dataset)
        if dataset_size < self.world_size * self.config.num_samples:
            # repeat the dataset to match the size use concat dataset
            repeat_num = math.ceil(
                self.world_size * self.config.num_samples / dataset_size
            )
            if isinstance(train_dataset, List):
                train_dataset = train_dataset * repeat_num
            else:
                train_dataset = ConcatDataset([train_dataset] * repeat_num)
        return train_dataset

    def _create_subset_from_training(self, train_dataset: Union[Dataset, List[Dict]]):
        """Create validation subset from training dataset"""
        if train_dataset is None:
            logger.warning(
                "No training dataset provided for validation subset creation"
            )
            return []
        assert (
            self.data_config is not None
        ), "data_config is required for using training dataset"
        train_dataset = self._repeat_datasets(train_dataset)

        self.selected_indices = sample_indices_per_rank(
            self.accelerator,
            len(train_dataset),
            self.config.num_samples,
            seed=self.config.seed,
            replacement=False,
            global_shuffle=False,
        )

        # Use first few samples as validation set
        self.selected_datasets = []  # List dict
        for i in self.selected_indices:
            data_item = train_dataset[i]
            new_data = {}
            if "prompt" in data_item:
                new_data["prompt"] = data_item["prompt"]
            if (
                "control" in data_item
            ):  # suppose control is a Union[list[tensor], tensor]
                new_data["control"] = data_item["control"]
            self.selected_datasets.append(data_item)

    def _load_dataset_from_path(self, dataset_path: str):
        """Load validation dataset from file path"""
        assert (
            self.data_config is not None
        ), "data_config is required to instantiated dataset class"
        init_args = self.data_config.init_args
        init_args["dataset_path"] = dataset_path
        dataset = ImageDataset(init_args)
        return self._create_subset_from_training(dataset)

    def _create_dataset_from_pairs(self, pairs: List[Dict]):
        """Create validation dataset from control-prompt pairs"""
        validation_samples = []  # List dict

        for pair in pairs:
            if "control" in pair and "prompt" in pair:
                prompt = pair["prompt"]
                with open(pair["prompt"], "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                control = pair["control"]
                control = cv2.imread(control)
                control = cv2.cvtColor(control, cv2.COLOR_BGR2RGB)
                control = control.transpose(2, 0, 1)
                validation_samples.append(
                    {
                        "control": control,
                        "prompt": prompt,
                    }
                )
        return self._create_subset_from_training(validation_samples)

    def cache_embeddings(self, trainer):
        """Cache embeddings for all validation samples using trainer's methods"""
        self.cached_embeddings = []
        for idx, sample in enumerate(self.selected_datasets):
            cached_sample = {"sample_idx": idx}
            cached_sample["control_latents"] = trainer.encode_vae_image_for_validation(
                sample["control"]
            )
            cached_sample["text_embeddings"] = trainer.encode_prompt_for_validation(
                sample["prompt"], sample["control"]
            )
            self.cached_embeddings.append(cached_sample)
        logging.info(
            f"rank [{self.local_rank}] Successfully cached embeddings for "
            f"{len(self.cached_embeddings)} samples"
        )
        self.embeddings_cached = True

    def should_run_validation(self, global_step: int) -> bool:
        """Check if validation should run at current step"""
        if self.config.validation_steps <= 0:
            return False
        return global_step % self.config.validation_steps == 0

    def run_validation_loop(self, global_step: int, trainer):
        """Main validation loop using cached embeddings and trainer's methods"""
        if not self.embeddings_cached:
            self.accelerator.print(
                "Warning: Embeddings not cached, skipping validation"
            )
            return

        try:
            # Sample from cached embeddings
            for data, cache_embedding in zip(
                self.selected_datasets, self.cached_embeddings
            ):
                prompt = data["prompt"]
                control = data["control"]  # C,H,W [0,222] np.ndarray
                control = torch.from_numpy(control).unsqueeze(0)
                control = (control - 127.5) / 255
                log_images_auto(
                    self.accelerator,
                    f"control_{self.local_rank}",
                    control,
                    global_step,
                    caption=prompt,
                )
                # Generate sample using cached embeddings and trainer's model
                generated_image = trainer.sampling_from_embeddings(cache_embedding)
                log_images_auto(
                    self.accelerator,
                    f"generated_image_{self.local_rank}",
                    generated_image,
                    global_step,
                    caption=prompt,
                )

        except Exception as e:
            logger.error(f"Validation sampling failed at step {global_step}: {e}")
            if not self._internal_config["skip_on_error"]:
                raise

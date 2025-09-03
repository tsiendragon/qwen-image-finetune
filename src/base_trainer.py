"""
Abstract Base Trainer for all trainer implementations.
Defines the core interface that all trainers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import torch
from accelerate import Accelerator
import logging

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all trainer implementations.
    Defines the core interface that all trainers must implement.
    """

    def __init__(self, config):
        """Initialize trainer with configuration."""
        self.config = config
        self.accelerator: Optional[Accelerator] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0

        # Common attributes that all trainers should have
        self.weight_dtype = torch.bfloat16
        self.batch_size = config.data.batch_size
        self.use_cache = config.cache.use_cache
        self.cache_dir = config.cache.cache_dir

        logger.info(f"Initialized {self.__class__.__name__} with config")

    @abstractmethod
    def load_model(self, **kwargs):
        """Load and initialize model components."""
        pass

    @abstractmethod
    def cache(self, train_dataloader):
        """Pre-compute and cache embeddings/latents for training efficiency."""
        pass

    @abstractmethod
    def fit(self, train_dataloader):
        """Main training loop implementation."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Inference/prediction method."""
        pass

    @abstractmethod
    def set_model_devices(self, mode: str = "train"):
        """Set model device allocation based on different modes."""
        pass

    @abstractmethod
    def encode_prompt(self, *args, **kwargs):
        """Encode text prompts to embeddings."""
        pass

    # Common methods that can be shared across implementations
    def setup_accelerator(self):
        """Initialize accelerator and logging configuration."""
        # This will be implemented by child classes with specific logic
        # but can contain common initialization code
        pass

    def save_checkpoint(self, epoch: int, global_step: int):
        """Save model checkpoint."""
        # Common checkpoint saving logic can be implemented here
        pass

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Common optimizer configuration can be implemented here
        pass

    def setup_versioned_logging_dir(self):
        """Set up versioned logging directory."""
        # Common logging directory setup can be implemented here
        pass

    def get_model_type(self) -> str:
        """Get the model type identifier."""
        return getattr(self.config.model, 'model_type', 'unknown')

    def get_precision_info(self) -> Dict[str, Any]:
        """Get precision and quantization information."""
        return {
            'weight_dtype': str(self.weight_dtype),
            'mixed_precision': getattr(self.config.train, 'mixed_precision', 'none'),
            'quantize': getattr(self.config.model, 'quantize', False)
        }

    def log_model_info(self):
        """Log model information."""
        logger.info(f"Model Type: {self.get_model_type()}")
        logger.info(f"Precision Info: {self.get_precision_info()}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Use Cache: {self.use_cache}")

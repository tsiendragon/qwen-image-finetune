"""Qwen Image Finetune - Parameter-efficient fine-tuning for Qwen image editing models"""

__version__ = "1.2.0"
__author__ = "Qwen Image Finetune Team"
__description__ = "A framework for fine-tuning Qwen image editing models with LoRA and quantization support"

from . import src

__all__ = ["src", "__version__"]
"""Qwen Image Finetune - Parameter-efficient fine-tuning for Qwen image editing models"""

from pathlib import Path

# Read version from VERSION file
_version_file = Path(__file__).parent / "VERSION"
if _version_file.exists():
    __version__ = _version_file.read_text().strip()
else:
    __version__ = "unknown"

__author__ = "Lilong"
__description__ = "A framework for fine-tuning Qwen image editing models with LoRA and quantization support"

try:
    from . import src
    __all__ = ["src", "__version__"]
except ImportError:
    # Handle case when imported directly
    __all__ = ["__version__"]
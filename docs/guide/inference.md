# Inference Guide

This guide covers how to use trained Qwen Image Edit models for inference and image generation.

## Quick Start

### Basic Inference

```python
from qflux.qwen_image_edit_trainer import QwenImageEditTrainer
from qflux.data.config import load_config_from_yaml
import numpy as np
from PIL import Image

# Load configuration
config = load_config_from_yaml("configs/qwen_image_edit_config.yaml")

# Initialize trainer
trainer = QwenImageEditTrainer(config)

# Setup for prediction
trainer.setup_predict()

# Load your input image
input_image = Image.open("path/to/your/image.jpg")

# Generate edited image
result = trainer.predict(
    prompt_image=input_image,
    prompt="Make the sky more dramatic and add golden hour lighting",
    negative_prompt="blurry, low quality, artifacts",
    num_inference_steps=20,
    true_cfg_scale=4.0
)

# Save result
result[0].save("output.jpg")
```

## Configuration for Inference

### Multi-GPU Setup

For optimal performance with multiple GPUs:

```yaml
# configs/inference_config.yaml
predict:
  devices:
    vae: "cuda:0"              # VAE encoder/decoder
    text_encoder: "cuda:1"     # Text encoder
    transformer: "cuda:2"      # Main transformer model

model:
  quantize: true               # Enable FP8 quantization for speed
```

## Inference API

### QwenImageEditTrainer.predict()

Main inference method with comprehensive parameters:

```python
def predict(
    self,
    prompt_image: np.ndarray,           # Input image as numpy array
    prompt: str,                        # Text prompt for editing
    negative_prompt: str = "",          # Negative prompt to avoid
    num_inference_steps: int = 20,      # Number of denoising steps
    true_cfg_scale: float = 4.0,        # Classifier-free guidance scale
)
```

#### Parameters

- **`prompt_image`**: Input image as numpy array (H, W, 3) in RGB format
- **`prompt`**: Text description of desired edits
- **`negative_prompt`**: Text describing what to avoid in the output
- **`num_inference_steps`**: More steps = higher quality but slower
- **`true_cfg_scale`**: Higher values = stronger prompt adherence

#### Returns

- **`PIL.Image`**: Generated image as PIL.Image

### Advanced Usage

#### Using Trained LoRA Models

```python
# Load and use trained LoRA weights
from qflux.qwen_image_edit_trainer import QwenImageEditTrainer
from qflux.data.config import load_config_from_yaml
from PIL import Image

# Load configuration
config = load_config_from_yaml("configs/face_seg_config.yaml")

# Initialize trainer
trainer = QwenImageEditTrainer(config)

# Load your trained LoRA weights
trainer.load_lora("outputs/lora_weights.pth")

# Setup for prediction
trainer.setup_predict()

# Use the trained model for inference
input_image = Image.open("data/face_seg/control_images/060002_4_028450_FEMALE_30.jpg")
result = trainer.predict(
    prompt_image=input_image,
    prompt="change the image from the face to the face segmentation mask",
    num_inference_steps=20,
    true_cfg_scale=4.0
)

# Save result
result[0].save("face_segmentation_output.png")
```

#### Batch Inference

```python
# Process multiple images
images = [img1, img2, img3]
prompts = ["edit1", "edit2", "edit3"]
result = trainer.predict(images, prompts)
```

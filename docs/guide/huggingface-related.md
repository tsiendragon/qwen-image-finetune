### Hugging Face Related Features Guide

This document describes how to use Hugging Face Hub for the following operations:
1. Upload and download image editing datasets
2. Upload and download LoRA model weights
3. Manage additional files (configuration files, etc.)

Implementation details can be found in `src/utils/huggingface.py` and example script `upload_dataset.py`.

---

## Directory Structure Standards

The data root directory should contain `train/` and/or `test/` subdirectories, each containing:

- `control_images/`: Control images (at least 1). Supports multiple control images with naming rules:
  - Main image: `<base>.*` (any extension)
  - Additional numbered images: `<base>_control_1.*`, `<base>_control_2.*`, ... (using _control_N format)
  - Optional mask: `<base>_mask.*` (if exists, will be used as `control_mask`)
- `training_images/`: Target images and text prompt pairs:
  - Required: `<base>.txt` (prompt)
  - Optional: `<base>.*` (target image, can be omitted in test sets)

Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` (case insensitive).

Example:

```
face_seg/
  train/
    control_images/
      sampleA.jpg
      sampleA_control_1.jpg
      sampleA_mask.png
    training_images/
      sampleA.txt
      sampleA.png
  test/
    control_images/
      sampleB.png
    training_images/
      sampleB.txt
```

---

## Environment Setup

Install dependencies:

```bash
pip install -U datasets huggingface_hub hf-transfer
```

Configure access token (choose one):

```bash
export HUGGINGFACE_HUB_TOKEN=hf_xxx
# or
export HF_TOKEN=hf_xxx
```

To accelerate uploads, enable:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## Dataset Schema (Features)

Unified HF Dataset schema:

- `id: string` (sample base name)
- `control_images: Sequence[Image]` (≥1 images)
- `control_mask: Image | None` (optional)
- `target_image: Image | None` (recommended for training sets, can be empty for test sets)
- `prompt: string` (read from `<base>.txt`)

---

## Upload to Hugging Face Hub

### Method 1: Standard Directory Structure Upload

Use script `upload_dataset.py` (internally calls `upload_editing_dataset`):

```python
from qflux.utils.huggingface import upload_editing_dataset

upload_editing_dataset(
    root_dir="face_seg",                 # contains train/ and/or test/
    repo_id="<org_or_user>/<dataset>",  # e.g.: "TsienDragon/face_segmentation_20"
    private=False                        # public or private
)
```

Or run the example script directly (modify as needed):

```bash
python upload_dataset.py
```

### Method 2: CSV Metadata Upload

For datasets with CSV metadata files, use `upload_editing_dataset_from_csv`:

```python
from qflux.utils.huggingface import upload_editing_dataset_from_csv

upload_editing_dataset_from_csv(
    root_dir="/path/to/dataset",         # contains train.csv and/or test.csv
    repo_id="<org_or_user>/<dataset>",  # e.g.: "TsienDragon/character-composition"
    private=True                         # public or private
)
```

**CSV Format Requirements:**

- **train.csv**: Should contain columns `image,control,control_1,prompt`
- **test.csv**: Should contain columns `control,control_1,prompt` (no target image)
- **Flexible columns**: Supports `control_2`, `control_3`, etc. for additional control images
- **Path format**: All paths should be relative to the root directory

**Example CSV structure:**

```
root_dir/
├── train.csv                    # metadata for training split
├── test.csv                     # metadata for test split
├── train/
│   ├── target/
│   │   ├── sample1.png         # target images
│   │   └── sample1_mask.png    # optional mask files
│   └── control/
│       ├── sample1.png         # main control images
│       └── sample1_control_1.webp  # additional control images
└── test/
    └── control/
        ├── sample2.webp        # test control images
        └── sample2_control_1.png
```

**Example train.csv:**
```csv
image,control,control_1,prompt
train/target/sample1.png,train/control/sample1.png,train/control/sample1_control_1.webp,Add character to scene
```

**Example test.csv:**
```csv
control,control_1,prompt
test/control/sample2.webp,test/control/sample2_control_1.png,Generate character composition
```

**Advanced Features:**

- **Multi-format Support**: Automatically detects and supports `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, `.tif` (case insensitive)
- **Format Auto-correction**: If CSV records incorrect format (e.g., `.png` but actual file is `.webp`), system automatically finds the correct file
- **Flexible Mask Detection**: Searches for mask files in both `control/` and `target/` directories
- **Mixed Format Support**: Each control image can have different formats (e.g., main control is `.png`, additional control is `.webp`)

**Notes:**

- First run will automatically create HF dataset repository (if it doesn't exist).
- Error will be raised if both `train/` and `test/` are missing; at least one must be provided.
- If a sample lacks control images, it will be skipped with a warning in the logs.
- CSV method is more flexible for complex datasets with mixed formats or non-standard directory structures.

---

## Download/Load from Hub

Unified loading interface:

```python
from qflux.utils.huggingface import load_editing_dataset

dsd = load_editing_dataset("<org_or_user>/<dataset>")  # returns DatasetDict
print(dsd)

# Get a sample from training set
sample = dsd["train"][0]
print(type(sample["control_images"]))
print(type(sample["control_mask"]))
print(type(sample["target_image"]))
print(type(sample["prompt"]))
```

For private datasets, set `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN` environment variable in advance.

Load only a single split if needed:

```python
train_ds = load_editing_dataset("<org_or_user>/<dataset>", split="train")
```

---

## Common Issues and Troubleshooting

- Missing token: Error message prompts to set `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN`.
- Split not found: Ensure `train/` or `test/` subdirectories exist under root directory.
- No valid samples:
  - Check if `training_images/` contains `<base>.txt` (prompt required).
  - Check if `control_images/` contains `<base>.*` or `<base>_control_1.*` etc. control images.
  - Confirm additional control images use correct naming format (e.g., `_control_1`, `_control_2`).
- Mask not effective: Confirm naming as `<base>_mask.<ext>` and located under `control_images/`.

---

## LoRA Model Upload and Download

### LoRA Model Upload

Use `upload_lora_safetensors` function to upload trained LoRA weights to Hugging Face Hub (model repository):

```python
from qflux.utils.huggingface import upload_lora_safetensors

# Basic upload (LoRA weights file only)
repo_url = upload_lora_safetensors(
    src_path="/path/to/pytorch_lora_weights.safetensors",  # local LoRA file path
    repo_id="<org_or_user>/<model_name>",                 # e.g.: "TsienDragon/qwen-image-edit-face-seg"
    private=True,                                          # private or public repository
    commit_message="Upload LoRA face segmentation model"
)

# Advanced upload (with additional configuration files)
repo_url = upload_lora_safetensors(
    src_path="/path/to/output/directory",                  # directory containing LoRA file
    repo_id="<org_or_user>/<model_name>",
    private=False,
    remote_name="pytorch_lora_weights.safetensors",        # custom remote filename (optional)
    commit_message="Upload trained LoRA model",
    extra_files={                                          # additional file uploads
        "/path/to/train_config.yaml": "train_config.yaml",
        "/path/to/README.md": "README.md",
        "/path/to/adapter_config.json": "adapter_config.json"
    }
)

print(f"Model uploaded to: {repo_url}")
```

**Parameter Description:**

- `src_path`: Local LoRA file path or directory containing the file
- `repo_id`: Hugging Face repository ID, format "username/repository-name"
- `private`: Whether to create private repository (default True)
- `remote_name`: Remote filename (optional, defaults to `pytorch_lora_weights.safetensors`)
- `commit_message`: Commit message
- `extra_files`: Additional file mapping dictionary `{local_path: remote_path}`

**Features:**

- **Smart Skip**: Automatically skips upload if remote file with same content exists
- **Auto Repository Creation**: Automatically creates repository if target doesn't exist
- **Multi-file Support**: Supports simultaneous upload of training configs, README, etc.
- **SHA256 Verification**: Ensures upload integrity through file hashing

### LoRA Model Download

Use `download_lora` function to download LoRA weights from Hugging Face Hub:

```python
from qflux.utils.huggingface import download_lora

# Download LoRA weights from specified repository
local_path = download_lora(
    repo_id="TsienDragon/qwen-image-edit-lora-face-segmentation",
    filename="pytorch_lora_weights.safetensors"  # optional, defaults to this filename
)

print(f"LoRA weights downloaded to: {local_path}")

# Load LoRA weights in trainer
trainer.load_lora(local_path)
```

### Usage Example

Complete workflow example:

```python
# 1. Upload LoRA model after training completion
lora_output_dir = "/tmp/my_face_seg_lora/output"
config_file = "/path/to/train_config.yaml"

repo_url = upload_lora_safetensors(
    src_path=lora_output_dir,
    repo_id="myusername/face-segmentation-lora",
    private=False,
    extra_files={
        config_file: "train_config.yaml"
    }
)

# 2. Download and use model elsewhere
lora_path = download_lora(
    repo_id="myusername/face-segmentation-lora"
)

# 3. Load into trainer for inference
trainer.load_lora(lora_path)
result = trainer.predict(...)
```

---

## Environment Configuration

### Authentication Setup

All operations require a Hugging Face access token. Set environment variable (choose one):

```bash
export HUGGINGFACE_HUB_TOKEN=hf_xxx
# or
export HF_TOKEN=hf_xxx
```

### Upload Acceleration

To improve large file upload speed, enable:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## Reference Implementation

- Dataset and model upload/download implementation: `src/utils/huggingface.py`
- Dataset upload example: `upload_dataset.py`

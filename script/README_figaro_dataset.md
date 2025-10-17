# Figaro Hair Segmentation Dataset Download Script

This script downloads and organizes the Figaro Hair Segmentation dataset from Hugging Face Hub (`Allison/figaro_hair_segmentation_1000`) into the directory structure required by the training pipeline.

## Quick Start

```bash
# 1. Set your HuggingFace token (required for upload)
export HF_TOKEN=hf_xxxxxxxxxxxxx

# 2. Download, organize, and upload to HuggingFace in one command
python script/download_figaro_dataset.py \
    --output_dir /tmp/figaro_hair_seg \
    --upload \
    --upload_repo_id TsienDragon/figaro_hair_segmentation_1k

# 3. Use in your training config
# data:
#   init_args:
#     dataset_path:
#       - split: train
#         repo_id: TsienDragon/figaro_hair_segmentation_1k
```

## Dataset Information

- **Source**: [Allison/figaro_hair_segmentation_1000](https://huggingface.co/datasets/Allison/figaro_hair_segmentation_1000)
- **Task**: Hair segmentation from images
- **Size**: 1,000 images with hair segmentation masks
- **Categories**: 7 hairstyle classes (straight, wavy, curly, kinky, braids, dreadlocks, short)

## Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
# or specifically:
pip install datasets huggingface_hub Pillow tqdm
```

## Usage

### Basic Usage

```bash
python script/download_figaro_dataset.py --output_dir /path/to/output
```

### Download and Upload to HuggingFace Hub

```bash
# Download, organize, and upload to TsienDragon/figaro_hair_segmentation_1k
python script/download_figaro_dataset.py \
    --output_dir /tmp/figaro_hair_seg \
    --upload

# Upload to custom repository
python script/download_figaro_dataset.py \
    --output_dir /tmp/figaro_hair_seg \
    --upload \
    --upload_repo_id your-org/your-dataset-name

# Upload as private dataset
python script/download_figaro_dataset.py \
    --output_dir /tmp/figaro_hair_seg \
    --upload \
    --upload_private
```

### With HuggingFace Token (for private datasets)

```bash
python script/download_figaro_dataset.py \
    --output_dir /path/to/output \
    --hf_token hf_xxxxxxxxxxxxx
```

### Custom Source Repository

```bash
python script/download_figaro_dataset.py \
    --output_dir /path/to/output \
    --repo_id your-org/your-dataset
```

### Complete Example

```bash
# Download, organize, and upload in one command
export HF_TOKEN=hf_xxxxxxxxxxxxx

python script/download_figaro_dataset.py \
    --output_dir /tmp/figaro_hair_seg \
    --upload \
    --upload_repo_id TsienDragon/figaro_hair_segmentation_1k

# Expected output:
# Downloading dataset from Allison/figaro_hair_segmentation_1000...
# Processing train split (800 samples) -> train/
# Processing test split (200 samples) -> test/
# Dataset organization complete!
# Uploading to HuggingFace Hub: TsienDragon/figaro_hair_segmentation_1k
# ✓ Successfully uploaded to: https://huggingface.co/datasets/TsienDragon/figaro_hair_segmentation_1k
```

## Output Structure

The script organizes the dataset according to the structure required by `docs/data-preparation.md`:

```
output_dir/
├── train/
│   ├── control_images/
│   │   ├── sample_00000.jpg          # Control: input photo
│   │   ├── sample_00001.jpg
│   │   └── ...
│   └── training_images/
│       ├── sample_00000.png          # Target: segmentation mask
│       ├── sample_00000.txt          # Prompt text
│       ├── sample_00001.png
│       ├── sample_00001.txt
│       └── ...
└── test/
    ├── control_images/
    │   └── ...
    └── training_images/
        └── ...
```

## Dataset Mapping

The script performs the following transformations:

1. **Control Image**: Photo (from 'image' field) saved to `control_images/<id>.jpg`
2. **Target Image**: Segmentation mask (from 'label' field) saved to `training_images/<id>.png`
3. **Prompt**: Auto-generated text prompt saved to `training_images/<id>.txt`

**Task**: Learn to generate hair segmentation masks from input photos.

### Prompt Generation

The script generates varied prompts for better generalization:
- "Generate hair segmentation mask from the image"
- "Create binary mask for hair region"
- "Segment the hair in this photo"
- "Extract hair segmentation from the image"
- "Produce hair mask for the given image"

## Configuration Example

After downloading, you can use the dataset in your training configuration:

### Option 1: Local Folder

```yaml
data:
  class_path: src.data.dataset.ImageDataset
  init_args:
    dataset_path:
      - /path/to/output/train
    use_cache: true
    cache_dir: /tmp/figaro_cache
```

### Option 2: Use from HuggingFace Hub (after upload)

After uploading with `--upload` flag:

```yaml
data:
  class_path: src.data.dataset.ImageDataset
  init_args:
    dataset_path:
      - split: train
        repo_id: TsienDragon/figaro_hair_segmentation_1k
    use_cache: true
    cache_dir: /tmp/figaro_cache
```

## Script Details

### Key Features

1. **Automatic Directory Setup**: Creates required `control_images/` and `training_images/` folders
2. **Split Handling**: Automatically maps splits (e.g., 'validation' → 'test')
3. **Progress Tracking**: Shows progress bar during download and processing
4. **Error Handling**: Skips problematic samples with warnings
5. **Summary Report**: Prints detailed statistics after completion
6. **HuggingFace Upload**: Optionally upload organized dataset directly to HuggingFace Hub with `--upload` flag

### What the Script Does

1. Downloads the dataset from HuggingFace Hub
2. Creates the required directory structure (train/test splits)
3. For each sample:
   - Saves photo (from 'image' field) as **control** image to `control_images/<id>.jpg`
   - Saves segmentation mask (from 'label' field) as **target** image to `training_images/<id>.png`
   - Generates and saves text prompt to `training_images/<id>.txt`
4. Validates and reports statistics
5. Optionally uploads organized dataset to HuggingFace Hub

## Validation

After running the script, verify the output:

```bash
# Check train split
ls /path/to/output/train/control_images/*.jpg | wc -l    # Control photos
ls /path/to/output/train/training_images/*.png | wc -l   # Target masks
ls /path/to/output/train/training_images/*.txt | wc -l   # Prompts

# Check test split
ls /path/to/output/test/control_images/*.jpg | wc -l
ls /path/to/output/test/training_images/*.png | wc -l
ls /path/to/output/test/training_images/*.txt | wc -l
```

Expected output (approximate):
```
800  # train control photos
800  # train target masks
800  # train prompts
200  # test control photos
200  # test target masks
200  # test prompts
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'datasets'"
- **Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: "Failed to download dataset"
- **Solution**: Check your internet connection and HuggingFace token (if required)
- **Solution**: Verify the dataset exists: https://huggingface.co/datasets/Allison/figaro_hair_segmentation_1000

### Issue: "Permission denied"
- **Solution**: Ensure you have write permissions for the output directory

### Issue: "Missing splits"
- **Solution**: The script automatically maps common split names (e.g., 'validation' -> 'test')

### Issue: Dataset structure is different
- **Solution**: Check the dataset schema on HuggingFace, the script expects 'image' and 'label' fields
- **Solution**: Modify the `save_sample()` function to match your dataset structure

### Issue: Upload fails with authentication error
- **Solution**: Set your HuggingFace token:
  ```bash
  export HF_TOKEN=hf_xxxxxxxxxxxxx
  # or
  export HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxxx
  ```
- **Solution**: Pass token via command line: `--hf_token hf_xxxxxxxxxxxxx`
- **Solution**: Ensure your token has write permissions

### Issue: Upload repo already exists
- **Solution**: The script will update the existing repository
- **Solution**: If you want a fresh start, delete the old repo on HuggingFace first

## Related Documentation

- [Data Preparation Guide](../docs/data-preparation.md)
- [HuggingFace Integration](../docs/huggingface-related.md)
- [Training Guide](../docs/training.md)

## Advanced Usage

### Customizing Prompts

Edit the `generate_prompt()` function to customize prompts:

```python
def generate_prompt(idx: int, split: str = 'train') -> str:
    """Generate a descriptive prompt for hair segmentation task."""
    prompts = [
        "Your custom prompt 1",
        "Your custom prompt 2",
        # Add more prompts...
    ]
    return prompts[idx % len(prompts)]
```

### Using Different Image Formats

The script saves control/target images as JPEG and masks as PNG by default. To change:

```python
# In save_sample() function, modify:
image.convert('RGB').save(control_path, 'PNG')  # Change to PNG
```

### Processing Only Specific Splits

Modify the script to process only certain splits:

```python
# In download_and_organize() function, filter splits:
for split_name in available_splits:
    if split_name not in ['train']:  # Process only train
        continue
    # ... rest of the code
```

# Figaro Hair Segmentation Dataset - Implementation Summary

## Overview

Successfully implemented a complete pipeline to download, organize, and upload the Figaro Hair Segmentation dataset from `Allison/figaro_hair_segmentation_1000` to `TsienDragon/figaro_hair_segmentation_1k`.

## Files Created

1. **`script/download_figaro_dataset.py`** (290 lines)
   - Main script for downloading and organizing dataset
   - Includes optional HuggingFace upload functionality

2. **`script/README_figaro_dataset.md`** (260+ lines)
   - Comprehensive documentation
   - Usage examples and troubleshooting guide

## Key Features Implemented

### 1. Dataset Download
- Downloads from `Allison/figaro_hair_segmentation_1000`
- Handles the dataset structure with `image` and `label` fields
- Supports train/test/validation splits
- Auto-maps validation → test split

### 2. Dataset Organization
Transforms the dataset to match `docs/data-preparation.md` structure:

```
Input (HuggingFace):              Output (Organized):
┌──────────────┐                 ┌─────────────────────────────────┐
│ image        │ ───────────────> │ control_images/sample_X.jpg     │
│ (photo)      │                 │ (control: input photo)          │
├──────────────┤                 ├─────────────────────────────────┤
│ label        │ ───────────────> │ training_images/sample_X.png    │
│ (mask)       │                 │ (target: segmentation mask)     │
├──────────────┤                 ├─────────────────────────────────┤
│ (generated)  │ ───────────────> │ training_images/sample_X.txt    │
└──────────────┘                 └─────────────────────────────────┘
```

**Task**: Learn to generate hair segmentation masks from input photos.

### 3. Prompt Generation
Auto-generates 5 varied prompts for better training generalization:
- "Generate hair segmentation mask from the image"
- "Create binary mask for hair region"
- "Segment the hair in this photo"
- "Extract hair segmentation from the image"
- "Produce hair mask for the given image"

### 4. HuggingFace Upload
- Optional upload with `--upload` flag
- Default repo: `TsienDragon/figaro_hair_segmentation_1k`
- Supports custom repo IDs and private uploads
- Uses existing `upload_editing_dataset()` utility

## Usage Examples

### Basic Download Only
```bash
python script/download_figaro_dataset.py \
    --output_dir /tmp/figaro_hair_seg
```

### Download + Upload (Recommended)
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx

python script/download_figaro_dataset.py \
    --output_dir /tmp/figaro_hair_seg \
    --upload \
    --upload_repo_id TsienDragon/figaro_hair_segmentation_1k
```

### All Options
```bash
python script/download_figaro_dataset.py \
    --output_dir /path/to/output \
    --hf_token hf_xxxxxxxxxxxxx \
    --repo_id Allison/figaro_hair_segmentation_1000 \
    --upload \
    --upload_repo_id TsienDragon/figaro_hair_segmentation_1k \
    --upload_private
```

## Output Structure

```
output_dir/
├── train/
│   ├── control_images/
│   │   ├── sample_00000.jpg          # Control: input photo
│   │   ├── sample_00001.jpg
│   │   └── ... (~1000 photos)
│   └── training_images/
│       ├── sample_00000.png          # Target: segmentation mask
│       ├── sample_00000.txt          # Prompt text
│       ├── sample_00001.png
│       ├── sample_00001.txt
│       └── ... (~1000 masks and prompts)
└── test/
    ├── control_images/
    │   └── ... (if test split exists)
    └── training_images/
        └── ... (if test split exists)
```

## Configuration for Training

### Using Local Dataset
```yaml
data:
  class_path: src.data.dataset.ImageDataset
  init_args:
    dataset_path:
      - /tmp/figaro_hair_seg/train
    use_cache: true
    cache_dir: /tmp/figaro_cache
```

### Using HuggingFace Hub (After Upload)
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

## Technical Details

### Dataset Schema Mapping
- **Source Field**: `image` (PIL.Image, photo) → **Target**: control image (`.jpg`)
- **Source Field**: `label` (PIL.Image, mask) → **Target**: target image (`.png`)
- **Generated**: Text prompts (5 variations, `.txt`)

### Error Handling
- Skips samples with missing control images or target labels
- Warns on non-PIL image types
- Reports statistics and skipped samples

### Code Quality
- ✅ Passes all linter checks (Black/Flake8)
- ✅ Type hints for function signatures
- ✅ Comprehensive logging
- ✅ Progress bars with tqdm
- ✅ Docstrings for all functions

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | **required** | Output directory path |
| `--hf_token` | str | None | HuggingFace token (optional) |
| `--repo_id` | str | `Allison/figaro_hair_segmentation_1000` | Source dataset repo |
| `--upload` | flag | False | Upload to HuggingFace after organizing |
| `--upload_repo_id` | str | `TsienDragon/figaro_hair_segmentation_1k` | Upload target repo |
| `--upload_private` | flag | False | Make uploaded dataset private |

## Dependencies

Required packages (already in `requirements.txt`):
- `datasets` - HuggingFace datasets library
- `huggingface_hub` - Hub interaction
- `Pillow` - Image processing
- `tqdm` - Progress bars

## Workflow

1. **Download**: Loads dataset from HuggingFace Hub
2. **Organize**: Creates directory structure and saves files
3. **Transform**: Converts segmentation dataset to editing format
4. **Validate**: Checks and reports statistics
5. **Upload** (optional): Pushes to HuggingFace Hub

## Expected Results

For the Figaro dataset (~1000 samples):
- **Train split**: ~800-900 samples
- **Test/Val split**: ~100-200 samples

Each split contains:
- Control images: `.jpg` format (input photos)
- Target images: `.png` format (segmentation masks, grayscale or RGB)
- Prompts: `.txt` format (UTF-8)

## Integration with Training Pipeline

The organized dataset follows the exact structure expected by:
- `src.data.dataset.ImageDataset`
- `src.utils.huggingface.upload_editing_dataset`
- All existing training configurations

## Testing Checklist

- [x] Script runs without syntax errors
- [x] Passes linter checks (Black/Flake8)
- [x] Help message displays correctly
- [x] Command-line arguments work
- [x] Import paths are correct
- [x] Dataset schema matches (image, label fields)
- [ ] Full end-to-end test (requires `datasets` package installation)
- [ ] Upload test (requires HF token)

## Next Steps for User

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set HuggingFace token**:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxx
   ```

3. **Run the script**:
   ```bash
   python script/download_figaro_dataset.py \
       --output_dir /tmp/figaro_hair_seg \
       --upload
   ```

4. **Use in training**:
   - Update config to use `TsienDragon/figaro_hair_segmentation_1k`
   - Run training with the organized dataset

## Documentation Links

- Main README: `script/README_figaro_dataset.md`
- Data Preparation: `docs/data-preparation.md`
- HuggingFace Integration: `docs/huggingface-related.md`
- Training Guide: `docs/training.md`

## Notes

- The script handles the specific schema of Figaro dataset (`image` + `label` fields)
- For other datasets with different schemas, modify the `save_sample()` function
- The upload uses the existing `upload_editing_dataset()` utility from `src/utils/huggingface.py`
- All file operations use pathlib for cross-platform compatibility


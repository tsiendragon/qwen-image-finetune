# Data Preparation Guide

This guide covers how to prepare and organize your dataset for training Qwen Image Edit models.

## Dataset Requirements

### Supported Formats

#### Images
- **Formats**: JPG, JPEG, PNG, WebP
- **Color Space**: RGB (3 channels)
- **Resolution**: Flexible (automatically resized during training)
- **Recommended Size**: 512x512 to 1024x1024 pixels

#### Text Prompts
- **Format**: Plain text files (.txt)
- **Encoding**: UTF-8
- **Length**: 10-200 words recommended
- **Content**: Descriptive editing instructions

### Dataset Sources

This project supports three dataset sources: local folder, Hugging Face Hub, and CSV metadata files.

#### 1) Folder (local directory)

Directory structure example:

```
dataset/
├── control_images/               # control/input images
│   ├── sampleA.jpg
│   ├── sampleA_control_1.jpg     # optional extra controls (_control_N)
│   └── sampleA_mask.png          # optional mask named <base>_mask.*
└── training_images/              # target images and prompt text
    ├── sampleA.png               # target image (recommended for train)
    └── sampleA.txt               # required prompt text
```

Key points:
- Control image naming: `<base>.*` for main, plus `<base>_control_1.*`, `<base>_control_2.*`, ...
- Mask naming: `<base>_mask.*`, located in either `control_images/` or `training_images/`.
- Target image is optional for test/validation, but `<base>.txt` prompt is required.

Config example (YAML):

```
data:
  class_path: qflux.data.dataset.ImageDataset
  init_args:
    dataset_path:
      - /path/to/dataset1
      - /path/to/dataset2
    use_cache: true
    cache_dir: /tmp/image_edit_lora/cache
```

#### 2) Hugging Face (Hub datasets)

Use the unified Hub dataset interface with lazy loading:

```
data:
  class_path: qflux.data.dataset.ImageDataset
  init_args:
    dataset_path:
      - split: train
        repo_id: TsienDragon/face_segmentation_20
```

Notes:
- `repo_id` points to the Hub repo, e.g., `OrgOrUser/dataset-name`.
- `split` optionally specifies `train`/`test`.
- See `docs/huggingface-related.md` for schema: `control_images`, `control_mask`, `target_image`, `prompt`.

#### 3) CSV (metadata file)

When using CSV, `dataset_path` should point directly to the `.csv` file. The loader will parse target/control images and prompt from CSV rows.

Minimal required columns:
- `path_target`: target image (used for training; optional for test)
- `path_control_*`: one or more control image columns, named `path_control`, `path_control_1`, `path_control_2`, ...
- `prompt`: prompt text
- Optional: `path_mask` mask path

CSV example (train.csv):

```csv
path_target,path_control,path_control_1,prompt,path_mask
train/target/sample1.png,train/control/sample1.png,train/control/sample1_control_1.webp,Add character to scene,train/target/sample1_mask.png
```

CSV example (test.csv, no target image):

```csv
path_control,path_control_1,prompt
test/control/sample2.webp,test/control/sample2_control_1.png,Generate character composition
```

Config example (pointing to CSV file):

```
data:
  class_path: qflux.data.dataset.ImageDataset
  init_args:
    dataset_path:
      - /path/to/train.csv
    use_cache: true
    cache_dir: /tmp/image_edit_lora/cache
```

For detailed Hub/CSV organization and upload, see [`docs/huggingface-related.md`](huggingface-related.md).

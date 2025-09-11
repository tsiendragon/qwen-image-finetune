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

### Dataset Structure

项目已提供了一个toy数据集示例在 `data/face_seg/` 目录下。数据集应该按照以下结构组织：

```
dataset/
├── control_images/         # 源/输入图像
│   ├── 060002_4_028450_FEMALE_30.png
│   ├── 060003_4_028451_FEMALE_65.png
│   └── ...
└── training_images/        # 目标图像和提示文本
    ├── 060002_4_028450_FEMALE_30.png    # 目标图像
    ├── 060002_4_028450_FEMALE_30.txt    # 编辑指令
    ├── 060003_4_028451_FEMALE_65.png
    ├── 060003_4_028451_FEMALE_65.txt
    └── ...
```

**注意**:
- `control_images/` 包含输入图像（通常为JPG格式）
- `training_images/` 包含目标图像（PNG格式）和对应的文本文件
- 文件名需要保持一致（除了扩展名）

We also support to integrate with Huggingface Dataset

Specify the dataset repo-id in the config as example in the following

```
init_args:
    dataset_path:
      - split: train
        repo_id: TsienDragon/face_segmentation_20
```
Refer [`docs/huggingface-dataset.md`](huggingface-dataset.md) for the preparing dataset to hugggingface or use dataset in huggingface.

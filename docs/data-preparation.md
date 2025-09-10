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

## 使用提供的Toy数据集

### Toy数据集示例

项目在 `data/face_seg/` 目录下提供了一个face segmentation的示例数据集，包含：

- **20个训练样本**
- **人脸到分割mask的转换任务**
- **统一的文本提示**: "change the image from the face to the face segmentation mask"

### 数据集验证

验证数据集结构：

```bash
# 检查数据集结构
ls data/face_seg/control_images/    # 输入图像
ls data/face_seg/training_images/   # 目标图像和文本

# 检查文件匹配
echo "控制图像数量: $(ls data/face_seg/control_images/*.jpg | wc -l)"
echo "训练图像数量: $(ls data/face_seg/training_images/*.png | wc -l)"
echo "文本文件数量: $(ls data/face_seg/training_images/*.txt | wc -l)"
```

## Manual Data Preparation

### Image Processing

#### Resize and Format Conversion

```python
from PIL import Image
import os

def process_images(input_dir, output_dir, target_size=(832, 576)):
    """Process images to consistent format and size"""

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Load image
            img_path = os.path.join(input_dir, filename)
            image = Image.open(img_path).convert('RGB')

            # Resize maintaining aspect ratio
            image.thumbnail(target_size, Image.Resampling.LANCZOS)

            # Create canvas with target size
            canvas = Image.new('RGB', target_size, (0, 0, 0))

            # Center the image
            x = (target_size[0] - image.width) // 2
            y = (target_size[1] - image.height) // 2
            canvas.paste(image, (x, y))

            # Save processed image
            output_path = os.path.join(output_dir, filename)
            canvas.save(output_path, 'JPEG', quality=95)
```

#### Batch Processing

```python
import multiprocessing as mp
from functools import partial

def process_single_image(filename, input_dir, output_dir, target_size):
    """Process a single image"""
    try:
        img_path = os.path.join(input_dir, filename)
        image = Image.open(img_path).convert('RGB')

        # Process image (resize, format, etc.)
        processed = resize_and_pad(image, target_size)

        # Save result
        output_path = os.path.join(output_dir, filename)
        processed.save(output_path, 'JPEG', quality=95)

        return f"Processed: {filename}"
    except Exception as e:
        return f"Error processing {filename}: {e}"

def batch_process_images(input_dir, output_dir, target_size=(832, 576), workers=8):
    """Process images in parallel"""

    filenames = [f for f in os.listdir(input_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    process_func = partial(
        process_single_image,
        input_dir=input_dir,
        output_dir=output_dir,
        target_size=target_size
    )

    with mp.Pool(workers) as pool:
        results = pool.map(process_func, filenames)

    for result in results:
        print(result)
```

### Text Prompt Processing

#### Prompt Cleaning and Validation

```python
import re

def clean_prompt(text):
    """Clean and normalize text prompts"""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s.,!?-]', '', text)

    # Ensure proper capitalization
    text = text.capitalize()

    # Add period if missing
    if not text.endswith(('.', '!', '?')):
        text += '.'

    return text

def validate_prompt(text, min_length=10, max_length=200):
    """Validate prompt quality"""

    # Check length
    if len(text) < min_length:
        return False, "Prompt too short"

    if len(text) > max_length:
        return False, "Prompt too long"

    # Check for meaningful content
    words = text.split()
    if len(words) < 3:
        return False, "Too few words"

    # Check for common issues
    if text.lower().count('edit') > 3:
        return False, "Too many 'edit' keywords"

    return True, "Valid"

def process_prompts(prompts_dir):
    """Process all prompt files"""

    for filename in os.listdir(prompts_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(prompts_dir, filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Clean prompt
            cleaned = clean_prompt(text)

            # Validate
            is_valid, message = validate_prompt(cleaned)

            if is_valid:
                # Save cleaned prompt
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                print(f"Processed: {filename}")
            else:
                print(f"Invalid prompt {filename}: {message}")
```

## Dataset Quality Control

### Validation Script

```python
import os
from PIL import Image

def validate_dataset(dataset_dir):
    """Comprehensive dataset validation"""

    images_dir = os.path.join(dataset_dir, 'training_images')
    control_dir = os.path.join(dataset_dir, 'control_images')
    prompts_dir = os.path.join(dataset_dir, 'training_images')  # 文本文件在同一目录

    issues = []
    valid_samples = 0

    # Get all basenames
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir)}
    control_files = {os.path.splitext(f)[0] for f in os.listdir(control_dir)}
    prompt_files = {os.path.splitext(f)[0] for f in os.listdir(prompts_dir)}

    # Find complete triplets
    complete_samples = image_files & control_files & prompt_files

    for basename in complete_samples:
        try:
            # Validate images
            img_path = find_file_with_extension(images_dir, basename)
            ctrl_path = find_file_with_extension(control_dir, basename)
            prompt_path = os.path.join(prompts_dir, basename + '.txt')

            # Check image validity
            with Image.open(img_path) as img:
                if img.mode not in ['RGB', 'RGBA']:
                    issues.append(f"{basename}: Image not RGB/RGBA")
                    continue

            with Image.open(ctrl_path) as ctrl:
                if ctrl.mode != 'RGB':
                    issues.append(f"{basename}: Control image not RGB")
                    continue

            # Check prompt
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                if len(prompt) < 10:
                    issues.append(f"{basename}: Prompt too short")
                    continue

            valid_samples += 1

        except Exception as e:
            issues.append(f"{basename}: {str(e)}")

    # Report results
    print(f"Valid samples: {valid_samples}")
    print(f"Issues found: {len(issues)}")

    if issues:
        print("\nIssues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")

    return valid_samples, issues

def find_file_with_extension(directory, basename):
    """Find file with any supported extension"""
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        filepath = os.path.join(directory, basename + ext)
        if os.path.exists(filepath):
            return filepath
    raise FileNotFoundError(f"No image found for {basename}")
```

### Dataset Statistics

```python
def analyze_dataset(dataset_dir):
    """Generate dataset statistics"""

    stats = {
        'total_samples': 0,
        'image_formats': {},
        'image_sizes': [],
        'prompt_lengths': [],
        'file_sizes': []
    }

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            continue

        images_dir = os.path.join(split_dir, 'images')
        prompts_dir = os.path.join(split_dir, 'prompts')

        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Image analysis
                img_path = os.path.join(images_dir, filename)

                with Image.open(img_path) as img:
                    # Format
                    fmt = img.format.lower()
                    stats['image_formats'][fmt] = stats['image_formats'].get(fmt, 0) + 1

                    # Size
                    stats['image_sizes'].append(img.size)

                    # File size
                    file_size = os.path.getsize(img_path)
                    stats['file_sizes'].append(file_size)

                # Prompt analysis
                basename = os.path.splitext(filename)[0]
                prompt_path = os.path.join(prompts_dir, basename + '.txt')

                if os.path.exists(prompt_path):
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                        stats['prompt_lengths'].append(len(prompt))

                stats['total_samples'] += 1

    # Calculate summary statistics
    if stats['image_sizes']:
        widths, heights = zip(*stats['image_sizes'])
        stats['avg_width'] = sum(widths) / len(widths)
        stats['avg_height'] = sum(heights) / len(heights)

    if stats['prompt_lengths']:
        stats['avg_prompt_length'] = sum(stats['prompt_lengths']) / len(stats['prompt_lengths'])

    if stats['file_sizes']:
        stats['avg_file_size'] = sum(stats['file_sizes']) / len(stats['file_sizes'])

    return stats
```

## Data Augmentation

### Image Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_augmentation_pipeline():
    """Create data augmentation pipeline"""

    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=5,
            p=0.3
        ),

        # Color transformations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),

        # Quality augmentations
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.Blur(blur_limit=3, p=0.1),

        # Normalize and convert
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def apply_augmentation(image, control_image, augment_pipeline):
    """Apply consistent augmentation to image pair"""

    # Create combined image for consistent transforms
    combined = np.concatenate([image, control_image], axis=1)

    # Apply augmentation
    augmented = augment_pipeline(image=combined)['image']

    # Split back
    width = image.shape[1]
    aug_image = augmented[:, :width]
    aug_control = augmented[:, width:]

    return aug_image, aug_control
```

### Text Augmentation

```python
import random

def augment_prompt(prompt, augmentation_ratio=0.2):
    """Apply text augmentation to prompts"""

    # Synonym replacement
    synonyms = {
        'bright': ['luminous', 'radiant', 'brilliant'],
        'dark': ['dim', 'shadowy', 'murky'],
        'beautiful': ['stunning', 'gorgeous', 'attractive'],
        'enhance': ['improve', 'boost', 'amplify'],
        'reduce': ['decrease', 'diminish', 'lessen']
    }

    words = prompt.split()
    augmented_words = []

    for word in words:
        if random.random() < augmentation_ratio and word.lower() in synonyms:
            # Replace with synonym
            augmented_words.append(random.choice(synonyms[word.lower()]))
        else:
            augmented_words.append(word)

    return ' '.join(augmented_words)

def generate_prompt_variations(prompt, num_variations=3):
    """Generate multiple prompt variations"""

    variations = [prompt]  # Include original

    # Style variations
    style_prefixes = [
        "In a photorealistic style, ",
        "With artistic flair, ",
        "In high definition, "
    ]

    # Quality additions
    quality_suffixes = [
        ", high quality, detailed",
        ", professional photography",
        ", sharp focus, vivid colors"
    ]

    for i in range(num_variations):
        variation = prompt

        # Add style prefix (sometimes)
        if random.random() < 0.3:
            variation = random.choice(style_prefixes) + variation.lower()

        # Add quality suffix (sometimes)
        if random.random() < 0.5:
            variation += random.choice(quality_suffixes)

        # Apply synonym replacement
        variation = augment_prompt(variation)

        variations.append(variation)

    return variations
```

## Best Practices

### Dataset Curation

1. **Diverse Content**: Include variety in subjects, styles, lighting
2. **Quality Control**: Remove blurry, corrupted, or low-quality images
3. **Balanced Distribution**: Ensure even representation across categories
4. **Consistent Quality**: Maintain similar resolution and format standards
5. **Clear Prompts**: Write specific, actionable editing instructions

### Performance Optimization

1. **Storage Format**: Use JPEG for photos, PNG for graphics with transparency
2. **Resolution**: Target 512-1024px for optimal training performance
3. **File Organization**: Use efficient directory structures
4. **Preprocessing**: Apply consistent preprocessing pipeline
5. **Validation**: Regularly validate dataset integrity

### Prompt Writing Guidelines

1. **Be Specific**: "Add warm golden hour lighting" vs "improve lighting"
2. **Action-Oriented**: Use verbs like "enhance", "add", "remove", "transform"
3. **Style Descriptions**: Include artistic style references when relevant
4. **Quality Indicators**: Mention desired quality level or technical aspects
5. **Avoid Negatives**: Focus on what to add rather than what to remove

This comprehensive data preparation guide should help you create high-quality datasets for optimal training results.

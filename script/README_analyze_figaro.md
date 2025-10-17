# Figaro Dataset Analysis Script

This script analyzes the image shapes, dimensions, and other statistics of the Figaro Hair Segmentation dataset (`Allison/figaro_hair_segmentation_1000`).

## Purpose

Before training, it's important to understand your dataset characteristics:
- Image dimensions and aspect ratios
- Size distribution and consistency
- Shape matching between control and target images
- Data quality and potential issues

## Usage

### Basic Analysis

```bash
python script/analyze_figaro_dataset.py
```

### Save Detailed Statistics

```bash
python script/analyze_figaro_dataset.py \
    --output_file figaro_stats.json
```

### Analyze Different Dataset

```bash
python script/analyze_figaro_dataset.py \
    --repo_id your-org/your-dataset \
    --hf_token hf_xxxxxxxxxxxxx
```

### Quiet Mode (No Progress Bars)

```bash
python script/analyze_figaro_dataset.py --quiet
```

## Output

### Console Summary

The script prints a human-readable summary:

```
================================================================================
FIGARO HAIR SEGMENTATION DATASET ANALYSIS
================================================================================

TRAIN SPLIT
----------------------------------------
Total samples: 1000
Processed: 1000
Skipped: 0

Control Images (Photos):
  Dimensions: 337x337 to 1000x1000
  Most common: (512, 512)
  Aspect ratio: 0.50 to 2.00 (mean: 1.00)
  Area: 113569 to 1000000 pixels (mean: 262144)
  Unique shapes: 15

Target Images (Masks):
  Dimensions: 337x337 to 1000x1000
  Most common: (512, 512)
  Aspect ratio: 0.50 to 2.00 (mean: 1.00)
  Area: 113569 to 1000000 pixels (mean: 262144)
  Unique shapes: 15

Shape Consistency:
  Matching pairs: 1000/1000 (100.0%)
```

### JSON Output

Detailed statistics are saved to a JSON file:

```json
{
  "train": {
    "split_name": "train",
    "total_samples": 1000,
    "processed_samples": 1000,
    "skipped_samples": 0,
    "control_images": {
      "width_count": 1000,
      "width_min": 337,
      "width_max": 1000,
      "width_mean": 512.0,
      "width_median": 512.0,
      "width_std": 45.2,
      "width_q25": 512.0,
      "width_q75": 512.0,
      "height_count": 1000,
      "height_min": 337,
      "height_max": 1000,
      "height_mean": 512.0,
      "height_median": 512.0,
      "height_std": 45.2,
      "height_q25": 512.0,
      "height_q75": 512.0,
      "aspect_ratio_count": 1000,
      "aspect_ratio_min": 0.5,
      "aspect_ratio_max": 2.0,
      "aspect_ratio_mean": 1.0,
      "aspect_ratio_median": 1.0,
      "aspect_ratio_std": 0.1,
      "aspect_ratio_q25": 1.0,
      "aspect_ratio_q75": 1.0,
      "area_count": 1000,
      "area_min": 113569,
      "area_max": 1000000,
      "area_mean": 262144,
      "area_median": 262144,
      "area_std": 46208,
      "area_q25": 262144,
      "area_q75": 262144,
      "unique_shapes": 15,
      "most_common_shape": [512, 512]
    },
    "target_images": {
      // Similar structure for target images
    },
    "shape_consistency": {
      "matching_shapes": 1000,
      "total_pairs": 1000,
      "match_percentage": 100.0
    }
  }
}
```

## Statistics Explained

### Control Images (Photos)
- **Dimensions**: Min/max width and height
- **Most common shape**: Most frequent (width, height) pair
- **Aspect ratio**: Width/height ratio statistics
- **Area**: Total pixel count (width × height)
- **Unique shapes**: Number of different dimension combinations

### Target Images (Masks)
- Same statistics as control images
- Should ideally match control image dimensions

### Shape Consistency
- **Matching pairs**: How many control-target pairs have identical dimensions
- **Match percentage**: Percentage of consistent pairs
- **Total pairs**: Total number of valid control-target pairs

## What to Look For

### ✅ Good Signs
- **High shape consistency** (>95% matching pairs)
- **Reasonable aspect ratios** (close to 1.0 for square images)
- **Consistent dimensions** (low standard deviation)
- **No skipped samples** (0 skipped)

### ⚠️ Warning Signs
- **Low shape consistency** (<90% matching pairs)
- **Extreme aspect ratios** (<0.5 or >2.0)
- **High dimension variance** (large std deviation)
- **Many skipped samples** (>5%)

### 🔧 Actions Based on Results

1. **If shapes don't match**: Consider preprocessing to resize images
2. **If aspect ratios vary widely**: Consider cropping or padding strategies
3. **If many samples skipped**: Check data quality and fix issues
4. **If dimensions are very large**: Consider downsampling for training efficiency

## Integration with Training

Use the analysis results to configure your training:

```yaml
# Example: If most images are 512x512
data:
  init_args:
    image_size: [512, 512]  # Match most common size

# Example: If aspect ratios vary, use dynamic resolution
data:
  init_args:
    image_size: [512, 512]  # Base size
    # Enable dynamic resolution if supported
```

## Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--repo_id` | str | `Allison/figaro_hair_segmentation_1000` | Source dataset repository |
| `--hf_token` | str | None | HuggingFace token for private datasets |
| `--output_file` | str | `figaro_dataset_stats.json` | Output JSON file path |
| `--quiet` | flag | False | Suppress progress bars |

## Dependencies

Required packages (already in `requirements.txt`):
- `datasets` - HuggingFace datasets library
- `Pillow` - Image processing
- `numpy` - Statistical calculations
- `tqdm` - Progress bars

## Examples

### Quick Analysis
```bash
python script/analyze_figaro_dataset.py --quiet
```

### Full Analysis with Custom Output
```bash
python script/analyze_figaro_dataset.py \
    --output_file /tmp/figaro_analysis.json \
    --repo_id Allison/figaro_hair_segmentation_1000
```

### Analyze After Download
```bash
# First download and organize
python script/download_figaro_dataset.py --output_dir /tmp/figaro

# Then analyze the organized dataset
python script/analyze_figaro_dataset.py \
    --repo_id TsienDragon/figaro_hair_segmentation_1k
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'datasets'"
- **Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: "Failed to load dataset"
- **Solution**: Check internet connection and HuggingFace token
- **Solution**: Verify dataset exists: https://huggingface.co/datasets/Allison/figaro_hair_segmentation_1000

### Issue: "All samples skipped"
- **Solution**: Check dataset schema - script expects 'image' and 'label' fields
- **Solution**: Verify images are PIL.Image objects

### Issue: "Memory error"
- **Solution**: Use `--quiet` flag to reduce memory usage
- **Solution**: Process smaller batches if needed

## Related Scripts

- `download_figaro_dataset.py` - Download and organize the dataset
- `README_figaro_dataset.md` - Dataset download documentation

## Notes

- The script analyzes the original HuggingFace dataset, not the organized version
- Statistics are calculated per split (train/test/validation)
- Image processing is done in-memory for accuracy
- Large datasets may take several minutes to analyze

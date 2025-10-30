# Validation Sampling Guide

This guide explains how to use the validation sampling feature introduced in v3.1.0 to monitor your model's training progress.

## Overview

Validation sampling allows you to visualize how your model improves during training by generating images from a validation dataset at regular intervals. These images are logged to TensorBoard, making it easy to track progress and identify issues early.

## Configuration

To enable validation sampling, add a `validation` section to your YAML configuration file. Here's an example from `test_example_fluxkontext_fp16.yaml`:

```yaml
validation:
  enabled: true        # Enable validation sampling
  steps: 100           # Run validation every 100 steps
  max_samples: 2       # Use up to 2 samples from validation dataset
  seed: 42             # Fixed seed for reproducible results
  dataset:             # Validation dataset configuration
    class_path: qflux.data.dataset.ImageDataset
    init_args:
      dataset_path:
        - split: test
          repo_id: TsienDragon/face_segmentation_20
      caption_dropout_rate: 0.0
      prompt_image_dropout_rate: 0.0
      use_edit_mask: true
      selected_control_indexes: [1]
      processor:
        class_path: qflux.data.preprocess.ImageProcessor
        init_args:
          process_type: center_crop
          resize_mode: bilinear
          target_size: [832, 576]
          controls_size: [[832, 576]]
```

### Key Parameters

- `enabled`: Set to `true` to enable validation sampling
- `steps`: How often to run validation (every N training steps)
- `max_samples`: Maximum number of validation samples to use
- `seed`: Optional fixed seed for reproducible sampling
- `dataset`: Configuration for the validation dataset (same format as training dataset)

## Viewing Validation Results

During training, validation samples are automatically logged to TensorBoard. To view them:

1. Start TensorBoard:
   ```bash
   tensorboard --logdir=/path/to/output/logs
   ```

2. Navigate to the "Images" tab in the TensorBoard UI

3. You'll see validation images organized by step, showing how your model's output improves over time

## Example Usage

Using the configuration from `test_example_fluxkontext_fp16.yaml`:

```bash
# First build cache (optional but recommended)
python -m qflux.main --config tests/test_configs/test_example_fluxkontext_fp16.yaml --cache

# Start training with validation
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config.yaml -m qflux.main --config tests/test_configs/test_example_fluxkontext_fp16.yaml
```

This will:
1. Train the model using the FLUX Kontext architecture
2. Run validation every 100 steps
3. Sample 2 images from the validation dataset
4. Log the results to TensorBoard

## Best Practices

- Use a small number of validation samples (2-5) to minimize overhead
- Set validation frequency based on your training length (every 100-500 steps)
- Use a fixed seed for consistent comparisons across runs
- Choose representative validation samples that cover your use cases

## Troubleshooting

- If validation is slow, reduce `max_samples` or increase `steps`
- If you see OOM errors during validation, reduce the validation batch size or image resolution
- Make sure your validation dataset has the same format as your training dataset

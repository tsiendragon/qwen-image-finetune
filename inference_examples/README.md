# Inference Examples

This folder contains inference scripts and examples for testing the image editing models.

## Files

- `test_inference.py` - Main inference script supporting both base model and LoRA fine-tuned models
- `test_inference_config.yaml` - Configuration file optimized for single GPU (A100) inference
- `kitchen.jpg` - Sample input image for testing
- `inference_output*.png/jpg` - Generated output images

## Usage

### From this directory:

```bash
cd inference_examples
python test_inference.py --mode base    # Test with base model
python test_inference.py --mode lora    # Test with pre-trained LoRA
```

### From project root:

```bash
python inference_examples/test_inference.py --mode base
python inference_examples/test_inference.py --mode lora
```

## Adding Your Own Images

1. **From local machine**, upload your image:
   ```bash
   scp /path/to/your/image.jpg ubuntu@YOUR_SERVER_IP:/home/ubuntu/qwen-image-finetune/inference_examples/your_image.jpg
   ```

2. **Modify `test_inference.py`** to use your image:
   ```python
   # Update line ~95 in test_inference_base_model():
   image_path = os.path.join(script_dir, 'your_image.jpg')
   ```

3. **Update the prompt** to match your editing task

## Configuration

The `test_inference_config.yaml` is configured for:
- Single GPU (`cuda:0`)
- Quantized model (NF4) for lower memory usage
- Batch size 2
- Mixed precision (bf16)

All device assignments use `cuda:0` to work with single GPU setups.

## Output

Generated images are saved in this directory with names like:
- `inference_output.png` - LoRA model results
- `inference_output_base.png` - Base model results


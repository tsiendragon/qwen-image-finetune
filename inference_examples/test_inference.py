"""
Simple inference script to test pre-trained models
Run from inference_examples directory: python test_inference.py
Or from project root: python inference_examples/test_inference.py
"""

import sys
import os

# Add src to path (go up one level from inference_examples to project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
from PIL import Image
from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml
from diffusers.utils import load_image

def test_inference_with_pretrained_lora():
    """Test inference using a pre-trained LoRA model from HuggingFace"""
    
    print("=" * 60)
    print("Testing FLUX Kontext with Face Segmentation LoRA")
    print("=" * 60)
    
    # Load configuration (using single GPU config)
    config_path = "test_inference_config.yaml"
    config = load_config_from_yaml(config_path)
    
    # Set pre-trained LoRA weights from HuggingFace
    config.model.lora.pretrained_weight = 'TsienDragon/flux-kontext-face-segmentation'
    
    # Initialize trainer
    print("\n1. Initializing trainer...")
    trainer = FluxKontextLoraTrainer(config)
    
    # Load a test image (using an online image)
    print("\n2. Loading test image...")
    IMAGE_URL = 'https://n.sinaimg.cn/ent/transform/775/w630h945/20201127/cee0-kentcvx8062290.jpg'
    prompt_image = load_image(IMAGE_URL)
    print(f"   Input image size: {prompt_image.size}")
    
    # Calculate output dimensions based on input (must be divisible by 16)
    input_width, input_height = prompt_image.size
    output_width = (input_width // 16) * 16
    output_height = (input_height // 16) * 16
    print(f"   Output dimensions: {output_width}x{output_height}")
    
    # Set the prompt
    prompt = 'change the image from the face to the face segmentation mask'
    print(f"\n3. Prompt: '{prompt}'")
    
    # Run inference
    print("\n4. Running inference...")
    print("   (This will download models on first run - may take a few minutes)")
    
    result = trainer.predict(
        image=prompt_image,
        prompt=prompt,
        num_inference_steps=20,
        true_cfg_scale=1.0,
        negative_prompt="",
        weight_dtype=torch.bfloat16,
        height=output_height,
        width=output_width,
        output_type='pil'
    )
    
    # Save result in inference_examples directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'inference_output.png')
    result[0].save(output_path)
    print(f"\n5. ✅ Success! Result saved to: {output_path}")
    print(f"   Output size: {result[0].size}")
    
    return result[0]

def test_inference_base_model():
    """Test inference using base model without LoRA"""
    
    print("\n" + "=" * 60)
    print("Testing Base FLUX Kontext (No LoRA)")
    print("=" * 60)
    
    # Load configuration (using single GPU config)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "test_inference_config.yaml")
    config = load_config_from_yaml(config_path)
    
    # Use quantized base model (lighter weight)
    config.model.pretrained_model_name_or_path = 'lrzjason/flux-kontext-nf4'
    
    # Initialize trainer
    print("\n1. Initializing trainer with base model...")
    trainer = FluxKontextLoraTrainer(config)
    
    # Load test image
    print("\n2. Loading test image...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'kitchen.jpg')
    prompt_image = Image.open(image_path)
    print(f"   Image loaded from: {image_path}")
    print(f"   Input image size: {prompt_image.size}")
    
    # Calculate output dimensions based on input (must be divisible by 16)
    input_width, input_height = prompt_image.size
    output_width = (input_width // 16) * 16
    output_height = (input_height // 16) * 16
    print(f"   Output dimensions: {output_width}x{output_height}")
    
    # Set the prompt
    prompt = 'Add 9 red balloons to the kitchen without changing anything else about the image. Make sure none of the balloons are touching each other or overlapping with each other.'
    print(f"\n3. Prompt: '{prompt}'")
    
    # Run inference
    print("\n4. Running inference...")
    result = trainer.predict(
        image=prompt_image,
        prompt=prompt,
        num_inference_steps=20,
        true_cfg_scale=1.0,
        negative_prompt="",
        weight_dtype=torch.bfloat16,
        height=output_height,
        width=output_width,
        output_type='pil'
    )
    
    # Save result in inference_examples directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'inference_output_base_5.png')
    result[0].save(output_path)
    print(f"\n5. ✅ Success! Result saved to: {output_path}")
    
    return result[0]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test inference with FLUX Kontext')
    parser.add_argument('--mode', type=str, default='lora', choices=['lora', 'base'],
                        help='Test with LoRA (lora) or base model (base)')
    args = parser.parse_args()
    
    try:
        if args.mode == 'lora':
            test_inference_with_pretrained_lora()
        else:
            test_inference_base_model()
            
        print("\n" + "=" * 60)
        print("🎉 Inference test completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()


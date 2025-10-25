
import pytest
import torch
import logging
from pathlib import Path
import cv2
import numpy as np
import PIL.Image as Image
from diffusers.utils import load_image
import os
import sys
from tqdm.rich import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.data.config import load_config_from_yaml
from datasets import load_dataset


def construct_trainer(lora_weight: str = "TsienDragon/flux-kontext-face-segmentation"):
    config_path = "tests/test_configs/test_example_fluxkontext_fp16.yaml"
    config = load_config_from_yaml(config_path)
    config.model.lora.pretrained_weight = lora_weight
    config.data.init_args.processor.init_args.process_type = "center_crop"
    config.data.init_args.processor.init_args.resize_mode = "bilinear"
    config.data.init_args.processor.init_args.target_size = [832, 576]
    config.data.init_args.processor.init_args.controls_size = [[832, 576]]
    trainer = FluxKontextLoraTrainer(config)
    return trainer


def predict_faceseg_data(trainer: FluxKontextLoraTrainer, sample_image: Image.Image):
    # Arrange
    prompt = "change the image from the face to the face segmentation mask"
    num_inference_steps = 20
    height = 832
    width = 576

    out = trainer.predict(
        prompt_image=sample_image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        controls_size=[[height, width]],
        guidance_scale=3.5,
        true_cfg_scale=1.0,
        negative_prompt="",
        weight_dtype=torch.bfloat16,
        height=height,
        width=width,
    )
    return out[0]


def predict_faceseg_dataset(
        lora_weight: str = 'TsienDragon/flux-kontext-face-segmentation',
        dataset_name= "TsienDragon/face_segmentation_20",
        split: str = "train",
        save_dir: str = "/tmp/flux_sampling_faceseg_dataset"
    ):
    trainer = construct_trainer(lora_weight)
    dataset = load_dataset(dataset_name, split=split)
    os.makedirs(save_dir, exist_ok=True)
    for sample in tqdm(dataset, desc="Predicting faceseg dataset", total=len(dataset)):
        control_images = sample['control_images']
        if isinstance(control_images, list) and len(control_images) > 0:
            prompt_image = control_images[0].convert('RGB')
        else:
            prompt_image = control_images.convert('RGB')
        out = predict_faceseg_data(trainer, prompt_image)

        save_path = os.path.join(save_dir, f"{sample['id']}_generate.png")
        target_image = sample['target_image'].convert('RGB')
        target_path = os.path.join(save_dir, f"{sample['id']}_target.png")
        target_image.save(target_path)
        out.save(save_path)


if __name__ == "__main__":
    lora_weight = '/tmp/image_edit_lora/faceSegFluxKontextFp16_origin/v0/checkpoint-last-347-3470-last/pytorch_lora_weights.safetensors',
    save_dir = '/tmp/flux_sampling_faceseg_dataset_origin_3470'

    lora_weight= '/tmp/image_edit_lora/faceSegFluxKontextFp16MultiRes/v0/checkpoint-last-1227-3683-last/pytorch_lora_weights.safetensors'
    save_dir = '/tmp/flux_sampling_faceseg_dataset_multires_3683'

    lora_weight = '/tmp/image_edit_lora/faceSegFluxKontextFp16/v7/checkpoint-last-752-3761-last/pytorch_lora_weights.safetensors'
    save_dir = '/tmp/flux_sampling_faceseg_dataset_new_code_fixres_v7_3761'

    lora_weight = '/tmp/image_edit_lora/faceSegFluxKontextFp16MultiRes/v1/checkpoint-last-1665-4998-last/pytorch_lora_weights.safetensors'
    save_dir = '/tmp/flux_sampling_faceseg_dataset_multires_4998'

    predict_faceseg_dataset(
        lora_weight=lora_weight,
        # '/tmp/image_edit_lora/faceSegFluxKontextFp16_origin/v0/checkpoint-last-347-3470-last/pytorch_lora_weights.safetensors',
        # TsienDragon/flux-kontext-face-segmentation',
        dataset_name='TsienDragon/face_segmentation_20',
        split='train',
        save_dir=save_dir
        # '/tmp/flux_sampling_faceseg_dataset_origin_3470'
    )

#!/usr/bin/env python3
"""
简单测试脚本，用于验证ImageDataset类的功能
"""

import sys
import os
sys.path.append('src')

from data.dataset import ImageDataset


def test_id_card_dataset():
    """测试ID卡数据集配置"""

    # 使用原始代码中的配置
    data_config = {
        'dataset_path': '/data/kyc_gen/id_card/',
        'image_size': (512, 312)
    }

    try:
        print("正在测试ID卡数据集配置...")
        dataset = ImageDataset(data_config)

        print(f"ID卡数据集大小: {len(dataset)}")

        if len(dataset) > 0:
            print("正在测试第一个ID卡样本...")
            sample = dataset[0]

            # 验证返回格式是字典
            assert isinstance(sample, dict), f"Expected dict, got {type(sample)}"
            assert 'image' in sample, "Missing 'image' key in sample"
            assert 'control' in sample, "Missing 'control' key in sample"
            assert 'prompt' in sample, "Missing 'prompt' key in sample"

            image = sample['image']
            control_image = sample['control']
            prompt = sample['prompt']

            print(f"ID卡 Image shape: {image.shape}")
            print(f"ID卡 Control image shape: {control_image.shape}")
            print(f"ID卡 Prompt: {prompt[:100]}...")

            # 验证tensor类型和形状
            import torch
            assert isinstance(image, torch.Tensor), f"Expected torch.Tensor, got {type(image)}"
            assert isinstance(control_image, torch.Tensor), f"Expected torch.Tensor, got {type(control_image)}"
            assert isinstance(prompt, str), f"Expected str, got {type(prompt)}"

            # 验证tensor维度
            assert len(image.shape) == 3, f"Expected 3D tensor, got {len(image.shape)}D"
            assert image.shape[0] == 3, f"Expected 3 channels, got {image.shape[0]}"

            # 验证图像尺寸是否正确 (应该是512x312)
            expected_height, expected_width = 312, 512  # PyTorch格式 (C, H, W)
            assert image.shape[1] == expected_height, f"Expected height {expected_height}, got {image.shape[1]}"
            assert image.shape[2] == expected_width, f"Expected width {expected_width}, got {image.shape[2]}"

            print("✓ ID卡数据集测试通过!")
        else:
            print("⚠️  ID卡数据集为空，请检查数据路径和文件结构")

    except Exception as e:
        print(f"❌ ID卡数据集测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_id_card_dataset()

#!/usr/bin/env python3
"""
获取Qwen-Image-Edit模型的详细配置参数
"""

import json

import torch

from qflux.models.transformer_qwenimage import QwenImageTransformer2DModel


def get_pretrained_model_config():
    """获取预训练模型配置"""
    print("🔍 正在获取Qwen-Image-Edit模型配置...")

    try:
        # 加载预训练模型
        model = QwenImageTransformer2DModel.from_pretrained(
            "Qwen/Qwen-Image-Edit", subfolder="transformer", torch_dtype=torch.bfloat16
        )

        print("✅ 模型加载成功！")

        # 获取配置
        config = model.config

        print("\n📋 模型配置参数：")
        print("=" * 60)

        # 打印所有配置参数
        if hasattr(config, "__dict__"):
            config_dict = config.__dict__
        else:
            config_dict = dict(config)

        for key, value in sorted(config_dict.items()):
            if not key.startswith("_"):
                print(f"{key:25}: {value}")

        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\n📊 参数统计：")
        print("=" * 60)
        print(f"{'总参数量':25}: {total_params:,}")
        print(f"{'可训练参数量':25}: {trainable_params:,}")
        print(f"{'模型大小 (MB)':25}: {total_params * 2 / 1024 / 1024:.2f}")  # bfloat16 = 2 bytes

        # 保存配置到JSON文件
        config_for_save = {}
        for key, value in config_dict.items():
            if not key.startswith("_"):
                # 转换为JSON兼容的格式
                if isinstance(value, torch.dtype):
                    config_for_save[key] = str(value)
                elif hasattr(value, "__dict__"):
                    config_for_save[key] = str(value)
                else:
                    config_for_save[key] = value

        with open("qwen_image_edit_config.json", "w", encoding="utf-8") as f:
            json.dump(config_for_save, f, indent=2, ensure_ascii=False)

        print("\n💾 配置已保存到: qwen_image_edit_config.json")

        # 打印关键的transformer架构参数
        print("\n🏗️ 关键架构参数：")
        print("=" * 60)
        key_params = [
            "num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "patch_size",
            "sample_size",
            "hidden_size",
            "num_single_layers",
            "pooled_projection_dim",
        ]

        for param in key_params:
            if param in config_dict:
                print(f"{param:25}: {config_dict[param]}")

        return config_dict

    except Exception as e:
        print(f"❌ 无法加载模型: {e}")
        print("\n可能的原因：")
        print("- 网络连接问题")
        print("- 需要Hugging Face认证")
        print("- 模型路径或名称错误")
        print("- 缺少依赖包")
        return None


def compare_with_local_config():
    """比较本地模型配置"""
    print("\n🔄 比较本地模型配置...")

    # 本地模型配置
    local_config = {
        "patch_size": 2,
        "in_channels": 64,
        "out_channels": 16,
        "num_layers": 60,
        "attention_head_dim": 128,
        "num_attention_heads": 24,
    }

    print("\n📝 本地模型配置：")
    print("=" * 60)
    for key, value in local_config.items():
        print(f"{key:25}: {value}")

    # 创建本地模型并计算参数
    try:
        local_model = QwenImageTransformer2DModel(**local_config)
        local_params = sum(p.numel() for p in local_model.parameters())
        print(f"{'本地模型参数量':25}: {local_params:,}")
    except Exception as e:
        print(f"❌ 创建本地模型失败: {e}")


if __name__ == "__main__":
    # 获取预训练模型配置
    pretrained_config = get_pretrained_model_config()

    # 比较本地配置
    compare_with_local_config()

    print("\n🏁 配置获取完成！")

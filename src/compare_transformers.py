#!/usr/bin/env python3
"""
比较两种transformer加载方式是否得到相同的模型
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.load_model import load_transformer
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline


def compare_models(model1, model2, model_name="models"):
    """比较两个模型是否相同"""
    print(f"🔍 比较 {model_name}...")

    # 1. 检查模型类型
    print(f"Model 1 类型: {type(model1)}")
    print(f"Model 2 类型: {type(model2)}")

    if type(model1) != type(model2):
        print("❌ 模型类型不同！")
    else:
        print("✅ 模型类型相同")

    # 2. 检查模型结构
    print(f"\nModel 1 参数数量: {sum(p.numel() for p in model1.parameters()):,}")
    print(f"Model 2 参数数量: {sum(p.numel() for p in model2.parameters()):,}")

    # 3. 检查state_dict键
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    print(f"\nModel 1 参数键数量: {len(keys1)}")
    print(f"Model 2 参数键数量: {len(keys2)}")

    if keys1 != keys2:
        print("❌ 参数键不完全相同！")
        print(f"Model 1 独有的键: {keys1 - keys2}")
        print(f"Model 2 独有的键: {keys2 - keys1}")
    else:
        print("✅ 参数键完全相同")

    # 4. 检查参数值是否相同
    print("\n🔍 检查参数值...")
    differences = 0
    max_diff = 0.0

    for key in keys1:
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        if param1.shape != param2.shape:
            print(f"❌ 参数 {key} 的形状不同: {param1.shape} vs {param2.shape}")
            differences += 1

        diff = torch.abs(param1 - param2).max().item()
        max_diff = max(max_diff, diff)

        if diff > 1e-6:  # 设置一个小的阈值
            print(f"⚠️  参数 {key} 有差异: 最大差值 = {diff}")
            differences += 1

    print(f"\n📊 比较结果:")
    print(f"参数差异数量: {differences}")
    print(f"最大差值: {max_diff}")

    if differences == 0:
        print("✅ 所有参数值完全相同！")
        return True
    else:
        print(f"❌ 发现 {differences} 个参数有差异")
        return False


def main():
    """主函数"""
    print("🚀 开始比较两种transformer加载方式...")

    model_path = "Qwen/Qwen-Image-Edit"
    weight_dtype = torch.bfloat16

    # 方法1：从QwenImageEditPipeline加载
    print(f"\n📥 方法1: 从QwenImageEditPipeline加载...")
    pipe = QwenImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=weight_dtype
    )
    transformer1 = pipe.transformer
    print("✅ 方法1加载成功")

    # 方法2：直接加载transformer
    print(f"\n📥 方法2: 直接加载transformer...")
    transformer2 = load_transformer(model_path, weight_dtype)
    print("✅ 方法2加载成功")

    # 比较两个模型
    print(f"\n" + "="*50)
    is_same = compare_models(transformer1, transformer2, "transformers")
    print("="*50)

    if is_same:
        print("🎉 结论: 两种加载方式得到的transformer模型完全相同！")
    else:
        print("⚠️  结论: 两种加载方式得到的transformer模型不完全相同")

    # 额外信息
    print(f"\n📋 额外信息:")
    print(f"Pipeline transformer device: {transformer1.device}")
    print(f"Direct load transformer device: {transformer2.device}")
    print(f"Pipeline transformer dtype: {next(transformer1.parameters()).dtype}")
    print(f"Direct load transformer dtype: {next(transformer2.parameters()).dtype}")

    # 检查配置
    if hasattr(transformer1, 'config') and hasattr(transformer2, 'config'):
        config1 = transformer1.config
        config2 = transformer2.config
        print(f"\n⚙️  配置比较:")
        print(f"Config 1: {config1}")
        print(f"Config 2: {config2}")
        print(f"配置相同: {config1 == config2}")

if __name__ == "__main__":
    main()

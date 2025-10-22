#!/usr/bin/env python3
"""
比较两种VAE加载方式是否得到相同的模型
"""
import torch
import sys
import os
# Add path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qflux.models.load_model import load_vae  # noqa: E402
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline  # noqa: E402
def compare_models(model1, model2, model_name="models"):
    """比较两个模型是否相同"""
    print(f"🔍 比较 {model_name}...")
    # 1. 检查模型类型
    print(f"Model 1 类型: {type(model1)}")
    print(f"Model 2 类型: {type(model2)}")
    if type(model1) is not type(model2):
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
        if key not in keys2:
            continue
        param1 = state_dict1[key]
        param2 = state_dict2[key]
        if param1.shape != param2.shape:
            print(f"❌ 参数 {key} 的形状不同: {param1.shape} vs {param2.shape}")
            differences += 1
            continue
        diff = torch.abs(param1 - param2).max().item()
        max_diff = max(max_diff, diff)
        if diff > 1e-6:  # 设置一个小的阈值
            print(f"⚠️  参数 {key} 有差异: 最大差值 = {diff}")
            differences += 1
    print("\n📊 比较结果:")
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
    print("🚀 开始比较两种VAE加载方式...")
    model_path = "Qwen/Qwen-Image-Edit"
    weight_dtype = torch.bfloat16
    # 方法1：从QwenImageEditPipeline加载
    print("\n📥 方法1: 从QwenImageEditPipeline加载...")
    pipe = QwenImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=weight_dtype
    )
    vae1 = pipe.vae
    print("✅ 方法1加载成功")
    # 方法2：直接加载VAE
    print("\n📥 方法2: 直接加载VAE...")
    vae2 = load_vae(model_path, weight_dtype)
    print("✅ 方法2加载成功")
    # 比较两个模型
    print("\n" + "="*50)
    is_same = compare_models(vae1, vae2, "VAE models")
    print("="*50)
    if is_same:
        print("🎉 结论: 两种加载方式得到的VAE模型完全相同！")
    else:
        print("⚠️  结论: 两种加载方式得到的VAE模型不完全相同")
    # 额外信息
    print("\n📋 额外信息:")
    print(f"Pipeline VAE device: {vae1.device}")
    print(f"Direct load VAE device: {vae2.device}")
    print(f"Pipeline VAE dtype: {next(vae1.parameters()).dtype}")
    print(f"Direct load VAE dtype: {next(vae2.parameters()).dtype}")
    # 检查配置
    if hasattr(vae1, 'config') and hasattr(vae2, 'config'):
        config1 = vae1.config
        config2 = vae2.config
        print("\n⚙️  配置比较:")
        print(f"Config 1: {config1}")
        print(f"Config 2: {config2}")
        print(f"配置相同: {config1 == config2}")
    # 检查VAE特有的属性
    print("\n🔧 VAE特有属性检查:")
    if hasattr(vae1, 'temperal_downsample') and hasattr(vae2, 'temperal_downsample'):
        print(f"Pipeline VAE temperal_downsample: {vae1.temperal_downsample}")
        print(f"Direct load VAE temperal_downsample: {vae2.temperal_downsample}")
        print(f"temperal_downsample相同: {vae1.temperal_downsample == vae2.temperal_downsample}")
    if hasattr(vae1, 'config') and hasattr(vae2, 'config'):
        config1 = vae1.config
        config2 = vae2.config
        # 检查重要的VAE配置参数
        vae_attrs = ['latents_mean', 'latents_std', 'z_dim']
        for attr in vae_attrs:
            if hasattr(config1, attr) and hasattr(config2, attr):
                val1 = getattr(config1, attr)
                val2 = getattr(config2, attr)
                print(f"VAE {attr}: {val1} vs {val2}, 相同: {val1 == val2}")


if __name__ == "__main__":
    main()

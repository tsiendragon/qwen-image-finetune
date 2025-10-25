"""
测试 load_transformer 函数
passed: 2025-10-22 10:00:00
"""

import pytest
import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline

from qflux.models.load_model import load_transformer


def compare_models(model1, model2, model_name="models", tolerance=1e-6):
    """
    比较两个模型是否相同

    Args:
        model1: 第一个模型
        model2: 第二个模型
        model_name: 模型名称，用于日志输出
        tolerance: 参数值差异的容忍度

    Returns:
        bool: 如果模型完全相同返回 True，否则返回 False
    """
    print(f"🔍 比较 {model_name}...")

    # 1. 检查模型类型
    print(f"Model 1 类型: {type(model1)}")
    print(f"Model 2 类型: {type(model2)}")

    # 2. 检查模型结构
    param_count1 = sum(p.numel() for p in model1.parameters())
    param_count2 = sum(p.numel() for p in model2.parameters())
    print(f"\nModel 1 参数数量: {param_count1:,}")
    print(f"Model 2 参数数量: {param_count2:,}")

    assert param_count1 == param_count2, f"参数数量不同: {param_count1:,} vs {param_count2:,}"

    # 3. 检查state_dict键
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    print(f"\nModel 1 参数键数量: {len(keys1)}")
    print(f"Model 2 参数键数量: {len(keys2)}")

    # 检查键是否完全相同
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1

    if only_in_1:
        print(f"❌ Model 1 独有的键: {only_in_1}")
    if only_in_2:
        print(f"❌ Model 2 独有的键: {only_in_2}")

    assert keys1 == keys2, "参数键不完全相同！"
    print("✅ 参数键完全相同")

    # 4. 检查参数值是否相同
    print("\n🔍 检查参数值...")
    differences = 0
    max_diff = 0.0
    diff_details = []

    for key in keys1:
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        # 检查形状
        if param1.shape != param2.shape:
            print(f"❌ 参数 {key} 的形状不同: {param1.shape} vs {param2.shape}")
            differences += 1
            continue

        # 检查值
        diff = torch.abs(param1 - param2).max().item()
        max_diff = max(max_diff, diff)

        if diff > tolerance:
            print(f"⚠️  参数 {key} 有差异: 最大差值 = {diff}")
            differences += 1
            diff_details.append((key, diff))

    print("\n📊 比较结果:")
    print(f"参数差异数量: {differences}")
    print(f"最大差值: {max_diff}")

    if differences == 0:
        print("✅ 所有参数值完全相同！")
        return True
    else:
        print(f"❌ 发现 {differences} 个参数有差异")
        # 显示差异最大的前5个参数
        if diff_details:
            diff_details.sort(key=lambda x: x[1], reverse=True)
            print("\n差异最大的参数:")
            for key, diff in diff_details[:5]:
                print(f"  {key}: {diff}")
        return False


@pytest.mark.integration
def test_load_transformer_consistency():
    """
    测试 load_transformer 函数是否与 QwenImageEditPipeline 加载的 transformer 一致

    比较两种加载方式:
    1. 从 QwenImageEditPipeline 加载
    2. 直接使用 load_transformer 加载

    确保两种方式得到的模型完全相同（类型、参数数量、参数键、参数值）
    """
    model_path = "Qwen/Qwen-Image-Edit"
    weight_dtype = torch.bfloat16

    # 方法1：从QwenImageEditPipeline加载
    print("\n📥 方法1: 从QwenImageEditPipeline加载...")
    pipe = QwenImageEditPipeline.from_pretrained(
        model_path,
        torch_dtype=weight_dtype
    )
    transformer1 = pipe.transformer
    print("✅ 方法1加载成功")

    # 方法2：直接加载transformer
    print("\n📥 方法2: 直接加载transformer...")
    transformer2 = load_transformer(model_path, weight_dtype)
    print("✅ 方法2加载成功")

    # 比较两个模型
    print("\n" + "="*50)
    is_same = compare_models(transformer1, transformer2, "transformers", tolerance=1e-6)
    print("="*50)

    # 额外信息
    print("\n📋 额外信息:")
    print(f"Pipeline transformer device: {transformer1.device}")
    print(f"Direct load transformer device: {transformer2.device}")
    print(f"Pipeline transformer dtype: {next(transformer1.parameters()).dtype}")
    print(f"Direct load transformer dtype: {next(transformer2.parameters()).dtype}")

    # 检查配置
    if hasattr(transformer1, 'config') and hasattr(transformer2, 'config'):
        keys_to_compare = ['patch_size', 'in_channels', 'out_channels', 'num_layers', 'attention_head_dim', 'num_attention_heads', 'joint_attention_dim', 'guidance_embeds', 'axes_dims_rope']
        config1 = transformer1.config
        config2 = transformer2.config
        for key in keys_to_compare:
            assert config1[key] == config2[key], f"配置 {key} 不同: {config1[key]} vs {config2[key]}"
    # 最终断言
    assert is_same, "两种加载方式得到的transformer模型不完全相同"
    print("🎉 结论: 两种加载方式得到的transformer模型完全相同！")


if __name__ == "__main__":
    """
    直接运行此文件进行测试
    """
    test_load_transformer_consistency()

#!/usr/bin/env python3
"""
测试脚本：比较两个text_encoder模型是否相同
"""

import torch
import sys
import os
from diffusers import QwenImageEditPipeline

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qflux.models.load_model import load_qwenvl


def compare_models(model1, model2, model1_name="Model1", model2_name="Model2", tolerance=1e-6):
    """
    比较两个模型，包括参数形状和数值差异
    Args:
        model1, model2: 要比较的模型
        model1_name, model2_name: 模型名称（用于日志）
        tolerance: 数值差异的容差
    Returns:
        bool: 模型是否相同
    """
    print(f"\n🔍 比较 {model1_name} vs {model2_name}")
    print("-" * 60)

    # 获取state_dict
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # 比较参数键
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    print("📊 参数统计:")
    print(f"  {model1_name}: {len(keys1)} 个参数")
    print(f"  {model2_name}: {len(keys2)} 个参数")

    # 检查缺失/额外的键
    missing_in_model2 = keys1 - keys2
    missing_in_model1 = keys2 - keys1
    common_keys = keys1 & keys2

    if missing_in_model2:
        print(f"\n❌ {model1_name}中有但{model2_name}中没有的参数:")
        for key in sorted(list(missing_in_model2)[:10]):  # 只显示前10个
            print(f"  - {key}")
        if len(missing_in_model2) > 10:
            print(f"  ... 还有 {len(missing_in_model2) - 10} 个")

    if missing_in_model1:
        print(f"\n❌ {model2_name}中有但{model1_name}中没有的参数:")
        for key in sorted(list(missing_in_model1)[:10]):  # 只显示前10个
            print(f"  - {key}")
        if len(missing_in_model1) > 10:
            print(f"  ... 还有 {len(missing_in_model1) - 10} 个")

    print(f"\n✅ 共同参数: {len(common_keys)}")

    if not common_keys:
        print("❌ 没有找到共同参数!")
        return False

    # 比较共同参数的形状和数值
    shape_mismatches = []
    value_differences = []
    identical_params = 0

    print(f"\n🔍 详细参数比较 (容差={tolerance}):")
    print("-" * 60)

    # 只显示前20个参数的详细比较，避免输出过长
    keys_to_check = sorted(list(common_keys))
    show_details = len(keys_to_check) <= 50

    for i, key in enumerate(keys_to_check):
        param1 = state_dict1[key]
        param2 = state_dict2[key]

        # 比较形状
        if param1.shape != param2.shape:
            shape_mismatches.append((key, param1.shape, param2.shape))
            if show_details or i < 20:
                print(f"❌ {key}: 形状不匹配 - {param1.shape} vs {param2.shape}")
            continue

        # 比较数值
        try:
            # 转换到相同设备和数据类型进行比较
            param1_cpu = param1.detach().cpu().float()
            param2_cpu = param2.detach().cpu().float()

            # 计算差异
            abs_diff = torch.abs(param1_cpu - param2_cpu)
            max_diff = torch.max(abs_diff).item()
            mean_diff = torch.mean(abs_diff).item()

            # 检查参数是否在容差范围内相同
            if max_diff <= tolerance:
                identical_params += 1
                if show_details and i < 10:  # 只显示前10个相同参数的详情
                    print(f"✅ {key}: 相同 (max_diff={max_diff:.2e})")
            else:
                value_differences.append((key, max_diff, mean_diff))
                if show_details or i < 20:
                    print(f"⚠️  {key}: 不同 - max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

        except Exception as e:
            if show_details or i < 20:
                print(f"❌ {key}: 比较数值时出错 - {e}")

    if not show_details:
        print(f"... (只显示前20个参数的详情，共检查了{len(keys_to_check)}个参数)")

    # 汇总
    print("\n📋 比较结果汇总:")
    print(f"  共同参数总数: {len(common_keys)}")
    print(f"  相同参数: {identical_params}")
    print(f"  形状不匹配: {len(shape_mismatches)}")
    print(f"  数值差异: {len(value_differences)}")

    if shape_mismatches:
        print("\n❌ 形状不匹配的参数 (前10个):")
        for key, shape1, shape2 in shape_mismatches[:10]:
            print(f"  {key}: {shape1} vs {shape2}")
        if len(shape_mismatches) > 10:
            print(f"  ... 还有 {len(shape_mismatches) - 10} 个")

    if value_differences:
        print("\n⚠️  数值差异最大的参数 (前10个):")
        # 按最大差异排序并显示前10个
        value_differences.sort(key=lambda x: x[1], reverse=True)
        for i, (key, max_diff, mean_diff) in enumerate(value_differences[:10]):
            print(f"  {i+1}. {key}: max={max_diff:.2e}, mean={mean_diff:.2e}")
        if len(value_differences) > 10:
            print(f"  ... 还有 {len(value_differences) - 10} 个")

    # 最终判断
    models_identical = (len(shape_mismatches) == 0 and len(value_differences) == 0)
    print(f"\n🎯 最终结论: {'完全相同' if models_identical else '存在差异'}")

    return models_identical


def main():
    """主函数：执行模型比较"""
    print("=" * 80)
    print("模型比较测试：text_encoder vs qwen_vl")
    print("=" * 80)

    try:
        print("\n🚀 正在加载模型...")

        # 加载完整pipeline并提取text_encoder
        print("  加载 QwenImageEditPipeline...")
        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit",
            torch_dtype=torch.bfloat16,
        )
        text_encoder = pipe.text_encoder

        # 单独加载qwen_vl
        print("  加载 qwen_vl...")
        qwen_vl = load_qwenvl("Qwen/Qwen-Image-Edit", torch.bfloat16)

        print("✅ 模型加载完成!")

        # 比较两个模型
        print("\n" + "=" * 80)
        print("比较 text_encoder (从pipe提取) vs qwen_vl (单独加载)")
        print("=" * 80)

        compare_models(
            text_encoder, qwen_vl,
            "text_encoder (从pipe)", "qwen_vl (单独加载)",
            tolerance=1e-6
        )

        # 如果qwen_vl有text_encoder属性，也比较一下
        if hasattr(qwen_vl, 'text_encoder'):
            print("\n" + "=" * 80)
            print("额外比较: text_encoder vs qwen_vl.text_encoder")
            print("=" * 80)

            compare_models(
                text_encoder, qwen_vl.text_encoder,
                "text_encoder (从pipe)", "qwen_vl.text_encoder",
                tolerance=1e-6
            )

        # 最终结论
        print("\n" + "=" * 80)
        print("🏁 测试完成!")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

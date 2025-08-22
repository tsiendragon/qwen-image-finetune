#!/usr/bin/env python3
"""
预计算嵌入脚本
用于预先计算并缓存训练数据集的嵌入，以加速训练过程
"""

import argparse
import sys
import os

# 添加源代码路径
sys.path.append('src')

from data.dataset import loader
from data.cache_hook import create_cache_hook
from utils.options import parse_args


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="预计算数据集嵌入")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--start_idx", type=int, default=0, help="开始索引")
    parser.add_argument("--end_idx", type=int, default=None, help="结束索引")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    args = parser.parse_args()

    # 解析训练配置
    config = parse_args(config_path=args.config)

    print("=== 嵌入预计算脚本 ===")
    print(f"配置文件: {args.config}")
    print(f"数据集路径: {config.data.init_args.get('dataset_path', 'N/A')}")
    print(f"缓存目录: {config.data.init_args.get('cache_dir', 'N/A')}")
    print(f"索引范围: {args.start_idx} - {args.end_idx or '末尾'}")

    # 创建数据加载器
    print("\n正在创建数据加载器...")
    train_dataloader = loader(**config.data.init_args)
    dataset = train_dataloader.dataset

    # 检查缓存配置
    if not dataset.use_cache:
        print("警告: 数据集未启用缓存，请检查配置")
        return

    # 显示数据集信息
    print(f"数据集大小: {len(dataset)}")
    if dataset.cache_manager:
        cache_stats = dataset.get_cache_stats()
        print(f"当前缓存统计: {cache_stats}")

    # 创建缓存钩子
    print("\n正在创建缓存钩子...")
    cache_hook = create_cache_hook(config)

    # 开始预计算
    print("\n开始预计算嵌入...")
    try:
        cache_hook.precompute_embeddings(
            dataset=dataset,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
        print("\n预计算完成!")

        # 显示最终统计
        if dataset.cache_manager:
            final_stats = dataset.get_cache_stats()
            print(f"最终缓存统计: {final_stats}")

    except KeyboardInterrupt:
        print("\n用户中断预计算")
    except Exception as e:
        print(f"\n预计算过程中出错: {e}")
        import traceback
        traceback.print_exc()


def test_cached_loading():
    """测试缓存加载功能"""
    print("\n=== 测试缓存加载 ===")

    # 使用默认配置
    config = parse_args(config_path="configs/qwen_image_edit_config.yaml")
    train_dataloader = loader(**config.data.init_args)
    dataset = train_dataloader.dataset

    if not dataset.use_cache:
        print("缓存未启用，跳过测试")
        return

    print(f"数据集大小: {len(dataset)}")

    # 测试加载第一个样本
    print("正在测试第一个样本...")
    sample = dataset[0]

    if sample.get('cached', False):
        print("✓ 成功从缓存加载!")
        print(f"  - pixel_latent shape: {sample['pixel_latent'].shape}")
        print(f"  - control_latent shape: {sample['control_latent'].shape}")
        print(f"  - prompt_embed shape: {sample['prompt_embed'].shape}")
        print(f"  - prompt_embeds_mask shape: {sample['prompt_embeds_mask'].shape}")
    else:
        print("× 未找到缓存，需要先运行预计算")
        print("请运行: python precompute_embeddings.py --config configs/qwen_image_edit_config.yaml")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有参数，运行测试
        test_cached_loading()
    else:
        # 否则运行预计算
        main()

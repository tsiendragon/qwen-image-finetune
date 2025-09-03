#!/usr/bin/env python3
"""
测试配置模块的功能
"""

import sys
import os
import tempfile
import yaml
sys.path.append('src')

from data.config import load_config_from_yaml


def test_config_loading():
    """测试配置文件加载功能"""
    print("Testing configuration loading...")

    # 使用项目中的实际配置文件
    config_file = '/home/lilong/repos/qwen-image-finetune/configs/qwen_image_edit_config.yaml'

    try:
        config = load_config_from_yaml(config_file)
        print("✓ Configuration loaded successfully")

        # 测试转换为扁平字典
        flat_config = config.to_flat_dict()
        print("✓ Configuration converted to flat dictionary")
        print(f"Total parameters: {len(flat_config)}")

        # 显示各个配置块
        print("\nConfiguration blocks:")
        model_params = [k for k in flat_config.keys() if k in ['pretrained_model_name_or_path', 'rank', 'lora_r', 'lora_alpha', 'lora_init_weights', 'lora_target_modules']]
        data_params = [k for k in flat_config.keys() if k in ['data_class_path', 'data_init_args']]
        logging_params = [k for k in flat_config.keys() if k in ['output_dir', 'logging_dir', 'report_to', 'tracker_project_name']]
        optimizer_params = [k for k in flat_config.keys() if k in ['optimizer_class_path', 'optimizer_init_args']]
        scheduler_params = [k for k in flat_config.keys() if k in ['lr_scheduler', 'lr_warmup_steps', 'lr_num_cycles', 'lr_power']]
        train_params = [k for k in flat_config.keys() if k in ['train_batch_size', 'gradient_accumulation_steps', 'max_train_steps', 'checkpointing_steps', 'checkpoints_total_limit', 'max_grad_norm', 'mixed_precision']]

        print(f"- Model: {len(model_params)} parameters")
        print(f"- Data: {len(data_params)} parameters")
        print(f"- Logging: {len(logging_params)} parameters")
        print(f"- Optimizer: {len(optimizer_params)} parameters")
        print(f"- LR Scheduler: {len(scheduler_params)} parameters")
        print(f"- Train: {len(train_params)} parameters")

        return config

    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_config_defaults():
    """测试配置默认值处理"""
    print("\nTesting default value handling...")

    # 创建临时目录用于测试文件
    with tempfile.TemporaryDirectory() as temp_dir:
        # 测试缺失参数的默认值处理
        minimal_config = {"optimizer": {"learning_rate": 0.001}}
        minimal_config_path = os.path.join(temp_dir, "minimal_config.yaml")

        with open(minimal_config_path, 'w') as f:
            yaml.dump(minimal_config, f)

        try:
            minimal_loaded = load_config_from_yaml(minimal_config_path)
            print("✓ Missing parameters handled with default values")
            return True
        except Exception as e:
            print(f"❌ Default value handling failed: {e}")
            return False


def test_config_validation():
    """测试配置验证功能"""
    print("\nTesting configuration validation...")

    # 创建临时目录用于测试文件
    with tempfile.TemporaryDirectory() as temp_dir:
        # 测试类型错误检测
        try:
            invalid_config = {"optimizer": {"learning_rate": "invalid"}}
            invalid_config_path = os.path.join(temp_dir, "invalid_config.yaml")

            with open(invalid_config_path, 'w') as f:
                yaml.dump(invalid_config, f)

            invalid_loaded = load_config_from_yaml(invalid_config_path)
            # 这应该触发验证错误
            print("✗ Type validation failed - this should not happen")
            return False
        except Exception as e:
            print("✓ Type validation working correctly")
            return True


def run_all_tests():
    """运行所有配置测试"""
    print("=" * 50)
    print("运行配置模块测试...")
    print("=" * 50)

    test_results = []

    # 测试配置加载
    config = test_config_loading()
    test_results.append(config is not None)

    # 测试默认值处理
    test_results.append(test_config_defaults())

    # 测试配置验证
    test_results.append(test_config_validation())

    # 总结测试结果
    passed_tests = sum(test_results)
    total_tests = len(test_results)

    print(f"\n{'=' * 50}")
    print(f"测试完成: {passed_tests}/{total_tests} 通过")

    if passed_tests == total_tests:
        print("✓ 所有配置测试通过!")
    else:
        print("❌ 部分测试失败")
    print("=" * 50)

    return passed_tests == total_tests


if __name__ == "__main__":
    run_all_tests()

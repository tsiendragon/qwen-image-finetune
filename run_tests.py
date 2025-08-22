#!/usr/bin/env python3
"""
运行模型比较测试的脚本
"""

import sys
import os
import unittest

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def run_model_comparison_tests():
    """运行模型比较测试"""
    print("🧪 开始运行Qwen Image模型比较测试...")
    print("="*80)

    # 发现并运行测试
    loader = unittest.TestLoader()

    # 加载特定的测试模块
    try:
        suite = loader.loadTestsFromName('tests.test_model_comparison')
    except ImportError as e:
        print(f"❌ 无法导入测试模块: {e}")
        return 1

    # 运行测试
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=False
    )

    result = runner.run(suite)

    # 输出测试结果摘要
    print("\n" + "="*80)
    print("🏁 测试完成！")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")

    if result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")

    if result.errors:
        print("\n💥 错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")

    if result.skipped:
        print(f"\n⏭️ 跳过的测试: {len(result.skipped)}")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")

    print("="*80)

    # 返回退出代码
    return 0 if result.wasSuccessful() else 1

def run_specific_test(test_name=None):
    """运行特定测试"""
    if test_name:
        print(f"🎯 运行特定测试: {test_name}")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(f'tests.test_model_comparison.{test_name}')
    else:
        print("🧪 运行所有模型比较测试...")
        suite = unittest.TestLoader().discover('tests', pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='运行Qwen Image模型测试')
    parser.add_argument('--test', '-t', help='运行特定测试方法', default=None)
    parser.add_argument('--list', '-l', action='store_true', help='列出所有可用测试')

    args = parser.parse_args()

    if args.list:
        print("📝 可用的测试:")
        print("  TestModelComparison:")
        print("    - test_local_model_creation")
        print("    - test_pretrained_model_loading")
        print("    - test_parameter_count_comparison")
        print("    - test_parameter_shapes_comparison")
        print("    - test_parameter_names_completeness")
        print("    - test_config_comparison")
        print("  TestModelFunctionality:")
        print("    - test_model_forward_pass")
        print("\n使用方法: python run_tests.py --test TestModelComparison.test_parameter_shapes_comparison")
        sys.exit(0)

    if args.test:
        exit_code = run_specific_test(args.test)
    else:
        exit_code = run_model_comparison_tests()

    sys.exit(exit_code)

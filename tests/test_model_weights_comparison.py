"""
测试函数：比较本地权重和HuggingFace仓库权重的参数差异
使用FluxKontextLoraTrainer加载两种不同来源的模型，检查参数是否相等
"""

import torch
import pytest
import logging
import os
from typing import Dict, Any
from unittest.mock import Mock

from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer
from qflux.utils.model_compare import compare_model_parameters
from qflux.data.config import load_config_from_yaml

logger = logging.getLogger(__name__)


class TestModelWeightsComparison:
    """测试模型权重比较功能"""

    def setup_method(self):
        """设置测试环境"""
        # 使用指定的配置文件路径
        self.config_path = 'tests/test_configs/test_example_fluxkontext_fp16_character_composition.yaml'

    def create_config_with_lora_weight(self, lora_weight: str = None):
        """从YAML配置文件创建配置对象并设置LoRA权重"""
        # 从YAML文件加载配置
        config = load_config_from_yaml(self.config_path)

        # 设置LoRA权重
        if lora_weight:
            config.model.lora.pretrained_weight = lora_weight

        return config

    def compare_lora_weights(
        self,
        local_weights: str,
        repo_id: str,
        tolerance: float = 1e-6,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        比较本地权重和HuggingFace仓库权重的参数差异

        Args:
            local_weights: 本地权重文件路径
            repo_id: HuggingFace仓库ID
            tolerance: 比较容差
            verbose: 是否打印详细信息

        Returns:
            包含比较结果的字典
        """
        if verbose:
            print(f"\n{'='*80}")
            print("开始比较模型权重:")
            print(f"本地权重: {local_weights}")
            print(f"HuggingFace仓库: {repo_id}")
            print(f"容差: {tolerance}")
            print(f"{'='*80}")

        # 检查本地权重文件是否存在
        if not os.path.exists(local_weights):
            raise FileNotFoundError(f"本地权重文件不存在: {local_weights}")

        try:
            # 1. 创建使用本地权重的trainer
            if verbose:
                print("\n📂 加载本地权重模型...")

            local_config = self.create_config_with_lora_weight(lora_weight=local_weights)
            local_trainer = FluxKontextLoraTrainer(local_config)

            # 设置预测模式加载模型
            local_trainer.setup_predict()
            local_model = local_trainer.dit

            if verbose:
                print("✅ 本地权重模型加载成功")
                print(f"   模型类型: {type(local_model)}")
                print(f"   参数数量: {sum(p.numel() for p in local_model.parameters()):,}")

            # 2. 创建使用HuggingFace仓库权重的trainer
            if verbose:
                print("\n🤗 加载HuggingFace仓库权重模型...")

            hf_config = self.create_config_with_lora_weight(lora_weight=repo_id)
            hf_trainer = FluxKontextLoraTrainer(hf_config)

            # 设置预测模式加载模型
            hf_trainer.setup_predict()
            hf_model = hf_trainer.dit

            if verbose:
                print("✅ HuggingFace权重模型加载成功")
                print(f"   模型类型: {type(hf_model)}")
                print(f"   参数数量: {sum(p.numel() for p in hf_model.parameters()):,}")

            # 3. 比较模型参数
            if verbose:
                print("\n🔍 开始比较模型参数...")

            comparison_results = compare_model_parameters(
                model1=local_model,
                model2=hf_model,
                model1_name="本地权重模型",
                model2_name="HuggingFace权重模型",
                relative_threshold=tolerance,
                verbose=verbose
            )

            # 4. 额外的LoRA特定比较
            if verbose:
                print("\n🎯 LoRA特定参数分析...")

            # 检查LoRA参数
            local_lora_params = self._extract_lora_parameters(local_model)
            hf_lora_params = self._extract_lora_parameters(hf_model)

            comparison_results['lora_analysis'] = self._compare_lora_parameters(
                local_lora_params, hf_lora_params, tolerance, verbose
            )

            # 5. 生成最终报告
            if verbose:
                self._print_final_report(comparison_results, tolerance)

            return comparison_results

        except Exception as e:
            error_msg = f"模型权重比较过程中发生错误: {str(e)}"
            logger.error(error_msg)
            if verbose:
                print(f"❌ {error_msg}")
            raise

    def _extract_lora_parameters(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """提取LoRA参数"""
        lora_params = {}
        for name, param in model.named_parameters():
            if any(lora_key in name.lower() for lora_key in ['lora_a', 'lora_b', 'adapter']):
                lora_params[name] = param.detach().cpu()
        return lora_params

    def _compare_lora_parameters(
        self,
        local_params: Dict[str, torch.Tensor],
        hf_params: Dict[str, torch.Tensor],
        tolerance: float,
        verbose: bool
    ) -> Dict[str, Any]:
        """比较LoRA特定参数"""
        lora_results = {
            'total_local_lora_params': len(local_params),
            'total_hf_lora_params': len(hf_params),
            'common_lora_params': 0,
            'identical_lora_params': 0,
            'different_lora_params': [],
            'missing_lora_params': {
                'in_local': list(set(hf_params.keys()) - set(local_params.keys())),
                'in_hf': list(set(local_params.keys()) - set(hf_params.keys()))
            }
        }

        common_lora_keys = set(local_params.keys()).intersection(set(hf_params.keys()))
        lora_results['common_lora_params'] = len(common_lora_keys)
        if verbose:
            print(f"   本地LoRA参数: {len(local_params)}")
            print(f"   HF LoRA参数: {len(hf_params)}")
            print(f"   共同LoRA参数: {len(common_lora_keys)}")

        for key in sorted(common_lora_keys):
            local_param = local_params[key]
            hf_param = hf_params[key]

            if local_param.shape == hf_param.shape:
                # 计算差异
                abs_diff = torch.abs(local_param - hf_param)
                max_diff = torch.max(abs_diff).item()
                mean_diff = torch.mean(abs_diff).item()

                if max_diff <= tolerance:
                    lora_results['identical_lora_params'] += 1
                    if verbose:
                        print(f"   ✅ {key}: 相同 (max_diff={max_diff:.2e})")
                else:
                    lora_results['different_lora_params'].append({
                        'key': key,
                        'max_diff': max_diff,
                        'mean_diff': mean_diff
                    })
                    if verbose:
                        long_line = f"   ⚠️  {key}: 不同 - "
                        long_line += f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
                        print(long_line)

        return lora_results

    def _print_final_report(self, results: Dict[str, Any], tolerance: float):
        """打印最终比较报告"""
        print(f"\n{'='*80}")
        print(f"最终比较报告 (容差: {tolerance})")
        print(f"{'='*80}")

        # 基本统计
        stats = results.get('statistics', {})
        print("📊 基本统计:")
        print(f"   总参数相同: {stats.get('identical_params', 0)}")
        print(f"   总参数不同: {len(results.get('value_differences', []))}")
        print(f"   形状不匹配: {len(results.get('shape_differences', []))}")

        # LoRA统计
        lora_stats = results.get('lora_analysis', {})
        if lora_stats:
            print("\n🎯 LoRA参数统计:")
            print(f"   LoRA参数相同: {lora_stats.get('identical_lora_params', 0)}")
            print(f"   LoRA参数不同: {len(lora_stats.get('different_lora_params', []))}")

        # 判断结果
        total_differences = (len(results.get('value_differences', [])) +
                             len(results.get('shape_differences', [])))
        lora_differences = len(lora_stats.get('different_lora_params', []))

        if total_differences == 0 and lora_differences == 0:
            print("\n🎉 结论: 本地权重和HuggingFace权重完全相同!")
        elif lora_differences == 0:
            print("\n✅ 结论: LoRA权重完全相同，基础模型可能有差异")
        else:
            print("\n⚠️  结论: 发现权重差异，请检查模型版本或权重文件")

        print(f"{'='*80}")


def test_compare_lora_weights_equal():
    """测试相同权重的比较 - 应该完全相等"""
    tester = TestModelWeightsComparison()

    # 这里需要提供实际的本地权重路径和对应的HuggingFace仓库ID
    # 示例:
    base_path = '/tmp/image_edit_lora/character_composition_fp16/'
    weights_file = 'characterCompositionFluxKontextFp16/pytorch_lora_weights.safetensors'
    local_weights = base_path + weights_file
    repo_id = 'TsienDragon/qwen-image-edit-character-composition'

    # 检查文件是否存在，如果不存在则跳过测试
    if not os.path.exists(local_weights):
        pytest.skip(f"本地权重文件不存在: {local_weights}")

    try:
        results = tester.compare_lora_weights(
            local_weights=local_weights,
            repo_id=repo_id,
            tolerance=1e-6,
            verbose=True
        )

        # 验证结果
        assert results is not None, "比较结果不能为空"
        assert 'lora_analysis' in results, "缺少LoRA分析结果"

        # 检查是否有显著差异
        value_differences = results.get('value_differences', [])
        shape_differences = results.get('shape_differences', [])
        lora_differences = results['lora_analysis'].get('different_lora_params', [])

        total_diffs = len(value_differences) + len(shape_differences) + len(lora_differences)
        if total_diffs > 0:
            print("\n⚠️  发现模型权重差异，这可能是正常的（不同的训练轮次或版本）")
        else:
            print("\n🎉 权重完全相同!")

    except Exception as e:
        pytest.fail(f"测试失败: {str(e)}")


def test_compare_lora_weights_function():
    """独立的权重比较函数测试"""

    def compare_lora_weights_standalone(local_weights: str, repo_id: str,
                                        tolerance: float = 1e-6) -> bool:
        """
        独立的权重比较函数

        Args:
            local_weights: 本地权重文件路径
            repo_id: HuggingFace仓库ID
            tolerance: 比较容差

        Returns:
            bool: True表示权重相等，False表示不相等
        """
        tester = TestModelWeightsComparison()
        tester.setup_method()

        try:
            results = tester.compare_lora_weights(
                local_weights=local_weights,
                repo_id=repo_id,
                tolerance=tolerance,
                verbose=True
            )

            # 检查是否存在显著差异
            value_differences = results.get('value_differences', [])
            shape_differences = results.get('shape_differences', [])
            lora_analysis = results.get('lora_analysis', {})
            lora_differences = lora_analysis.get('different_lora_params', [])

            # 权重相等的条件：没有值差异、形状差异和LoRA差异
            weights_equal = (
                len(value_differences) == 0 and
                len(shape_differences) == 0 and
                len(lora_differences) == 0
            )

            return weights_equal

        except Exception as e:
            logger.error(f"权重比较失败: {str(e)}")
            return False

    # 示例使用
    base_path = '/tmp/image_edit_lora/character_composition_fp16/'
    weights_file = 'characterCompositionFluxKontextFp16/pytorch_lora_weights.safetensors'
    local_weights = base_path + weights_file
    repo_id = 'TsienDragon/qwen-image-edit-character-composition'

    if os.path.exists(local_weights):
        are_equal = compare_lora_weights_standalone(local_weights, repo_id)
        print(f"\n权重比较结果: {'相等' if are_equal else '不相等'}")
    else:
        print(f"本地权重文件不存在，跳过测试: {local_weights}")


def compare_weights_simple(local_weights: str, repo_id: str, tolerance: float = 1e-6) -> bool:
    """
    简单的权重比较函数

    Args:
        local_weights: 本地权重文件路径
        repo_id: HuggingFace仓库ID
        tolerance: 比较容差

    Returns:
        bool: True表示权重相等，False表示不相等
    """
    config_path = 'tests/test_configs/test_example_fluxkontext_fp16_character_composition.yaml'

    print(f"\n{'='*80}")
    print("开始比较模型权重:")
    print(f"本地权重: {local_weights}")
    print(f"HuggingFace仓库: {repo_id}")
    print(f"配置文件: {config_path}")
    print(f"容差: {tolerance}")
    print(f"{'='*80}")

    # 检查本地权重文件是否存在
    if not os.path.exists(local_weights):
        print(f"❌ 本地权重文件不存在: {local_weights}")
        return False

    try:
        # 1. 加载本地权重模型
        print("\n📂 加载本地权重模型...")
        local_config = load_config_from_yaml(config_path)
        local_config.model.lora.pretrained_weight = local_weights
        local_trainer = FluxKontextLoraTrainer(local_config)
        local_trainer.setup_predict()
        local_model = local_trainer.dit

        print("✅ 本地权重模型加载成功")
        print(f"   模型类型: {type(local_model)}")
        print(f"   参数数量: {sum(p.numel() for p in local_model.parameters()):,}")

        # 2. 加载HuggingFace权重模型
        print("\n🤗 加载HuggingFace仓库权重模型...")
        hf_config = load_config_from_yaml(config_path)
        hf_config.model.lora.pretrained_weight = repo_id
        hf_trainer = FluxKontextLoraTrainer(hf_config)
        hf_trainer.setup_predict()
        hf_model = hf_trainer.dit

        print("✅ HuggingFace权重模型加载成功")
        print(f"   模型类型: {type(hf_model)}")
        print(f"   参数数量: {sum(p.numel() for p in hf_model.parameters()):,}")

        # 3. 比较模型参数
        print("\n🔍 开始比较模型参数...")

        local_state_dict = local_model.state_dict()
        hf_state_dict = hf_model.state_dict()

        local_keys = set(local_state_dict.keys())
        hf_keys = set(hf_state_dict.keys())
        common_keys = local_keys.intersection(hf_keys)

        print(f"本地模型参数: {len(local_keys)}")
        print(f"HF模型参数: {len(hf_keys)}")
        print(f"共同参数: {len(common_keys)}")

        if len(local_keys - hf_keys) > 0:
            print(f"本地独有参数: {len(local_keys - hf_keys)}")
        if len(hf_keys - local_keys) > 0:
            print(f"HF独有参数: {len(hf_keys - local_keys)}")

        # 比较参数值
        identical_params = 0
        different_params = 0
        shape_mismatches = 0

        for key in sorted(common_keys):
            local_param = local_state_dict[key]
            hf_param = hf_state_dict[key]

            # 检查形状
            if local_param.shape != hf_param.shape:
                shape_mismatches += 1
                print(f"❌ {key}: 形状不匹配 - {local_param.shape} vs {hf_param.shape}")
                continue

            # 比较值
            try:
                local_param_cpu = local_param.detach().cpu().float()
                hf_param_cpu = hf_param.detach().cpu().float()

                abs_diff = torch.abs(local_param_cpu - hf_param_cpu)
                max_diff = torch.max(abs_diff).item()

                if max_diff <= tolerance:
                    identical_params += 1
                else:
                    different_params += 1
                    if different_params <= 10:  # 只显示前10个不同的参数
                        print(f"⚠️  {key}: 不同 - max_diff={max_diff:.2e}")
                    elif different_params == 11:
                        print("   ... (更多差异参数未显示)")

            except Exception as e:
                print(f"❌ {key}: 比较时出错 - {e}")

        # 结果统计
        print(f"\n📊 比较结果:")
        print(f"   相同参数: {identical_params}")
        print(f"   不同参数: {different_params}")
        print(f"   形状不匹配: {shape_mismatches}")

        # 判断结果
        weights_equal = (different_params == 0 and shape_mismatches == 0)

        if weights_equal:
            print("\n🎉 结论: 本地权重和HuggingFace权重完全相同!")
        else:
            print(f"\n⚠️  结论: 发现权重差异 - {different_params} 个参数不同，{shape_mismatches} 个形状不匹配")

        print(f"{'='*80}")
        return weights_equal

    except Exception as e:
        error_msg = f"模型权重比较过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        return False


if __name__ == "__main__":
    # 直接运行测试
    print("运行模型权重比较测试...")

    # 示例使用
    local_weights = '/tmp/image_edit_lora/character_composition_fp16/characterCompositionFluxKontextFp16/pytorch_lora_weights.safetensors'
    repo_id = 'TsienDragon/qwen-image-edit-character-composition'

    if os.path.exists(local_weights):
        are_equal = compare_weights_simple(local_weights, repo_id)
        print(f"\n最终结果: 权重{'相等' if are_equal else '不相等'}")
    else:
        print(f"本地权重文件不存在，跳过测试: {local_weights}")

    # 可选：运行pytest测试
    # pytest -v tests/test_model_weights_comparison.py::test_compare_lora_weights_equal

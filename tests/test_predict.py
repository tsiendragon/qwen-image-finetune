#!/usr/bin/env python3
"""
测试脚本：使用QwenImageEditTrainer进行预测测试
支持带/不带LoRA权重的对比测试
"""

import os
import sys
import torch
import argparse
import logging
import copy
from PIL import Image
import numpy as np
from pathlib import Path
import time
# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
print('project_root', project_root)
sys.path.insert(0, str(project_root))
try:
    from qflux.data.config import load_config_from_yaml
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str):
    """加载配置文件"""
    return load_config_from_yaml(config_path)


def load_test_data(image_path: str, prompt_path: str):
    """加载测试数据"""
    # 加载图像
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert('RGB')
    logger.info(f"加载图像: {image_path}, 尺寸: {image.size}")

    # 加载提示词
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    else:
        prompt = prompt_path

    logger.info(f"加载提示词: {prompt[:100]}...")

    return image, prompt


def create_config_for_predict(base_config, lora_weight_path: str = None):
    """为预测创建配置"""
    predict_config = copy.deepcopy(base_config)

    # 设置预测模式的LoRA权重
    if lora_weight_path and os.path.exists(lora_weight_path):
        predict_config.model.lora.pretrained_weight = lora_weight_path
        logger.info(f"使用LoRA权重: {lora_weight_path}")
    else:
        predict_config.model.lora.pretrained_weight = None
        logger.info("不使用LoRA权重（基础模型）")

    return predict_config


def run_prediction(trainer,
                   image: Image.Image,
                   prompt: str,
                   num_inference_steps: int = 20,
                   cfg_scale: float = 4.0) -> np.ndarray:
    """运行预测"""
    logger.info(f"开始预测，推理步数: {num_inference_steps}, CFG scale: {cfg_scale}")

    start_time = time.time()

    # 运行预测
    print('prompt', prompt)
    print('number of inference steps', num_inference_steps)
    print('cfg_scale', cfg_scale)
    height, width = image.size
    height = int(height//16)*16
    width = int(width//16)*16
    result = trainer.predict(
        prompt_image=image,
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=num_inference_steps,
        true_cfg_scale=cfg_scale,
        weight_dtype=torch.bfloat16,
        height=height,
        width=width
    )

    end_time = time.time()
    logger.info(f"预测完成，耗时: {end_time - start_time:.2f}秒")

    return result


def save_result(result, output_path: str, suffix: str = ""):
    """保存预测结果"""
    # 处理不同的结果类型
    if isinstance(result, list):
        # 如果是列表，取第一个元素
        result_image = result[0]
    elif isinstance(result, Image.Image):
        # 如果是单个PIL图像
        result_image = result
    elif isinstance(result, np.ndarray):
        # 如果是numpy数组，转换为PIL图像
        if result.ndim == 4:
            # 4维数组: (batch, height, width, channels) -> 取第一个
            result = result[0]

        if result.ndim == 3:
            result = result*255
            result_image = Image.fromarray(result.astype(np.uint8))
        else:
            raise ValueError(f"Unexpected result shape: {result.shape}")
    else:
        raise ValueError(f"Unexpected result type: {type(result)}")

    # 确保是PIL Image对象
    if not isinstance(result_image, Image.Image):
        raise ValueError(f"Expected PIL Image, got {type(result_image)}")

    # 生成输出文件名
    base_name = Path(output_path).stem
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir / f"{base_name}{suffix}.jpg"
    result_image.save(final_path, quality=95)
    logger.info(f"结果已保存: {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Qwen Image Edit 预测测试")
    parser.add_argument("--config", type=str,
                        default="configs/qwen_image_edit_config_r2.yaml",
                        help="配置文件路径")
    parser.add_argument("--image", type=str,
                        default="/raid/lilong/data/kyc_gen/aligned/control_images/row_fff37bba.jpg",
                        help="输入图像路径")
    parser.add_argument("--prompt", type=str,
                        default="/raid/lilong/data/kyc_gen/aligned/training_images/row_fff37bba.txt",
                        help="提示词文件路径")
    parser.add_argument("--prompt-text", type=str,
                        default=None,
                        help="直接提供的提示词文本")
    parser.add_argument("--lora-weight", type=str,
                        default=None,
                        help="LoRA权重路径 (可选)")
    parser.add_argument("--output-dir", type=str,
                        default="tests/outputs",
                        help="输出目录")
    parser.add_argument("--steps", type=int, default=20,
                        help="推理步数")
    parser.add_argument("--cfg-scale", type=float, default=2.5,
                        help="CFG guidance scale")
    parser.add_argument("--compare", action="store_true",
                        help="对比测试（基础模型 vs LoRA模型）")

    args = parser.parse_args()

    logger.info("开始Qwen Image Edit预测测试")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输入图像: {args.image}")
    logger.info(f"提示词文件: {args.prompt}")

    try:
        # 加载配置文件
        config_path = project_root / args.config
        base_config = load_config(config_path)
        from qflux.main import import_trainer

        Trainer = import_trainer(base_config)
        image, prompt = load_test_data(args.image, args.prompt)

        # 创建输出目录
        output_dir = project_root / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存输入图像用于对比
        input_copy_path = output_dir / "input_image.jpg"
        image.save(input_copy_path, quality=95)
        logger.info(f"输入图像副本已保存: {input_copy_path}")

        # 保存提示词
        prompt_path = output_dir / "prompt.txt"
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        logger.info(f"提示词已保存: {prompt_path}")

        if args.compare and args.lora_weight:
            # 对比测试模式
            logger.info("=" * 60)
            logger.info("开始对比测试：基础模型 vs LoRA模型")
            logger.info("=" * 60)


            logger.info("\n--- 测试基础模型 ---")
            base_config_predict = create_config_for_predict(base_config, None)
            lora_config_predict = create_config_for_predict(base_config, args.lora_weight)

            base_trainer = Trainer(base_config_predict)

            base_result = run_prediction(
                base_trainer, image, prompt,
                args.steps, args.cfg_scale
            )

            base_output_path = save_result(
                base_result,
                str(output_dir / "result_base_model.jpg"),
                ""
            )

            # 清理基础模型
            del base_trainer
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            # weight some time
            time.sleep(10)

            # 测试2: LoRA模型
            logger.info("\n--- 测试LoRA模型 ---")
            lora_config_predict = create_config_for_predict(base_config, args.lora_weight)
            print('with lora weight', args.lora_weight)

            lora_trainer = Trainer(lora_config_predict)


            lora_result = run_prediction(
                lora_trainer, image, prompt,
                args.steps, args.cfg_scale
            )

            lora_output_path = save_result(
                lora_result,
                str(output_dir / "result_lora_model.jpg"),
                ""
            )

            # 清理LoRA模型
            del lora_trainer
            torch.cuda.empty_cache()

            logger.info("\n对比测试完成!")
            logger.info(f"基础模型结果: {base_output_path}")
            logger.info(f"LoRA模型结果: {lora_output_path}")

        else:
            # 单一测试模式
            logger.info("=" * 60)
            logger.info("开始单一模型测试")
            logger.info("=" * 60)

            predict_config = create_config_for_predict(base_config, args.lora_weight)

            trainer = Trainer(predict_config)
            trainer.setup_predict()

            result = run_prediction(
                trainer, image, prompt,
                args.steps, args.cfg_scale
            )

            model_type = "lora" if args.lora_weight else "base"
            output_path = save_result(
                result,
                str(output_dir / f"result_{model_type}_model.jpg"),
                ""
            )

            logger.info(f"\n测试完成! 结果保存至: {output_path}")

        # 生成测试报告
        report_path = output_dir / "test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Qwen Image Edit 预测测试报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入图像: {args.image}\n")
            f.write(f"提示词文件: {args.prompt}\n")
            f.write(f"推理步数: {args.steps}\n")
            f.write(f"CFG Scale: {args.cfg_scale}\n")
            f.write(f"测试模式: {'对比测试' if args.compare else '单一测试'}\n")
            if args.lora_weight:
                f.write(f"LoRA权重: {args.lora_weight}\n")
            f.write(f"\n提示词内容:\n{prompt}\n")

        logger.info(f"测试报告已保存: {report_path}")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        raise

    logger.info("测试完成!")


if __name__ == "__main__":
    main()

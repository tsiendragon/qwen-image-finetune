#!/bin/bash
# Qwen Image Edit 预测测试脚本

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 测试配置
IMAGE_PATH="/raid/lilong/data/kyc_gen/aligned/control_images/row_fff37bba.jpg"
PROMPT_TEXT="Change Name from \"MAKSIMOV DMITRII\" to \"ALBERT ROBISON\", Change the NIE from \"Y5428731P\" to \"Y0123456B\", change the face from male to female with glasses. Keep the ID card structure, layout, fingerprint, biometric marks, or any other patterns associated to this card type."
# LORA_WEIGHT="/raid/lilong/data/kyc_gen/logs/id_card_qwen_image_lora/checkpoint-1-900/pytorch_lora_weights.safetensors"
# LORA_WEIGHT='/raid/lilong/data/kyc_gen/logs/id_card_qwen_image_lora/checkpoint-0-900/pytorch_lora_weights.safetensors'
IMAGE_PATH='test_prompt_image.png'
IMAGE_PATH='data/face_seg/control_images/060020_3_024801_NONE_28.jpg'
PROMPT_TEXT="change the image from the face to the face segmentation mask"
LORA_WEIGHT='/raid/lilong/data/experiment/qwen-edit-face_seg_lora/checkpoint-89-900/pytorch_lora_weights.safetensors'
LORA_WEIGHT='/raid/lilong/data/experiment/qwen-edit-face_seg_lora_fp4/checkpoint-9-100/pytorch_lora_weights.safetensors'
OUTPUT_DIR="tests/outputs/$(date +%Y%m%d_%H%M%S)"
config_file='configs/face_seg_config.yaml'
config_file='configs/face_seg_fp4_config.yaml'
echo "=== Qwen Image Edit 预测测试 ==="

# 检查文件存在性
[ ! -f "$IMAGE_PATH" ] && echo "错误: 图像文件不存在: $IMAGE_PATH" && exit 1

mkdir -p "$OUTPUT_DIR"
echo "输出目录: $OUTPUT_DIR"

# 运行测试
if [ -f "$LORA_WEIGHT" ]; then
    echo "运行对比测试 (基础模型 vs LoRA模型)"
    python tests/test_predict.py \
        --image "$IMAGE_PATH" \
        --prompt-text "$PROMPT_TEXT" \
        --lora-weight "$LORA_WEIGHT" \
        --output-dir "$OUTPUT_DIR" \
        --config "$config_file" \
        --compare
else
    echo "LoRA权重不存在，仅测试基础模型"
    python tests/test_predict.py \
        --image "$IMAGE_PATH" \
        --prompt-text "$PROMPT_TEXT" \
        --output-dir "$OUTPUT_DIR" \
        --config "$config_file"
fi

echo "=== 测试完成 ==="
echo "结果保存在: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

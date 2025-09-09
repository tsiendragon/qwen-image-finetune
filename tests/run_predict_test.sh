#!/bin/bash
# Qwen Image Edit 预测测试脚本

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 测试配置
IMAGE_PATH="/raid/lilong/data/kyc_gen/aligned/control_images/row_fff37bba.jpg"
PROMPT_TEXT="Change Name from \"MAKSIMOV DMITRII\" to \"ALBERT ROBISON\", Change the NIE from \"Y5428731P\" to \"Y0123456B\", change the face from male to female with glasses. Keep the ID card structure, layout, fingerprint, biometric marks, or any other patterns associated to this card type."
# LORA_WEIGHT="/raid/lilong/data/kyc_gen/logs/id_card_qwen_image_lora/checkpoint-1-900/pytorch_lora_weights.safetensors"
# LORA_WEIGHT='/raid/lilong/data/kyc_gen/logs/id_card_qwen_image_lora/checkpoint-0-900/pytorch_lora_weights.safetensors'
IMAGE_PATH='data/test_person.png'
PROMPT_TEXT="change the image from the face to the face segmentation mask"
LORA_WEIGHT='/raid/lilong/data/experiment/qwen-edit-face_seg_lora/checkpoint-89-900/pytorch_lora_weights.safetensors'
# LORA_WEIGHT='/raid/lilong/data/experiment/qwen-edit-face_seg_lora_fp4/checkpoint-9-100/pytorch_lora_weights.safetensors'
# LORA_WEIGHT='/raid/lilong/data/experiment/qwen-edit-face_seg_lora_fp4/checkpoint-99-1000/pytorch_lora_weights.safetensors'
LORA_WEIGHT='/raid/lilong/data/experiment/qwen-edit-face_seg_lora_fp4-4090/checkpoint-99-500/pytorch_lora_weights.safetensors'
LORA_WEIGHT='/raid/lilong/data/experiment/qwen-edit-face_seg_lora_fp4-4090/checkpoint-79-400/pytorch_lora_weights.safetensors'
LORA_WEIGHT='/data/lilong/kyc_gen/logs/id_card_qwen_image_lora_inpainting/checkpoint-0-100/pytorch_lora_weights.safetensors'
LORA_WEIGHT='/data/lilong/kyc_gen/logs/id_card_qwen_image_lora_inpainting/checkpoint-1-2100/pytorch_lora_weights.safetensors'
LORA_WEIGHT='/data/lilong/kyc_gen/logs/id_card_qwen_image_lora_inpainting/lora_test3/v3/checkpoint-3-3700/model.safetensors'
LORA_WEIGHT='/data/lilong/kyc_gen/logs/id_card_qwen_image_lora_inpainting/lora_test3/v3/checkpoint-4-4600/model.safetensors'
# LORA_WEIGHT='/data/lilong/kyc_gen/logs/id_card_qwen_image_lora_inpainting/lora_test3/v3/checkpoint-0-900/model.safetensors'
# LORA_WEIGHT='/data/lilong/kyc_gen/logs/id_card_qwen_image_lora_inpainting/lora_test3/v3/checkpoint-1-2100/model.safetensors'
# LORA_WEIGHT='/data/lilong/kyc_gen/logs/id_card_qwen_image_lora_inpainting/checkpoint-2-2900/pytorch_lora_weights.safetensors'
# LORA_WEIGHT='/data/lilong/kyc_gen/logs/id_card_qwen_image_lora_inpainting/checkpoint-2-3000/pytorch_lora_weights.safetensors'
IMAGE_PATH='/data/lilong/ktp/ktp/dataset1/control_images/OCR_KTP_CHECK-f40e57645e144433_20190529023443445_2094134257_original_sample_040.jpg'

PROMPT_TEXT='/data/lilong/ktp/ktp/dataset1/training_images/OCR_KTP_CHECK-f40e57645e144433_20190529023443445_2094134257_original_sample_040.txt'
PROMPT_TEXT='/home/lilong/repos/qwen-image-finetune/data/test_ic_prompt.txt'
IMAGE_PATH='/home/lilong/repos/qwen-image-finetune/data/test_ic.png'
IMAGE_PATH='/data/lilong/ktp/test_dataset/control_images/test_047_OCR_KTP_CHECK-0feb983e5a3f2a61_20190529025939148_2904398016.jpg'
PROMPT_TEXT='/data/lilong/ktp/test_dataset/training_images/test_047_OCR_KTP_CHECK-0feb983e5a3f2a61_20190529025939148_2904398016.txt'
IMAGE_PATH='/data/lilong/ktp/test_dataset/control_images/test_009_OCR_KTP_CHECK-6541f4a1bc55408d_20190529181925938_9190064570.jpg'
PROMPT_TEXT='/data/lilong/ktp/test_dataset/training_images/test_009_OCR_KTP_CHECK-6541f4a1bc55408d_20190529181925938_9190064570.txt'
PROMPT_TEXT='/home/lilong/repos/qwen-image-finetune/tests/outputs/20250905_025133/prompt.txt'

LORA_WEIGHT='/raid/lilong/data/experiment/flux-kontext-face_seg_lora_fp16/face_segmentation_lora/v0/checkpoint-19-200/model.safetensors'
LORA_WEIGHT='/raid/lilong/data/experiment/flux-kontext-face_seg_lora_fp16/face_segmentation_lora/v1/checkpoint-99-300/model.safetensors'
IMAGE_PATH='/mnt/nas/public2/lilong/repos/qwen-image-finetune/data/test_person.png'
PROMPT_TEXT='/mnt/nas/public2/lilong/repos/qwen-image-finetune/data/test_prompt.txt'

cfg_scale=4.5
OUTPUT_DIR="tests/outputs/$(date +%Y%m%d_%H%M%S)"
config_file='configs/face_seg_config.yaml'
config_file='configs/face_seg_fp4_4090.yaml'
config_file='configs/qwen_image_edit_config_inpainting.yaml'
config_file='configs/face_seg_flux_kontext_fp16.yaml'
# config_file='configs/face_seg_fp4_config.yaml'
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
        --lora-weight "$LORA_WEIGHT" \
        --output-dir "$OUTPUT_DIR" \
        --config "$config_file" \
        --cfg-scale $cfg_scale \
        --prompt $PROMPT_TEXT \
        --compare
else
    echo "LoRA权重不存在，仅测试基础模型"
    python tests/test_predict.py \
        --image "$IMAGE_PATH" \
        --prompt "$PROMPT_TEXT" \
        --output-dir "$OUTPUT_DIR" \
        --config "$config_file" \
        --cfg-scale $cfg_scale
fi

# --prompt "$PROMPT_TEXT" \

echo "=== 测试完成 ==="
echo "结果保存在: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
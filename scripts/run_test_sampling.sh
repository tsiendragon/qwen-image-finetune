# #!/bin/bash
# # Test sampling script for multi-resolution hair segmentation
# # Usage: ./scripts/run_test_sampling.sh

# set -e

# # Default configuration
# CONFIG="tests/test_configs/test_example_fluxkontext_multiresolution.yaml"
# CONFIG="tests/test_configs/test_example_fluxkontext_resize.yaml"
# DATASET_NAME="TsienDragon/figaro_hair_segmentation_1k"
# SPLIT="test"
# NUM_SAMPLES=50  # Process first 50 samples, set to empty for all
# STEPS=20
# CFG_SCALE=1.0
# DEVICE="cuda:1"
# RESOLUTION="512x512"

# # Optional: LoRA weight path
# # Uncomment and set the path if you have a trained LoRA checkpoint
# # LORA_WEIGHT="/path/to/checkpoint/model.safetensors"
# LORA_WEIGHT="/tmp/image_edit_lora/figaroHairSegFluxKontextFp4MultiRes/v0/checkpoint-last-5-2585-last/pytorch_lora_weights.safetensors"
# # LORA_WEIGHT='/tmp/image_edit_lora/figaroHairSegFluxKontextFp4MultiRes/v0/checkpoint-1-800/pytorch_lora_weights.safetensors'
# # LORA_WEIGHT='/tmp/image_edit_lora/figaroHairSegFluxKontextFp4MultiRes/v0/checkpoint-3-2000/pytorch_lora_weights.safetensors'
# LORA_WEIGHT='/tmp/image_edit_lora/figaroHairSegFluxKontextFp4MultiRes/v1/checkpoint-3-200/pytorch_lora_weights.safetensors'
# LORA_WEIGHT='/tmp/image_edit_lora/figaroHairSegFluxKontextFp4MultiRes/v1/checkpoint-12-800/pytorch_lora_weights.safetensors'
# LORA_WEIGHT='/tmp/image_edit_lora/figaroHairSegFluxKontextResize/v0/checkpoint-last-20-855-last/pytorch_lora_weights.safetensors'

# # Output folder with timestamp
# TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# SAVE_FOLDER="/tmp/flux_kontext_sampling_multiresolution/${TIMESTAMP}"

# echo "=================================================="
# echo "Multi-Resolution Hair Segmentation Sampling Test"
# echo "=================================================="
# echo "Config: ${CONFIG}"
# echo "Dataset: ${DATASET_NAME} (${SPLIT})"
# echo "Output: ${SAVE_FOLDER}"
# echo "Samples: ${NUM_SAMPLES:-all}"
# echo "Steps: ${STEPS}"
# echo "CFG Scale: ${CFG_SCALE}"
# echo "Device: ${DEVICE}"
# echo "Resolution: ${RESOLUTION}"
# if [ -n "${LORA_WEIGHT}" ]; then
#     echo "LoRA Weight: ${LORA_WEIGHT}"
# fi
# echo "=================================================="

# # Build command
# CMD="python scripts/test_sampling_multiresolution.py \
#     --config ${CONFIG} \
#     --dataset-name ${DATASET_NAME} \
#     --split ${SPLIT} \
#     --save-folder ${SAVE_FOLDER} \
#     --steps ${STEPS} \
#     --cfg-scale ${CFG_SCALE} \
#     --device ${DEVICE} \
#     --resolution ${RESOLUTION}"

# # Add optional arguments
# if [ -n "${NUM_SAMPLES}" ]; then
#     CMD="${CMD} --num-samples ${NUM_SAMPLES}"
# fi

# if [ -n "${LORA_WEIGHT}" ] && [ -f "${LORA_WEIGHT}" ]; then
#     CMD="${CMD} --lora-weight ${LORA_WEIGHT}"
# fi

# # Run the script
# echo "Running: ${CMD}"
# echo ""
# eval ${CMD}

# echo ""
# echo "=================================================="
# echo "Sampling complete!"
# echo "Results saved to: ${SAVE_FOLDER}"
# echo "=================================================="

# # Show output structure
# echo ""
# echo "Output structure:"
# ls -lh ${SAVE_FOLDER}/

# echo ""
# echo "Sample counts:"
# for dir in images prompts; do
#     if [ -d "${SAVE_FOLDER}/${dir}" ]; then
#         count=$(ls -1 ${SAVE_FOLDER}/${dir} | wc -l)
#         echo "  ${dir}: ${count} files"
#     fi
# done

# python scripts/test_sampling_multiresolution.py \
#   --config tests/test_configs/test_example_fluxkontext_fp16_faceseg_multiresolution.yaml \
#   --dataset-name TsienDragon/face_segmentation_20 \
#   --split train \
#   --num-samples 10 \
#   --steps 20 \
#   --device cuda:1 \
#   --save-folder /tmp/flux_kontext_faceseg_results_flux_image_edit_lora \
#   --lora-weight /home/lilong/.cache/huggingface/hub/models--TsienDragon--flux-kontext-face-segmentation/snapshots/a8406e69a343de1b92b902ce5812fe7d7b44153c/pytorch_lora_weights.safetensors

#/home/lilong/.cache/huggingface/hub/models--TsienDragon--qwen-image-edit-lora-face-segmentation/snapshots/847ede43e3aaa4174e1ede1435f513acd7d0456b/pytorch_lora_weights.safetensors
#/tmp/image_edit_lora/faceSegFluxKontextFp16MultiRes/v0/checkpoint-599-1800/pytorch_lora_weights.safetensors
#/tmp/image_edit_lora/faceSegFluxKontextFp16MultiRes/v0/checkpoint-last-1227-3683-last/pytorch_lora_weights.safetensors

lora_weight='/tmp/image_edit_lora/figaroHairSegFluxKontextFp16MultiRes/v0/checkpoint-last-37-6215-last/pytorch_lora_weights.safetensors'
config='tests/test_configs/test_example_fluxkontext_multiresolution.yaml'
python scripts/test_sampling_multiresolution.py \
  --config $config \
  --dataset-name TsienDragon/figaro_hair_segmentation_1k \
  --split test \
  --num-samples 20 \
  --steps 20 \
  --device cuda:1 \
  --save-folder /tmp/flux_kontext_figaro_hair_segmentation \
  --lora-weight $lora_weight

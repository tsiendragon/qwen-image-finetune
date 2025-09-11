config_file='configs/qwen_image_edit_config.yaml'
config_file='configs/qwen_image_edit_config_r2.yaml'
config_file='configs/face_seg_config.yaml'
config_file='configs/face_seg_fp4_config.yaml'
config_file='configs/face_seg_fp4_4090.yaml'
config_file='configs/qwen_image_edit_config_inpainting.yaml'
config_file='configs/qwen_image_edit_config_inpainting_edit_mask.yaml'
config_file='configs/face_seg_flux_kontext_fp16.yaml'
config_file='configs/face_seg_flux_kontext_fp16_prodigy.yaml'
# config_file='configs/face_seg_flux_kontext_fp8.yaml'
config_file='configs/face_seg_flux_kontext_fp4.yaml'
echo "Used config file: $config_file"

# cache data (using GPU1)
# echo "Caching data..."
# CUDA_VISIBLE_DEVICES=0,1
# python3 -m src.main --config $config_file --cache

# use accelerate to train (using GPU1)
# echo "Using accelerate to train..."
# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config_file

# NCCL_P2P_DISABLE only for RTX4090, not for A100

# python3 -m src.main --config $config_file --cache
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config_file

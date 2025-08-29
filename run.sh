config_file='configs/qwen_image_edit_config.yaml'
config_file='configs/qwen_image_edit_config_r2.yaml'
config_file='configs/face_seg_config.yaml'
config_file='configs/face_seg_fp4_config.yaml'

echo "Used config file: $config_file"

# cache data (using GPU1)
# echo "Caching data..."
# CUDA_VISIBLE_DEVICES=0 python3 -m src.main --config $config_file --cache

# use accelerate to train (using GPU1)
# echo "Using accelerate to train..."
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config_file


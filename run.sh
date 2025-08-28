config_file='configs/qwen_image_edit_config.yaml'
config_file='configs/qwen_image_edit_config_r2.yaml'
config_file='configs/face_seg_config.yaml'

echo "配置文件: $config_file"

# 缓存数据（使用GPU1）
# echo "正在缓存数据..."
# CUDA_VISIBLE_DEVICES=1 python3 -m src.main --config $config_file --cache

# 使用accelerate进行训练（GPU1）
# echo "开始使用accelerate进行训练..."
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config_file


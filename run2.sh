config='configs/qwen_image_edit_config_inpainting_v2.yaml'
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config

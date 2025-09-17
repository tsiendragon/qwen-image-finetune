config='configs/ktp_flux_kontext_fp16yaml'
config='tests/test_configs/test_example_fluxkontext_fp16.yaml'
config='tests/test_configs/test_example_fluxkontext_fp16_character_composition.yaml'
config='tests/test_configs/test_example_qwen_image_edit_fp16_character_composition.yaml'
# python3 -m src.main --config $config --cache
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config

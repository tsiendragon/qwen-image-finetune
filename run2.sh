config='configs/ktp_flux_kontext_fp16yaml'
config='tests/test_configs/test_example_fluxkontext_fp16.yaml'
python3 -m src.main --config $config --cache
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config

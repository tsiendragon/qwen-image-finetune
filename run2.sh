config='configs/ktp_flux_kontext_fp16yaml'
config='tests/test_configs/test_example_fluxkontext_fp16.yaml'
config='tests/test_configs/test_example_fluxkontext_fp16_character_composition.yaml'
config='/mnt/nas/public2/lilong/repos/qwen-image-finetune/tests/test_configs/test_example_qwen_image_edit_fp16_character_composition.yaml'
# config='tests/test_configs/test_example_qwen_image_edit_fp16.yaml'
# python3 -m src.main --config $config --cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# 仅排查时用（会让错尽快抛出）：
# export TORCH_NCCL_BLOCKING_WAIT=1
# 看到过 torch/_compile 参与就禁用 Dynamo：
# export TORCHDYNAMO_DISABLE=1
# 对 A100/H100 常能降低抢占抖动：
export CUDA_DEVICE_MAX_CONNECTIONS=1
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config


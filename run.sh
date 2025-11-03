# config_file='tests/test_configs/test_example_fluxkontext_fp4.yaml'
# config_file='tests/test_configs/test_example_fluxkontext_fp16.yaml'
# config_file='tests/test_configs/test_example_qwen_image_edit_fp16.yaml'
# config_file='tests/test_configs/test_example_fluxkontext_fp16.yaml'
# config_file='tests/test_configs/test_example_fluxkontext_fp16_faceseg_multiresolution.yaml'
# config_file='tests/test_configs/test_example_fluxkontext_multiresolution.yaml'
config_file='../tests/test_configs/test_example_fluxkontext_fp16.yaml'
config_file='../tests/test_configs/test_example_qwen_image_edit_fp16.yaml'
config_file='../tests/test_configs/test_example_qwen_image_edit_plus_fp16.yaml'
# config_file='../tests/test_configs/test_example_qwen_image_edit_plus_fp4.yaml'
# config_file='../tests/test_configs/test_example_qwen_image_edit_plus_fp4.yaml'
config_file='../tests/test_configs/test_dreamomni2_fp16.yaml'
echo "Used config file: $config_file"

cd src/
# python3 -m qflux.main --config $config_file --cache

# # NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
# # NCCL_P2P_DISABLE only for RTX4090, not for A100
# export WANDB_MODE=offline && \
# export NCCL_BLOCKING_WAIT=1 && \
# export NCCL_DEBUG=WARN && \
CUDA_VISIBLE_DEVICES=2 \
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  -m qflux.main --config $config_file

# upload test resources to huggingface
# python scripts/upload_test_resources.py --resources-dir test_resources_organized/ --repo-id TsienDragon/qwen-image-finetune-test-resources

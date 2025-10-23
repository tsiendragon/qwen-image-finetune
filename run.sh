# config_file='tests/test_configs/test_example_fluxkontext_fp4.yaml'
# config_file='tests/test_configs/test_example_fluxkontext_fp16.yaml'
# config_file='tests/test_configs/test_example_qwen_image_edit_fp16.yaml'
# config_file='tests/test_configs/test_example_fluxkontext_fp16.yaml'
# config_file='tests/test_configs/test_example_fluxkontext_fp16_faceseg_multiresolution.yaml'
# config_file='tests/test_configs/test_example_fluxkontext_multiresolution.yaml'
# echo "Used config file: $config_file"

# # cache data (using GPU1)
# # echo "Caching data..."

# # use accelerate to train (using GPU1)
# # echo "Using accelerate to train..."
# # NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
# # CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config_file

# # NCCL_P2P_DISABLE only for RTX4090, not for A100

# python3 -m src.main --config $config_file --cache

# CUDA_VISIBLE_DEVICES=0,1,3 \
# accelerate launch \
#   --num_processes 3 \
#   --mixed_precision bf16 \
#   -m src.main --config $config_file

# # CUDA_VISIBLE_DEVICES=2 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config_file

# upload test resources to huggingface
python scripts/upload_test_resources.py --resources-dir test_resources_organized/ --repo-id TsienDragon/qwen-image-finetune-test-resources

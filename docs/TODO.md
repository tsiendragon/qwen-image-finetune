# TODO List

## Completed Features
- [x] Flash attention speedup
- [x] DDP training on RTX 4090
- [x] Edit Mask Loss support (v1.4.0)
- [x] Dataset: cache hash calculated
- [x] Multi dataset_path supported
- [x] Resume training
- [x] Fix LoRA training without cache function
- [x] Non-caching training mode with memory efficiency
- [x] TensorBoard log dir add version when resume training
- [x] Flux Kontext LoRA (v1.5.0)
- [x] HuggingFace Dataset support (v1.5.2)
- [x] Fix additional control image naming rule documentation (v1.5.3)
- [x] Training step dtype conversion simplification
- [x] Validation Sampling During Training (v1.6.0)
- [x] HuggingFace compatible LoRA format (v1.6.0)
- [x] Automatic LoRA weight upload functionality (v1.6.0)
- [x] Save final checkpoint on training completion (v1.6.0)
- [x] FSDP training support (v2.1.0)
- [x] Multi-control image training with multiple input images (v2.0.0)
- [x] Enhanced HuggingFace dataset upload for multi-control scenarios (v2.1.0)
- [x] CSV数据格式支持 - 支持基于CSV元数据文件的数据集管理 (v2.2.0)
- [x] Qwen-Image-Edit-Plus (2509) support - Complete support for enhanced model architecture (v2.3.0)

## Planned Features
- [ ] Customized pipeline based on diffusers pipeline with multi-control compatibility
- [ ] Sampling for training monitoring with TensorBoard
- [ ] Online quantization for training and inference
- [ ] Release character-composition dataset on HuggingFace
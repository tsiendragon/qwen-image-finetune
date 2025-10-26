# TODO Summary

Track ongoing development, documentation, and operations tasks. Update completion states whenever progress changes.

## In Progress

- [ ] FSDP优化实现，解决FluxKontext BF16训练的内存溢出问题
- [ ] FSDP与LoRA兼容性测试与适配
- [ ] FSDP checkpoint保存与加载功能
- [ ] 对比BF16 DDP、FP4 DDP和BF16 FSDP三种配置的训练效率与性能
- [ ] Customized pipeline based on diffusers pipeline with multi-control compatibility
- [ ] Sampling for training monitoring with TensorBoard
- [ ] Online quantization for training and inference
- [ ] Release character-composition dataset on HuggingFace

## Completed

- [x] Flash attention speedup
- [x] DDP training on RTX 4090
- [x] Edit Mask Loss support (v1.4.0)
- [x] Dataset cache hash calculated
- [x] Multi dataset_path supported
- [x] Resume training
- [x] Fix LoRA training without cache function
- [x] Non-caching training mode with memory efficiency
- [x] TensorBoard log dir add version when resume training
- [x] Flux Kontext LoRA (v1.5.0)
- [x] HuggingFace Dataset support (v1.5.2)
- [x] Fix additional control image naming rule documentation (v1.5.3)
- [x] Training step dtype conversion simplification
- [x] Validation sampling during training (v1.6.0)
- [x] HuggingFace compatible LoRA format (v1.6.0)
- [x] Automatic LoRA weight upload functionality (v1.6.0)
- [x] Save final checkpoint on training completion (v1.6.0)
- [x] FSDP training support (v2.1.0)
- [x] Multi-control image training with multiple input images (v2.0.0)
- [x] Enhanced HuggingFace dataset upload for multi-control scenarios (v2.1.0)
- [x] CSV metadata dataset support (v2.2.0)
- [x] Qwen-Image-Edit-Plus (2509) architecture integration (v2.3.0)
- [x] Dynamic shape support with pixel-based resizing for Plus version (v2.4.0)
- [x] Documentation improvements and guide updates (v2.4.1)
- [x] Multi-resolution mixed training (v3.0.0)
- [x] Test resources management with HuggingFace Hub integration (v3.0.1)
- [x] E2E tests for Qwen-Image-Edit and Qwen-Image-Edit-Plus sampling (v3.0.1)
- [x] Pre-commit hooks configuration (v3.0.1)

## Notes

- Keep entries concise: include scope + version/target if applicable.
- Close or archive completed tasks when they are fully delivered and deployed.

# Changelog Overview

This document summarizes the key highlights for each released version. Detailed change descriptions live in the per-version files under this folder.

## Version Index

- [3.3.9](./v3.3.9.md) — Fixed Flux2LoraTrainer I2I prediction preprocessing mismatch with Flux2Pipeline.
- [3.3.8](./v3.3.8.md) — Fixed Flux2LoraTrainer system message mismatch causing prompt_embeds differences.
- [3.3.7](./v3.3.7.md) — Fixed configuration validation errors in flux2_i2i_fp16_config.yaml.
- [3.3.6](./v3.3.6.md) — Memory-optimized pipeline loading to reduce CUDA OOM errors in comparison script.
- [3.3.5](./v3.3.5.md) — Optimized cross-device comparison to automatically use GPU for tensor comparisons.
- [3.3.4](./v3.3.4.md) — Added separate device support for Pipeline and Trainer in comparison script.
- [3.3.3](./v3.3.3.md) — Fixed indentation error in Flux2LoraTrainer.decode_vae_latent() method.
- [3.3.2](./v3.3.2.md) — Added CUDA device selection support for comparison test script.
- [3.3.1](./v3.3.1.md) — Fixed Flux2LoraTrainer tokenizer parameter issue for T2I prediction.
- [3.3.0](./v3.3.0.md) — DreamOmni2 trainer support with cumulative offset positioning and VLM prompt optimization.
- [3.2.1](./v3.2.1.md) — Bug fixes: scheduler state isolation and directory cleanup improvements.
- [3.2.0](./v3.2.0.md) — Multi-logger support (TensorBoard/wandb/SwanLab) with unified LoggerManager interface.
- [3.1.0](./v3.1.0.md) — Validation sampling during training with TensorBoard visualization.
- [3.0.2](./v3.0.2.md) — Fixed FSDP LoRA checkpoint saving issues and performance benchmarking.
- [3.0.1](./v3.0.1.md) — Bug fixes, test infrastructure improvements, and dependency updates.
- [3.0.0](./v3.0.0.md) — Multi-resolution mixed training pipeline, dynamic resolution candidate selection.
- [2.4.1](./v2.4.1.md) — Documentation polish and dataset guide updates.
- [2.4.0](./v2.4.0.md) — Dynamic shape support for Qwen-Image-Edit-Plus.
- [2.3.0](./v2.3.0.md) — Full Qwen-Image-Edit-Plus architecture integration.
- [2.2.0](./v2.2.0.md) — CSV metadata dataset support.
- [2.1.0](./v2.1.0.md) — FSDP training path and distributed optimizations.
- [2.0.0](./v2.0.0.md) — Major refactor with revised trainer stack.
- [1.6.0](./v1.6.0.md) — Training workflow improvements and LoRA upload automation.
- Older versions are tracked in the [`legacy`](./legacy/) subfolder when applicable.

## Unreleased

- Track pending features and bugfixes that target the next release.
- Update this section when pull requests land without an immediate release tag.

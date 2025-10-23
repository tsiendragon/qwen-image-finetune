# Specification Documentation

This directory contains detailed technical specifications for the codebase, organized by module.

## Directory Structure

- **`data/`** — Data loading, preprocessing, and caching system specifications
  - [`cache_system.md`](./data/cache_system.md) — Embedding cache architecture and tensor shapes
- **`losses/`** — Loss function implementations and design rationale
  - [`attention_mask_mse_loss.md`](./losses/attention_mask_mse_loss.md) — Channel-invariant token loss for multi-resolution training
  - [`image_edit_mask_loss.md`](./losses/image_edit_mask_loss.md) — Mask-weighted loss for focused editing regions
- **`models/`** — Model architecture specifications and custom implementations
  - [`flux_kontext.md`](./models/flux_kontext.md) — FLUX Kontext transformer architecture and LoRA integration
  - [`qwen_image_model.md`](./models/qwen_image_model.md) — Qwen-Image-Edit base model architecture
  - [`qwen_image_edit_plus.md`](./models/qwen_image_edit_plus.md) — Enhanced Qwen-Image-Edit-Plus (2509) architecture
- **`trainer/`** — Training loop and trainer implementations
  - [`trainer_architecture.md`](./trainer/trainer_architecture.md) — Base trainer design and extension patterns
- **`pipeline/`** — Inference pipeline specifications (future)
- **`utils/`** — Utility module specifications (future)

## Maintenance Guidelines

- Update corresponding spec files when merging code changes that affect module interfaces or behavior.
- Include tensor shapes, configuration schemas, and API contracts in each spec.
- Reference spec files in pull requests when introducing new components.

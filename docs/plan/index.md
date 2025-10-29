# Development Plans Overview

This index summarizes active and upcoming initiatives. Each entry links to a detailed plan covering scope, milestones, and open questions.

| Plan | Description | Status | Details |
| --- | --- | --- | --- |
| Multi-Logger Integration | Add support for SwanLab alongside TensorBoard and W&B | Planning | [multi_logger_integration.md](./multi_logger_integration.md) |
| Validation Sampling During Training | Add validation sampling with TensorBoard visualization | Planning | [validation_sampling_during_training.md](./validation_sampling_during_training.md) |
| FSDP Memory Optimization | Implement FSDP to reduce memory usage for FluxKontext BF16 training | In progress | [fsdp_memory_optimization.md](./fsdp_memory_optimization.md) |
| Training Performance Benchmarks | Track memory usage and throughput across different configurations | In progress | [training_performance_benchmarks.md](./training_performance_benchmarks.md) |
| Flux Kontext Enhancements | Extend Kontext-based training, sampling, and deployment workflows | In progress | [flux_kontext_implement.md](./flux_kontext_implement.md) |
| Per-Sample RoPE Experiments | Evaluate per-sample rotary positional encoding improvements | Exploring | [flux-per-sample-rope-implementation.md](./flux-per-sample-rope-implementation.md) |
| Multi-Control Training | Support multiple control inputs with consistent scheduling | Completed | [multi_control.md](./multi_control.md) |
| Multi-Resolution Padding | Validate padding strategies for dynamic resolution batches | Completed | [multi-resolution-padding-mask-training.plan.md](./multi-resolution-padding-mask-training.plan.md) |
| HF Dataset Integration | Build complete HF dataset ingestion and sharing pipeline | Completed | [v1.5.2_add_huggingface_dataset.md](./v1.5.2_add_huggingface_dataset.md) |
| Sampling During Training | Add validation sampling hooks for live monitoring | Completed | [v1.6.0:sampling_during_train.md](./v1.6.0:sampling_during_train.md) |
| Dynamic Resolution Roadmap | Track v2.4.0+ dynamic resolution feature rollout | Completed | [v2.4.0_dynamic_resolution.md](./v2.4.0_dynamic_resolution.md) |

## Process Guidance

- Record new initiatives here once they pass initial review.
- Keep status in sync with the latest execution state (e.g., Exploring → In progress → Completed).
- Move fully delivered plans to a "Completed" mark but keep them listed for historical reference.

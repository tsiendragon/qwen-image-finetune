# FSDP Memory Optimization for FluxKontext Training

**Status**: In Progress
**Target Version**: TBD
**Last Updated**: 2025-10-26

## 背景与问题

当前在4×RTX 4090环境下训练FluxKontext模型时，使用BF16精度的DDP模式会导致内存溢出(OOM)问题。RTX 4090显卡有24GB显存，而BF16 DDP训练需要超过24GB的内存，无法直接运行。这限制了我们在有限硬件资源下的训练能力，特别是对于较大模型或批量大小。

## 目标

实现PyTorch的Fully Sharded Data Parallel (FSDP)以优化内存使用，使FluxKontext模型能够在4×RTX 4090环境下使用BF16精度训练而不发生OOM。

## 技术方案

### 1. FSDP基本原理

FSDP通过以下机制减少内存使用：
- 模型参数分片：将模型参数分布到多个GPU上
- 梯度分片：每个GPU只保存自己负责部分的梯度
- 优化器状态分片：优化器状态也分布在多个设备上
- 按需通信：仅在前向/反向传播需要时进行通信

### 2. 实现步骤

1. **基础配置调整**
   - 修改训练配置以支持FSDP选项
   - 添加FSDP特定参数（分片策略、通信优化等）

2. **模型封装**
   - 识别合适的模型层级进行分片
   - 实现FSDP封装逻辑，特别关注Transformer层

3. **内存优化策略**
   - 实现混合精度训练与FSDP结合
   - 配置CPU卸载选项用于极限场景
   - 优化激活检查点(activation checkpointing)策略

4. **通信优化**
   - 实现梯度累积与FSDP结合
   - 优化all-gather和reduce-scatter操作

### 3. 配置参数设计

```bash
# FSDP 配置
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  --use_fsdp \
  --fsdp_sharding_strategy 1 \  # 1 = FULL_SHARD
  -m qflux.main --config $config_file
```

```yaml
# 模型配置
model:
  pretrained_model_name_or_path: black-forest-labs/FLUX.1-Kontext-dev
  quantize: false
```

## 预期效果

1. **内存使用降低**：
   - 预计可减少40-60%的GPU内存使用
   - 使BF16训练在4×RTX 4090上可行
   - 初步测试结果：FP16 FSDP配置下每GPU内存占用约10GB，每GPU分配约2.99B参数

2. **性能影响**：
   - 通信开销增加，训练速度可能降低5-15%
   - 需要调整batch size找到最佳平衡点
   - 初步测试结果：FP16 FSDP在4×RTX 4090上达到1.7 FPS

3. **扩展性提升**：
   - 支持更大模型规模训练
   - 为未来更复杂架构提供基础

## 测试计划

1. **基准测试**：
   - 测量相同配置下FSDP vs DDP的内存使用
   - 比较训练吞吐量和迭代时间
   - 对比BF16 DDP、FP4 DDP和BF16 FSDP三种配置的训练效率、内存占用和收敛性能
   - 初步测试结果：
     - BF16 DDP: 内存需求超过24GB，在RTX 4090上直接OOM
     - FP4 DDP: 每GPU内存占用约25GB，吞吐量约0.4 FPS
     - BF16 FSDP: 每GPU内存占用约10GB，吞吐量约1.7 FPS

2. **功能验证**：
   - 验证模型收敛性是否受影响
   - 确认checkpoint保存/加载功能正常

3. **极限测试**：
   - 测试最大可用batch size
   - 验证CPU卸载功能在低内存场景下的效果

## 里程碑

1. **阶段一：基础实现**
   - FSDP配置框架集成
   - 基本功能测试

2. **阶段二：优化与调优**
   - 内存使用优化
   - 性能调优
   - 完整测试套件

3. **阶段三：文档与示例**
   - 更新用户指南
   - 提供最佳实践配置示例

## 风险与缓解

| 风险 | 影响 | 缓解措施 | 状态 |
|------|------|---------|------|
| 通信瓶颈导致训练速度显著下降 | 中 | 实现梯度累积和通信优化，调整分片粒度 | 进行中 |
| 与现有功能冲突（如LoRA） | 高 | 专门测试FSDP与LoRA的兼容性，必要时实现适配层 | 进行中 |
| Checkpoint兼容性问题 | 中 | 实现专用的保存/加载逻辑，支持FSDP模型转换为常规模型 | 已解决(v3.0.2) |
| 调试难度增加 | 低 | 添加详细日志和内存分析工具 | 进行中 |

## 后续工作

- 探索ZeRO-3与FSDP结合的可能性
- 研究更细粒度的分片策略
- 添加自动内存优化建议功能
- 基于BF16 DDP、FP4 DDP和BF16 FSDP三种策略的对比实验结果，形成最佳实践指南

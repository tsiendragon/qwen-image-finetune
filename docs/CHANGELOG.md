# 更新日志

所有值得注意的项目变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 新增
- 待添加...

### 改进
- 待添加...

### 修复
- 待添加...

---

## [1.5.1] - 2025-01-11

### 改进
- **优化器选择文档**: 在 `docs/training.md` 中新增优化器选择章节
  - 支持四种优化器：AdamW（默认）、8-bit Adam（内存高效）、Prodigy（自动学习率）、SGD（经典）
  - 提供详细的 YAML 配置示例和使用场景说明
  - 添加快速参考表格，对比各优化器的内存占用、自动学习率功能和最佳使用场景
  - 在 README.md 技术支持部分添加优化器选择的快速链接

### 修复
- 修正了 bitsandbytes 优化器的类路径（从 `bnb.optim.Adam8bit` 改为 `bitsandbytes.optim.Adam8bit`）

---

## [1.5.0] - 2025-01-10

### 新增
- **FLUX Kontext LoRA 训练支持**: 完整实现了 FLUX Kontext 模型的 LoRA 微调功能
  - 支持三种精度级别：FP16（最高质量）、FP8（平衡性能）、FP4（最大效率）
  - 提供预配置的训练配置文件：`face_seg_flux_kontext_fp16.yaml`、`face_seg_flux_kontext_fp8.yaml`、`face_seg_flux_kontext_fp4.yaml`
  - 实现了 `FluxKontextLoraTrainer` 类，继承自 `BaseTrainer` 确保接口一致性
  - 支持双文本编码器架构（CLIP + T5），优化多设备分配策略
  - 完整的缓存系统支持，包括 VAE、CLIP 和 T5 编码器的独立设备配置

### 改进
- **文档完善**: 在 `docs/training.md` 中新增详细的 FLUX Kontext 训练指导章节
  - 详细的精度对比表格，包含质量、训练速度、显存需求和使用场景
  - 完整的训练工作流程和多GPU训练配置
  - 设备分配策略和内存优化建议
  - FLUX Kontext 推理代码示例和最佳实践
- **框架描述更新**: 更新 README.md 以反映双模型架构支持
  - 强调对 Qwen-Image-Edit 和 FLUX Kontext 的完整支持
  - 突出多精度训练能力（FP16/FP8/FP4）
  - 更新技术支持部分，添加 FLUX Kontext 训练指南链接

### 技术细节
- FLUX Kontext 模型架构：基于 Transformer 的扩散模型，支持图像和文本的联合理解
- 精度性能对比：FP16（参考质量）、FP8（95%质量，1.5x速度）、FP4（85%质量，2.5x速度）
- 显存需求优化：FP16（24GB训练/12GB推理）→ FP8（18GB/8GB）→ FP4（12GB/5GB）
- 支持的预训练模型：
  - `black-forest-labs/FLUX.1-Kontext-dev` (FP16)
  - `camenduru/flux1-kontext-dev_fp8_e4m3fn_diffusers` (FP8)
  - `eramth/flux-kontext-4bit-fp4` (FP4)

---

## [1.4.0] - 2025-01-03

### 新增
- **Edit Mask Loss 支持**: 实现了基于编辑掩码的损失函数，提升图像编辑训练效果
  - 新增 `MaskEditLoss` 类，支持对编辑区域和背景区域的差异化权重
  - 实现 `map_mask_to_latent` 函数，将图像空间掩码映射到潜在空间序列权重
  - 支持可配置的前景权重 (`foreground_weight`) 和背景权重 (`background_weight`)
  - 完全向后兼容，现有训练管道在 `mask_loss: false` 时保持不变

### 改进
- **数据集支持**: 扩展数据集结构以支持可选的编辑掩码文件 (`*_mask.png`)
- **配置系统**: 新增 `loss.mask_loss`、`loss.foreground_weight`、`loss.background_weight` 配置选项
- **训练效果**: 通过聚焦编辑区域的损失计算，提升模型在关键编辑区域的收敛速度和质量

### 技术细节
- 掩码处理管道：图像空间 → VAE下采样 → 2×2补丁合并 → 潜在空间序列对齐
- 双重损失计算：原始损失 + 掩码加权损失的组合
- 支持时间步权重与掩码权重的联合应用
- 详细的实现文档：`docs/prd/image_edit_mask_loss.md`

---

## [0.2.0] - 2025-08-27

### 新增
- **QwenImageEditTrainer.predict() 方法**: 实现了完整的图像生成推理功能
  - 支持单张或批量图像处理
  - 完整的 CFG (Classifier-Free Guidance) 支持，包括负面提示
  - 多设备 GPU 分布式推理支持
  - 遵循原始 QwenImageEditPipeline 的完整推理逻辑
  - 支持可配置的推理步数和引导强度
  - 包含完整的去噪循环和时间步调度
  - 输出高质量的 RGB 格式图像

### 改进
- **推理性能**: 新的 predict 方法现在能够正确生成高质量的图像样本
- **设备管理**: 优化了多GPU推理时的设备分配策略
- **内存效率**: 实现了 transformer 缓存上下文以提升推理效率

### 技术细节
- predict 方法完全重写，确保与 QwenImageEditPipeline 的行为一致
- 实现了正确的潜在向量处理和图像解码流程
- 添加了完整的批量处理支持
- 集成了进度条显示以跟踪推理进度
![alt text](docs/images/image.png)
---

## 历史版本

### [0.1.0] - 2025-08-27
- 初始版本发布
- 基础训练功能实现
- 缓存系统支持


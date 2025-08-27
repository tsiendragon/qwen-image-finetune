# 更新日志

所有值得注意的项目变更都会记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

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

### [0.1.0] - 2024-XX-XX
- 初始版本发布
- 基础训练功能实现
- 缓存系统支持


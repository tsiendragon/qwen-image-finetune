# Qwen Image Finetune Documentation

Welcome to the comprehensive documentation for Qwen Image Finetune project. This documentation provides detailed guides and references for setup, training, and usage of the Qwen Image editing model.

## 📚 Documentation Overview

### 🚀 Getting Started
- **[Setup Guide](docs/setup.md)** - Complete installation and environment setup
- **[Quick Start](#quick-start)** - Get running in minutes
- **[Data Preparation](docs/data-preparation.md)** - Prepare your dataset for training

### 📖 User Guides
- **[Training Guide](docs/training.md)** - Complete training workflow and best practices
- **[Inference Guide](docs/inference.md)** - Running predictions and model inference
- **[Configuration Guide](docs/configuration.md)** - Configuration parameters and examples

### 🔧 Advanced Topics
- **[Cache System](docs/cache-system.md)** - Embedding cache system for training acceleration
- **[Trainer Architecture](docs/trainer-architecture.md)** - QwenImageEditTrainer comprehensive documentation
- **[Model Architecture](docs/architecture/qwen_image_model_architecture.md)** - Deep dive into model internals

### 🛠️ Tools & Utilities
- **[Storage Checker](script/README_storage_checker.md)** - Monitor storage usage during training
- **[Change Log](docs/CHANGELOG.md)** - Project updates and version history

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- CUDA 12.0+ (for GPU training)
- 16GB+ RAM, 8GB+ VRAM recommended

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/qwen-image-finetune.git
cd qwen-image-finetune

# Automated setup
./setup.sh

# Or with custom path and HF token
./setup.sh /your/path hf_your_token_here
```

### Basic Usage with Toy Dataset
```bash
# 1. 项目已提供toy数据集在 data/face_seg/
ls data/face_seg/control_images/    # 输入图像 (20个样本)
ls data/face_seg/training_images/   # 目标图像和文本 (20个样本)

# 2. 配置训练
cp configs/face_seg_config.yaml configs/my_config.yaml

# 3. 缓存嵌入（推荐）
CUDA_VISIBLE_DEVICES=1 python -m src.main --config configs/my_config.yaml --cache

# 4. 开始训练
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml
```

## 📋 Documentation Roadmap

### For New Users
1. **Start Here**: [Setup Guide](docs/setup.md) - Get your environment ready
2. **Prepare Data**: [Data Preparation](docs/data-preparation.md) - Format your dataset
3. **First Training**: [Training Guide](docs/training.md) - Run your first training
4. **Generate Images**: [Inference Guide](docs/inference.md) - Use your trained model

### For Developers
1. **Architecture**: [Model Architecture](docs/architecture/qwen_image_model_architecture.md) - Understand the model
2. **Implementation**: [Trainer Architecture](docs/trainer-architecture.md) - Training implementation details
3. **Optimization**: [Cache System](docs/cache-system.md) - Performance optimization

### For Advanced Users
1. **Performance**: [Cache System](docs/cache-system.md) - 2-3x training speedup
2. **Monitoring**: [Storage Checker](script/README_storage_checker.md) - Resource monitoring
3. **Customization**: [Configuration Guide](docs/configuration.md) - Advanced settings

## 🎯 Key Features Covered

### Training Optimization
- **Embedding Cache System**: 2-3x training acceleration
- **LoRA Fine-tuning**: Memory-efficient parameter updates
- **Multi-GPU Support**: Distributed training capabilities
- **Gradient Checkpointing**: Memory optimization techniques

### Model Capabilities
- **Multimodal Processing**: Combined text and image understanding
- **Flexible Architecture**: Support for various image editing tasks
- **Quality Control**: Advanced inference parameters and optimization
- **Production Ready**: Deployment guides and API examples

### Developer Tools
- **Comprehensive Testing**: Full test suite and validation tools
- **Monitoring**: Real-time storage and performance monitoring
- **Configuration**: Flexible YAML-based configuration system
- **Documentation**: Complete API reference and examples

## 🤝 Contributing to Documentation

We welcome contributions to improve this documentation:

1. **Found an Error?** Open an issue or submit a PR
2. **Missing Information?** Suggest additions or improvements
3. **Want to Help?** Contact the maintainers for contribution guidelines

### Documentation Standards
- Use clear, concise language
- Include practical examples
- Provide complete code snippets
- Add troubleshooting sections
- Keep content up to date

## 📞 Getting Help

### Documentation Issues
- **Missing Information**: Check if it's covered in another section
- **Outdated Content**: Open an issue to report outdated information
- **Unclear Instructions**: Suggest improvements via issues or PRs

### Technical Support
- **Training Issues**: See [Training Guide](docs/training.md) troubleshooting
- **Setup Problems**: Check [Setup Guide](docs/setup.md) common issues
- **Performance**: Review [Cache System](docs/cache-system.md) optimization
- **General Questions**: Open a GitHub issue with detailed description

## 📚 External Resources

### Related Projects
- [Qwen Official Repository](https://github.com/QwenLM/Qwen)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

### Community
- [GitHub Discussions](../../discussions) - General discussions and Q&A
- [Issues](../../issues) - Bug reports and feature requests
- [Pull Requests](../../pulls) - Code contributions

---

**📝 Note**: This documentation is continuously updated. Last updated: 2025/08/28

**⭐ Tip**: Use the navigation links above to jump to specific topics, or browse sequentially for a complete understanding of the framework.

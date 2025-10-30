# Setup Guide

This guide provides detailed instructions for setting up the Qwen Image Finetune environment, from basic installation to advanced configuration.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS, or Windows with WSL2
- **Python**: 3.12 or higher
- **CUDA**: 12.0 or higher (for GPU training)
- **Memory**: 16GB+ RAM recommended
- **Storage**: 50GB+ free space for models and cache

### Hardware Requirements

#### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or equivalent)
- **CPU**: 8+ cores
- **RAM**: 16GB
- **Storage**: 50GB SSD space

#### Recommended Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 4090, A100, or equivalent)
- **CPU**: 16+ cores
- **RAM**: 32GB+
- **Storage**: 100GB+ NVMe SSD

## Quick Setup (Recommended)

### Automated Installation

Use our automated setup script for the fastest installation:

```bash
# Clone the repository
git clone https://github.com/yourusername/qwen-image-finetune.git
cd qwen-image-finetune

# Quick setup with default settings
./setup.sh

# Setup with custom path
./setup.sh /your/custom/path

# Setup with Hugging Face token
./setup.sh ~/.local hf_your_token_here
```

### What the Script Does

The `setup.sh` script automatically:

1. **Checks and installs Miniforge3** (if not present)
2. **Configures conda environment directories**
3. **Creates Python 3.12 environment** named 'myenv'
4. **Sets up Hugging Face cache** in optimized location
5. **Installs PyTorch** with CUDA 12.9 support
6. **Configures Hugging Face authentication**
7. **Installs all dependencies** from requirements.txt

**⚠️ Note**: The setup script installs `diffusers>=0.36.0` from requirements.txt. If you need Qwen-Image-Edit-Plus (2509) support, you must manually upgrade diffusers after setup:

```bash
pip install --upgrade "git+https://github.com/huggingface/diffusers.git"
```

### Script Options

```bash
# Basic usage
./setup.sh [BASE_PATH] [HF_TOKEN]

# Examples
./setup.sh                              # Use default ~/.local
./setup.sh /custom/path                 # Custom installation path
./setup.sh ~/.local hf_token_here       # With HF token
./setup.sh /custom/path hf_token_here   # Custom path + token
```

### Environment Variables

You can also configure the setup using environment variables:

```bash
# Set Hugging Face token
export HF_TOKEN=hf_your_token_here

# Set custom cache directory
export HF_HOME=/custom/cache/path

# Run setup
./setup.sh
```

## Manual Installation

If you prefer manual installation or need custom configuration:

### 1. Create Conda Environment

```bash
# Install conda (if not present)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# Create environment
conda create -n qwen-finetune python=3.12
conda activate qwen-finetune
```

### 2. Install PyTorch

```bash
# For CUDA 12.x
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 3. Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install additional packages for development
pip install -r requirements-dev.txt  # If available
```

**⚠️ Important for Qwen-Image-Edit-Plus (2509) Users**: If you plan to use Qwen-Image-Edit-Plus (2509) model, you need to install the latest version of diffusers from GitHub:

```bash
pip install --upgrade "git+https://github.com/huggingface/diffusers.git"
```

If you don't need Qwen-Image-Edit-Plus (2509) support, you can use the older version specified in `requirements.txt` (`diffusers>=0.36.0`).

### 4. Configure Hugging Face

```bash
# Login to Hugging Face
huggingface-cli login

# Or set token directly
export HF_TOKEN=your_token_here
```

### 5. Verify Installation

```bash
# Test basic functionality
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"

# Test model loading
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B')
print('Qwen model access successful')
"
```

## Configuration

### Environment Configuration

Create a `.env` file in the project root for persistent configuration:

```bash
# .env file
HF_TOKEN=your_huggingface_token
HF_HOME=/path/to/hf/cache
CUDA_VISIBLE_DEVICES=0,1,2,3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Cache Directory Setup

Configure optimal cache directories for best performance:

```bash
# Set Hugging Face cache
export HF_HOME=/fast/ssd/path/.cache/huggingface

# Set custom model cache
export TRANSFORMERS_CACHE=/fast/ssd/path/.cache/transformers

# Set project cache directory
export QWEN_CACHE_DIR=/fast/ssd/path/.cache/qwen
```

### GPU Configuration

#### Single GPU Setup
```bash
export CUDA_VISIBLE_DEVICES=0
```

#### Multi-GPU Setup
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configure memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Storage Configuration

Create optimized storage structure:

```bash
# Create directory structure
mkdir -p /path/to/project/{data,cache,models,logs,outputs}

# Set permissions
chmod -R 755 /path/to/project
```

## Advanced Setup

### Docker Installation

For containerized deployment:

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-pip python3.12-dev \
    git wget curl build-essential

# Install conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda
ENV PATH="/opt/conda/bin:$PATH"

# Create environment
RUN conda create -n qwen-finetune python=3.12 -y
SHELL ["conda", "run", "-n", "qwen-finetune", "/bin/bash", "-c"]

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project
COPY . /app
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["conda", "run", "-n", "qwen-finetune"]
```

Build and run:

```bash
# Build container
docker build -t qwen-finetune .

# Run container
docker run --gpus all -v /data:/data -v /cache:/cache qwen-finetune
```

### Cluster Setup

For distributed training on clusters:

#### SLURM Configuration

```bash
#!/bin/bash
#SBATCH --job-name=qwen-finetune
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load modules
module load cuda/12.1
module load conda

# Activate environment
conda activate qwen-finetune

# Run distributed training
srun python src/main.py --config configs/distributed_config.yaml
```

#### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen-finetune
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qwen-finetune
  template:
    metadata:
      labels:
        app: qwen-finetune
    spec:
      containers:
      - name: qwen-finetune
        image: qwen-finetune:latest
        resources:
          limits:
            nvidia.com/gpu: 4
          requests:
            nvidia.com/gpu: 4
        volumeMounts:
        - name: data-volume
          mountPath: /data
        - name: cache-volume
          mountPath: /cache
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
      - name: cache-volume
        persistentVolumeClaim:
          claimName: cache-pvc
```

## Troubleshooting

### Common Issues

#### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Memory Issues
```bash
# Reduce batch size
export BATCH_SIZE=8

# Enable gradient checkpointing
export GRADIENT_CHECKPOINTING=true

# Use memory efficient attention
export USE_MEMORY_EFFICIENT_ATTENTION=true
```

#### Permission Issues
```bash
# Fix cache permissions
sudo chown -R $USER:$USER ~/.cache/huggingface

# Fix conda permissions
sudo chown -R $USER:$USER ~/miniforge3
```

#### Network Issues
```bash
# Use mirror for faster downloads
export HF_ENDPOINT=https://hf-mirror.com

# Configure proxy if needed
export HTTP_PROXY=http://proxy:8080
export HTTPS_PROXY=http://proxy:8080
```

### Performance Optimization

#### System Optimization
```bash
# Increase file descriptor limits
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# Optimize GPU performance
nvidia-smi -pm 1  # Enable persistence mode
nvidia-smi -acp 0  # Enable auto-boost
```

#### Storage Optimization
```bash
# Use SSD for cache
ln -s /fast/ssd/cache ~/.cache/huggingface

# Enable write caching
echo 'vm.dirty_ratio = 15' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio = 5' >> /etc/sysctl.conf
```

## Validation

### Installation Validation

Run comprehensive validation tests:

```bash
# Test basic installation
python tests/test_installation.py

# Test model loading
python tests/test_model_loading.py

# Test training setup
python tests/test_training_setup.py

# Test GPU functionality
python tests/test_gpu.py
```

### Performance Benchmarks

```bash
# Run performance benchmarks
python tests/benchmark_training.py
python tests/benchmark_inference.py
python tests/benchmark_cache.py
```

### Health Check Script

Create a health check script:

```python
#!/usr/bin/env python
# health_check.py

import torch
import sys
from transformers import AutoTokenizer

def check_cuda():
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"✅ GPU count: {torch.cuda.device_count()}")
    return True

def check_model_access():
    try:
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B')
        print("✅ Model access successful")
        return True
    except Exception as e:
        print(f"❌ Model access failed: {e}")
        return False

def check_disk_space():
    import shutil
    free_gb = shutil.disk_usage('.').free / (1024**3)
    if free_gb < 10:
        print(f"⚠️  Low disk space: {free_gb:.1f}GB")
        return False
    print(f"✅ Disk space: {free_gb:.1f}GB")
    return True

if __name__ == "__main__":
    checks = [check_cuda(), check_model_access(), check_disk_space()]
    if all(checks):
        print("\n🎉 All checks passed!")
        sys.exit(0)
    else:
        print("\n❌ Some checks failed!")
        sys.exit(1)
```

Run health check:

```bash
python health_check.py
```

## install flash-atten

```bash

conda install -n myenv -c conda-forge cuda-nvcc=12.2 # install correct cuda-nvcc

export CUDA_HOME=$HOME/.local/envs/myenv
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install

```

## Next Steps

After successful setup:

1. **Configure Training**: Edit `configs/qwen_image_edit_config.yaml`
2. **Prepare Data**: Follow [Data Preparation Guide](data-preparation.md)
3. **Start Training**: See [Training Guide](training.md)
4. **Monitor Progress**: Use [Storage Checker](../script/README_storage_checker.md)

For additional help, see [Troubleshooting Guide](troubleshooting.md) or open an issue on GitHub.

#!/bin/bash

# Qwen Image Finetune Setup Script
# Usage: ./setup.sh [BASE_PATH] [HF_TOKEN]
# Example: ./setup.sh ~/.local hf_your_token_here
# Or set HF_TOKEN in .env file or environment variable

# 加载.env文件（如果存在）
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# 解析参数
BASE_PATH=${1:-"$HOME/.local"}
HF_TOKEN=${2:-$HF_TOKEN}

echo "Setting up Qwen Image Finetune environment..."
echo "Using base path: $BASE_PATH"

# 设置Hugging Face缓存路径
HF_CACHE_PATH="$BASE_PATH/.cache"
export HF_HOME="$HF_CACHE_PATH"
echo "Setting Hugging Face cache path to: $HF_CACHE_PATH"
mkdir -p "$HF_CACHE_PATH"

# 检查是否已经安装了Miniforge3
if [ -d "$BASE_PATH/miniforge3" ] && [ -f "$BASE_PATH/miniforge3/bin/conda" ]; then
    echo "Miniforge3 already installed at $BASE_PATH/miniforge3, skipping installation..."
else
    # 下载并安装 Miniforge3
    echo "Downloading Miniforge3..."
    curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

    echo "Installing Miniforge3 to $BASE_PATH/miniforge3..."
    bash Miniforge3-Linux-x86_64.sh -b -p "$BASE_PATH/miniforge3"

    # 清理下载的安装文件
    echo "Cleaning up installation files..."
    rm -f Miniforge3-Linux-x86_64.sh
fi

# 初始化当前shell
echo "Initializing conda for current shell..."
eval "$($BASE_PATH/miniforge3/bin/conda shell.bash hook)"

# 添加conda初始化到~/.bashrc
echo "Adding conda initialization to ~/.bashrc..."
"$BASE_PATH/miniforge3/bin/conda" init bash

# 添加Hugging Face缓存路径到~/.bashrc
echo "Adding Hugging Face cache path to ~/.bashrc..."
if ! grep -q "export HF_HOME=" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Hugging Face cache path" >> ~/.bashrc
    echo "export HF_HOME=\"$HF_CACHE_PATH\"" >> ~/.bashrc
fi
source ~/.bashrc

# 创建环境目录并配置
echo "Setting up conda environments directory..."
mkdir -p "$BASE_PATH/envs"
conda config --prepend envs_dirs "$BASE_PATH/envs"

# 检查conda环境是否已存在
if conda env list | grep -q "^myenv "; then
    echo "Conda environment 'myenv' already exists, skipping creation..."
else
    # 创建Python 3.12环境
    echo "Creating conda environment 'myenv' with Python 3.12..."
    conda create -y -n myenv python=3.12 pip --no-default-packages
fi

# 激活环境
echo "Activating conda environment 'myenv'..."
conda activate myenv

# 升级pip并安装PyTorch
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch with CUDA 12.9 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt

# 配置Hugging Face Token
echo ""
echo "Configuring Hugging Face authentication..."

# 检查token是否提供
if [ -z "$HF_TOKEN" ]; then
    echo "Hugging Face token not provided."
    echo "You can:"
    echo "1. Create .env file with: HF_TOKEN=hf_your_token_here"
    echo "2. Set HF_TOKEN environment variable"
    echo "3. Pass token as second parameter: ./setup.sh $BASE_PATH your_token"
    echo "4. Enter token now (will be hidden)"
    echo ""
    read -s -p "Please enter your Hugging Face token (or press Enter to skip): " HF_TOKEN
    echo ""
fi

# 如果提供了token，进行登录
if [ -n "$HF_TOKEN" ]; then
    echo "Setting up Hugging Face authentication..."
    pip install huggingface_hub
    python -c "
from huggingface_hub import login
import sys
try:
    login(token='$HF_TOKEN')
    print('✓ Successfully authenticated with Hugging Face')
except Exception as e:
    print(f'✗ Failed to authenticate: {e}')
    sys.exit(1)
"
    if [ $? -eq 0 ]; then
        echo "Hugging Face token configured successfully!"
    else
        echo "Warning: Failed to configure Hugging Face token"
    fi
else
    echo "Skipping Hugging Face authentication (no token provided)"
    echo "You can set it up later with: huggingface-cli login"
fi

sudo mkdir /raid
sudo mkdir /raid/lilong
sudo ln -s /data/lilong /raid/lilong/data
echo "ln -s /data/lilong /raid/lilong/data"
echo "Done!"

echo "Setup completed successfully!"
echo ""
echo "Conda has been initialized in ~/.bashrc."
echo "To use conda in the current terminal, either:"
echo "1. Restart your terminal, or"
echo "2. Run: source ~/.bashrc"
echo ""
echo "Then activate the environment with:"
echo "conda activate myenv"

echo "Done!"

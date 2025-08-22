# Qwen Image Finetune

## 快速开始

使用提供的安装脚本来自动化设置过程：

```bash
# 使用默认路径 ~/.local
./setup.sh

# 指定自定义基础路径
./setup.sh /your/custom/path

# 同时提供Hugging Face token
./setup.sh ~/.local hf_your_token_here

# 或者创建.env文件
echo "HF_TOKEN=hf_your_token_here" > .env
./setup.sh

# 或者设置环境变量
export HF_TOKEN=hf_your_token_here
./setup.sh
```

脚本将自动：
- 检查并安装 Miniforge3（如果未安装）
- 配置 conda 环境目录
- 创建名为 'myenv' 的 Python 3.12 环境（如果不存在）
- 设置 Hugging Face 缓存路径到 `BASE_PATH/.cache/`
- 安装 PyTorch (CUDA 12.9 支持)
- 配置 Hugging Face 认证（如果提供了token）
\
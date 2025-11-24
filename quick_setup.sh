#!/bin/bash

echo "=========================================="
echo "Quick Setup for Qwen Image Finetune"
echo "=========================================="

# Run main setup
echo "Running main setup script..."
./setup.sh

# Initialize conda for zsh
echo "Initializing conda for zsh..."
~/.local/miniforge3/bin/conda init zsh

# Source zshrc
echo "Sourcing ~/.zshrc..."
source ~/.zshrc

# Note: Need to activate conda in the new shell
echo ""
echo "Activating conda environment..."
eval "$(~/.local/miniforge3/bin/conda shell.zsh hook)"
conda activate myenv

# Install diffusers from GitHub
echo ""
echo "Installing diffusers from GitHub..."
pip install --upgrade "git+https://github.com/huggingface/diffusers.git"

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install python-dotenv transformers==4.52.4 pyyaml oyaml tensorboardX einops accelerate pydantic omegaconf tqdm huggingface_hub opencv-python matplotlib==3.10.1 setuptools==69.5.1 peft tabulate psutil ipykernel optimum-quanto bitsandbytes packaging ninja wheel sentencepiece datasets prodigyopt blake3 pillow rich wandb swanlab ImageHash

# Prompt for HuggingFace token
echo ""
echo "=========================================="
echo "HuggingFace Authentication"
echo "=========================================="
if [ -f ".env" ]; then
    echo ".env file already exists."
    read -p "Do you want to update your HuggingFace token? (y/n): " update_token
    if [ "$update_token" != "y" ]; then
        echo "Keeping existing .env file."
    else
        read -s -p "Enter your HuggingFace token (input will be hidden): " HF_TOKEN
        echo ""
        echo "HF_TOKEN=$HF_TOKEN" > .env
        echo "✓ Token saved to .env file"
    fi
else
    read -s -p "Enter your HuggingFace token (input will be hidden): " HF_TOKEN
    echo ""
    echo "HF_TOKEN=$HF_TOKEN" > .env
    echo "✓ Token saved to .env file"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Close and reopen your terminal, OR run: source ~/.zshrc"
echo "2. Activate environment: conda activate myenv"
echo "3. Upload your test image to inference_examples/ if needed"
echo "4. Run inference:"
echo "   cd inference_examples"
echo "   python test_inference.py --mode base"
echo ""
echo "Note: Your HuggingFace token is stored in .env file (not committed to git)"


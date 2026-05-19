#!/bin/bash
# Speculative Decoding 实验环境安装脚本
# 用法: bash setup.sh

set -e

echo "=========================================="
echo "  Speculative Decoding 实验环境安装"
echo "=========================================="

# 1. 检查 CUDA
echo ""
echo "[1/5] 检查 CUDA..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未检测到 NVIDIA GPU 或 CUDA 未安装"
    exit 1
fi
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "CUDA 检测通过"

# 2. 安装 Miniconda（如果未安装）
echo ""
echo "[2/5] 检查 Miniconda..."
if ! command -v conda &> /dev/null; then
    echo "安装 Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    echo "Miniconda 安装完成"
else
    echo "Miniconda 已安装: $(conda --version)"
fi

# 3. 克隆第三方仓库
echo ""
echo "[3/5] 克隆第三方仓库..."
mkdir -p third_party

if [ ! -d "third_party/FLy-main" ]; then
    echo "克隆 FLy..."
    git clone git@github.com:AMD-AGI/FLy.git third_party/FLy-main
else
    echo "FLy 已存在"
fi

if [ ! -d "third_party/fsd" ]; then
    echo "克隆 FSD..."
    git clone git@github.com:maxholsman/fsd.git third_party/fsd
else
    echo "FSD 已存在"
fi

# 4. 创建 conda 环境
echo ""
echo "[4/5] 创建 conda 环境..."

# FLy 环境
if [ ! -d "envs/fly" ]; then
    echo "创建 FLy 环境..."
    conda create --prefix ./envs/fly python=3.12 -y
    conda run -p ./envs/fly pip install -e third_party/FLy-main
    echo "FLy 环境创建完成"
else
    echo "FLy 环境已存在"
fi

# FSD 环境
if [ ! -d "envs/fsd" ]; then
    echo "创建 FSD 环境..."
    conda create --prefix ./envs/fsd python=3.12 -y
    conda run -p ./envs/fsd pip install transformers==4.44.0 torch accelerate
    echo "FSD 环境创建完成"
else
    echo "FSD 环境已存在"
fi

# 通用环境（自回归 + 推测解码）
if [ ! -d "envs/speculative" ]; then
    echo "创建推测解码环境..."
    conda create --prefix ./envs/speculative python=3.12 -y
    conda run -p ./envs/speculative pip install transformers torch accelerate
    echo "推测解码环境创建完成"
else
    echo "推测解码环境已存在"
fi

# 5. 创建模型目录
echo ""
echo "[5/5] 创建模型目录..."
mkdir -p models
echo "模型目录已创建，请运行 download_models.sh 下载模型"

echo ""
echo "=========================================="
echo "  安装完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 下载模型: bash download_models.sh"
echo "  2. 激活环境: conda activate ./envs/fly"
echo "  3. 运行测试: python src/main.py"
echo ""
echo "可用环境:"
echo "  - FLy:          conda activate ./envs/fly"
echo "  - FSD:          conda activate ./envs/fsd"
echo "  - 推测解码:     conda activate ./envs/speculative"

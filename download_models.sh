#!/bin/bash
# 模型下载脚本
# 用法: bash download_models.sh

set -e

echo "=========================================="
echo "  模型下载"
echo "=========================================="

mkdir -p models

# 检查 huggingface-cli 或 modelscope
if command -v huggingface-cli &> /dev/null; then
    echo "使用 HuggingFace 下载..."
    
    echo ""
    echo "[1/2] 下载 Target 模型: Qwen2.5-72B-Instruct-AWQ..."
    huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ --local-dir ./models/Qwen2.5-72B-Instruct-AWQ
    
    echo ""
    echo "[2/2] 下载 Draft 模型: Qwen2.5-7B-Instruct-GPTQ-Int4..."
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 --local-dir ./models/Qwen2.5-7B-Instruct-GPTQ-Int4

elif command -v python &> /dev/null; then
    echo "使用 ModelScope 下载..."
    
    python -c "
from modelscope import snapshot_download

print('[1/2] 下载 Target 模型: Qwen2.5-72B-Instruct-AWQ...')
snapshot_download('Qwen/Qwen2.5-72B-Instruct-AWQ', cache_dir='./models')

print('[2/2] 下载 Draft 模型: Qwen2.5-7B-Instruct-GPTQ-Int4...')
snapshot_download('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', cache_dir='./models')
"
else
    echo "错误: 请先安装 huggingface-cli 或 modelscope"
    echo "  pip install huggingface_hub"
    echo "  或 pip install modelscope"
    exit 1
fi

echo ""
echo "=========================================="
echo "  模型下载完成！"
echo "=========================================="
echo "模型目录: ./models/"
ls -lh ./models/

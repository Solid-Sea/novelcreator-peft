#!/bin/bash
echo "Starting Qwen finetuning..."
# 切换到脚本所在目录，确保路径正确
cd "$(dirname "$0")"
python ../src/train.py
echo "Finetuning finished."
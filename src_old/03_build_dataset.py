#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集构建脚本
功能：
1. 读取带instruction的文本块
2. 转换为最终的.jsonl格式
3. 生成微调数据集
"""

import os
import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_chunks_with_instructions(file_path: str) -> list:
    """加载带instruction的文本块"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"成功加载 {len(chunks)} 个文本块")
        return chunks
    except Exception as e:
        logger.error(f"加载文件失败 {file_path}: {e}")
        return []


def convert_to_training_format(chunks: list) -> list:
    """转换为训练格式"""
    training_data = []
    
    for chunk in chunks:
        if "instruction" not in chunk:
            continue
            
        training_item = {
            "instruction": chunk["instruction"],
            "input": "",
            "output": chunk["text"],
            "system": "你是一个功底深厚的小说家，请根据指令进行创作。",
            "history": []
        }
        training_data.append(training_item)
    
    return training_data


def save_jsonl(data: list, file_path: str):
    """保存为JSONL格式"""
    # 确保目录存在
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"数据集已保存到: {file_path}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")


def main():
    """主函数"""
    logger.info("开始构建数据集...")
    
    # 文件路径
    input_file = "data/processed/chunks/all_chunks_with_instructions.json"
    output_file = "dataset/novel_finetuning_dataset.jsonl"
    
    # 检查输入文件
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 加载数据
    chunks = load_chunks_with_instructions(input_file)
    if not chunks:
        logger.error("没有找到数据")
        return
    
    # 转换格式
    training_data = convert_to_training_format(chunks)
    logger.info(f"成功转换 {len(training_data)} 条训练数据")
    
    # 保存数据集
    save_jsonl(training_data, output_file)
    
    logger.info("数据集构建完成!")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"训练样本数: {len(training_data)}")


if __name__ == "__main__":
    main()
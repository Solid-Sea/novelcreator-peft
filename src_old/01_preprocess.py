#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小说数据清洗与分段脚本
功能：
1. 设置目录结构
2. 文本清洗
3. 文本分段
4. 日志记录
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """设置目录结构"""
    directories = [
        'data/raw_novels/long',
        'data/raw_novels/short', 
        'data/processed/cleaned',
        'data/processed/chunks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {directory}")


def clean_text(text: str) -> str:
    """清洗文本内容"""
    # 移除常见广告语和网址
    ad_patterns = [
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'www\.[\w\.-]+\.\w+',
        r'[\w\.-]+@[\w\.-]+\.\w+',
        r'QQ[:：]\s*\d+',
        r'微信[:：]\s*[\w\d_-]+',
        r'关注.*公众号',
        r'更多.*小说.*请.*访问',
        r'本书.*首发.*网站',
        r'.*书友.*群.*\d+',
        r'.*VIP.*章节.*',
        r'.*订阅.*支持.*作者.*'
    ]
    
    for pattern in ad_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 合并多个连续换行符为一个
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 移除段落开头和结尾的空白字符
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    return '\n'.join(cleaned_lines)


def split_into_sentences(text: str) -> List[str]:
    """将文本分割成句子"""
    # 中文句子分割符
    sentence_endings = r'[。！？；…]'
    sentences = re.split(sentence_endings, text)
    
    # 过滤空句子并保留标点
    result = []
    for i, sentence in enumerate(sentences[:-1]):  # 最后一个通常是空的
        if sentence.strip():
            result.append(sentence.strip())
    
    return result


def create_chunks(sentences: List[str], min_length: int = 400, max_length: int = 1200) -> List[str]:
    """按句子聚合分割，确保片段在指定字数范围内"""
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果加上当前句子不会超过最大长度，就添加
        if len(current_chunk + sentence) <= max_length:
            current_chunk += sentence
        else:
            # 如果当前块已经达到最小长度，保存它
            if len(current_chunk) >= min_length:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                # 如果当前块太短，继续添加句子
                current_chunk += sentence
    
    # 添加最后一个块（如果足够长）
    if len(current_chunk) >= min_length:
        chunks.append(current_chunk)
    elif chunks and current_chunk:
        # 如果最后一块太短，合并到前一块
        chunks[-1] += current_chunk
    
    return chunks


def process_novel_file(file_path: str, source_type: str) -> List[Dict]:
    """处理单个小说文件"""
    try:
        # 尝试不同编码读取文件
        encodings = ['utf-8', 'gbk', 'gb2312']
        text = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if text is None:
            logger.error(f"无法读取文件: {file_path}")
            return []
        
        # 清洗文本
        cleaned_text = clean_text(text)
        
        # 保存清洗后的文本
        filename = Path(file_path).name
        cleaned_path = f"data/processed/cleaned/{filename}"
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # 分割成句子
        sentences = split_into_sentences(cleaned_text)
        
        # 创建文本块
        chunks = create_chunks(sentences)
        
        # 创建JSON格式的块数据
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_info = {
                "chunk_id": f"{Path(filename).stem}_{i+1:03d}",
                "source_file": filename,
                "text": chunk,
                "length": len(chunk)
            }
            chunk_data.append(chunk_info)
        
        logger.info(f"处理文件 {filename}: 生成 {len(chunk_data)} 个文本块")
        return chunk_data
        
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        return []


def main():
    """主函数"""
    logger.info("开始小说数据预处理...")
    
    # 设置目录结构
    setup_directories()
    
    # 处理长篇小说
    long_novels_dir = "data/长篇（9篇）"
    short_novels_dir = "data/中短篇（43篇）"
    
    all_chunks = []
    processed_files = 0
    
    # 处理长篇小说
    if os.path.exists(long_novels_dir):
        for filename in os.listdir(long_novels_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(long_novels_dir, filename)
                chunks = process_novel_file(file_path, "long")
                all_chunks.extend(chunks)
                processed_files += 1
    
    # 处理中短篇小说
    if os.path.exists(short_novels_dir):
        for filename in os.listdir(short_novels_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(short_novels_dir, filename)
                chunks = process_novel_file(file_path, "short")
                all_chunks.extend(chunks)
                processed_files += 1
    
    # 保存所有文本块到JSON文件
    chunks_file = "data/processed/chunks/all_chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"预处理完成!")
    logger.info(f"处理文件数: {processed_files}")
    logger.info(f"生成文本块数: {len(all_chunks)}")
    logger.info(f"文本块保存至: {chunks_file}")


if __name__ == "__main__":
    main()
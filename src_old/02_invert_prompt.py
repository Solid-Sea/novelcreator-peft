#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提示词逆推脚本
功能：
1. 读取预处理后的文本块
2. 使用Meta-Prompt调用LLM API生成instruction
3. 支持断点续传
4. 保存带instruction的文本块
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Optional
import time
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('invert_prompt.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Meta-Prompt模板
META_PROMPT_TEMPLATE = """你是一个专业的提示词工程师。请根据以下文本内容，逆推出一个合适的instruction（指令），这个指令应该能够引导AI生成类似的文本内容。

文本内容：
{chunk_text}

请生成一个简洁、准确的instruction，要求：
1. 指令应该清晰明确
2. 能够指导AI生成相似风格和内容的文本
3. 长度控制在50字以内
4. 只返回instruction内容，不要其他解释

Instruction:"""


def load_config() -> Dict:
    """加载配置文件"""
    config_path = "config.yaml"
    default_config = {
        "api_key": "your-api-key-here",
        "model_name": "gpt-3.5-turbo",
        "api_base_url": "https://api.openai.com/v1",
        "max_retries": 3,
        "retry_delay": 1
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"已加载配置文件: {config_path}")
            return {**default_config, **config}
        except Exception as e:
            logger.warning(f"加载配置文件失败，使用默认配置: {e}")
    else:
        logger.info("配置文件不存在，使用默认配置")
    
    return default_config


def call_llm_api(chunk_text: str, config: Dict) -> Optional[str]:
    """调用LLM API生成instruction"""
    # 从环境变量获取API密钥
    api_key = os.getenv('CEREBRAS_API_KEY')
    if not api_key:
        logger.error("错误: 缺少CEREBRAS_API_KEY环境变量")
        return None
    
    # 定义模型列表，按优先级排序
    models = [
        {"name": "gpt-oss-120b", "max_tokens": 65536},
        {"name": "llama-3.3-70b", "max_tokens": 65536}
    ]
    
    for model_info in models:
        try:
            # 使用Cerebras API
            client = Cerebras(api_key=api_key)
            prompt = META_PROMPT_TEMPLATE.format(chunk_text=chunk_text)
            
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model_info["name"],
                max_completion_tokens=model_info["max_tokens"],
                temperature=0.7,
                top_p=0.8
            )
            logger.info(f"使用模型 {model_info['name']} 成功生成instruction")
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"使用模型 {model_info['name']} 时API调用失败: {e}")
            # 如果是最后一个模型，返回None
            if model_info == models[-1]:
                logger.error("所有模型都调用失败")
                return None
            else:
                logger.info(f"尝试切换到下一个模型...")
            
    return None


def load_chunks(file_path: str) -> List[Dict]:
    """加载文本块数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"成功加载 {len(chunks)} 个文本块")
        return chunks
    except Exception as e:
        logger.error(f"加载文件失败 {file_path}: {e}")
        return []


def load_existing_results(file_path: str) -> List[Dict]:
    """加载已有的结果（支持断点续传）"""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"加载已有结果: {len(results)} 个")
            return results
        except Exception as e:
            logger.warning(f"加载已有结果失败: {e}")
    return []


def save_results(results: List[Dict], file_path: str):
    """保存结果到文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {file_path}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")


def process_chunks(chunks: List[Dict], config: Dict, output_file: str) -> List[Dict]:
    """处理文本块，生成instruction"""
    # 加载已有结果
    existing_results = load_existing_results(output_file)
    processed_ids = {item["chunk_id"] for item in existing_results if "instruction" in item}
    
    results = existing_results.copy()
    total_chunks = len(chunks)
    processed_count = len(processed_ids)
    
    logger.info(f"开始处理文本块，总数: {total_chunks}，已处理: {processed_count}")
    
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        
        # 跳过已处理的块
        if chunk_id in processed_ids:
            continue
        
        logger.info(f"处理进度: {processed_count + 1}/{total_chunks} - {chunk_id}")
        
        # 调用API生成instruction
        instruction = call_llm_api(chunk["text"], config)
        
        if instruction:
            # 添加instruction到chunk
            chunk_with_instruction = chunk.copy()
            chunk_with_instruction["instruction"] = instruction
            results.append(chunk_with_instruction)
            processed_count += 1
            
            # 每处理10个块保存一次
            if processed_count % 10 == 0:
                save_results(results, output_file)
                logger.info(f"中间保存完成，已处理: {processed_count}")
        else:
            logger.warning(f"跳过块 {chunk_id}，API调用失败")
    
    return results


def main():
    """主函数"""
    logger.info("开始提示词逆推处理...")
    
    # 加载配置
    config = load_config()
    
    # 输入输出文件路径
    input_file = "data/processed/chunks/all_chunks.json"
    output_file = "data/processed/chunks/all_chunks_with_instructions.json"
    
    # 检查输入文件
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 确保输出目录存在
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 加载文本块
    chunks = load_chunks(input_file)
    if not chunks:
        logger.error("没有找到文本块数据")
        return
    
    # 处理文本块
    results = process_chunks(chunks, config, output_file)
    
    # 最终保存
    save_results(results, output_file)
    
    # 统计信息
    with_instructions = [r for r in results if "instruction" in r]
    logger.info(f"处理完成!")
    logger.info(f"总文本块数: {len(chunks)}")
    logger.info(f"已生成instruction数: {len(with_instructions)}")
    logger.info(f"结果保存至: {output_file}")


if __name__ == "__main__":
    main()
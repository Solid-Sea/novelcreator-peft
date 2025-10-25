#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理脚本
功能：
1. 从 config.yaml 读取配置
2. 加载基础模型和微调后的 LoRA 权重
3. 提供交互式文本生成界面
"""

import os
import yaml
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def load_model_and_tokenizer(config):
    """加载模型和tokenizer，并合并LoRA权重"""
    model_name_or_path = config.get('model_name_or_path', 'microsoft/DialoGPT-medium')
    fine_tuned_model_path = config.get('output_dir', 'output/fine_tuned_model') + '/final_model'
    
    logger.info(f"正在加载基础模型: {model_name_or_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 合并LoRA权重
    if os.path.exists(fine_tuned_model_path):
        logger.info(f"正在加载LoRA权重: {fine_tuned_model_path}")
        model = PeftModel.from_pretrained(model, fine_tuned_model_path)
        logger.info("LoRA权重加载完成")
    else:
        logger.warning(f"未找到微调模型路径: {fine_tuned_model_path}，使用基础模型")
    
    return model, tokenizer


def generate_text(model, tokenizer, instruction: str, max_length: int = 512):
    """生成文本"""
    system_prompt = "你是一个功底深厚的小说家，请根据指令进行创作。"
    
    # 构建prompt
    prompt = f"{system_prompt}\n\n指令：{instruction}\n\n输出："
    
    # Tokenize
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码并去除输入部分
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    output_text = generated_text[len(prompt):].strip()
    
    return output_text


def main():
    """主函数 - 交互式界面"""
    logger.info("正在启动推理脚本...")
    
    # 加载配置
    config = load_config()
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(config)
    
    print("\n=== 小说创作AI助手 ===")
    print("输入指令来生成小说内容，输入 'quit' 或 'exit' 退出")
    print("-" * 50)
    
    while True:
        try:
            # 获取用户输入
            instruction = input("\n请输入指令: ").strip()
            
            # 检查退出条件
            if instruction.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not instruction:
                print("请输入有效的指令")
                continue
            
            # 生成文本
            print("\n正在生成...")
            generated_text = generate_text(model, tokenizer, instruction)
            
            # 输出结果
            print(f"\n生成结果:\n{generated_text}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            logger.error(f"生成过程中出错: {e}")
            print(f"生成失败: {e}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型对比推理脚本
功能：对比微调前后模型的生成能力
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

# 硬编码测试指令
TEST_INSTRUCTION_CN = "请以第三人称视角，描写主角在一个雨夜进入一座废弃古堡进行探索的场景。着重刻画他内心的恐惧与好奇交织的复杂情绪，并营造出一种哥特式的恐怖氛围。"
TEST_INSTRUCTION_EN = "Tell me a story."
TEST_INSTRUCTION_SIMPLE = "Hello, how are you?"


def load_config(config_path: str = "config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_base_model(model_name_or_path):
    """加载基础模型"""
    logger.info(f"正在加载基础模型: {model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer


def load_finetuned_model(base_model, tokenizer, fine_tuned_model_path):
    """加载微调模型"""
    if os.path.exists(fine_tuned_model_path):
        logger.info(f"正在加载微调模型: {fine_tuned_model_path}")
        finetuned_model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
        return finetuned_model
    else:
        logger.warning(f"未找到微调模型路径: {fine_tuned_model_path}")
        return None


def generate_text(model, tokenizer, instruction: str, max_new_tokens: int = 200):
    """生成文本"""
    # 使用对话格式，更适合DialoGPT
    prompt = f"用户: {instruction}\n助手:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取生成的部分
    generated_part = generated_text[len(prompt):].strip()
    
    # 如果生成的文本为空，尝试不同的方法
    if not generated_part:
        # 尝试简单的prompt
        simple_prompt = instruction[:50] + "..."
        inputs = tokenizer(simple_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = generated_text[len(simple_prompt):].strip()
    
    return generated_part if generated_part else "[模型未生成任何文本]"


def main():
    """主函数"""
    logger.info("开始模型对比测试...")
    
    # 加载配置
    config = load_config()
    model_name_or_path = config.get('model_name_or_path', 'microsoft/DialoGPT-medium')
    fine_tuned_model_path = config.get('output_dir', 'output/fine_tuned_model') + '/final_model'
    
    # 加载基础模型
    base_model, tokenizer = load_base_model(model_name_or_path)
    
    # 加载微调模型
    finetuned_model = load_finetuned_model(base_model, tokenizer, fine_tuned_model_path)
    
    print("\n" + "="*80)
    print("模型生成能力对比测试")
    print("="*80)
    
    # 测试中文指令
    print(f"\n中文测试指令：\n{TEST_INSTRUCTION_CN}")
    print("\n" + "-"*80)
    
    # 基础模型生成（中文）
    print("\n【基础模型输出 - 中文】:")
    base_output_cn = generate_text(base_model, tokenizer, TEST_INSTRUCTION_CN)
    print(base_output_cn)
    
    # 微调模型生成（中文）
    if finetuned_model:
        print("\n【微调模型输出 - 中文】:")
        finetuned_output_cn = generate_text(finetuned_model, tokenizer, TEST_INSTRUCTION_CN)
        print(finetuned_output_cn)
    else:
        print("\n【微调模型输出 - 中文】:")
        print("微调模型未找到，无法进行对比")
    
    print("\n" + "="*80)
    
    # 测试英文指令
    print(f"\n英文测试指令：\n{TEST_INSTRUCTION_EN}")
    print("\n" + "-"*80)
    
    # 基础模型生成（英文）
    print("\n【基础模型输出 - 英文】:")
    base_output_en = generate_text(base_model, tokenizer, TEST_INSTRUCTION_EN)
    print(base_output_en)
    
    # 微调模型生成（英文）
    if finetuned_model:
        print("\n【微调模型输出 - 英文】:")
        finetuned_output_en = generate_text(finetuned_model, tokenizer, TEST_INSTRUCTION_EN)
        print(finetuned_output_en)
    else:
        print("\n【微调模型输出 - 英文】:")
        print("微调模型未找到，无法进行对比")
    
    print("\n" + "="*80)
    
    # 测试简单对话
    print(f"\n简单对话测试：\n{TEST_INSTRUCTION_SIMPLE}")
    print("\n" + "-"*80)
    
    # 基础模型生成（简单对话）
    print("\n【基础模型输出 - 简单对话】:")
    base_output_simple = generate_text(base_model, tokenizer, TEST_INSTRUCTION_SIMPLE)
    print(base_output_simple)
    
    # 微调模型生成（简单对话）
    if finetuned_model:
        print("\n【微调模型输出 - 简单对话】:")
        finetuned_output_simple = generate_text(finetuned_model, tokenizer, TEST_INSTRUCTION_SIMPLE)
        print(finetuned_output_simple)
    else:
        print("\n【微调模型输出 - 简单对话】:")
        print("微调模型未找到，无法进行对比")
    
    print("\n" + "="*80)
    print("对比测试完成")
    print("="*80)


if __name__ == "__main__":
    main()
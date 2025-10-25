#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终模型评估脚本
功能：加载微调后的Qwen模型并进行结构化能力测试
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


def load_model_and_tokenizer():
    """加载微调模型和tokenizer"""
    model_name_or_path = "Qwen/Qwen1.5-1.8B-Chat"
    fine_tuned_model_path = "output/qwen_fine_tuned_model/final_model/"
    
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
        logger.error(f"未找到微调模型路径: {fine_tuned_model_path}")
        raise FileNotFoundError(f"微调模型不存在: {fine_tuned_model_path}")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=512):
    """生成文本"""
    # Tokenize输入
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def main():
    """主函数 - 执行模型评估"""
    logger.info("开始最终模型评估...")
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer()
    
    # 定义格式化提示词
    prompt = """
[PROMPT_VERSION]: 1.0
[TASK]: GENERATE_CHAPTER_TEXT
[CONTEXT]:
- 主要人物: [宇航员李维]
[INPUT]:
- ChapterNumber: [2]
- ChapterSynopsis: [李维试图修复通讯系统，期间发现了一本日记，日记内容让他产生怀疑。]
- SupplementaryContent: [请在结尾处设置一个悬念，暗示日记的内容可能被篡改过。]
[INSTRUCTIONS]:
- RequiredPlotPoints: [李维必须找到工具箱。]
- QuotesToInclude: ["在这片死寂的太空中，唯一的敌人就是谎言。"]
[OUTPUT_FORMAT]:
- ChapterTitle: [章节标题]
- Content: [完整的章节正文]
[START_OUTPUT]
"""
    
    print("=" * 80)
    print("最终模型评估测试")
    print("=" * 80)
    
    print("\n【输入提示词】:")
    print(prompt)
    
    print("\n【正在生成...】")
    
    # 执行生成
    generated_text = generate_text(model, tokenizer, prompt)
    
    print("\n【模型输出结果】:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)
    
    logger.info("模型评估完成")


if __name__ == "__main__":
    main()
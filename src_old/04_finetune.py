#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微调训练脚本
功能：
1. 从 config.yaml 读取训练配置
2. 加载基础模型和 Tokenizer
3. 加载并预处理数据集
4. 配置 LoRA 微调
5. 开始训练并保存模型

依赖：
- transformers
- peft
- datasets
- torch
- accelerate
- yaml
"""

import os
import json
import logging
import yaml
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
from pathlib import Path
import glob

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败 {config_path}: {e}")
        raise


def load_and_prepare_dataset(dataset_path: str):
    """加载并预处理数据集"""
    try:
        # 加载 JSONL 数据集
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        logger.info(f"成功加载数据集: {dataset_path}, 样本数: {len(dataset)}")
        
        # 显示数据集结构
        logger.info(f"数据集列名: {dataset.column_names}")
        
        return dataset
    except Exception as e:
        logger.error(f"加载数据集失败 {dataset_path}: {e}")
        raise


def preprocess_function(examples, tokenizer, max_length=512):
    """
    预处理函数：将 instruction, input, output 拼接成完整的 prompt
    """
    # 构建完整的提示词
    processed_examples = []
    
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i] if examples['input'][i] else ""
        output_text = examples['output'][i]
        system_prompt = examples.get('system', ["你是一个功底深厚的小说家，请根据指令进行创作。"])[i] if 'system' in examples else "你是一个功底深厚的小说家，请根据指令进行创作。"
        
        # 构建完整的对话格式
        if input_text:
            full_prompt = f"{system_prompt}\n\n指令：{instruction}\n\n输入：{input_text}\n\n输出：{output_text}{tokenizer.eos_token}"
        else:
            full_prompt = f"{system_prompt}\n\n指令：{instruction}\n\n输出：{output_text}{tokenizer.eos_token}"
        
        # Tokenize 完整的 prompt
        tokenized = tokenizer(
            full_prompt,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        # 为 input 和 instruction 部分设置 -100，只计算 output 部分的 loss
        labels = tokenized['input_ids'].copy()
        
        # 创建 prompt 部分的 token
        if input_text:
            prompt_part = f"{system_prompt}\n\n指令：{instruction}\n\n输入：{input_text}\n\n输出："
        else:
            prompt_part = f"{system_prompt}\n\n指令：{instruction}\n\n输出："
        
        prompt_tokens = tokenizer.encode(prompt_part, add_special_tokens=False)
        
        # 将 prompt 部分的标签设置为 -100，只计算 output 部分的 loss
        labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)
        
        processed_examples.append({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        })
    
    # 重新组织为字典格式
    result = {
        'input_ids': [ex['input_ids'] for ex in processed_examples],
        'attention_mask': [ex['attention_mask'] for ex in processed_examples],
        'labels': [ex['labels'] for ex in processed_examples]
    }
    
    return result


def prepare_model_and_tokenizer(model_name_or_path: str):
    """加载模型和 Tokenizer"""
    logger.info(f"正在加载模型: {model_name_or_path}")
    
    try:
        # 尝试加载 Tokenizer，设置更长的超时时间
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=False,
            trust_remote_code=True,
            use_fast=True
        )
    except Exception as e:
        logger.warning(f"从远程加载tokenizer失败: {e}")
        logger.info("尝试使用本地缓存...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=True
            )
        except Exception as e2:
            logger.error(f"从本地缓存加载tokenizer也失败: {e2}")
            raise e2
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("设置 pad_token 为 eos_token")
    
    try:
        # 尝试加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",  # 自动分配到可用设备
            local_files_only=False,
            trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"从远程加载模型失败: {e}")
        logger.info("尝试使用本地缓存...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e2:
            logger.error(f"从本地缓存加载模型也失败: {e2}")
            raise e2
    
    logger.info(f"模型加载完成，模型大小: {model.get_memory_footprint() / 1024**2:.2f} MB")
    
    return model, tokenizer


def setup_lora_model(model, lora_config):
    """配置 LoRA 微调"""
    logger.info("正在配置 LoRA...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 32),
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        target_modules=lora_config.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj",
                                                          "gate_proj", "up_proj", "down_proj",
                                                          "lm_head"]),
        bias=lora_config.get('bias', "none")
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 确保 LoRA 参数需要梯度
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    
    logger.info("LoRA 配置完成")
    
    return model


def find_latest_checkpoint(output_dir: str) -> str:
    """查找最新的检查点"""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    # 按检查点编号排序，返回最新的
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    latest_checkpoint = checkpoints[-1]
    logger.info(f"找到最新检查点: {latest_checkpoint}")
    return latest_checkpoint


def main():
    """主函数"""
    logger.info("开始微调训练...")
    
    # 加载配置
    config = load_config()
    
    # 获取训练参数
    model_name_or_path = config.get('model_name_or_path', 'microsoft/DialoGPT-medium')
    dataset_path = config.get('dataset_path', 'dataset/novel_finetuning_dataset.jsonl')
    output_dir = config.get('output_dir', 'output/fine_tuned_model')
    resume_from_checkpoint = config.get('resume_from_checkpoint', True)
    
    # 获取 LoRA 配置
    lora_config = config.get('lora_config', {
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'target_modules': ["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj", "lm_head"],
        'bias': "none"
    })
    
    # 获取训练参数
    training_args_config = config.get('training_args', {
        'num_train_epochs': 3,
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 1,
        'learning_rate': 5e-5,
        'logging_steps': 10,
        'save_steps': 500,
        'save_total_limit': 3,
        'save_strategy': "steps",
        'warmup_steps': 100,
        'report_to': None, # 禁用 wandb 等日志上报
        'output_dir': output_dir,
        'overwrite_output_dir': True,
        'remove_unused_columns': False,
        'push_to_hub': False,
        'bf16': torch.cuda.is_available(),  # 如果有 GPU，使用混合精度
        'dataloader_pin_memory': True,
        'dataloader_num_workers': 0 # 避免多进程问题
    })
    
    # 确保数值参数被正确转换为适当的类型
    training_args_config['num_train_epochs'] = float(training_args_config.get('num_train_epochs', 3))
    training_args_config['per_device_train_batch_size'] = int(training_args_config.get('per_device_train_batch_size', 2))
    training_args_config['gradient_accumulation_steps'] = int(training_args_config.get('gradient_accumulation_steps', 4))
    training_args_config['learning_rate'] = float(training_args_config.get('learning_rate', 5e-5))
    training_args_config['logging_steps'] = int(training_args_config.get('logging_steps', 50))
    training_args_config['save_steps'] = int(training_args_config.get('save_steps', 200))
    training_args_config['save_total_limit'] = int(training_args_config.get('save_total_limit', 5))
    training_args_config['warmup_steps'] = int(training_args_config.get('warmup_steps', 100))
    training_args_config['max_grad_norm'] = float(training_args_config.get('max_grad_norm', 1.0))
    training_args_config['eval_steps'] = int(training_args_config.get('eval_steps', 200))
    
    # 确保布尔值正确处理
    training_args_config['gradient_checkpointing'] = bool(training_args_config.get('gradient_checkpointing', True))
    training_args_config['logging_first_step'] = bool(training_args_config.get('logging_first_step', True))
    
    # 更新输出目录
    training_args_config['output_dir'] = output_dir
    
    logger.info(f"基础模型: {model_name_or_path}")
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"LoRA 配置: {lora_config}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 检查是否有检查点可以恢复
    checkpoint_to_resume = None
    if resume_from_checkpoint:
        checkpoint_to_resume = find_latest_checkpoint(output_dir)
        if checkpoint_to_resume:
            logger.info(f"将从检查点恢复训练: {checkpoint_to_resume}")
        else:
            logger.info("未找到检查点，将从头开始训练")
    
    # 加载模型和 Tokenizer
    model, tokenizer = prepare_model_and_tokenizer(model_name_or_path)
    
    # 加载数据集
    dataset = load_and_prepare_dataset(dataset_path)
    
    # 应用预处理
    logger.info("正在预处理数据集...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names  # 移除原始列，保留 tokenized 列
    )
    
    logger.info(f"数据集预处理完成，样本数: {len(tokenized_dataset)}")
    
    # 设置 LoRA
    model = setup_lora_model(model, lora_config)
    
    # 启用梯度检查点以节省内存（在 LoRA 配置后）
    if training_args_config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        # 确保梯度检查点与 LoRA 兼容
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        logger.info("已启用梯度检查点")
    
    # 设置训练参数
    training_args = TrainingArguments(**training_args_config)
    
    # 设置数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 # 优化 GPU 计算
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("开始训练...")
    
    # 开始训练（如果有检查点则从检查点恢复）
    trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"训练完成！最终模型已保存到: {final_model_path}")
    
    # 保存训练参数到配置文件
    training_config_path = os.path.join(final_model_path, "training_config.json")
    with open(training_config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name_or_path': model_name_or_path,
            'dataset_path': dataset_path,
            'lora_config': lora_config,
            'training_args': training_args_config
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"训练配置已保存到: {training_config_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGLM3 QLoRA 量化微调训练脚本
功能：
1. 从 config_qlora.yaml 读取训练配置
2. 使用 BitsAndBytesConfig 进行4位量化加载 ChatGLM3 模型
3. 配置适合 ChatGLM3 的 LoRA 参数
4. 执行量化微调训练并保存模型

依赖：
- transformers
- peft
- datasets
- torch
- accelerate
- bitsandbytes
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
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from pathlib import Path
import glob

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config_qlora.yaml"):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败 {config_path}: {e}")
        raise


def create_quantization_config(quant_config):
    """创建量化配置"""
    logger.info("正在创建量化配置...")
    
    # 处理计算数据类型
    compute_dtype = quant_config.get('bnb_4bit_compute_dtype', 'bfloat16')
    if compute_dtype == 'bfloat16':
        compute_dtype = torch.bfloat16
    elif compute_dtype == 'float16':
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.get('load_in_4bit', True),
        bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
    )
    
    logger.info(f"量化配置: {quantization_config}")
    return quantization_config


def load_and_prepare_dataset(dataset_path: str):
    """加载并预处理数据集"""
    try:
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        logger.info(f"成功加载数据集: {dataset_path}, 样本数: {len(dataset)}")
        logger.info(f"数据集列名: {dataset.column_names}")
        return dataset
    except Exception as e:
        logger.error(f"加载数据集失败 {dataset_path}: {e}")
        raise


def preprocess_function_chatglm3(examples, tokenizer, max_length=2048):
    """ChatGLM3 专用预处理函数"""
    processed_examples = []
    
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i] if examples['input'][i] else ""
        output_text = examples['output'][i]
        system_prompt = examples.get('system', ["你是一个功底深厚的小说家，请根据指令进行创作。"])[i] if 'system' in examples else "你是一个功底深厚的小说家，请根据指令进行创作。"
        
        # ChatGLM3 对话格式
        if input_text:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{instruction}\n\n{input_text}"},
                {"role": "assistant", "content": output_text}
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output_text}
            ]
        
        # 使用 ChatGLM3 的 apply_chat_template 方法
        try:
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"使用 apply_chat_template 失败: {e}, 使用备用格式")
            if input_text:
                full_prompt = f"<|system|>\n{system_prompt}<|user|>\n{instruction}\n\n{input_text}<|assistant|>\n{output_text}"
            else:
                full_prompt = f"<|system|>\n{system_prompt}<|user|>\n{instruction}<|assistant|>\n{output_text}"
        
        # Tokenize
        tokenized = tokenizer(
            full_prompt,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        # 设置标签
        labels = tokenized['input_ids'].copy()
        assistant_start_tokens = tokenizer.encode("<|assistant|>", add_special_tokens=False)
        if assistant_start_tokens:
            input_ids = tokenized['input_ids']
            for j in range(len(input_ids) - len(assistant_start_tokens) + 1):
                if input_ids[j:j+len(assistant_start_tokens)] == assistant_start_tokens:
                    labels[:j+len(assistant_start_tokens)] = [-100] * (j+len(assistant_start_tokens))
                    break
        else:
            prompt_without_output = full_prompt.split(output_text)[0]
            prompt_tokens = tokenizer.encode(prompt_without_output, add_special_tokens=False)
            labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)
        
        processed_examples.append({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        })
    
    return {
        'input_ids': [ex['input_ids'] for ex in processed_examples],
        'attention_mask': [ex['attention_mask'] for ex in processed_examples],
        'labels': [ex['labels'] for ex in processed_examples]
    }


def prepare_model_and_tokenizer(model_name_or_path: str, quantization_config, model_config):
    """加载量化模型和 Tokenizer"""
    logger.info(f"正在加载量化模型: {model_name_or_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=model_config.get('trust_remote_code', True),
            use_fast=False
        )
        logger.info("Tokenizer 加载完成")
    except Exception as e:
        logger.error(f"加载 Tokenizer 失败: {e}")
        raise
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
        logger.info(f"设置 pad_token 为: {tokenizer.pad_token}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map=model_config.get('device_map', 'auto'),
            trust_remote_code=model_config.get('trust_remote_code', True),
            torch_dtype=model_config.get('torch_dtype', 'auto')
        )
        logger.info("量化模型加载完成")
    except Exception as e:
        logger.error(f"加载量化模型失败: {e}")
        raise
    
    model = prepare_model_for_kbit_training(model)
    logger.info("模型已准备好进行量化训练")
    
    return model, tokenizer


def setup_lora_model(model, lora_config):
    """配置 LoRA 微调（适配 ChatGLM3）"""
    logger.info("正在配置 LoRA（ChatGLM3 适配）...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config.get('r', 32),
        lora_alpha=lora_config.get('lora_alpha', 64),
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        target_modules=lora_config.get('target_modules', ["query_key_value"]),
        bias=lora_config.get('bias', "none")
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    
    logger.info("LoRA 配置完成（ChatGLM3 适配）")
    return model


def find_latest_checkpoint(output_dir: str) -> str:
    """查找最新的检查点"""
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    latest_checkpoint = checkpoints[-1]
    logger.info(f"找到最新检查点: {latest_checkpoint}")
    return latest_checkpoint


def main():
    """主函数"""
    logger.info("开始 ChatGLM3 QLoRA 量化微调训练...")
    
    config = load_config()
    
    model_name_or_path = config.get('model_name_or_path', 'THUDM/chatglm3-6b')
    dataset_path = config.get('dataset_path', 'dataset/novel_finetuning_dataset.jsonl')
    output_dir = config.get('output_dir', 'output/chatglm3_qlora_model')
    resume_from_checkpoint = config.get('resume_from_checkpoint', True)
    
    quantization_config_dict = config.get('quantization_config', {
        'load_in_4bit': True,
        'bnb_4bit_quant_type': "nf4",
        'bnb_4bit_compute_dtype': "bfloat16",
        'bnb_4bit_use_double_quant': True
    })
    
    model_config = config.get('model_config', {
        'trust_remote_code': True,
        'torch_dtype': 'auto',
        'device_map': 'auto',
        'max_length': 2048
    })
    
    lora_config = config.get('lora_config', {
        'r': 32,
        'lora_alpha': 64,
        'lora_dropout': 0.1,
        'target_modules': ["query_key_value"],
        'bias': "none"
    })
    
    training_args_config = config.get('training_args', {})
    
    # 设置默认训练参数
    training_args_config.setdefault('num_train_epochs', 3)
    training_args_config.setdefault('per_device_train_batch_size', 1)
    training_args_config.setdefault('gradient_accumulation_steps', 8)
    training_args_config.setdefault('learning_rate', 2e-4)
    training_args_config.setdefault('logging_steps', 10)
    training_args_config.setdefault('save_steps', 100)
    training_args_config.setdefault('save_total_limit', 3)
    training_args_config.setdefault('save_strategy', "steps")
    training_args_config.setdefault('warmup_steps', 50)
    training_args_config.setdefault('report_to', None)
    training_args_config.setdefault('overwrite_output_dir', False)
    training_args_config.setdefault('remove_unused_columns', False)
    training_args_config.setdefault('push_to_hub', False)
    training_args_config.setdefault('bf16', True)
    training_args_config.setdefault('dataloader_pin_memory', False)
    training_args_config.setdefault('dataloader_num_workers', 0)
    training_args_config.setdefault('optim', "paged_adamw_32bit")
    training_args_config.setdefault('lr_scheduler_type', "cosine")
    training_args_config.setdefault('max_grad_norm', 1.0)
    training_args_config.setdefault('gradient_checkpointing', True)
    training_args_config.setdefault('logging_first_step', True)
    training_args_config['output_dir'] = output_dir
    
    # 确保数值参数为正确的数据类型
    numeric_params = ['learning_rate', 'weight_decay', 'max_grad_norm', 'warmup_ratio']
    for param in numeric_params:
        if param in training_args_config and isinstance(training_args_config[param], str):
            try:
                training_args_config[param] = float(training_args_config[param])
            except ValueError:
                logger.warning(f"无法转换参数 {param} 为浮点数: {training_args_config[param]}")
    
    # 确保整数参数为正确的数据类型
    int_params = ['num_train_epochs', 'per_device_train_batch_size', 'gradient_accumulation_steps',
                  'logging_steps', 'save_steps', 'save_total_limit', 'warmup_steps',
                  'dataloader_num_workers', 'max_steps']
    for param in int_params:
        if param in training_args_config and isinstance(training_args_config[param], str):
            try:
                training_args_config[param] = int(training_args_config[param])
            except ValueError:
                logger.warning(f"无法转换参数 {param} 为整数: {training_args_config[param]}")
    
    logger.info(f"基础模型: {model_name_or_path}")
    logger.info(f"数据集路径: {dataset_path}")
    logger.info(f"输出目录: {output_dir}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint_to_resume = None
    if resume_from_checkpoint:
        checkpoint_to_resume = find_latest_checkpoint(output_dir)
        if checkpoint_to_resume:
            logger.info(f"将从检查点恢复训练: {checkpoint_to_resume}")
        else:
            logger.info("未找到检查点，将从头开始训练")
    
    quantization_config = create_quantization_config(quantization_config_dict)
    model, tokenizer = prepare_model_and_tokenizer(model_name_or_path, quantization_config, model_config)
    dataset = load_and_prepare_dataset(dataset_path)
    
    logger.info("正在预处理数据集（ChatGLM3 格式）...")
    max_length = model_config.get('max_length', 2048)
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function_chatglm3(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"数据集预处理完成，样本数: {len(tokenized_dataset)}")
    
    model = setup_lora_model(model, lora_config)
    
    if training_args_config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        logger.info("已启用梯度检查点")
    
    training_args = TrainingArguments(**training_args_config)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("开始 QLoRA 训练...")
    trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"QLoRA 训练完成！最终模型已保存到: {final_model_path}")
    
    training_config_path = os.path.join(final_model_path, "training_config.json")
    with open(training_config_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name_or_path': model_name_or_path,
            'dataset_path': dataset_path,
            'quantization_config': quantization_config_dict,
            'lora_config': lora_config,
            'training_args': training_args_config,
            'model_config': model_config
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"训练配置已保存到: {training_config_path}")
    
    lora_adapter_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(lora_adapter_path)
    logger.info(f"LoRA 适配器权重已保存到: {lora_adapter_path}")


if __name__ == "__main__":
    main()
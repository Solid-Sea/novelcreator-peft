#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen QLoRA 量化微调训练脚本
功能：
1. 从 ../config.yaml 读取训练配置
2. 使用 Unsloth 高效加载 Qwen 模型
3. 配置适合 Qwen 的 LoRA 参数
4. 执行量化微调训练并保存模型

依赖：
- unsloth
- transformers
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
import gc
import importlib
from unsloth import FastLanguageModel
# 可选：使用 accelerate 的低内存加载辅助方法（init_empty_weights, load_checkpoint_and_dispatch）
try:
    from accelerate import init_empty_weights
    try:
        # 新版 accelerate 将 load_checkpoint_and_dispatch 放在 accelerate module
        from accelerate import load_checkpoint_and_dispatch
    except Exception:
        # 老版可能在不同位置或不可用
        load_checkpoint_and_dispatch = None
except Exception:
    init_empty_weights = None
    load_checkpoint_and_dispatch = None
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from pathlib import Path
import glob
import argparse

# 尝试按需导入 bitsandbytes 的 8-bit 优化器（如果可用）
try:
    from bitsandbytes.optim import Adam8bit  # newer versions may provide Adam8bit
    AdamW8bit = Adam8bit
except Exception:
    try:
        from bitsandbytes.optim import AdamW8bit
    except Exception:
        AdamW8bit = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 针对 Windows 平台的 Unsloth 兼容性补丁
import platform
if platform.system() == "Windows":
    try:
        import unsloth_zoo.rl_environments
        from contextlib import contextmanager

        @contextmanager
        def dummy_time_limit(seconds):
            yield

        unsloth_zoo.rl_environments.time_limit = dummy_time_limit
        logger.info("Applied Windows compatibility patch for Unsloth's time_limit function.")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not apply Unsloth Windows patch, continuing without it. Error: {e}")

# 针对 Unsloth 内部 ZeroDivisionError 的补丁
try:
    from unsloth_zoo.fused_losses import cross_entropy_loss as ce_loss
    original_get_chunk_multiplier = ce_loss._get_chunk_multiplier

    def patched_get_chunk_multiplier(vocab_size, target_gb):
        if target_gb is None or target_gb == 0:
            target_gb = 0.2  # 使用一个安全的默认值避免除零
        # 直接重新实现原始逻辑，而不是再次调用它
        return (vocab_size * 4 / 1024 / 1024 / 1024) / target_gb

    ce_loss._get_chunk_multiplier = patched_get_chunk_multiplier
    logger.info("Applied patch for Unsloth's _get_chunk_multiplier to prevent ZeroDivisionError.")
except (ImportError, AttributeError) as e:
    logger.warning(f"Could not apply Unsloth _get_chunk_multiplier patch, continuing without it. Error: {e}")


def load_config(config_path: str):
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
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        logger.info(f"成功加载数据集: {dataset_path}, 样本数: {len(dataset)}")
        logger.info(f"数据集列名: {dataset.column_names}")
        return dataset
    except Exception as e:
        logger.error(f"加载数据集失败 {dataset_path}: {e}")
        raise


def preprocess_function_qwen(examples, tokenizer, max_length=2048):
    """Qwen 专用预处理函数"""
    processed_examples = []
    
    for i in range(len(examples['instruction'])):
        instruction = examples['instruction'][i]
        input_text = examples['input'][i] if examples['input'][i] else ""
        output_text = examples['output'][i]
        system_prompt = examples.get('system', ["你是一个功底深厚的小说家，请根据指令进行创作。"])[i] if 'system' in examples else "你是一个功底深厚的小说家，请根据指令进行创作。"
        
        # Qwen ChatML 对话格式
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
        
        # 使用 Qwen 的 apply_chat_template 方法
        try:
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        except Exception as e:
            logger.warning(f"使用 apply_chat_template 失败: {e}, 使用备用格式")
            # Qwen ChatML 格式备用方案
            if input_text:
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
            else:
                full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        
        # Tokenize
        tokenized = tokenizer(
            full_prompt,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        
        # 设置标签 - 只对 assistant 部分计算损失
        labels = tokenized['input_ids'].copy()
        
        # 查找 assistant 开始位置
        assistant_start_tokens = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        if assistant_start_tokens:
            input_ids = tokenized['input_ids']
            for j in range(len(input_ids) - len(assistant_start_tokens) + 1):
                if input_ids[j:j+len(assistant_start_tokens)] == assistant_start_tokens:
                    # 将 assistant 开始标记之前的所有 token 设为 -100
                    labels[:j+len(assistant_start_tokens)] = [-100] * (j+len(assistant_start_tokens))
                    break
        else:
            # 备用方案：根据文本分割来确定标签位置
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


def prepare_model_and_tokenizer(model_name_or_path, model_config):
    """
    加载 Unsloth 优化后的模型和 Tokenizer（新版可靠逻辑）
    """
    force_local_load = model_config.get("force_local_load", False)
    local_model_path = model_config.get("local_model_path")
    cache_dir = model_config.get("model_cache_dir")

    model_to_load = model_name_or_path
    local_files_only_flag = False
    
    if force_local_load:
        print("模式: 强制本地加载已启用。")
        if not local_model_path or not os.path.isdir(local_model_path):
            raise ValueError(
                f"强制本地加载失败：当 'force_local_load' 为 true 时, 'local_model_path' "
                f"必须被设置并且是一个有效的目录, 但当前值为: '{local_model_path}'"
            )
        
        model_to_load = local_model_path
        local_files_only_flag = True
        # 在纯本地加载模式下，我们不应依赖 Hugging Face 的缓存目录，以避免行为混淆
        cache_dir = None
        print(f"将严格从本地路径加载模型: {model_to_load}")

    else:
        print("模式: 将从 Hugging Face Hub 或缓存加载。")
        print(f"模型标识: {model_to_load}")
        if cache_dir:
            print(f"使用缓存目录: {cache_dir}")

    # 动态获取并转换 dtype
    dtype_str = model_config.get("dtype", "float16")
    model_dtype = getattr(torch, dtype_str, torch.float16)
    print(f"模型将使用数据类型: {model_dtype}")

    # 调用 Unsloth 加载函数
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_to_load,
        max_seq_length=model_config.get('max_length', 2048),
        dtype=model_dtype,
        load_in_4bit=True,
        local_files_only=local_files_only_flag,
        cache_dir=cache_dir,
    )
    
    print("Unsloth 模型和 Tokenizer 加载完成。")
    return model, tokenizer


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
    parser = argparse.ArgumentParser(description="Qwen QLoRA 量化微调训练脚本")
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../config.yaml"),
        help='配置文件的路径'
    )
    args = parser.parse_args()

    logger.info("开始 Qwen QLoRA 量化微调训练...")
    
    config = load_config(args.config)
    
    model_name_or_path = config.get('model_name_or_path', 'Qwen/Qwen2.5-7B-Instruct')
    dataset_path = config.get('dataset_path', 'dataset/novel_finetuning_dataset.jsonl')
    output_dir = config.get('output_dir', 'output/qwen_qlora_model')
    resume_from_checkpoint = config.get('resume_from_checkpoint', True)
    
    model_config = config.get('model_config', {
        'trust_remote_code': True,
        'torch_dtype': 'auto',
        'device_map': 'auto',
        'max_length': 2048
    })
    
    lora_config = config.get('lora_config', {
        'r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    # 可选项：使用 8-bit 优化器以减少优化器状态内存占用（需要 bitsandbytes）
    training_args_config.setdefault('use_8bit_optimizer', False)
    
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
    
    model, tokenizer = prepare_model_and_tokenizer(model_name_or_path, model_config)
    dataset = load_and_prepare_dataset(dataset_path)
    
    logger.info("正在预处理数据集（Qwen 格式）...")
    max_length = model_config.get('max_length', 2048)
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function_qwen(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"数据集预处理完成，样本数: {len(tokenized_dataset)}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('lora_alpha', 16),
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        bias=lora_config.get('bias', "none"),
    )
    
    if training_args_config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        logger.info("已启用梯度检查点")
    
    # 如果配置要求使用 8-bit 优化器，尝试创建并将其传入 Trainer（否则让 Trainer 使用默认优化器）
    use_8bit_opt = training_args_config.pop('use_8bit_optimizer', False)
    training_args = TrainingArguments(**training_args_config)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )

    optimizer_to_pass = None
    if use_8bit_opt:
        if AdamW8bit is None:
            logger.warning("配置要求使用 8-bit 优化器，但 bitsandbytes 未安装或不支持 AdamW8bit，回退到默认优化器。")
        else:
            try:
                # 只优化可训练参数（LoRA 的参数通常是 requires_grad=True）
                trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
                if len(trainable_params) == 0:
                    logger.warning("未找到任何可训练参数以用于 8-bit 优化器，回退到默认优化器。")
                else:
                    lr = training_args_config.get('learning_rate', 2e-4)
                    weight_decay = training_args_config.get('weight_decay', 0.0)
                    optimizer_to_pass = AdamW8bit(trainable_params, lr=lr, weight_decay=weight_decay)
                    logger.info("已创建 8-bit 优化器（bitsandbytes.AdamW8bit）以减小优化器内存占用。")
            except Exception as e:
                logger.warning(f"创建 8-bit 优化器失败，回退到默认优化器: {e}")

    if optimizer_to_pass is not None:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer_to_pass, None)
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    
    # 在训练前尝试回收无用内存并清理 CUDA 缓存，以降低 OOM 风险
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

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
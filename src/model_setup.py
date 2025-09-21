import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import os


def setup_model(model_path: str, tokenizer_name: str, r: int = 32, lora_alpha: int = 64, 
                target_modules: list = None):
    """
    设置模型和tokenizer
    
    Args:
        model_path: 模型路径
        tokenizer_name: tokenizer名称
        r: LoRA秩值
        lora_alpha: LoRA alpha值
        target_modules: 目标模块列表
        
    Returns:
        model, tokenizer: 配置好的模型和tokenizer
    """
    print("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # 如果tokenizer没有pad_token，设置一个
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("正在加载模型...")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 使用半精度以节省内存
        device_map="auto",  # 自动分配设备
        trust_remote_code=True
    )
    
    # 配置LoRA
    print("正在配置LoRA...")
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]  # 默认目标模块
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules
    )
    
    # 应用LoRA配置
    model = get_peft_model(model, peft_config)
    
    print("模型设置完成!")
    return model, tokenizer


# 测试代码
if __name__ == "__main__":
    print("模型设置模块已实现!")
import os
import torch
from peft import PeftModel


def save_lora_adapter(model, save_path: str, adapter_name: str = "default"):
    """
    保存LoRA适配器权重
    
    Args:
        model: 配置了LoRA的模型
        save_path: 保存路径
        adapter_name: 适配器名称
    """
    print(f"正在保存LoRA适配器 '{adapter_name}' 到 {save_path}")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 保存LoRA适配器
    model.save_pretrained(save_path, adapter_name=adapter_name)
    
    print("LoRA适配器保存完成!")


def save_model_weights(model, save_path: str, save_full_model: bool = False):
    """
    保存模型权重
    
    Args:
        model: 模型
        save_path: 保存路径
        save_full_model: 是否保存完整模型（包括基础模型）
    """
    print(f"正在保存模型权重到 {save_path}")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    if save_full_model:
        # 保存完整模型
        model.save_pretrained(save_path)
    else:
        # 只保存LoRA权重（如果模型是PeftModel）
        if isinstance(model, PeftModel):
            model.save_pretrained(save_path)
        else:
            # 对于普通模型，保存状态字典
            torch.save(model.state_dict(), os.path.join(save_path, "model_weights.pth"))
    
    print("模型权重保存完成!")


def save_training_state(model, optimizer, scheduler, epoch: int, save_path: str):
    """
    保存训练状态（模型、优化器、调度器状态）
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 调度器
        epoch: 当前epoch
        save_path: 保存路径
    """
    print(f"正在保存训练状态到 {save_path}")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 准备保存的状态字典
    state_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    
    # 保存状态字典
    torch.save(state_dict, os.path.join(save_path, "training_state.pth"))
    
    print("训练状态保存完成!")


# 测试代码
if __name__ == "__main__":
    print("模型保存功能模块已实现!")
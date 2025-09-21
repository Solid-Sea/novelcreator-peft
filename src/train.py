import os
import sys
import argparse
from data_preprocessing import preprocess_data, get_data_loaders
from model_setup import setup_model
from training_loop import train_model


def main():
    """主训练函数"""
    print("开始LoRA微调训练...")
    print("=" * 50)
    
    # 设置参数
    data_dir = "../data"  # 数据目录
    model_path = "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit"  # 模型路径
    tokenizer_name = "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit"  # Tokenizer名称
    output_dir = "../output"  # 输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 数据预处理
    print("步骤1: 数据预处理")
    train_dataset, val_dataset = preprocess_data(
        data_dir=data_dir,
        tokenizer_name=tokenizer_name,
        max_length=256,  # 减少序列长度以节省内存
        train_ratio=0.9
    )
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=1  # 批次大小
    )
    
    print(f"训练数据批次数量: {len(train_loader)}")
    print(f"验证数据批次数量: {len(val_loader)}")
    
    # 2. 模型设置
    print("\n步骤2: 模型设置")
    model, tokenizer = setup_model(
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        r=32,  # LoRA秩值
        lora_alpha=64,  # LoRA alpha值
        target_modules=["q_proj", "v_proj"]  # 目标模块
    )
    
    # 3. 训练模型
    print("\n步骤3: 开始训练")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,  # 训练轮数
        learning_rate=2e-4,  # 学习率
        weight_decay=0.01,  # 权重衰减
        gradient_clip=1.0,  # 梯度裁剪
        save_dir=os.path.join(output_dir, "checkpoints"),  # 模型保存目录
        save_every=1  # 每个epoch保存一次
    )
    
    # 4. 保存最终模型
    print("\n步骤4: 保存最终模型")
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    print("\n训练完成!")


if __name__ == "__main__":
    main()
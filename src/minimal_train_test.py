import os
import sys
import argparse
from data_preprocessing import preprocess_data, get_data_loaders, read_novel_files, clean_text, split_text_into_chunks
from model_setup import setup_model
from training_loop import train_model
from torch.utils.data import Dataset
import random


class MinimalNovelDataset(Dataset):
    """极小规模小说数据集类"""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def minimal_preprocess_data(data_dir: str, tokenizer_name: str = "gpt2",
                           max_length: int = 128, train_ratio: float = 0.5, sample_count: int = 3) -> tuple:
    """
    极小规模数据预处理函数
    
    Args:
        data_dir: 数据目录路径
        tokenizer_name: tokenizer名称
        max_length: 序列最大长度
        train_ratio: 训练集比例
        sample_count: 采样文件数量
        
    Returns:
        训练数据集和验证数据集
    """
    # 读取小说文件
    print("正在读取小说文件...")
    texts = read_novel_files(data_dir)
    print(f"共读取 {len(texts)} 个文件")
    
    # 只取少量样本
    if len(texts) > sample_count:
        texts = random.sample(texts, sample_count)
        print(f"采样 {sample_count} 个文件用于训练测试")
    
    # 文本清洗和分段
    print("正在进行文本清洗和分段...")
    all_chunks = []
    for text in texts:
        # 清洗文本
        cleaned_text = clean_text(text)
        if cleaned_text:
            # 分段
            chunks = split_text_into_chunks(cleaned_text, max_length)
            all_chunks.extend(chunks[:2])  # 每个文件只取前2个段落
    
    print(f"共生成 {len(all_chunks)} 个文本段落")
    
    # 如果段落数量大于5，只取前5个
    if len(all_chunks) > 5:
        all_chunks = all_chunks[:5]
        print(f"限制段落数量为5个")
    
    # 划分训练集和验证集
    random.shuffle(all_chunks)
    split_idx = int(len(all_chunks) * train_ratio)
    
    train_texts = all_chunks[:max(1, split_idx)]  # 确保至少有一个训练样本
    val_texts = all_chunks[split_idx:] if split_idx < len(all_chunks) else all_chunks[-1:]
    
    print(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}")
    
    # 初始化tokenizer
    print("正在加载tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # 如果tokenizer没有pad_token，设置一个
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    train_dataset = MinimalNovelDataset(train_texts, tokenizer, max_length)
    val_dataset = MinimalNovelDataset(val_texts, tokenizer, max_length)
    
    return train_dataset, val_dataset


def main():
    """极小规模训练测试主函数"""
    print("开始极小规模LoRA微调训练测试...")
    print("=" * 50)
    
    # 设置参数
    data_dir = "../organized_novels"  # 数据目录
    model_path = "gpt2"  # 模型路径
    tokenizer_name = "gpt2"  # Tokenizer名称
    output_dir = "../output/minimal_test"  # 输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 数据预处理（极小规模）
    print("步骤1: 极小规模数据预处理")
    train_dataset, val_dataset = minimal_preprocess_data(
        data_dir=data_dir,
        tokenizer_name=tokenizer_name,
        max_length=128,  # 较短的序列长度
        train_ratio=0.5,  # 训练集比例
        sample_count=3  # 只使用3个样本文件
    )
    
    # 获取数据加载器（极小批次大小）
    train_loader, val_loader = get_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=1  # 极小批次大小
    )
    
    print(f"训练数据批次数量: {len(train_loader)}")
    print(f"验证数据批次数量: {len(val_loader)}")
    
    # 2. 模型设置
    print("\n步骤2: 模型设置")
    model, tokenizer = setup_model(
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        r=8,  # 较小的LoRA秩值以减少内存使用
        lora_alpha=16,  # 对应的alpha值
        target_modules=["c_attn"]  # 适合gpt2模型的目标模块
    )
    
    # 3. 训练模型（极小规模）
    print("\n步骤3: 开始极小规模训练")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,  # 限制训练轮数
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
    
    print("\n极小规模训练测试完成!")


if __name__ == "__main__":
    main()
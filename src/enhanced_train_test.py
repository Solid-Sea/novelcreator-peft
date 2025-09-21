import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data, get_data_loaders, read_novel_files, clean_text, split_text_into_chunks
from model_setup import setup_model
from training_loop import train_model
from torch.utils.data import Dataset
import random
import torch


class EnhancedNovelDataset(Dataset):
    """增强版小说数据集类"""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 256):
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


def enhanced_preprocess_data(data_dir: str, tokenizer_name: str = "gpt2",
                           max_length: int = 256, train_ratio: float = 0.9, sample_count: int = 10) -> tuple:
    """
    增强版数据预处理函数
    
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
    
    # 采样文件（减少到10个以适应内存限制）
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
            # 每个文件取前3个段落
            all_chunks.extend(chunks[:3])
    
    print(f"共生成 {len(all_chunks)} 个文本段落")
    
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
    train_dataset = EnhancedNovelDataset(train_texts, tokenizer, max_length)
    val_dataset = EnhancedNovelDataset(val_texts, tokenizer, max_length)
    
    return train_dataset, val_dataset


def save_loss_plot(train_losses, val_losses, save_path):
    """
    保存损失函数变化图表
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='训练损失', marker='o')
    plt.plot(epochs, val_losses, 'r-', label='验证损失', marker='s')
    plt.title('训练和验证损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path)
    plt.close()
    print(f"损失函数变化图表已保存到: {save_path}")


def save_training_report(train_losses, val_losses, output_dir):
    """
    保存训练报告
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        output_dir: 输出目录
    """
    report = {
        "训练配置": {
            "训练轮数": len(train_losses),
            "训练集大小": "根据数据集确定",
            "验证集大小": "根据数据集确定",
            "批次大小": 2,
            "序列长度": 256,
            "学习率": 2e-4,
            "LoRA配置": {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["c_attn"]
            }
        },
        "训练过程": {
            "每个epoch的损失": []
        }
    }
    
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        report["训练过程"]["每个epoch的损失"].append({
            "epoch": i + 1,
            "训练损失": train_loss,
            "验证损失": val_loss
        })
    
    # 保存报告为JSON文件
    report_path = os.path.join(output_dir, "training_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"训练报告已保存到: {report_path}")
    
    # 保存损失图表
    plot_path = os.path.join(output_dir, "loss_curve.png")
    save_loss_plot(train_losses, val_losses, plot_path)


def test_model_output(model, tokenizer, test_prompts, output_dir):
    """
    测试模型输出能力
    
    Args:
        model: 训练好的模型
        tokenizer: tokenizer
        test_prompts: 测试提示列表
        output_dir: 输出目录
    """
    from text_generator import TextGenerator
    
    # 创建文本生成器
    generator = TextGenerator(model, tokenizer)
    
    # 设置生成参数
    generator.set_generation_params(max_length=200, temperature=0.7, top_p=0.9, top_k=50)
    
    # 生成文本示例
    generated_texts = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n正在生成第 {i+1} 个示例...")
        try:
            generated_text = generator.generate(prompt)
            generated_texts.append({
                "提示": prompt,
                "生成结果": generated_text
            })
            print(f"提示: {prompt}")
            print(f"生成结果: {generated_text}")
            print("-" * 50)
        except Exception as e:
            print(f"生成文本时出错: {e}")
            generated_texts.append({
                "提示": prompt,
                "生成结果": f"生成失败: {e}"
            })
    
    # 保存生成的文本示例
    examples_path = os.path.join(output_dir, "generated_examples.json")
    with open(examples_path, 'w', encoding='utf-8') as f:
        json.dump(generated_texts, f, ensure_ascii=False, indent=2)
    
    print(f"生成的文本示例已保存到: {examples_path}")


def main():
    """增强版训练测试主函数"""
    print("开始增强版LoRA微调训练测试...")
    print("=" * 50)
    
    # 设置参数
    data_dir = "../organized_novels"  # 数据目录
    model_path = "gpt2"  # 使用较小的模型以适应内存限制
    tokenizer_name = "gpt2"  # Tokenizer名称
    output_dir = "../output/enhanced_test"  # 输出目录
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 数据预处理（增强版）
    print("步骤1: 增强版数据预处理")
    train_dataset, val_dataset = enhanced_preprocess_data(
        data_dir=data_dir,
        tokenizer_name=tokenizer_name,
        max_length=256,  # 减少序列长度到256
        train_ratio=0.9,
        sample_count=10 # 使用10个样本文件
    )
    
    # 获取数据加载器（减少批次大小）
    train_loader, val_loader = get_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=2  # 减少批次大小到2
    )
    
    print(f"训练数据批次数量: {len(train_loader)}")
    print(f"验证数据批次数量: {len(val_loader)}")
    
    # 2. 模型设置
    print("\n步骤2: 模型设置")
    model, tokenizer = setup_model(
        model_path=model_path,
        tokenizer_name=tokenizer_name,
        r=8,  # 减少LoRA秩值到8
        lora_alpha=16,  # 对应的alpha值
        target_modules=["c_attn"]  # 适合gpt2模型的目标模块
    )
    
    # 3. 训练模型（增强版）
    print("\n步骤3: 开始增强版训练")
    
    # 训练参数
    num_epochs = 3  # 减少训练轮数到3轮
    
    # 训练模型并获取损失值
    train_losses = []
    val_losses = []
    
    # 修改训练循环以记录损失
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    # 计算总训练步数
    total_steps = len(train_loader) * num_epochs
    
    # 设置学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 设置损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    # 创建保存目录
    save_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            # 打印进度
            if (batch_idx + 1) % 5 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"平均训练损失: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
                total_val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"平均验证损失: {avg_val_loss:.4f}")
        
        # 保存模型
        save_path = os.path.join(save_dir, f"epoch_{epoch + 1}")
        os.makedirs(save_path, exist_ok=True)
        
        # 保存LoRA适配器
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")
    
    print("\n训练完成!")
    
    # 4. 保存最终模型
    print("\n步骤4: 保存最终模型")
    final_model_path = os.path.join(output_dir, "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 5. 保存训练报告和损失图表
    print("\n步骤5: 生成训练报告")
    save_training_report(train_losses, val_losses, output_dir)
    
    # 6. 测试模型输出能力
    print("\n步骤6: 测试模型输出能力")
    
    # 定义测试提示
    test_prompts = [
        "在遥远的未来，人类已经掌握了星际旅行的技术",
        "在一个被遗忘的小镇上，住着一位神秘的老人",
        "当第一缕阳光穿过云层，照亮了大地"
    ]
    
    test_model_output(model, tokenizer, test_prompts, output_dir)
    
    print("\n增强版训练测试完成!")


if __name__ == "__main__":
    main()
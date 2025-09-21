import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model_saver import save_training_state
import os


def train_model(model, train_loader, val_loader, num_epochs: int, learning_rate: float,
                weight_decay: float, gradient_clip: float, save_dir: str, save_every: int):
    """
    训练模型主循环
    
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        gradient_clip: 梯度裁剪值
        save_dir: 模型保存目录
        save_every: 每多少个epoch保存一次
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 计算总训练步数
    total_steps = len(train_loader) * num_epochs
    
    # 设置学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建保存目录
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_loader)
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
        print(f"平均验证损失: {avg_val_loss:.4f}")
        
        # 保存模型
        if (epoch + 1) % save_every == 0:
            save_path = os.path.join(save_dir, f"epoch_{epoch + 1}")
            save_training_state(model, optimizer, scheduler, epoch + 1, save_path)
            print(f"模型已保存到: {save_path}")
    
    print("\n训练完成!")


# 测试代码
if __name__ == "__main__":
    print("训练循环模块已实现!")
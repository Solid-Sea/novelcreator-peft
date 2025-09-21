import os
import re
import random
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class NovelDataset(Dataset):
    """小说数据集类"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
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


def read_novel_files(data_dir: str) -> List[str]:
    """
    读取data目录下的所有小说文件
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        所有小说文件内容的列表
    """
    texts = []
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    # 尝试不同的编码方式读取文件
                    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
                    content = None
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content is not None and content.strip():  # 只添加非空内容
                        texts.append(content)
                    elif content is None:
                        print(f"无法解码文件 {file_path}，跳过该文件")
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")
    
    return texts


def clean_text(text: str) -> str:
    """
    文本清洗函数
    
    Args:
        text: 原始文本
        
    Returns:
        清洗后的文本
    """
    # 去除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 去除特殊字符（保留中文、英文、数字和常见标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\u3000-\u303f\uff00-\uffef]+', '', text)
    
    # 去除首尾空白
    text = text.strip()
    
    return text


def split_text_into_chunks(text: str, chunk_size: int = 512) -> List[str]:
    """
    将文本分段
    
    Args:
        text: 输入文本
        chunk_size: 每段的最大长度
        
    Returns:
        分段后的文本列表
    """
    # 按句子分割（简单按句号、问号、感叹号分割）
    sentences = re.split(r'[。！？]', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 如果当前段落加上新句子超过chunk_size，则保存当前段落
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += sentence + "。"  # 添加句号
    
    # 添加最后一个段落
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def preprocess_data(data_dir: str, tokenizer_name: str = "Qwen/Qwen3-8B", 
                   max_length: int = 512, train_ratio: float = 0.9) -> Tuple[Dataset, Dataset]:
    """
    数据预处理主函数
    
    Args:
        data_dir: 数据目录路径
        tokenizer_name: tokenizer名称
        max_length: 序列最大长度
        train_ratio: 训练集比例
        
    Returns:
        训练数据集和验证数据集
    """
    # 读取小说文件
    print("正在读取小说文件...")
    texts = read_novel_files(data_dir)
    print(f"共读取 {len(texts)} 个文件")
    
    # 文本清洗和分段
    print("正在进行文本清洗和分段...")
    all_chunks = []
    for text in texts:
        # 清洗文本
        cleaned_text = clean_text(text)
        if cleaned_text:
            # 分段
            chunks = split_text_into_chunks(cleaned_text, max_length)
            all_chunks.extend(chunks)
    
    print(f"共生成 {len(all_chunks)} 个文本段落")
    
    # 划分训练集和验证集
    random.shuffle(all_chunks)
    split_idx = int(len(all_chunks) * train_ratio)
    
    train_texts = all_chunks[:split_idx]
    val_texts = all_chunks[split_idx:]
    
    print(f"训练集大小: {len(train_texts)}, 验证集大小: {len(val_texts)}")
    
    # 初始化tokenizer
    print("正在加载tokenizer...")
    # 尝试从本地模型文件加载tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("../DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf", trust_remote_code=True)
    except Exception as e:
        print(f"从本地模型加载tokenizer失败: {e}")
        # 如果失败，尝试从HuggingFace加载
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # 如果tokenizer没有pad_token，设置一个
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    train_dataset = NovelDataset(train_texts, tokenizer, max_length)
    val_dataset = NovelDataset(val_texts, tokenizer, max_length)
    
    return train_dataset, val_dataset


def get_data_loaders(train_dataset: Dataset, val_dataset: Dataset, 
                    batch_size: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    获取数据加载器
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: 批次大小
        
    Returns:
        训练和验证数据加载器
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 在Windows上使用0以避免问题
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


# 测试代码
if __name__ == "__main__":
    # 示例用法
    train_dataset, val_dataset = preprocess_data("../data", train_ratio=0.9)
    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset, batch_size=4)
    
    print("数据预处理完成！")
    print(f"训练批次数量: {len(train_loader)}")
    print(f"验证批次数量: {len(val_loader)}")
import torch
from typing import Optional


class TextGenerator:
    """文本生成器类"""
    
    def __init__(self, model, tokenizer):
        """
        初始化文本生成器
        
        Args:
            model: 加载的模型
            tokenizer: 加载的tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device if hasattr(model, 'device') else torch.device("cpu")
        
        # 默认生成参数
        self.max_length = 512
        self.temperature = 0.7
        self.top_p = 0.9
        self.top_k = 50
    
    def set_generation_params(self, max_length: int = 512, temperature: float = 0.7, 
                             top_p: float = 0.9, top_k: int = 50):
        """
        设置生成参数
        
        Args:
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: top-p采样参数
            top_k: top-k采样参数
        """
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
    
    def post_process_text(self, text: str, prompt: str) -> str:
        """
        后处理生成的文本
        
        Args:
            text: 生成的文本
            prompt: 提示文本
            
        Returns:
            处理后的文本
        """
        # 移除提示部分，只保留生成的部分
        if text.startswith(prompt):
            text = text[len(prompt):]
        
        # 移除开头和结尾的空白字符
        text = text.strip()
        
        # 移除可能的特殊标记
        special_tokens = ['<|endoftext|>', '<|end|>', '</s>']
        for token in special_tokens:
            text = text.replace(token, '')
        
        # 移除多余的换行符
        lines = text.split('\n')
        processed_lines = []
        empty_line_count = 0
        
        for line in lines:
            if line.strip() == '':
                empty_line_count += 1
                # 最多保留两个连续的空行
                if empty_line_count <= 2:
                    processed_lines.append(line)
            else:
                empty_line_count = 0
                processed_lines.append(line)
        
        text = '\n'.join(processed_lines).strip()
        
        return text
    
    def generate(self, prompt: str, max_length: Optional[int] = None, 
                temperature: Optional[float] = None, top_p: Optional[float] = None, 
                top_k: Optional[int] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示文本
            max_length: 最大生成长度（可选，覆盖默认值）
            temperature: 温度参数（可选，覆盖默认值）
            top_p: top-p采样参数（可选，覆盖默认值）
            top_k: top-k采样参数（可选，覆盖默认值）
            
        Returns:
            生成的文本
        """
        # 使用传入的参数或默认参数
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        
        # 编码提示文本
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # 检查输入长度，确保不超过最大长度
        input_length = inputs.shape[1]
        if input_length >= max_length:
            print(f"警告: 输入长度({input_length})已达到或超过最大长度({max_length})")
            max_length = input_length + 10  # 至少生成一些内容
        
        # 生成文本
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 后处理文本
        processed_text = self.post_process_text(generated_text, prompt)
        
        return processed_text


# 测试代码
if __name__ == "__main__":
    print("文本生成器模块已实现!")
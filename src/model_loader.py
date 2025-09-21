import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os


class ModelLoader:
    """模型加载器类"""
    
    def __init__(self, base_model_path: str = "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit", lora_weights_path: str = None):
        """
        初始化模型加载器
        
        Args:
            base_model_path: 基础模型路径
            lora_weights_path: LoRA权重路径（可选）
        """
        self.base_model_path = base_model_path
        self.lora_weights_path = lora_weights_path
        self.model = None
        self.tokenizer = None
    
    def load_base_model(self):
        """加载基础模型"""
        print("正在加载基础模型...")
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        
        # 如果tokenizer没有pad_token，设置一个
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_path, trust_remote_code=True)
        
        
        print("基础模型加载完成!")
    
    def load_lora_weights(self):
        """加载LoRA权重"""
        if self.lora_weights_path is None:
            print("未指定LoRA权重路径，跳过加载")
            return
        
        if not os.path.exists(self.lora_weights_path):
            raise FileNotFoundError(f"LoRA权重文件不存在: {self.lora_weights_path}")
        
        print("正在加载LoRA权重...")
        self.model = PeftModel.from_pretrained(
            self.model, 
            self.lora_weights_path,
            torch_dtype=torch.float16
        )
        print("LoRA权重加载完成!")
    
    def setup_inference(self, max_length: int = 512, temperature: float = 0.7, 
                       top_p: float = 0.9, top_k: int = 50):
        """
        设置推理参数
        
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
        print("推理参数设置完成!")
    
    def load_model(self):
        """加载完整模型（基础模型 + LoRA权重）"""
        # 加载基础模型
        self.load_base_model()
        
        # 加载LoRA权重（如果指定了路径）
        if self.lora_weights_path:
            self.load_lora_weights()
        
        print("模型加载完成!")
        return self.model, self.tokenizer


# 测试代码
if __name__ == "__main__":
    # 示例用法
    base_model_path = "../DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf"
    # lora_weights_path = "../output/final_model"  # 如果有微调后的权重
    
    loader = ModelLoader(base_model_path)
    model, tokenizer = loader.load_model()
    loader.setup_inference(max_length=512, temperature=0.7)
    
    print("模型加载器测试完成!")
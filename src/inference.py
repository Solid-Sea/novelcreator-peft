import os
import sys
from model_loader import ModelLoader
from text_generator import TextGenerator
from cli_interface import CLIInterface, parse_arguments


def create_generator(model_path: str, lora_path: str = None, max_length: int = 512, 
                    temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50):
    """
    创建文本生成器
    
    Args:
        model_path: 基础模型路径
        lora_path: LoRA权重路径（可选）
        max_length: 最大生成长度
        temperature: 温度参数
        top_p: top-p采样参数
        top_k: top-k采样参数
        
    Returns:
        TextGenerator: 文本生成器实例
    """
    # 加载模型
    loader = ModelLoader(model_path, lora_path)
    model, tokenizer = loader.load_model()
    
    # 设置推理参数
    loader.setup_inference(max_length, temperature, top_p, top_k)
    
    # 创建文本生成器
    generator = TextGenerator(model, tokenizer)
    generator.set_generation_params(max_length, temperature, top_p, top_k)
    
    return generator


def generate_text_wrapper(generator: TextGenerator):
    """
    创建文本生成包装函数
    
    Args:
        generator: 文本生成器实例
        
    Returns:
        包装后的生成函数
    """
    def generate(prompt: str) -> str:
        return generator.generate(prompt)
    
    return generate


def main():
    """主函数"""
    print("正在启动小说创作助手...")
    
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        # 创建文本生成器
        # 如果没有指定模型路径，使用默认的4bit模型
        model_path = args.model_path if args.model_path != "../DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf" else "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit"
        
        generator = create_generator(
            model_path=model_path,
            lora_path=args.lora_path,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        
        # 创建CLI界面
        cli_interface = CLIInterface(generate_text_wrapper(generator))
        
        # 运行CLI界面
        cli_interface.run()
        
    except Exception as e:
        print(f"启动过程中出现错误: {e}")
        sys.exit(1)


# 测试代码
if __name__ == "__main__":
    main()
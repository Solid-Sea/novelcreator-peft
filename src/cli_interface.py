import sys
import argparse
from typing import Callable


class CLIInterface:
    """CLI交互界面类"""
    
    def __init__(self, generator_func: Callable):
        """
        初始化CLI界面
        
        Args:
            generator_func: 文本生成函数
        """
        self.generator_func = generator_func
        self.running = True
    
    def print_welcome(self):
        """打印欢迎信息"""
        print("=" * 60)
        print("欢迎使用小说创作助手!")
        print("=" * 60)
        print("您可以输入提示文本，AI将为您续写小说内容。")
        print("输入 'quit' 或 'exit' 退出程序。")
        print("输入 'help' 查看帮助信息。")
        print("=" * 60)
    
    def print_help(self):
        """打印帮助信息"""
        print("\n帮助信息:")
        print("-" * 30)
        print("1. 输入任意文本作为提示，AI将为您续写小说内容")
        print("2. 输入 'quit' 或 'exit' 退出程序")
        print("3. 输入 'help' 查看此帮助信息")
        print("4. 输入 'clear' 清除屏幕")
        print("-" * 30)
    
    def get_user_input(self) -> str:
        """
        获取用户输入
        
        Returns:
            用户输入的文本
        """
        try:
            user_input = input("\n请输入提示文本: ").strip()
            return user_input
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。")
            return "quit"
        except EOFError:
            print("\n\n输入结束。")
            return "quit"
    
    def process_command(self, user_input: str):
        """
        处理用户命令
        
        Args:
            user_input: 用户输入
        """
        if user_input.lower() in ['quit', 'exit']:
            self.running = False
            print("感谢使用小说创作助手，再见!")
            return
        
        if user_input.lower() == 'help':
            self.print_help()
            return
        
        if user_input.lower() == 'clear':
            # 清除屏幕（跨平台）
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            self.print_welcome()
            return
        
        if user_input:
            # 调用文本生成函数
            self.generate_text(user_input)
        else:
            print("输入不能为空，请重新输入。")
    
    def generate_text(self, prompt: str):
        """
        生成文本
        
        Args:
            prompt: 提示文本
        """
        print("\n正在生成文本，请稍候...")
        try:
            generated_text = self.generator_func(prompt)
            print("\n生成结果:")
            print("-" * 30)
            print(generated_text)
            print("-" * 30)
        except Exception as e:
            print(f"\n生成文本时出错: {e}")
    
    def run(self):
        """运行CLI界面"""
        self.print_welcome()
        
        while self.running:
            user_input = self.get_user_input()
            self.process_command(user_input)


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description="小说创作助手CLI")
    parser.add_argument(
        "--model-path",
        type=str,
        default="../DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf",
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="LoRA权重路径"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="最大生成长度"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="top-p采样参数"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="top-k采样参数"
    )
    
    return parser.parse_args()


# 测试代码
if __name__ == "__main__":
    print("CLI交互界面模块已实现!")
import torch
import transformers
from peft import PeftModel
import yaml
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def generate_text(model, tokenizer, prompt, max_new_tokens=256):
    """使用给定的模型和分词器生成文本"""
    # 格式化输入
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 将文本编码为输入ID
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成文本
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解码生成的ID
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 提取生成的回复部分
    # 注意：这里的切片逻辑是为了去除输入提示，可能需要根据模型的实际输出来微调
    try:
        # 找到 assistant 回复的起始位置
        assistant_prompt = "assistant\n"
        start_index = decoded.rfind(assistant_prompt)
        if start_index != -1:
            response = decoded[start_index + len(assistant_prompt):]
        else: # 如果模板不匹配，则使用原始的长度切片作为备用
            response = decoded[len(text)-len(prompt):] # 尝试一个更鲁棒的切片
    except:
        response = "Failed to decode or slice the response."

    return response.strip()

def main():
    """主函数，执行模型评测流程"""
    # 1. 设置命令行参数
    parser = argparse.ArgumentParser(description="评测微调后的 Qwen 模型")
    parser.add_argument(
        '--adapter_path',
        type=str,
        default="./output/qwen_qlora_model/final_model",
        help="LoRA 适配器权重路径"
    )
    args = parser.parse_args()

    # 2. 加载配置文件
    # 获取脚本所在目录，并拼接配置文件路径
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, '../config.yaml')
        config = load_config(config_path)
    except Exception as e:
        print(f"错误：无法加载配置文件 '../config.yaml'。请确保它与脚本在同一目录下。")
        print(f"详细错误: {e}")
        return

    # 从配置中获取模型缓存目录
    model_config = config.get("model_config", {})
    model_cache_dir = model_config.get("model_cache_dir")

    # 3. 模型加载
    # 3.1 创建量化配置
    try:
        compute_dtype = getattr(torch, config['quantization_config']['bnb_4bit_compute_dtype'])
    except AttributeError:
        print(f"错误: 无效的 torch dtype '{config['quantization_config']['bnb_4bit_compute_dtype']}'")
        return
        
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization_config']['load_in_4bit'],
        bnb_4bit_quant_type=config['quantization_config']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config['quantization_config']['bnb_4bit_use_double_quant'],
    )

    # 3.2 加载基础模型和分词器
    print("正在加载基础模型和分词器...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            config['model_name_or_path'],
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=model_cache_dir
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config['model_name_or_path'],
            trust_remote_code=True,
            cache_dir=model_cache_dir
        )
    except Exception as e:
        print(f"错误：加载基础模型 '{config['model_name_or_path']}' 失败。")
        print(f"详细错误: {e}")
        return
    
    # 3.3 加载微调后的模型
    print(f"正在从 '{args.adapter_path}' 加载 LoRA 适配器...")
    if not os.path.exists(args.adapter_path):
        print(f"错误：适配器路径 '{args.adapter_path}' 不存在。请检查路径或运行训练脚本。")
        return
        
    try:
        finetuned_model = PeftModel.from_pretrained(base_model, args.adapter_path)
        finetuned_model.eval()
    except Exception as e:
        print(f"错误：加载 PeftModel 失败。")
        print(f"详细错误: {e}")
        return
        
    print("模型加载完成。")

    # 4. 定义测试提示词
    test_prompts = [
        "续写这个故事：在一个被魔法遗忘的世界里，一位年轻的铁匠发现了一块会唱歌的古老金属...",
        "描写一个场景：黄昏时分，一座悬浮在云海之上的古老城市，归来的飞船穿梭在楼宇之间。",
        "以'剑气纵横三万里，一剑光寒十九洲'为开头，写一段武侠小说的打斗场面。",
        "创作一个角色：一个靠收集和出售“记忆”为生的神秘商人，他有什么样的过去？",
        "请你写一篇关于未来赛博朋克城市的小说开头，主角是一个刚刚失业的义体改造师。"
    ]

    # 5. 生成与对比
    for i, prompt in enumerate(test_prompts):
        print("="*80)
        print(f"提示词 {i+1}: {prompt}")
        print("="*80)

        print("\n--- 基础模型输出 ---")
        try:
            base_output = generate_text(base_model, tokenizer, prompt)
            print(base_output)
        except Exception as e:
            print(f"生成文本时出错: {e}")

        print("\n--- 微调模型输出 ---")
        try:
            finetuned_output = generate_text(finetuned_model, tokenizer, prompt)
            print(finetuned_output)
        except Exception as e:
            print(f"生成文本时出错: {e}")
        
        print("\n\n")

if __name__ == "__main__":
    main()
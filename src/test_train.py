#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Qwen 训练脚本的基本功能
验证配置加载、模块导入等基础功能
"""

import sys
import os
import tempfile
import yaml

# 设置控制台编码
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    try:
        import train
        print("[成功] 成功导入 train 模块")
        return True
    except ImportError as e:
        print(f"[失败] 导入 train 模块失败: {e}")
        return False

def test_config_loading():
    """测试配置加载功能"""
    print("测试配置加载...")
    
    # 创建临时配置文件
    test_config = {
        'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct',
        'dataset_path': 'test_dataset.jsonl',
        'output_dir': 'test_output',
        'quantization_config': {
            'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4'
        },
        'lora_config': {
            'r': 16,
            'lora_alpha': 32
        },
        'training_args': {
            'num_train_epochs': 1,
            'per_device_train_batch_size': 1
        }
    }
    
    # 保存临时配置文件
    temp_config_path = os.path.join(os.path.dirname(__file__), 'config_test.yaml')
    try:
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_config, f, default_flow_style=False, allow_unicode=True)
        
        # 临时替换配置文件路径进行测试
        import train
        original_config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
        
        # 备份原配置文件（如果存在）
        backup_needed = os.path.exists(original_config_path)
        if backup_needed:
            backup_path = original_config_path + '.backup'
            os.rename(original_config_path, backup_path)
        
        # 复制测试配置文件
        os.rename(temp_config_path, original_config_path)
        
        try:
            config = train.load_config()
            print("[成功] 成功加载配置文件")
            print(f"  - 模型路径: {config.get('model_name_or_path')}")
            print(f"  - 数据集路径: {config.get('dataset_path')}")
            print(f"  - 输出目录: {config.get('output_dir')}")
            return True
        except Exception as e:
            print(f"[失败] 加载配置文件失败: {e}")
            return False
        finally:
            # 恢复原配置文件
            if backup_needed:
                os.rename(original_config_path, temp_config_path)
                os.rename(backup_path, original_config_path)
            else:
                os.remove(original_config_path)
                
    except Exception as e:
        print(f"[失败] 创建测试配置文件失败: {e}")
        return False

def test_quantization_config():
    """测试量化配置创建"""
    print("测试量化配置创建...")
    try:
        import train
        
        test_quant_config = {
            'load_in_4bit': True,
            'bnb_4bit_quant_type': 'nf4',
            'bnb_4bit_compute_dtype': 'bfloat16',
            'bnb_4bit_use_double_quant': True
        }
        
        quantization_config = train.create_quantization_config(test_quant_config)
        print("[成功] 成功创建量化配置")
        print(f"  - 4位量化: {quantization_config.load_in_4bit}")
        print(f"  - 量化类型: {quantization_config.bnb_4bit_quant_type}")
        return True
    except Exception as e:
        print(f"[失败] 创建量化配置失败: {e}")
        return False

def test_lora_config():
    """测试 LoRA 配置"""
    print("测试 LoRA 配置...")
    try:
        from peft import LoraConfig, TaskType
        
        test_lora_config = {
            'r': 32,
            'lora_alpha': 64,
            'lora_dropout': 0.1,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
            'bias': "none"
        }
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=test_lora_config.get('r', 32),
            lora_alpha=test_lora_config.get('lora_alpha', 64),
            lora_dropout=test_lora_config.get('lora_dropout', 0.1),
            target_modules=test_lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias=test_lora_config.get('bias', "none")
        )
        
        print("[成功] 成功创建 LoRA 配置")
        print(f"  - LoRA rank: {peft_config.r}")
        print(f"  - LoRA alpha: {peft_config.lora_alpha}")
        print(f"  - 目标模块: {peft_config.target_modules}")
        return True
    except Exception as e:
        print(f"[失败] 创建 LoRA 配置失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("Qwen 训练脚本功能测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_quantization_config,
        test_lora_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        print("-" * 30)
    
    print()
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("[成功] 所有测试通过！训练脚本基础功能正常")
        return True
    else:
        print("[失败] 部分测试失败，请检查相关依赖和配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
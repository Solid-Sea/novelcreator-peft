import json

# 验证原始文件
try:
    with open('novel_creator_lora_finetuning.ipynb', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('原始文件 JSON 格式正确')
except json.JSONDecodeError as e:
    print(f'原始文件 JSON 格式错误: {e}')
except FileNotFoundError:
    print('原始文件不存在')

# 验证修复后的文件
try:
    with open('novel_creator_lora_finetuning_fixed.ipynb', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('修复后的文件 JSON 格式正确')
except json.JSONDecodeError as e:
    print(f'修复后的文件 JSON 格式错误: {e}')
except FileNotFoundError:
    print('修复后的文件不存在')
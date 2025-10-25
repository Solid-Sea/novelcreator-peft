# Qwen QLoRA 微调训练管道

这是一个专门为 Qwen 模型设计的 QLoRA（量化低秩适应）微调训练脚本，支持配置文件驱动的训练流程。

## 功能特性

- ✅ **Qwen 模型专用适配**：针对 Qwen 系列模型优化的 ChatML 格式处理
- ✅ **4位量化训练**：使用 BitsAndBytesConfig 进行内存高效的量化训练
- ✅ **LoRA 微调**：支持低秩适应微调，大幅减少训练参数
- ✅ **配置文件驱动**：通过 YAML 配置文件管理所有训练参数
- ✅ **检查点恢复**：支持从中断点恢复训练
- ✅ **灵活的数据格式**：支持标准的指令微调数据格式

## 文件结构

```
qwen_finetune_pipeline/
├── train.py          # 主训练脚本
├── config.yaml       # 训练配置文件
├── test_train.py     # 功能测试脚本
└── README.md         # 说明文档
```

## 安装依赖

```bash
pip install torch transformers peft datasets accelerate bitsandbytes pyyaml
```

## 配置文件说明

`config.yaml` 包含以下主要配置项：

### 基础配置
```yaml
model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"  # 基础模型路径
dataset_path: "dataset/novel_finetuning_dataset.jsonl"  # 数据集路径
output_dir: "output/qwen_qlora_model"  # 输出目录
resume_from_checkpoint: true  # 是否从检查点恢复
```

### 量化配置
```yaml
quantization_config:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
```

### LoRA 配置
```yaml
lora_config:
  r: 32  # LoRA 秩
  lora_alpha: 64  # LoRA 缩放参数
  lora_dropout: 0.1  # Dropout 率
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]  # 目标模块
  bias: "none"
```

### 训练参数
```yaml
training_args:
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  logging_steps: 10
  save_steps: 100
  # ... 更多参数
```

## 数据格式

训练数据应为 JSONL 格式，每行包含以下字段：

```json
{
  "instruction": "请写一个关于春天的短诗",
  "input": "",
  "output": "春风轻拂柳絮飞，\n桃花满树映朝晖。\n燕子归来筑新巢，\n万物复苏展生机。",
  "system": "你是一个功底深厚的小说家，请根据指令进行创作。"
}
```

字段说明：
- `instruction`: 指令内容（必需）
- `input`: 输入内容（可选，为空时可省略）
- `output`: 期望输出（必需）
- `system`: 系统提示（可选，有默认值）

## 使用方法

### 1. 准备配置文件
复制并修改 `config.yaml` 文件，设置你的模型路径、数据集路径等参数。

### 2. 运行功能测试
```bash
cd qwen_finetune_pipeline
python test_train.py
```

### 3. 开始训练
```bash
cd qwen_finetune_pipeline
python train.py
```

### 4. 监控训练进度
训练日志会实时显示，包括：
- 损失值变化
- 学习率调整
- 检查点保存
- 训练进度

## 输出文件

训练完成后，会在输出目录生成：

```
output/qwen_qlora_model/
├── final_model/              # 最终合并的模型
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── training_config.json  # 训练配置记录
├── lora_adapter/             # LoRA 适配器权重
│   ├── adapter_config.json
│   └── adapter_model.bin
└── checkpoint-*/             # 训练检查点
```

## 高级功能

### 检查点恢复
如果训练中断，脚本会自动检测最新的检查点并恢复训练：

```python
# 在 config.yaml 中设置
resume_from_checkpoint: true
```

### 自定义 LoRA 目标模块
根据不同的 Qwen 模型版本，可能需要调整目标模块：

```yaml
lora_config:
  target_modules: 
    - "q_proj"
    - "k_proj" 
    - "v_proj"
    - "o_proj"
    - "gate_proj"  # 对于某些 Qwen 版本
    - "up_proj"
    - "down_proj"
```

### 内存优化
对于显存较小的 GPU，可以调整以下参数：

```yaml
training_args:
  per_device_train_batch_size: 1  # 减小批次大小
  gradient_accumulation_steps: 16  # 增加梯度累积
  gradient_checkpointing: true  # 启用梯度检查点
  dataloader_pin_memory: false  # 关闭内存锁定
```

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   - 减小 `per_device_train_batch_size`
   - 增加 `gradient_accumulation_steps`
   - 启用 `gradient_checkpointing`

2. **模型加载失败**
   - 检查 `model_name_or_path` 是否正确
   - 确认网络连接（如果从 HuggingFace 下载）
   - 验证本地模型文件完整性

3. **数据集格式错误**
   - 确认 JSONL 格式正确
   - 检查必需字段是否存在
   - 验证文件编码为 UTF-8

### 调试模式
启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能建议

- **GPU 内存**: 建议至少 8GB 显存用于 7B 模型的 4 位量化训练
- **系统内存**: 建议至少 16GB RAM
- **存储空间**: 预留足够空间存储检查点和最终模型

## 许可证

本项目遵循 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个训练管道。
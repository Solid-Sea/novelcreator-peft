# novelcreator-peft — Qwen QLoRA 微调训练管道

这是仓库 `novelcreator-peft` 中的 QLoRA（量化低秩适应）微调训练管道，主要针对 Qwen 系列模型，采用配置文件驱动的训练流程。

## 功能特性

- ✅ **Qwen 模型专用适配**：针对 Qwen 系列模型优化的 ChatML 格式处理
- ✅ **4位量化训练**：使用 BitsAndBytesConfig 进行内存高效的量化训练
- ✅ **LoRA 微调**：支持低秩适应微调，大幅减少训练参数
- ✅ **配置文件驱动**：通过 YAML 配置文件管理所有训练参数
- ✅ **检查点恢复**：支持从中断点恢复训练
- ✅ **灵活的数据格式**：支持标准的指令微调数据格式

## 仓库结构（概要）

```
.
├── config.yaml                  # 主配置（位于项目根）
├── config_test.yaml             # 测试 / 示例配置
├── novel_finetuning_dataset.jsonl
├── README.md
├── scripts/                     # 辅助脚本（run/train/eval 等）
├── src/
│   ├── train.py                 # 主训练脚本（从根目录的 config.yaml 加载配置）
│   ├── test_train.py            # 简单功能测试脚本
│   └── evaluate.py
└── src_old/                      # 旧版脚本备份
```

## 先决条件与依赖

推荐使用 Python 3.8+（建议 3.10+）。建议在虚拟环境中安装依赖。

示例（Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch transformers peft datasets accelerate bitsandbytes pyyaml
```

示例（bash / Linux / macOS）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers peft datasets accelerate bitsandbytes pyyaml
```

如果你有 `requirements.txt`，可以替换为 `pip install -r requirements.txt`。

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

## 使用方法（快速开始）

1) 准备配置文件

复制并根据需要修改根目录下的 `config.yaml`（或使用 `config_test.yaml` 做试验）。注意：`src/train.py` 会从仓库根目录读取 `config.yaml`，因此请确保根目录下存在该文件。

2) 运行功能测试

在项目根目录运行：

```powershell
# Windows PowerShell
python .\src\test_train.py
```

或在 bash：

```bash
python3 ./src/test_train.py
```

3) 开始训练

在项目根目录运行训练脚本（脚本会加载根目录的 `config.yaml`）：

```powershell
python .\src\train.py
```

或在 bash：

```bash
python3 ./src/train.py
```

4) 监控训练进度

训练过程中会输出日志信息，包括：损失、学习率、检查点保存与训练进度等。你也可以结合 `scripts/` 中的脚本来包装运行或监控（例如在远程机器上使用 tmux / screen）。

## 输出文件示例

训练结束后，会在 `config.yaml` 中指定的 `output_dir`（例如 `output/qwen_qlora_model`）下生成：

```
output/qwen_qlora_model/
├── final_model/              # 最终合并的模型
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── training_config.json  # 训练配置记录（自动保存）
├── lora_adapter/             # LoRA 适配器权重（可单独加载/分发）
└── checkpoint-*/             # 训练检查点（按 save_steps 保存）
```

## 高级功能

### 检查点恢复
当 `config.yaml` 中设置 `resume_from_checkpoint: true` 时，`src/train.py` 会在 `output_dir` 下查找最新的 `checkpoint-*` 并尝试从中恢复训练。

示例（config.yaml）：

```yaml
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

## 故障排除（常见问题）

1. CUDA 内存不足
  - 减小 `per_device_train_batch_size`
  - 增加 `gradient_accumulation_steps`
  - 启用 `gradient_checkpointing`

2. 模型加载失败
  - 检查 `model_name_or_path` 是否正确（本地路径或 HuggingFace 路径）
  - 确认网络连接以便从远程下载模型或检查缓存目录
  - 验证本地模型文件完整性

3. 数据集格式错误
  - 确认 JSONL 每行为合法 JSON
  - 检查每条记录包含 `instruction` 与 `output`（`input` 与 `system` 为可选）
  - 验证文件编码为 UTF-8

调试：在需要时启用更高日志等级，例如在代码中设置：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能建议

- GPU 内存: 建议至少 8GB 显存用于 7B 模型的 4 位量化训练（量化和 LoRA 可显著降低内存需求）
- 系统内存: 建议至少 16GB RAM
- 存储: 为检查点与最终模型预留充足空间（每个检查点可能占用数 GB）

## 许可证

本项目遵循 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request。如需贡献：

1. Fork 本仓库并新建分支
2. 在 `src/` 添加或修改功能，并编写相应说明
3. 提交 PR，描述改动与验证步骤

如需进一步文档改进或添加示例配置（如 `requirements.txt`、Dockerfile、或更详尽的训练脚本包装），请在 Issue 中提出。
# 小说创作助手 CLI 推理系统

这是一个基于微调模型的CLI交互式推理系统，允许用户与模型进行交互以生成小说内容。

## 功能特性

1. **模型加载模块**:
   - 加载基础模型 `DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf`
   - 加载微调后的LoRA权重（可选）
   - 实现模型推理设置（如最大生成长度、温度等参数）

2. **CLI交互界面**:
   - 命令行交互界面
   - 支持用户输入提示文本
   - 提供退出命令和帮助信息

3. **文本生成功能**:
   - 实现文本生成逻辑
   - 支持用户输入提示并生成续写
   - 处理生成文本的后处理

4. **主推理脚本**:
   - 整合所有模块，提供端到端的推理流程
   - 支持命令行参数配置

## 文件结构

```
src/
├── model_loader.py      # 模型加载模块
├── cli_interface.py     # CLI交互界面
├── text_generator.py    # 文本生成模块
├── inference.py         # 主推理脚本
├── model_setup.py       # 模型设置模块
├── training_loop.py     # 训练循环模块
├── model_saver.py       # 模型保存模块
├── data_preprocessing.py # 数据预处理模块
└── train.py             # 训练主脚本
```

## 使用方法

### 启动推理系统

```bash
cd src
python inference.py
```

### 命令行参数

```bash
python inference.py --help
```

可用参数:
- `--model-path`: 基础模型路径 (默认: ../DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf)
- `--lora-path`: LoRA权重路径 (可选)
- `--max-length`: 最大生成长度 (默认: 512)
- `--temperature`: 生成温度 (默认: 0.7)
- `--top-p`: top-p采样参数 (默认: 0.9)
- `--top-k`: top-k采样参数 (默认: 50)

### CLI交互命令

- 输入任意文本作为提示，AI将为您续写小说内容
- 输入 `quit` 或 `exit` 退出程序
- 输入 `help` 查看帮助信息
- 输入 `clear` 清除屏幕

## 开发说明

### 训练模型

如果需要训练模型，请运行:

```bash
cd src
python train.py
```

训练后的模型将保存在 `../output/` 目录中。

### 模块说明

1. **model_loader.py**: 负责加载基础模型和LoRA权重
2. **cli_interface.py**: 提供命令行交互界面
3. **text_generator.py**: 实现文本生成和后处理逻辑
4. **inference.py**: 主推理脚本，整合所有模块
5. **model_setup.py**: 模型设置模块，用于训练
6. **training_loop.py**: 训练循环模块
7. **model_saver.py**: 模型保存模块
8. **data_preprocessing.py**: 数据预处理模块
9. **train.py**: 训练主脚本

## 系统要求

- Python 3.8+
- transformers
- torch
- peft
- 其他依赖项请参考项目环境

## 注意事项

1. 基础模型文件 `DeepSeek-R1-0528-Qwen3-8B-Q4_0.gguf` 需要放置在项目根目录
2. 微调后的LoRA权重是可选的，如果没有可以只使用基础模型
3. 根据硬件配置调整生成参数以获得最佳性能
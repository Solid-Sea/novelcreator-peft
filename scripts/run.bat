@echo off
echo Starting Qwen finetuning...
:: 切换到批处理文件所在目录
cd /d "%~dp0"
python ../src/train.py
echo Finetuning finished.
pause
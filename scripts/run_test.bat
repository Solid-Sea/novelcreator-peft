@echo off
echo Starting Qwen finetuning test...
:: 切换到批处理文件所在目录
cd /d "%~dp0"
python ../src/train.py --config ../config_test.yaml
echo Finetuning test finished.
pause
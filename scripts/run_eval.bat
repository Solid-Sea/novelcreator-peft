@echo off
echo Starting model evaluation...
cd /d "%~dp0"
python ../src/evaluate.py
echo Evaluation finished.
pause
@echo off
cd /d "%~dp0"
python test_eval.py --turns 15 --profile qwen36_local %*
pause

@echo off
cd /d "%~dp0"
python test_eval.py --turns 4 --profile sonnet_local
pause

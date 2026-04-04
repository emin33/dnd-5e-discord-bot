@echo off
cd /d "%~dp0"
python test_eval.py --turns 15 --profile groq_full %*
pause

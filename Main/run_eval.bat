@echo off
REM venv, not bare `python`: system python is 3.10.11, below this
REM project's requires-python >= 3.11. These are the runs that cost
REM money, so they must not execute on an interpreter the suite itself
REM refuses. Same convention as run_tests.bat / run_typecheck.bat.
cd /d "%~dp0"
venv\Scripts\python.exe test_eval.py --turns 15 --profile qwen36_local %*
pause

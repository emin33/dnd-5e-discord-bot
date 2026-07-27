@echo off
REM venv-backed longform / soak runner. These runs cost money and their
REM results are the evidence behind release decisions, so they must not
REM execute on the interpreter tests/conftest.py refuses.
REM
REM   run_longform.bat --scenario emergent_callback --turns 22
REM   run_longform.bat --scenario deep_seeded_callback --profile deepseek_v4_flash_qwen9b
cd /d "%~dp0"
venv\Scripts\python.exe test_long_horizon.py %*
exit /b %ERRORLEVEL%

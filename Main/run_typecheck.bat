@echo off
cd /d "%~dp0"
venv\Scripts\python.exe -m mypy dnd_bot %*
exit /b %ERRORLEVEL%

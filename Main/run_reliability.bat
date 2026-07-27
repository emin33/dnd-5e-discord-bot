@echo off
REM venv-backed tool-reliability / lore_recall runner. Same reasoning as
REM run_longform.bat: model spend, and the artifacts feed soak_gate.
REM
REM   run_reliability.bat --scenario lore_recall
REM   run_reliability.bat --scenario player_state_sweep
cd /d "%~dp0"
venv\Scripts\python.exe test_tool_reliability.py %*
exit /b %ERRORLEVEL%

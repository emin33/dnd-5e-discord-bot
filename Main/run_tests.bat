@echo off
REM The authoritative way to run the suite. Use this, not a bare `pytest`.
REM
REM The system interpreter is Python 3.10.11 -- BELOW this project's own
REM requires-python (">=3.11") -- and its site-packages carries py-cord 2.7.1
REM AND discord.py 2.7.1, two distributions competing for the same `discord`
REM namespace. The result is not a clean failure: the suite runs, reports
REM green, and silently collects FOUR TESTS FEWER than it should, because
REM the Discord frontend tests cannot import `discord.ApplicationContext`.
REM
REM The venv is Python 3.13.14 with py-cord alone: 1617 collected against
REM the system interpreter's 1613. Same convention as run_typecheck.bat, and
REM the two now agree on which interpreter is authoritative.
cd /d "%~dp0"
venv\Scripts\python.exe -m pytest tests %*
exit /b %ERRORLEVEL%

@echo off
REM The authoritative way to run the suite. Use this, not a bare `pytest`.
REM
REM The system interpreter is Python 3.10.11 -- BELOW this project's own
REM requires-python (">=3.11") -- and its site-packages carries py-cord 2.7.1
REM AND discord.py 2.7.1, two distributions competing for the same `discord`
REM namespace. The result is not a clean failure: the suite runs, reports
REM green, and silently collects FOUR TESTS FEWER than it should, because
REM the Discord frontend tests cannot import `discord.ApplicationContext`.
REM Measured on one commit: system 1614, venv 1618 -- exactly those four.
REM tests/conftest.py now refuses the bad environment outright.
REM
cd /d "%~dp0"
REM `pytest %*` and not `pytest tests %*`: pyproject's testpaths already
REM defaults to tests/, and prepending it made a requested single file
REM collect the whole suite anyway.
venv\Scripts\python.exe -m pytest %*
exit /b %ERRORLEVEL%

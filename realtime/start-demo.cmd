@echo off
cd /d "%~dp0.."
".venv-realtime\Scripts\python.exe" -m realtime.scripts.windows_demo
if errorlevel 1 pause

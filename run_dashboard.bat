@echo off
cd /d "%~dp0"
.venv\Scripts\streamlit.exe run src\fpl_agent\ui\app.py
pause

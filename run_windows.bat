@echo off
REM Runs DocClean straight from source (no .exe needed).
cd /d "%~dp0"
if not exist .venv-win (
  python -m venv .venv-win
  call .venv-win\Scripts\activate.bat
  pip install -r requirements.txt
) else (
  call .venv-win\Scripts\activate.bat
)
start "" pythonw app.py

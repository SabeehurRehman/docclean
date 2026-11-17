@echo off
REM Builds DocClean.exe on Windows. Requires Python 3.10+ from python.org
REM (tick "Add python.exe to PATH" during install). Double-click this file.
cd /d "%~dp0"
echo Creating build environment...
python -m venv .venv-win || goto :error
call .venv-win\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller || goto :error
echo Building DocClean.exe ...
pyinstaller --noconfirm --clean --onefile --windowed --name DocClean ^
  --icon icon.ico --add-data "icon.ico;." --add-data "models\u2netp.onnx;models" app.py || goto :error
echo.
echo Done!  Your app is at:  %~dp0dist\DocClean.exe
explorer dist
pause
exit /b 0
:error
echo.
echo Build failed - see the messages above.
pause
exit /b 1

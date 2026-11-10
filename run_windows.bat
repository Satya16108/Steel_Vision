@echo off
REM SteelVision Desktop App — Windows launcher
REM Double-click this file to start the app

echo.
echo  ================================================================
echo   🏭  SteelVision — Steel Defect Detection App
echo  ================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

REM Change to app directory
cd /d "%~dp0"

REM Check if venv exists
if exist "venv\Scripts\activate.bat" (
    echo  Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo  No venv found — using system Python
)

REM Check if gradio is installed
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo  Installing requirements...
    pip install -r requirements.txt
)

echo.
echo  Starting app — browser will open automatically...
echo  URL: http://127.0.0.1:7860
echo.
python app.py

pause

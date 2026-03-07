@echo off
echo ====================================
echo Radar Backend - Start Script
echo ====================================
echo.

cd /d "%~dp0backend"

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting Backend Server...
echo.
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Health:    http://localhost:8000/health
echo.
echo Username: admin
echo Password: admin123
echo.
echo Press Ctrl+C to stop
echo ====================================
echo.

python main.py

pause

@echo off
echo ====================================
echo Radar Simulation Platform
echo ====================================
echo.

echo [1/4] Checking backend...
cd backend

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.8+
    pause
    exit /b 1
)

if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat

python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -q -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install deps
        pause
        exit /b 1
    )
)

if not exist ".env" copy .env.example .env >nul 2>&1

echo.
echo [2/4] Starting backend...
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.

start "Radar Backend" cmd /k "cd /d %CD% && venv\Scripts\activate.bat && python main.py"

timeout /t 3 /nobreak >nul

curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo WARNING: Backend may not be running
) else (
    echo Backend started!
)

cd ..

echo.
echo [3/4] Checking frontend...
cd frontend

where npm >nul 2>&1
if errorlevel 1 (
    echo ERROR: npm not found. Install Node.js
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo Installing frontend deps...
    call npm install
    if errorlevel 1 (
        echo ERROR: npm install failed
        pause
        exit /b 1
    )
)

echo.
echo [4/4] Starting frontend...
echo Frontend: http://localhost:5173
echo.

start "Radar Frontend" cmd /k "cd /d %CD% && npm run dev"

cd ..

echo.
echo ====================================
echo Started!
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo API Docs: http://localhost:8000/docs
echo.
echo Username: admin
echo Password: admin123
echo.
echo Press any key to close this window...
echo Services will keep running
echo ====================================
pause >nul

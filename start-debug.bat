@echo off
chcp 65001 >nul
echo ====================================
echo 雷达仿真平台启动脚本 (调试版)
echo ====================================
echo.

echo [步骤 1] 检查当前目录...
cd /d "%~dp0"
echo 当前目录: %CD%
echo.

echo [步骤 2] 检查 Python...
python --version 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python
    echo 请安装 Python 3.8 或更高版本
    pause
    exit /b 1
)
echo Python 已安装
echo.

echo [步骤 3] 进入后端目录...
cd backend
if errorlevel 1 (
    echo 错误: backend 目录不存在
    pause
    exit /b 1
)
echo 当前目录: %CD%
echo.

echo [步骤 4] 检查 main.py...
if not exist "main.py" (
    echo 错误: main.py 不存在
    pause
    exit /b 1
)
echo main.py 存在
echo.

echo [步骤 5] 启动后端服务...
echo.
echo ====================================
echo 后端服务正在启动...
echo 地址: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 可停止服务
echo ====================================
echo.

REM 直接启动，不在新窗口
python main.py

echo.
echo 后端服务已停止
pause

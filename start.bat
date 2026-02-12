@echo off
REM 相控阵雷达仿真平台 - 启动脚本

echo ========================================
echo 相控阵雷达仿真平台
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.10+
    pause
    exit /b 1
)

echo [1/3] 检查依赖...
pip show numpy >nul 2>&1
if errorlevel 1 (
    echo [信息] 依赖未安装，正在安装...
    pip install -r requirements.txt
)

echo [2/3] 启动后端系统...
python -m radar.main

echo.
echo 后端已关闭
pause

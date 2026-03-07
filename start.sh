#!/bin/bash

echo "===================================="
echo "雷达仿真平台启动脚本"
echo "===================================="

# 检查后端环境
echo ""
echo "[1/4] 检查后端环境..."
cd backend

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3，请先安装"
    exit 1
fi

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖（如果需要）
if ! python -c "import fastapi" &> /dev/null; then
    echo "安装后端依赖..."
    pip install -q -r requirements.txt
fi

# 创建.env文件（如果不存在）
if [ ! -f ".env" ]; then
    echo "创建配置文件..."
    cp .env.example .env 2>/dev/null || true
fi

echo ""
echo "[2/4] 启动后端服务..."
echo "后端服务运行在 http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo ""

# 启动后端（在后台）
python main.py &
BACKEND_PID=$!

# 等待后端启动
sleep 3

# 检查后端是否启动
if curl -s http://localhost:8000/health > /dev/null; then
    echo "后端服务启动成功！"
else
    echo "警告: 后端服务可能未正常启动"
fi

cd ..

echo ""
echo "[3/4] 检查前端环境..."
cd frontend

# 检查npm
if ! command -v npm &> /dev/null; then
    echo "错误: 未找到 npm，请先安装 Node.js"
    exit 1
fi

# 安装依赖（如果需要）
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    npm install
fi

echo ""
echo "[4/4] 启动前端服务..."
echo "前端服务运行在 http://localhost:5173"
echo ""

# 启动前端（在后台）
npm run dev &
FRONTEND_PID=$!

cd ..

echo ""
echo "===================================="
echo "启动完成！"
echo ""
echo "后端服务: http://localhost:8000"
echo "前端界面: http://localhost:5173"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "默认账号: admin"
echo "默认密码: admin123"
echo ""
echo "按 Ctrl+C 停止服务"
echo "===================================="

# 等待用户中断
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

wait

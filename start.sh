#!/bin/bash
# 相控阵雷达仿真平台 - 启动脚本

echo "========================================"
echo "相控阵雷达仿真平台"
echo "========================================"
echo ""

# 检查Python版本
if ! command -v python3 &> /dev/null
then
    echo "[错误] 未找到Python3"
    exit 1
fi

echo "[1/3] 检查依赖..."
if ! python3 -c "import numpy" &> /dev/null
then
    echo "[信息] 依赖未安装，正在安装..."
    pip3 install -r requirements.txt
fi

echo "[2/3] 启动后端系统..."
python3 -m radar.main

echo ""
echo "后端已关闭"

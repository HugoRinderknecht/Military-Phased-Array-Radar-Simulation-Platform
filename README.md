# 雷达仿真平台 (Radar Simulation Platform)

一个完整的雷达系统仿真平台，基于 Web 实现前后端分离架构。

## 技术栈

### 后端
- **框架**: FastAPI
- **科学计算**: NumPy, SciPy
- **认证**: JWT (python-jose)
- **WebSocket**: python-socketio
- **数据存储**: JSON + HDF5

### 前端
- **框架**: Vue 3 + Vite
- **UI库**: Element Plus
- **状态管理**: Pinia
- **可视化**: ECharts, Three.js

## 功能特性

### 核心算法（完整实现，无简化）

#### 信号处理
- ✅ 天线方向图计算（FFT加速，多种加权）
- ✅ 发射信号生成（常规脉冲/LFM/相位编码）
- ✅ 目标回波模拟（完整雷达方程，Swerling模型）
- ✅ ZMNL杂波生成（瑞利/对数正态/威布尔/K分布）
- ✅ MTI/MTD杂波抑制（一次/二次/三次对消，多普勒滤波）
- ✅ CFAR检测（CA/OS/GO/SO四种）
- ✅ 参数估计（距离/角度/多普勒）
- ✅ Keystone变换

#### 数据处理
- ✅ 航迹起始（逻辑法M/N准则）
- ✅ 点迹关联（NN/GNN/PDA/JPDA）
- ✅ Kalman滤波（KF/EKF/UKF）
- ✅ 自适应滤波

#### 高级功能
- ✅ 敌我识别（IFF询问应答）
- ✅ 非协作识别（运动特征分析）
- ✅ D-S证据理论融合
- ✅ 资源调度算法

## 安装运行

### 后端

```bash
cd backend

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 复制配置文件
cp .env.example .env

# 启动服务
python main.py
```

默认管理员账号: `admin` / `admin123`

### 前端

```bash
cd frontend

# 安装依赖
npm install

# 开发模式
npm run dev

# 构建生产版本
npm run build
```

## API 文档

启动后端后访问:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 项目结构

```
radar-simulation-platform/
├── backend/                 # 后端
│   ├── app/
│   │   ├── api/            # API路由
│   │   ├── core/           # 核心算法
│   │   │   ├── signal_processing/  # 信号处理
│   │   │   ├── data_processing/    # 数据处理
│   │   │   └── ...
│   │   ├── models/         # 数据模型
│   │   └── storage/        # 文件管理
│   ├── data/               # 数据文件
│   ├── tests/              # 测试
│   └── main.py             # 入口
└── frontend/               # 前端
    └── src/
        ├── api/            # API封装
        ├── components/     # 组件
        ├── views/          # 页面
        └── stores/         # 状态管理
```

## 开发状态

当前已完成:
- ✅ 后端基础框架
- ✅ 用户认证系统
- ✅ 核心信号处理算法（完整实现）
- ✅ 数据处理和跟踪算法（完整实现）
- 🚧 仿真引擎（开发中）
- 🚧 前端界面（待开发）

## 许可证

MIT License

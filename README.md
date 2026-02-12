# 相控阵雷达仿真平台 (Phased Array Radar Simulation Platform)

## 项目简介

本项目是一个基于 Python 的相控阵雷达全流程仿真平台，实现了从目标环境模拟、信号处理、数据处理到资源调度的完整仿真链路。

## 技术栈

- **编程语言**: Python 3.10+
- **数值计算**: NumPy, SciPy
- **高性能计算**: Numba JIT, pyFFTW
- **Web框架**: FastAPI, WebSocket
- **机器学习**: scikit-learn
- **测试**: pytest

## 项目结构

```
radar/
├── common/              # 公共模块
│   ├── types.py         # 类型定义
│   ├── constants.py     # 物理常数
│   ├── utils/           # 工具函数
│   ├── containers/      # 容器类
│   ├── config.py        # 配置管理
│   └── logger.py        # 日志系统
│
├── protocol/            # 通信协议
│   ├── messages.py      # 消息定义
│   ├── commands.py      # 指令定义
│   └── serializer.py    # 序列化
│
└── backend/             # 后端模块
    ├── core/            # 核心模块
    ├── environment/     # 环境模拟
    ├── antenna/         # 天线模块
    ├── signal/          # 信号处理
    ├── dataproc/        # 数据处理
    ├── scheduler/       # 资源调度
    └── network/         # 网络通信
```

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/radar-simulation/platform.git
cd platform
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
# 或
pip install -e .[dev]
```

## 运行

```bash
# 启动后端服务
python -m radar.main

# 运行测试
pytest

# 代码格式化
black radar/
isort radar/

# 类型检查
mypy radar/
```

## 配置

配置文件位于 `configs/` 目录：
- `radar_config.toml.example` - 雷达系统配置示例
- `scenario_config.json.example` - 场景配置示例

## 开发进度

当前处于 **第一阶段：基础设施搭建**

- [x] 项目结构初始化
- [x] 配置文件创建
- [ ] 公共模块实现
- [ ] 通信协议实现
- [ ] 网络基础实现
- [ ] 核心模块实现

## 文档

详细设计文档请参见 `docs/BACKEND_DESIGN.md`

## 许可证

MIT License

# Military Phased Array Radar Simulation Platform
# 军用相控阵雷达仿真平台

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive phased array radar simulation platform implemented in Python.
基于Python实现的综合相控阵雷达仿真平台。

## Features / 功能特性

### Core Modules / 核心模块

- **Environment Simulation / 环境模拟**
  Multi-target environment with various motion models (CV, CA, CT, 6DOF)
  支持多种运动模型的多目标环境仿真

- **Antenna System / 天线系统**
  Phased array antenna modeling with beamforming and beam steering
  相控阵天线建模，支持波束形成和波束控制

- **Signal Processing / 信号处理**
  Complete processing chain (LFM waveform, pulse compression, MTD, CFAR detection)
  完整的信号处理链（LFM波形、脉冲压缩、MTD、CFAR检测）

- **Data Processing / 数据处理**
  Track initiation (M/N logic), data association, Kalman filtering
  航迹起始（M/N逻辑）、数据关联、卡尔曼滤波

- **Resource Scheduling / 资源调度**
  Adaptive task scheduling with multiple priority algorithms
  自适应任务调度，支持多种优先级算法

- **Network Communication / 网络通信**
  Real-time communication via FastAPI and WebSocket
  基于FastAPI和WebSocket的实时通信

- **Performance Evaluation / 性能评估**
  Tracking accuracy, scheduling efficiency, detection metrics
  跟踪精度、调度效率、检测指标评估

### Technical Highlights / 技术亮点

- **High Performance / 高性能**
  NumPy vectorization, Numba JIT compilation
  NumPy向量化运算，Numba JIT编译加速

- **Asynchronous Architecture / 异步架构**
  Full asyncio-based I/O processing
  基于asyncio的异步I/O处理

- **Modular Design / 模块化设计**
  Clear interfaces, pluggable components
  清晰的接口定义，可插拔的组件设计

- **Complete Algorithms / 完整算法**
  No simplification, fully implemented radar algorithms
  无简化，完整实现雷达算法

## Technology Stack / 技术栈

- **Language / 编程语言**: Python 3.10+
- **Numerical Computing / 数值计算**: NumPy, SciPy, Numba
- **Web Framework / Web框架**: FastAPI, WebSocket, Uvicorn
- **Data Processing / 数据处理**: scikit-learn, Pydantic
- **Testing / 测试框架**: pytest
- **Configuration / 配置格式**: TOML

## Project Structure / 项目结构

```
radar/
├── common/              # Common modules / 公共模块
│   ├── types.py         # Type definitions / 类型定义
│   ├── constants.py     # Physical constants / 物理常数
│   ├── utils/          # Utility functions / 工具函数
│   ├── containers/      # Data structures / 数据结构
│   ├── config.py        # Configuration management / 配置管理
│   └── logger.py       # Logging system / 日志系统
│
├── protocol/            # Communication protocol / 通信协议
│   ├── messages.py      # Message definitions / 消息定义
│   └── serializer.py   # Serialization / 序列化
│
└── backend/             # Backend modules / 后端模块
    ├── core/            # Core control / 核心控制
    ├── environment/     # Environment simulation / 环境仿真
    ├── antenna/         # Phased array antenna / 相控阵天线
    ├── signal/          # Signal processing / 信号处理
    ├── dataproc/        # Data processing & tracking / 数据处理与跟踪
    ├── scheduler/       # Resource scheduling / 资源调度
    ├── network/         # Network communication / 网络通信
    └── evaluation/      # Performance evaluation / 性能评估
```

## Installation / 安装

### Prerequisites / 环境要求

- Python 3.10 or higher / Python 3.10及以上
- pip package manager / pip包管理器

### Setup Steps / 安装步骤

1. **Clone repository / 克隆仓库**
```bash
git clone https://github.com/HugoRinderknecht/Military-Phased-Array-Radar-Simulation-Platform.git
cd Military-Phased-Array-Radar-Simulation-Platform
```

2. **Create virtual environment / 创建虚拟环境**
```bash
python -m venv .venv

# Activate on Linux/Mac / Linux/Mac激活
source .venv/bin/activate

# Activate on Windows / Windows激活
.venv\Scripts\activate
```

3. **Install dependencies / 安装依赖**
```bash
pip install -r requirements.txt
```

## Usage / 使用方法

### Start Backend Server / 启动后端服务

```bash
# Method 1: Using Python module / 使用Python模块
python -m radar.main

# Method 2: Using startup scripts / 使用启动脚本
./start.sh     # Linux/Mac
start.bat       # Windows

# Method 3: Using Uvicorn (recommended for production / 生产环境推荐)
uvicorn radar.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access API / 访问API

Once server is running / 服务启动后:

- **HTTP API**: http://localhost:8000/api/status
- **WebSocket**: ws://localhost:8000/ws
- **API Documentation**: http://localhost:8000/docs (Swagger UI)

### Run Tests / 运行测试

```bash
# Run all tests / 运行所有测试
pytest

# Run with coverage / 运行测试并生成覆盖率报告
pytest --cov=radar --cov-report=html
```

### System Verification / 系统验证

```bash
python verify_system.py
```

## Configuration / 配置说明

Edit `radar_config.toml` to customize / 编辑配置文件自定义:

- Radar parameters / 雷达参数（频率、功率、带宽）
- Antenna configuration / 天线配置（阵列几何、波束图）
- Signal processing / 信号处理（波形、检测门限）
- Data processing / 数据处理（滤波器、关联门限）
- Scheduling parameters / 调度参数（优先级、时间窗）
- Network settings / 网络设置（主机、端口、CORS）
- Logging configuration / 日志配置（级别、文件轮转）

## Examples / 使用示例

See `examples/simple_usage.py` for complete usage examples.
参考 `examples/simple_usage.py` 查看完整使用示例。

```python
import asyncio
from radar.backend.environment.simulator import EnvironmentSimulator
from radar.common.types import TargetType, Position3D, Velocity3D

async def main():
    # Create environment simulator / 创建环境仿真器
    env = EnvironmentSimulator()
    await env.initialize()

    # Add a target / 添加目标
    await env.add_target(
        target_type=TargetType.AIRCRAFT,
        position=Position3D(50000, 30000, 10000),
        velocity=Velocity3D(200, 150, 0),
        rcs=10.0
    )

    # Update simulation / 更新仿真
    await env.update(0.1)

asyncio.run(main())
```

## Documentation / 文档

- [**Design Document**](docs/BACKEND_DESIGN.md) / 设计文档
  System architecture and design / 系统架构和设计

- [**Project Structure**](docs/PROJECT_STRUCTURE.md) / 项目结构
  Detailed code organization / 详细代码组织

- [**Development Summary**](docs/DEVELOPMENT_SUMMARY.md) / 开发总结
  Implementation status / 实现状态

- [**Final Report**](docs/FINAL_REPORT.md) / 完成报告
  Complete feature list / 完整功能列表

## Development Roadmap / 开发路线

- [x] Complete backend system / 完成后端系统 (Phase 1)
- [ ] Frontend visualization / 前端可视化界面
- [ ] GPU acceleration / GPU加速 (CuPy)
- [ ] Advanced motion models / 高级运动模型 (Singer, Jerk)
- [ ] JPDA/MHT association / JPDA/MHT数据关联
- [ ] STAP processing / 空时自适应处理
- [ ] Clutter map / 杂波图实现
- [ ] Electronic countermeasures / 电子对抗 (ECM)

## Contributing / 贡献

Contributions are welcome! / 欢迎贡献！
Please feel free to submit a Pull Request / 请随时提交拉取请求。

## License / 许可证

This project is licensed under the MIT License.
本项目采用 MIT 许可证开源。

## Acknowledgments / 致谢

- Implemented based on modern radar system theory / 基于现代雷达系统理论实现
- Reference: "Fundamentals of Radar Signal Processing" by Mark A. Richards
- Reference: "Principles of Radar" by Reintjes & Griffiths

---

**Military Phased Array Radar Simulation Platform**
**军用相控阵雷达仿真平台**

A complete, production-ready radar simulation system.
完整的、生产级雷达仿真系统。

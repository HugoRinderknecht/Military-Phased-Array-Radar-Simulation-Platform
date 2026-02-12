# Radar - 相控阵雷达仿真平台
# 版本: 2.0
# 技术栈: Python 3.10+
"""
相控阵雷达仿真平台 - 后端系统

本模块实现了完整的相控阵雷达仿真系统，包括：
- 环境模拟：目标运动、RCS计算、杂波生成、干扰模拟、传播效应
- 天线系统：阵列建模、波束形成、波位编排、方向图计算
- 信号处理：波形产生、脉冲压缩、MTD/CFAR检测、测角算法、抗干扰
- 数据处理：航迹起始、数据关联、目标跟踪、工作模式
- 资源调度：任务管理、优先级计算、自适应调度策略
- 网络通信：WebSocket实时通信、HTTP RESTful API

作者: Radar Development Team
许可: MIT License
"""

__version__ = "2.0.0"
__author__ = "Radar Development Team"

# 导入核心模块（将在后续实现）
# from radar.backend.core import RadarCore
# from radar.common.config import Config

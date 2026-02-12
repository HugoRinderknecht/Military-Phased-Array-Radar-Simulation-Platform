# 相控阵雷达仿真平台 - 后端设计文档

**版本**：2.0  
**技术栈**：Python 3.10+

---

## 目录

- [1. 技术选型](#1-技术选型)
- [2. 模块结构](#2-模块结构)
- [3. 系统架构](#3-系统架构)
- [4. 核心模块设计](#4-核心模块设计)
- [5. 核心算法实现](#5-核心算法实现)
- [6. 接口设计](#6-接口设计)
- [7. 实施计划](#7-实施计划)

---

## 1. 技术选型

### 1.1 核心依赖

| 技术领域 | 技术方案 | 版本要求 | 选型理由 |
|----------|----------|----------|----------|
| 编程语言 | Python | 3.10+ | 快速开发、丰富的科学计算生态 |
| 数值计算 | NumPy | 1.24+ | 高效数组运算，C扩展底层 |
| 科学计算 | SciPy | 1.10+ | 信号处理、优化、统计 |
| 高性能计算 | Numba | 0.57+ | JIT编译，接近C性能 |
| 并行计算 | multiprocessing + joblib | 内置/1.3+ | 多进程并行 |
| FFT加速 | pyFFTW | 0.13+ | FFTW的Python绑定 |
| 机器学习 | scikit-learn | 1.2+ | SVM分类器 |
| Web框架 | FastAPI | 0.100+ | 异步高性能Web框架 |
| WebSocket | websockets | 11.0+ | 异步WebSocket通信 |
| 序列化 | msgpack / orjson | 1.0+ / 3.9+ | 高效二进制序列化 |
| 配置管理 | Pydantic | 2.0+ | 数据验证和设置管理 |
| 日志系统 | loguru | 0.7+ | 简洁强大的日志 |
| 异步IO | asyncio | 内置 | 异步编程支持 |
| 单元测试 | pytest | 7.4+ | 成熟测试框架 |
| 类型检查 | mypy | 1.4+ | 静态类型检查 |

### 1.2 requirements.txt

```
# 核心数值计算
numpy>=1.24.0
scipy>=1.10.0
numba>=0.57.0
pyfftw>=0.13.0

# 机器学习
scikit-learn>=1.2.0

# Web框架
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
websockets>=11.0
aiofiles>=23.0

# 数据处理
pydantic>=2.0.0
msgpack>=1.0.0
orjson>=3.9.0

# 日志
loguru>=0.7.0

# 配置
python-dotenv>=1.0.0
tomli>=2.0.0

# 测试
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-benchmark>=4.0.0

# 代码质量
black>=23.0.0
isort>=5.12.0
pylint>=2.17.0
flake8>=6.0.0
mypy>=1.4.0
```

---

## 2. 模块结构

```
radar/
├── __init__.py
├── main.py                         # 后端入口
│
├── common/                         # 公共模块
│   ├── __init__.py
│   ├── types.py                    # 类型定义
│   ├── constants.py                # 物理常数
│   ├── utils/                      # 工具函数
│   │   ├── __init__.py
│   │   ├── math_utils.py           # 数学工具
│   │   ├── coord_transform.py      # 坐标变换
│   │   └── signal_utils.py         # 信号工具
│   ├── containers/                 # 容器类
│   │   ├── __init__.py
│   │   ├── ring_buffer.py          # 环形缓冲区
│   │   └── object_pool.py          # 对象池
│   ├── config.py                   # 配置管理
│   ├── logger.py                   # 日志系统
│   └── exceptions.py               # 异常定义
│
├── protocol/                       # 通信协议
│   ├── __init__.py
│   ├── messages.py                 # 消息定义
│   ├── commands.py                 # 指令定义
│   ├── serializer.py               # 序列化
│   └── constants.py                # 协议常量
│
└── backend/                        # 后端模块
    ├── __init__.py
    │
    ├── core/                       # 核心模块
    │   ├── __init__.py
    │   ├── radar_core.py           # 雷达核心
    │   ├── simulation_engine.py    # 仿真引擎
    │   ├── time_manager.py         # 时间管理
    │   └── state_manager.py        # 状态管理
    │
    ├── environment/                # 环境模拟
    │   ├── __init__.py
    │   ├── simulator.py            # 环境模拟器
    │   ├── target/                 # 目标模型
    │   │   ├── __init__.py
    │   │   ├── target.py           # 目标类
    │   │   ├── target_manager.py   # 目标管理
    │   │   ├── motion_model.py     # 运动模型
    │   │   ├── six_dof_model.py    # 六自由度
    │   │   ├── trajectory.py       # 轨迹生成
    │   │   └── attitude.py         # 姿态计算
    │   ├── rcs/                    # RCS模型
    │   │   ├── __init__.py
    │   │   ├── rcs_model.py        # RCS基类
    │   │   ├── swerling_model.py   # Swerling模型
    │   │   ├── gtd_model.py        # GTD模型
    │   │   └── dynamic_rcs.py      # 动态RCS
    │   ├── clutter/                # 杂波模型
    │   │   ├── __init__.py
    │   │   ├── clutter_model.py    # 杂波基类
    │   │   ├── ground_clutter.py   # 地杂波
    │   │   ├── sea_clutter.py      # 海杂波
    │   │   ├── weather_clutter.py  # 气象杂波
    │   │   ├── amplitude_dist.py   # 幅度分布
    │   │   ├── power_spectrum.py   # 功率谱
    │   │   └── correlated_clutter.py # 相关杂波
    │   ├── jammer/                 # 干扰模型
    │   │   ├── __init__.py
    │   │   ├── jammer_model.py     # 干扰基类
    │   │   ├── noise_jammer.py     # 噪声干扰
    │   │   ├── rgpo.py             # 距离拖引
    │   │   └── vgpo.py             # 速度拖引
    │   ├── propagation/            # 传播模型
    │   │   ├── __init__.py
    │   │   ├── propagation_model.py # 传播基类
    │   │   ├── free_space.py       # 自由空间
    │   │   ├── atmospheric.py      # 大气损耗
    │   │   └── ionosphere.py       # 电离层效应
    │   ├── echo/                   # 回波生成
    │   │   ├── __init__.py
    │   │   ├── echo_generator.py   # 回波生成器
    │   │   ├── pulse_echo.py       # 脉冲回波
    │   │   └── pd_blind_zone.py    # PD遮挡
    │   └── scenario/               # 场景管理
    │       ├── __init__.py
    │       ├── scenario.py         # 场景类
    │       └── scenario_loader.py  # 场景加载
    │
    ├── antenna/                    # 天线模块
    │   ├── __init__.py
    │   ├── antenna_system.py       # 天线系统
    │   ├── array/                  # 阵列建模
    │   │   ├── __init__.py
    │   │   ├── array_antenna.py    # 阵列基类
    │   │   ├── linear_array.py     # 线阵
    │   │   ├── planar_array.py     # 面阵
    │   │   ├── circular_array.py   # 圆阵
    │   │   ├── element.py          # 阵元
    │   │   ├── density_weighting.py # 密度加权
    │   │   └── sparse_array.py     # 稀布阵
    │   ├── beam/                   # 波束控制
    │   │   ├── __init__.py
    │   │   ├── beamformer.py       # 波束形成
    │   │   ├── beam_pattern.py     # 波束方向图
    │   │   ├── sum_beam.py         # 和波束
    │   │   ├── diff_beam.py        # 差波束
    │   │   └── beam_controller.py  # 波束控制器
    │   ├── scheduler/              # 波位编排
    │   │   ├── __init__.py
    │   │   ├── beam_scheduler.py   # 波位调度
    │   │   ├── sine_space.py       # 正弦空间
    │   │   ├── beam_position.py    # 波位定义
    │   │   ├── column_layout.py    # 列状编排
    │   │   ├── staggered_layout.py # 交错编排
    │   │   └── low_loss_layout.py  # 低损耗编排
    │   └── pattern/                # 方向图计算
    │       ├── __init__.py
    │       ├── pattern_calc.py     # 方向图计算
    │       └── sidelobe_analysis.py # 旁瓣分析
    │
    ├── signal/                     # 信号处理
    │   ├── __init__.py
    │   ├── signal_processor.py     # 信号处理器
    │   ├── waveform/               # 波形产生
    │   │   ├── __init__.py
    │   │   ├── waveform_gen.py     # 波形生成器
    │   │   ├── lfm_waveform.py     # LFM波形
    │   │   ├── phase_code_waveform.py # 相位编码
    │   │   ├── nlfm_waveform.py    # 非线性调频
    │   │   └── waveform_lib.py     # 波形库
    │   ├── receiver/               # 接收处理
    │   │   ├── __init__.py
    │   │   ├── receiver_chain.py   # 接收链
    │   │   ├── ionosphere_comp.py  # 电离层补偿
    │   │   └── agc.py              # 自动增益
    │   ├── compression/            # 脉冲压缩
    │   │   ├── __init__.py
    │   │   ├── pulse_compressor.py # 脉冲压缩
    │   │   ├── matched_filter.py   # 匹配滤波
    │   │   └── window_function.py  # 加窗函数
    │   ├── mtd/                    # 动目标检测
    │   │   ├── __init__.py
    │   │   ├── mtd_processor.py    # MTD处理器
    │   │   ├── mti_filter.py       # MTI滤波
    │   │   ├── doppler_bank.py     # 多普勒滤波组
    │   │   └── clutter_map.py      # 杂波图
    │   ├── detection/              # 检测
    │   │   ├── __init__.py
    │   │   ├── detector.py         # 检测器基类
    │   │   ├── cfar.py             # CFAR基类
    │   │   ├── ca_cfar.py          # CA-CFAR
    │   │   ├── os_cfar.py          # OS-CFAR
    │   │   ├── go_cfar.py          # GO-CFAR
    │   │   └── so_cfar.py          # SO-CFAR
    │   ├── integration/            # 积累
    │   │   ├── __init__.py
    │   │   ├── integrator.py       # 积累器
    │   │   ├── coherent_integrator.py # 相参积累
    │   │   ├── noncoherent_integrator.py # 非相参积累
    │   │   └── keystone.py         # Keystone变换
    │   ├── angle/                  # 测角
    │   │   ├── __init__.py
    │   │   ├── angle_estimator.py  # 测角基类
    │   │   ├── monopulse.py        # 单脉冲测角
    │   │   ├── amplitude_comp.py   # 比幅法
    │   │   ├── phase_comp.py       # 比相法
    │   │   ├── doa_estimator.py    # DOA估计
    │   │   ├── music.py            # MUSIC算法
    │   │   ├── imusic.py           # 改进MUSIC
    │   │   └── esprit.py           # ESPRIT算法
    │   ├── eccm/                   # 抗干扰
    │   │   ├── __init__.py
    │   │   ├── eccm_processor.py   # 抗干扰处理器
    │   │   ├── slc.py              # 旁瓣对消
    │   │   └── slc_lms.py          # LMS旁瓣对消
    │   └── plot/                   # 点迹处理
    │       ├── __init__.py
    │       ├── plot_extractor.py   # 点迹提取
    │       ├── plot_centroid.py    # 点迹质心
    │       └── plot_quality.py     # 点迹质量
    │
    ├── dataproc/                   # 数据处理
    │   ├── __init__.py
    │   ├── data_processor.py       # 数据处理器
    │   ├── preprocessing/          # 预处理
    │   │   ├── __init__.py
    │   │   ├── preprocessor.py     # 预处理器
    │   │   ├── svm_classifier.py   # SVM分类
    │   │   └── plot_filter.py      # 点迹过滤
    │   ├── track/                  # 航迹管理
    │   │   ├── __init__.py
    │   │   ├── track_manager.py    # 航迹管理器
    │   │   ├── track.py            # 航迹类
    │   │   ├── track_state.py      # 航迹状态
    │   │   └── track_quality.py    # 航迹质量
    │   ├── initiation/             # 航迹起始
    │   │   ├── __init__.py
    │   │   ├── track_initiator.py  # 起始器
    │   │   ├── rule_based_init.py  # 规则法
    │   │   ├── logic_init.py       # 逻辑法
    │   │   └── hough_init.py       # Hough变换
    │   ├── association/            # 数据关联
    │   │   ├── __init__.py
    │   │   ├── associator.py       # 关联器基类
    │   │   ├── nearest_neighbor.py # 最近邻
    │   │   ├── pda.py              # PDA
    │   │   ├── jpda.py             # JPDA
    │   │   ├── gnn.py              # GNN
    │   │   ├── tomht.py            # TOMHT
    │   │   └── hypothesis_tree.py  # 假设树
    │   ├── filter/                 # 滤波算法
    │   │   ├── __init__.py
    │   │   ├── base_filter.py      # 滤波器基类
    │   │   ├── kalman_filter.py    # 卡尔曼滤波
    │   │   ├── ekf.py              # 扩展卡尔曼
    │   │   ├── ukf.py              # 无迹卡尔曼
    │   │   ├── imm.py              # 交互多模型
    │   │   ├── imm_model.py        # IMM模型
    │   │   ├── cv_model.py         # CV模型
    │   │   ├── ca_model.py         # CA模型
    │   │   ├── ct_model.py         # CT模型
    │   │   └── auxiliary_filter.py # 辅助滤波器
    │   ├── phd/                    # PHD滤波
    │   │   ├── __init__.py
    │   │   ├── phd_filter.py       # PHD基类
    │   │   ├── gm_phd.py           # GM-PHD
    │   │   └── gm_component.py     # GM分量
    │   └── mode/                   # 工作模式
    │       ├── __init__.py
    │       ├── mode_controller.py  # 模式控制器
    │       ├── tws_mode.py         # TWS模式
    │       ├── tas_mode.py         # TAS模式
    │       └── hybrid_mode.py      # 复合模式
    │
    ├── scheduler/                  # 资源调度
    │   ├── __init__.py
    │   ├── scheduler.py            # 调度器基类
    │   ├── task/                   # 任务定义
    │   │   ├── __init__.py
    │   │   ├── task.py             # 任务基类
    │   │   ├── search_task.py      # 搜索任务
    │   │   ├── track_task.py       # 跟踪任务
    │   │   ├── verify_task.py      # 确认任务
    │   │   ├── acquire_task.py     # 截获任务
    │   │   └── guidance_task.py    # 制导任务
    │   ├── priority/               # 优先级
    │   │   ├── __init__.py
    │   │   ├── priority_calc.py    # 优先级计算
    │   │   ├── static_priority.py  # 静态优先级
    │   │   ├── dynamic_priority.py # 动态优先级
    │   │   ├── edf.py              # EDF
    │   │   ├── mhpf.py             # MHPF
    │   │   ├── medf.py             # MEDF
    │   │   ├── hpedf.py            # HPEDF
    │   │   └── threat_eval.py      # 威胁度评估
    │   ├── strategy/               # 调度策略
    │   │   ├── __init__.py
    │   │   ├── schedule_strategy.py # 策略基类
    │   │   ├── template_schedule.py # 模板调度
    │   │   ├── adaptive_schedule.py # 自适应调度
    │   │   ├── time_window.py      # 时间窗
    │   │   └── time_pointer.py     # 时间指针
    │   └── result/                 # 调度结果
    │       ├── __init__.py
    │       ├── schedule_result.py  # 调度结果
    │       └── schedule_stats.py   # 调度统计
    │
    ├── network/                    # 网络通信
    │   ├── __init__.py
    │   ├── network_manager.py      # 网络管理
    │   ├── websocket_server.py     # WebSocket服务器
    │   ├── http_server.py          # HTTP服务器
    │   ├── session.py              # 会话管理
    │   ├── critical_region.py      # 临界区
    │   ├── data_buffer.py          # 数据缓冲
    │   ├── command_dispatcher.py   # 指令分发
    │   └── data_publisher.py       # 数据发布
    │
    └── evaluation/                 # 效能评估
        ├── __init__.py
        ├── evaluator.py            # 评估器
        ├── tracking_error.py       # 跟踪误差
        ├── schedule_analysis.py    # 调度分析
        └── doa_evaluation.py       # DOA评估
```

---

## 3. 系统架构

### 3.1 架构分层

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           业务逻辑层 (Business Logic Layer)                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │                         RadarCore (核心控制器)                               ││
│  │                      Python + asyncio 异步框架                               ││
│  ├─────────────────────────────────────────────────────────────────────────────┤│
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         ││
│  │  │ 资源调度器  │  │ 时间管理器  │  │ 状态管理器  │                         ││
│  │  │ Scheduler   │  │ TimeManager │  │ StateManager│                         ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘                         ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────────────┐
│                           功能组件层 (Component Layer)                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                   │
│  │  环境模拟组件   │ │   天线组件      │ │  信号处理组件   │                   │
│  │ Environment     │ │   Antenna       │ │   SignalProc    │                   │
│  │ - 目标运动      │ │ - 阵列建模      │ │ - 波形产生      │                   │
│  │ - RCS计算       │ │ - 波束形成      │ │ - 脉冲压缩      │                   │
│  │ - 杂波生成      │ │ - 波位编排      │ │ - MTD/CFAR      │                   │
│  │ - 干扰模拟      │ │ - 方向图        │ │ - 测角算法      │                   │
│  │ - 传播效应      │ │                 │ │ - 抗干扰        │                   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘                   │
│  ┌─────────────────┐ ┌─────────────────┐                                       │
│  │  数据处理组件   │ │   效能评估      │                                       │
│  │ DataProc        │ │   Evaluation    │                                       │
│  │ - 航迹起始      │ │ - 精度统计      │                                       │
│  │ - 数据关联      │ │ - 调度分析      │                                       │
│  │ - 目标跟踪      │ │ - DOA评估       │                                       │
│  │ - 工作模式      │ │                 │                                       │
│  └─────────────────┘ └─────────────────┘                                       │
│                         NumPy + SciPy + Numba 高性能计算                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
┌───────────────────────────────────┼─────────────────────────────────────────────┐
│                           基础设施层 (Infrastructure Layer)                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │  数值计算   │ │  日志系统   │ │  配置管理   │ │  并行计算   │               │
│  │  NumPy      │ │  loguru     │ │  Pydantic   │ │  asyncio    │               │
│  │  SciPy      │ │             │ │  TOML/JSON  │ │  joblib     │               │
│  │  Numba JIT  │ │             │ │             │ │             │               │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │  Web框架    │ │  序列化     │ │  类型检查   │ │  单元测试   │               │
│  │  FastAPI    │ │  msgpack    │ │  mypy       │ │  pytest     │               │
│  │  Uvicorn    │ │  orjson     │ │             │ │  pytest-asyncio│             │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 后端异步架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              后端异步架构 (asyncio)                              │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                           主事件循环 (Main Event Loop)                     │ │
│  │  职责：协调所有异步任务、处理网络IO                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  asyncio.run(main())                                                 │  │ │
│  │  │  - 创建事件循环                                                      │  │ │
│  │  │  - 启动各异步任务                                                    │  │ │
│  │  │  - 管理任务生命周期                                                  │  │ │
│  │  └─────────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                          │
│          ┌───────────────────────────┼───────────────────────────┐             │
│          │                           │                           │             │
│          ▼                           ▼                           ▼             │
│  ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐    │
│  │  HTTP服务协程      │     │  WebSocket协程     │     │  仿真主循环协程    │    │
│  │ (FastAPI + Uvicorn)│     │ (websockets)      │     │ (Simulation Loop) │    │
│  ├───────────────────┤     ├───────────────────┤     ├───────────────────┤    │
│  │ 职责：             │     │ 职责：             │     │ 职责：             │    │
│  │ - RESTful API     │     │ - 实时数据推送     │     │ - 环境模拟         │    │
│  │ - 静态文件服务     │     │ - 指令接收         │     │ - 信号处理         │    │
│  │ - 配置管理API     │     │ - 心跳检测         │     │ - 数据处理         │    │
│  │                   │     │ - 会话管理         │     │ - 资源调度         │    │
│  │                   │     │                   │     │ - 效能评估         │    │
│  └─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘    │
│            │                         │                         │               │
│            └─────────────────────────┼─────────────────────────┘               │
│                                      │                                          │
│                        ┌─────────────┴─────────────┐                           │
│                        ▼                           ▼                           │
│              ┌───────────────────────────────────────────────┐                 │
│              │         共享数据区 (CriticalRegion)            │                 │
│              │  ┌─────────────────────────────────────────┐  │                 │
│              │  │         asyncio.Queue (异步队列)         │  │                 │
│              │  ├─────────────────────────────────────────┤  │                 │
│              │  │  指令队列:                              │  │                 │
│              │  │  - asyncio.Queue[Command] cmd_queue     │  │                 │
│              │  ├─────────────────────────────────────────┤  │                 │
│              │  │  数据缓冲:                              │  │                 │
│              │  │  - asyncio.Queue[PlotReport] plot_queue │  │                 │
│              │  │  - asyncio.Queue[TrackReport] track_queue│ │                 │
│              │  │  - asyncio.Queue[BeamStatus] beam_queue │  │                 │
│              │  ├─────────────────────────────────────────┤  │                 │
│              │  │  状态信息 (带锁):                       │  │                 │
│              │  │  - SimulationState current_state        │  │                 │
│              │  │  - asyncio.Lock state_lock              │  │                 │
│              │  └─────────────────────────────────────────┘  │                 │
│              └───────────────────────────────────────────────┘                 │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                         计算密集型任务 (多进程)                            │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  ProcessPoolExecutor (信号处理并行)                                  │  │ │
│  │  │  - Numba JIT加速的数值计算                                           │  │ │
│  │  │  - 独立进程池执行FFT、滤波等                                         │  │ │
│  │  │  - 通过队列与主进程通信                                              │  │ │
│  │  └─────────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 仿真主循环

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              仿真主循环流程                                      │
│                                                                                 │
│  ┌─────────┐                                                                    │
│  │  开始   │                                                                    │
│  └────┬────┘                                                                    │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          初始化阶段                                      │   │
│  │  1. 加载配置文件 (radar_config.toml, scenario_config.json)              │   │
│  │  2. 初始化各模块 (Environment, Antenna, Signal, DataProc, Scheduler)    │   │
│  │  3. 启动网络服务 (FastAPI + WebSocket)                                  │   │
│  │  4. 预热Numba JIT编译                                                   │   │
│  │  5. 初始化时间管理器                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          主循环 (asyncio异步)                            │   │
│  │                                                                         │   │
│  │  async def simulation_loop():                                          │   │
│  │      while self.m_running:                                             │   │
│  │          ┌─────────────────────────────────────────────────────────┐   │   │
│  │          │ Step 1: 处理前端指令 (从异步队列获取)                     │   │   │
│  │          │   cmd = await self.m_cmd_queue.get()                     │   │   │
│  │          │   await self.process_command(cmd)                        │   │   │
│  │          └─────────────────────────────────────────────────────────┘   │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │          ┌─────────────────────────────────────────────────────────┐   │   │
│  │          │ Step 2: 调度决策 (Scheduler)                             │   │   │
│  │          │   result = await self.m_scheduler.schedule(current_time)│   │   │
│  │          │   execute_queue = result.execute_queue                  │   │   │
│  │          └─────────────────────────────────────────────────────────┘   │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │          ┌─────────────────────────────────────────────────────────┐   │   │
│  │          │ Step 3: 环境模拟 (Environment)                           │   │   │
│  │          │   await self.m_environment.update(delta_time)           │   │   │
│  │          │   # 更新目标位置、计算RCS、生成杂波                      │   │   │
│  │          └─────────────────────────────────────────────────────────┘   │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │          ┌─────────────────────────────────────────────────────────┐   │   │
│  │          │ Step 4: 执行调度任务                                     │   │   │
│  │          │   for task in execute_queue:                             │   │   │
│  │          │       beam = self.m_antenna.steer_beam(task.direction)  │   │   │
│  │          │       echo = self.m_environment.generate_echo(beam)     │   │   │
│  │          │       plots = self.m_signal_proc.process(echo)          │   │   │
│  │          │       tracks = self.m_data_proc.process(plots)          │   │   │
│  │          └─────────────────────────────────────────────────────────┘   │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │          ┌─────────────────────────────────────────────────────────┐   │   │
│  │          │ Step 5: 效能评估                                         │   │   │
│  │          │   await self.m_evaluator.evaluate(tracks, true_targets) │   │   │
│  │          └─────────────────────────────────────────────────────────┘   │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │          ┌─────────────────────────────────────────────────────────┐   │   │
│  │          │ Step 6: 数据发布 (写入异步队列)                          │   │   │
│  │          │   await self.m_plot_queue.put(plots)                    │   │   │
│  │          │   await self.m_track_queue.put(tracks)                  │   │   │
│  │          │   await self.m_beam_queue.put(beam_status)              │   │   │
│  │          └─────────────────────────────────────────────────────────┘   │   │
│  │                              │                                         │   │
│  │                              ▼                                         │   │
│  │          ┌─────────────────────────────────────────────────────────┐   │   │
│  │          │ Step 7: 帧同步                                           │   │   │
│  │          │   await asyncio.sleep(frame_interval)                   │   │   │
│  │          │   self.m_time_manager.update()                          │   │   │
│  │          └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                              清理阶段                                    │   │
│  │  1. 停止所有异步任务                                                     │   │
│  │  2. 关闭网络连接                                                         │   │
│  │  3. 保存状态/日志                                                        │   │
│  │  4. 释放资源                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────┐                                                                    │
│  │  结束   │                                                                    │
│  └─────────┘                                                                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 核心模块设计

### 4.1 RadarCore 核心控制器

```python
from abc import ABC, abstractmethod
from typing import Optional, List
import asyncio

from radar.common.config import Config
from radar.protocol.commands import Command
from radar.protocol.messages import PlotReport, TrackReport, BeamStatus


class IRadarCore(ABC):
    """雷达核心控制器接口"""
    
    @abstractmethod
    async def initialize(self, config: Config) -> bool:
        """初始化雷达系统
        
        Args:
            config: 系统配置
            
        Returns:
            初始化是否成功
        """
        pass
    
    @abstractmethod
    async def run(self) -> None:
        """运行仿真主循环"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """关闭系统"""
        pass
    
    @abstractmethod
    async def pause(self) -> None:
        """暂停仿真"""
        pass
    
    @abstractmethod
    async def resume(self) -> None:
        """恢复仿真"""
        pass
    
    @abstractmethod
    def get_state(self) -> 'SimulationState':
        """获取当前仿真状态
        
        Returns:
            当前仿真状态
        """
        pass
    
    @abstractmethod
    async def process_command(self, cmd: Command) -> None:
        """处理前端指令
        
        Args:
            cmd: 控制指令
        """
        pass


class IEnvironmentSimulator(ABC):
    """环境模拟器接口"""
    
    @abstractmethod
    def initialize(self, config: 'EnvironmentConfig') -> None:
        """初始化环境模拟器"""
        pass
    
    @abstractmethod
    async def update(self, delta_time: float) -> None:
        """更新环境状态
        
        Args:
            delta_time: 时间步长（秒）
        """
        pass
    
    @abstractmethod
    def generate_echo(self, beam_info: 'BeamInfo') -> 'EchoData':
        """生成回波数据
        
        Args:
            beam_info: 当前波束信息
            
        Returns:
            回波复数数据
        """
        pass
    
    @abstractmethod
    def get_targets(self) -> List['Target']:
        """获取所有目标"""
        pass


class ISignalProcessor(ABC):
    """信号处理器接口"""
    
    @abstractmethod
    def initialize(self, config: 'SignalConfig') -> None:
        """初始化信号处理器"""
        pass
    
    @abstractmethod
    def process(self, echo_data: 'EchoData') -> List['Plot']:
        """处理回波数据
        
        Args:
            echo_data: 回波数据
            
        Returns:
            检测到的点迹列表
        """
        pass
    
    @abstractmethod
    def set_waveform(self, params: 'WaveformParams') -> None:
        """设置波形参数"""
        pass
    
    @abstractmethod
    def set_cfar_threshold(self, threshold: float) -> None:
        """设置CFAR检测门限"""
        pass


class IDataProcessor(ABC):
    """数据处理器接口"""
    
    @abstractmethod
    def initialize(self, config: 'DataProcConfig') -> None:
        """初始化数据处理器"""
        pass
    
    @abstractmethod
    def process(self, plots: List['Plot']) -> List['Track']:
        """处理点迹数据
        
        Args:
            plots: 点迹列表
            
        Returns:
            更新后的航迹列表
        """
        pass
    
    @abstractmethod
    def set_mode(self, mode: 'WorkMode') -> None:
        """设置工作模式"""
        pass
    
    @abstractmethod
    def manual_track(self, plot_id: int) -> None:
        """手动指派跟踪"""
        pass
    
    @abstractmethod
    def cancel_track(self, track_id: int) -> None:
        """取消跟踪"""
        pass


class IScheduler(ABC):
    """资源调度器接口"""
    
    @abstractmethod
    def initialize(self, config: 'ScheduleConfig') -> None:
        """初始化调度器"""
        pass
    
    @abstractmethod
    async def schedule(self, current_time: int) -> 'ScheduleResult':
        """执行调度决策
        
        Args:
            current_time: 当前仿真时间（微秒）
            
        Returns:
            调度结果
        """
        pass
    
    @abstractmethod
    def add_task(self, task: 'Task') -> None:
        """添加任务"""
        pass
    
    @abstractmethod
    def remove_task(self, task_id: int) -> None:
        """移除任务"""
        pass
    
    @abstractmethod
    def update_priority(self, task_id: int, priority: float) -> None:
        """更新任务优先级"""
        pass
```

---

## 5. 核心算法实现

### 5.1 目标运动模型

```python
@dataclass
class Target:
    """目标类 - 存储目标的所有属性和状态"""
    
    # 基本属性
    id: int                          # 目标ID
    name: str                        # 目标名称
    target_type: TargetType          # 目标类型 (AIRCRAFT/MISSILE/SHIP)
    
    # 位置信息 (ENU坐标系, 单位:米)
    position: np.ndarray             # [x, y, z] 位置向量
    velocity: np.ndarray             # [vx, vy, vz] 速度向量 (m/s)
    acceleration: np.ndarray         # [ax, ay, az] 加速度向量 (m/s²)
    
    # 姿态信息 (欧拉角, 单位:弧度)
    attitude: Attitude               # (roll, pitch, yaw)
    angular_velocity: np.ndarray     # [ωx, ωy, ωz] 角速度 (rad/s)
    
    # RCS属性
    mean_rcs: float                  # 平均RCS (m²)
    rcs_model: SwerlingModel         # Swerling起伏模型
    rcs_table: Optional[dict]        # 视角-RCS查表 (可选)
    
    # 运动模型
    motion_model: MotionModel        # 运动模型类型
    trajectory: Optional[Trajectory] # 预设轨迹 (可选)
    
    # 其他属性
    is_active: bool                  # 是否激活
    creation_time: float             # 创建时间 (仿真时间)


class MotionModel(Enum):
    """运动模型类型"""
    CONSTANT_VELOCITY = "cv"        # 匀速直线
    CONSTANT_ACCELERATION = "ca"    # 匀加速
    COORDINATED_TURN = "ct"         # 协调转弯
    CIRCULAR = "circular"           # 匀速圆周
    SIX_DOF = "6dof"                # 六自由度
    WAYPOINT = "waypoint"           # 航点跟随
    PREDEFINED = "predefined"       # 预设轨迹
```

### 5.2 RCS模型

```python
class SwerlingModel(Enum):
    """Swerling起伏模型"""
    I = 1     # 指数分布，扫描间相关
    II = 2    # 指数分布，脉冲间相关
    III = 3   # 4阶chi2分布，扫描间相关
    IV = 4    # 4阶chi2分布，脉冲间相关


def sample_swerling_rcs(mean_rcs: float, 
                        model: SwerlingModel,
                        is_new_scan: bool) -> float:
    """采样RCS值"""
    if model in [SwerlingModel.I, SwerlingModel.II]:
        # 指数分布: σ = -σ̄ * ln(U)
        u = np.random.uniform(0, 1)
        return -mean_rcs * np.log(u)
    elif model in [SwerlingModel.III, SwerlingModel.IV]:
        # Chi-square分布 (k=4): 使用两个独立指数分布的和
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        return -mean_rcs / 2 * (np.log(u1) + np.log(u2))
```

### 5.3 LFM波形生成

```python
def generate_lfm(
    sample_rate: float,      # 采样率 (Hz)
    pulse_width: float,      # 脉冲宽度 (s)
    bandwidth: float,        # 带宽 (Hz)
    center_freq: float = 0   # 中心频率 (Hz)，基带为0
) -> np.ndarray:
    """生成LFM波形"""
    n_samples = int(sample_rate * pulse_width)
    t = np.arange(n_samples) / sample_rate - pulse_width / 2
    
    # 调频斜率
    k = bandwidth / pulse_width
    
    # 相位
    phase = 2 * np.pi * (center_freq * t + 0.5 * k * t**2)
    
    # 复信号
    signal = np.exp(1j * phase)
    
    # 加窗 (可选)
    window = np.hamming(n_samples)
    signal = signal * window
    
    return signal
```

### 5.4 脉冲压缩

```python
class PulseCompressor:
    """脉冲压缩器"""
    
    def __init__(self, waveform: np.ndarray):
        self.waveform_length = len(waveform)
        
        # 计算FFT长度 (2的幂次)
        self.fft_size = 2 ** int(np.ceil(np.log2(2 * self.waveform_length)))
        
        # 计算参考信号FFT
        self.reference_fft = np.fft.fft(waveform, self.fft_size)
        
        # 匹配滤波器 (共轭)
        self.matched_filter = np.conj(self.reference_fft)
    
    def compress(self, received_signal: np.ndarray, 
                 window_type: str = 'hamming') -> np.ndarray:
        """脉冲压缩"""
        # 加窗
        if window_type:
            window = get_window(window_type, len(received_signal))
            received_signal = received_signal * window
        
        # FFT
        signal_fft = np.fft.fft(received_signal, self.fft_size)
        
        # 频域匹配滤波
        output_fft = signal_fft * self.matched_filter
        
        # IFFT
        compressed = np.fft.ifft(output_fft)
        
        # 取模
        return np.abs(compressed[:self.waveform_length])
```

### 5.5 CA-CFAR检测

```python
class CA_CFAR:
    """单元平均恒虚警检测器"""
    
    def __init__(self, n_train: int = 20, n_guard: int = 2, p_fa: float = 1e-6):
        self.n_train = n_train
        self.n_guard = n_guard
        self.p_fa = p_fa
        
        # 计算门限因子
        self.alpha = n_train * (p_fa ** (-1 / n_train) - 1)
    
    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CFAR检测"""
        detections = []
        thresholds = []
        
        half_train = self.n_train // 2
        half_guard = self.n_guard // 2
        
        for i in range(half_train + half_guard, 
                       len(data) - half_train - half_guard):
            
            # 前沿参考单元
            lead_cells = data[i - half_train - half_guard : i - half_guard]
            
            # 后沿参考单元
            lag_cells = data[i + half_guard + 1 : i + half_guard + half_train + 1]
            
            # 杂波功率估计
            noise_level = (np.mean(lead_cells) + np.mean(lag_cells)) / 2
            
            # 门限
            threshold = self.alpha * noise_level
            thresholds.append(threshold)
            
            # 检测判决
            if data[i] > threshold:
                detections.append(i)
        
        return np.array(detections), np.array(thresholds)
```

### 5.6 卡尔曼滤波

```python
class KalmanFilter:
    """卡尔曼滤波器"""
    
    def __init__(self, F: np.ndarray, H: np.ndarray, 
                 Q: np.ndarray, R: np.ndarray, 
                 x0: np.ndarray, P0: np.ndarray):
        self.F = F  # 状态转移矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 过程噪声协方差
        self.R = R  # 观测噪声协方差
        self.x = x0  # 状态估计
        self.P = P0  # 状态协方差
    
    def predict(self) -> None:
        """预测步"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, z: np.ndarray) -> None:
        """更新步"""
        # 新息
        y = z - self.H @ self.x
        
        # 新息协方差
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 状态更新
        self.x = self.x + K @ y
        
        # 协方差更新
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P
```

### 5.7 IMM滤波器

```python
class IMMFilter:
    """交互多模型滤波器"""
    
    def __init__(self, models: List[DynamicModel],
                 transition_matrix: np.ndarray,
                 initial_probs: np.ndarray):
        self.models = models
        self.P = transition_matrix
        self.mu = initial_probs  # 模型概率
        self.filters = [KalmanFilter(m) for m in models]
    
    def interaction(self) -> None:
        """输入交互"""
        n = len(self.models)
        c_bar = self.P.T @ self.mu  # 归一化常数
        
        # 计算混合权重
        self.mu_ij = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.mu_ij[i, j] = self.P[i, j] * self.mu[i] / c_bar[j]
        
        # 计算混合状态
        for j in range(n):
            x_mix = np.zeros_like(self.filters[0].x)
            for i in range(n):
                x_mix += self.mu_ij[i, j] * self.filters[i].x
            
            P_mix = np.zeros_like(self.filters[0].P)
            for i in range(n):
                diff = self.filters[i].x - x_mix
                P_mix += self.mu_ij[i, j] * (self.filters[i].P + diff @ diff.T)
            
            self.filters[j].x = x_mix
            self.filters[j].P = P_mix
    
    def update(self, z: np.ndarray) -> None:
        """各模型更新"""
        likelihoods = []
        for f in self.filters:
            f.update(z)
            # 计算似然
            v = f.innovation
            S = f.innovation_cov
            L = np.exp(-0.5 * v.T @ np.linalg.inv(S) @ v) / \
                np.sqrt(2 * np.pi * np.linalg.det(S))
            likelihoods.append(L)
        
        # 更新模型概率
        c = np.dot(self.mu, likelihoods)
        self.mu = self.mu * likelihoods / c
    
    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """融合输出"""
        x_est = np.zeros_like(self.filters[0].x)
        P_est = np.zeros_like(self.filters[0].P)
        
        for i, f in enumerate(self.filters):
            x_est += self.mu[i] * f.x
        
        for i, f in enumerate(self.filters):
            diff = f.x - x_est
            P_est += self.mu[i] * (f.P + diff @ diff.T)
        
        return x_est, P_est
```

### 5.8 自适应调度器

```python
class AdaptiveScheduler:
    """自适应资源调度器"""
    
    def schedule(self, tasks: List[Task], schedule_period: float,
                 current_time: int) -> ScheduleResult:
        """执行自适应调度"""
        # 按优先级排序
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        execute_queue = []
        delay_queue = []
        delete_queue = []
        
        time_pointer = 0
        period_end = schedule_period
        
        for task in sorted_tasks:
            # 计算时间窗
            earliest = max(time_pointer, task.time_window.front)
            latest = min(period_end - task.dwell_time, task.time_window.back)
            
            if earliest <= latest:
                # 可以安排
                task.scheduled_time = earliest
                execute_queue.append(task)
                time_pointer = earliest + task.dwell_time
            
            elif task.can_delay:
                # 延迟到下一周期
                task.delay_count += 1
                if task.delay_count < task.max_delay:
                    delay_queue.append(task)
                else:
                    delete_queue.append(task)
            else:
                # 删除
                delete_queue.append(task)
        
        # 统计
        stats = ScheduleStats(
            total_time=period_end,
            used_time=time_pointer,
            utilization=time_pointer / period_end,
            scheduled_count=len(execute_queue),
            delayed_count=len(delay_queue),
            deleted_count=len(delete_queue)
        )
        
        return ScheduleResult(execute_queue, delay_queue, delete_queue, stats)
```

---

## 6. 接口设计

### 6.1 RESTful API

```
POST /api/simulation/start      - 开始仿真
POST /api/simulation/stop       - 停止仿真
POST /api/simulation/pause      - 暂停仿真
GET  /api/simulation/status     - 获取状态
POST /api/config/radar          - 设置雷达参数
POST /api/config/scenario       - 加载场景
POST /api/track/manual          - 手动跟踪
POST /api/track/cancel          - 取消跟踪
```

### 6.2 WebSocket消息类型

```python
# 数据推送频率:
# - plot_update: 20Hz
# - track_update: 10Hz
# - beam_update: 20Hz
# - heartbeat: 1Hz

MESSAGE_TYPES = {
    'plot_update': {
        'type': 'plot_update',
        'timestamp': int,
        'data': [
            {'id': int, 'range': float, 'azimuth': float, 'velocity': float}
        ]
    },
    'track_update': {
        'type': 'track_update',
        'timestamp': int,
        'data': [
            {'id': int, 'position': [x, y, z], 'velocity': [vx, vy, vz]}
        ]
    },
    'beam_update': {
        'type': 'beam_update',
        'timestamp': int,
        'data': {
            'azimuth': float,
            'elevation': float,
            'taskType': str
        }
    }
}
```

---

## 7. 实施计划

### 7.1 阶段划分

| 阶段 | 名称 | 周期 | 主要任务 |
|------|------|------|----------|
| 1 | 基础设施搭建 | 3周 | 项目框架、通信协议、公共模块 |
| 2 | 场景与环境模拟 | 4周 | 目标运动、RCS、杂波、干扰模型 |
| 3 | 天线与波束控制 | 3周 | 阵列建模、波束形成、波位编排 |
| 4 | 信号处理 | 5周 | 波形产生、脉冲压缩、MTD、CFAR、测角 |
| 5 | 数据处理 | 4周 | 航迹起始、数据关联、目标跟踪 |
| 6 | 资源调度 | 3周 | 任务管理、优先级、自适应调度 |

**后端总工期：22周**

### 7.2 详细任务分解

#### 第一阶段：基础设施搭建（3周）

| 周次 | 任务 | 产出 |
|------|------|------|
| 1 | 项目初始化、依赖集成 | 可编译项目框架 |
| 1-2 | 公共模块：类型定义、常量、工具函数 | common模块 |
| 2 | 通信协议：消息结构、序列化 | protocol模块 |
| 2-3 | 网络基础：FastAPI、WebSocket服务器 | network模块 |
| 3 | Mock测试：前后端通信验证 | 通信联调通过 |

#### 第二阶段：场景与环境模拟（4周）

| 周次 | 任务 | 产出 |
|------|------|------|
| 1 | 目标模型：6DOF运动、轨迹生成 | TargetModel类 |
| 1-2 | RCS模型：Swerling、GTD | RCSModel类 |
| 2-3 | 杂波模型：幅度分布、相关杂波 | ClutterModel类 |
| 3 | 干扰模型：噪声干扰、RGPO/VGPO | JammerModel类 |
| 3-4 | 传播模型：自由空间、大气、电离层 | PropagationModel类 |
| 4 | 回波生成器、单元测试 | 完整环境模拟 |

#### 第三阶段：天线与波束控制（3周）

| 周次 | 任务 | 产出 |
|------|------|------|
| 1 | 阵列建模：线阵、面阵、密度加权 | ArrayAntenna类 |
| 1-2 | 波束形成：和差波束、移相器 | Beamformer类 |
| 2-3 | 波位编排：正弦空间、多种编排策略 | BeamScheduler类 |
| 3 | 方向图计算、旁瓣分析 | PatternCalculator类 |

#### 第四阶段：信号处理（5周）

| 周次 | 任务 | 产出 |
|------|------|------|
| 1 | 波形产生：LFM、相位编码 | WaveformGenerator类 |
| 1-2 | 脉冲压缩：FFT匹配滤波、加窗 | PulseCompressor类 |
| 2-3 | MTD/MTI：多普勒滤波组 | MTDProcessor类 |
| 3 | CFAR检测：CA-CFAR、OS-CFAR | CFAR类 |
| 3-4 | Keystone变换 | KeystoneTransform类 |
| 4 | 单脉冲测角、MUSIC | AngleEstimator类 |
| 4-5 | 旁瓣对消、集成测试 | 完整信号处理链 |

#### 第五阶段：数据处理（4周）

| 周次 | 任务 | 产出 |
|------|------|------|
| 1 | SVM分类器、点迹过滤 | SVMClassifier类 |
| 1-2 | 航迹起始：M/N逻辑法、Hough | TrackInitiator类 |
| 2-3 | 数据关联：NN、JPDA、GM-PHD | Associator类 |
| 3-4 | 滤波算法：KF、EKF、IMM | Filter类 |
| 4 | 工作模式：TWS、TAS | ModeController类 |

#### 第六阶段：资源调度（3周）

| 周次 | 任务 | 产出 |
|------|------|------|
| 1 | 任务定义、优先级计算 | Task、PriorityCalc类 |
| 1-2 | 自适应调度算法 | AdaptiveScheduler类 |
| 2-3 | 威胁度评估、调度统计 | 完整调度模块 |

---

## 8. 性能指标

| 性能指标 | 目标值 | 说明 |
|----------|--------|------|
| 仿真实时性 | 延迟 < 50ms | 100批次目标典型场景 |
| 跟踪容量 | ≥ 30批 | 稳定跟踪目标数 |
| 搜索容量 | ≥ 100批 | 同时搜索处理目标数 |
| 调度成功率 | ≥ 99% | 高优先级任务 |
| 时间利用率 | ≥ 90% | 雷达时间资源 |
| 测距精度 | ≤ c/(2B) | 取决于信号带宽 |
| 测角精度 | ≤ 波束宽度/10 | 取决于SNR和测角算法 |

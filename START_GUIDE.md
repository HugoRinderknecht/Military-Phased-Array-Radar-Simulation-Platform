# 雷达仿真平台 - 启动指南

## 最新修复 (2026-02-23)

**启动脚本闪退问题已解决！**

### 问题原因
1. ✅ **缺少虚拟环境** - 首次运行需要创建虚拟环境
2. ✅ **依赖包错误** - `requirements.txt` 中包含无效的 `python-cors` 包（已移除）
3. ✅ **循环导入问题** - `file_manager` 模块已修复为延迟初始化
4. ✅ **邮箱验证问题** - 默认邮箱从 `.local` 改为 `.com`

### 快速启动（推荐方式）

**Windows 用户 - 双击启动:**
```
双击 start.bat
```

脚本会自动：
1. 检查 Python 环境
2. 创建虚拟环境（如果不存在）
3. 安装所有依赖包
4. 启动后端服务

**首次启动需要等待几分钟来下载和安装依赖包。**

---

## 快速启动

### Windows 用户

**方法1：双击启动脚本**
```
双击 start.bat
```

**方法2：命令行启动**
```batch
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Linux/Mac 用户

```bash
chmod +x start.sh
./start.sh
```

---

## 访问地址

启动成功后，访问以下地址：

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端界面 | http://localhost:5173 | 主界面 |
| 后端API | http://localhost:8000 | API根路径 |
| API文档 | http://localhost:8000/docs | Swagger文档 |
| 健康检查 | http://localhost:8000/health | 状态检查 |

**默认账号**: `admin`
**默认密码**: `admin123`

---

## 冒烟测试结果 ✅

**测试时间**: 2026-02-23
**测试状态**: 全部通过

### API 接口测试
| 接口 | 方法 | 状态 | 说明 |
|------|------|------|------|
| `/health` | GET | ✅ PASS | 健康检查正常 |
| `/api/auth/login` | POST | ✅ PASS | 登录成功，返回JWT Token |
| `/api/auth/me` | GET | ✅ PASS | 获取当前用户信息 |
| `/api/radars` | POST | ✅ PASS | 创建雷达模型成功 |
| `/api/radars` | GET | ✅ PASS | 查询雷达模型列表 |
| `/api/radars/{id}` | GET | ✅ PASS | 获取雷达模型详情 |
| `/api/radars/materials/list` | GET | ✅ PASS | 材料库数据正常 |
| `/api/scenes` | GET | ✅ PASS | 场景列表接口正常 |

### 核心算法测试

所有8大核心算法模块都已通过测试：

| 算法模块 | 状态 | 说明 |
|---------|------|------|
| 天线方向图 | ✅ OK | FFT加速，支持多种加权 |
| 发射信号生成 | ✅ OK | LFM/Barker码/M序列 |
| ZMNL杂波生成 | ✅ OK | 瑞利/对数正态/威布尔/K分布 |
| MTI/MTD | ✅ OK | 一次/二次/三次对消 + 多普勒滤波 |
| CFAR检测 | ✅ OK | CA/OS/GO/SO四种检测器 |
| Kalman滤波 | ✅ OK | KF/EKF/UKF |
| 点迹关联 | ✅ OK | NN/GNN/PDA/JPDA |
| 参数估计 | ✅ OK | 距离/角度/多普勒 |

---

## 项目结构

```
radar-simulation-platform/
├── backend/                    # 后端（Python）
│   ├── app/
│   │   ├── api/                # API接口（认证/雷达/场景/仿真）
│   │   ├── core/               # 核心算法
│   │   │   ├── signal_processing/  # 信号处理（8个模块）
│   │   │   └── data_processing/    # 数据处理（3个模块）
│   │   ├── models/             # 数据模型
│   │   ├── storage/            # 文件管理
│   │   └── utils/              # 工具函数
│   ├── data/                   # 数据文件目录
│   ├── main.py                 # 后端入口
│   ├── requirements.txt        # 依赖包
│   └── test_algorithms.py      # 算法测试
│
├── frontend/                   # 前端（Vue 3）
│   ├── src/
│   │   ├── views/              # 页面组件
│   │   ├── stores/             # 状态管理
│   │   ├── router/             # 路由配置
│   │   ├── api/                # API封装
│   │   └── main.js             # 前端入口
│   ├── package.json
│   └── vite.config.js
│
├── start.bat                   # Windows启动脚本
├── start.sh                    # Linux/Mac启动脚本
└── README.md                   # 项目说明
```

---

## 依赖安装

如果手动安装，需要以下依赖：

**Python包**（requirements.txt）：
- fastapi, uvicorn
- numpy, scipy, h5py
- python-jose, passlib
- pydantic, pydantic-settings
- python-socketio
- slowapi

**Node.js包**（package.json）：
- vue, vite, vue-router, pinia
- element-plus
- axios, socket.io-client
- echarts, three

---

## 常见问题

### Q1: 启动脚本闪退
**A**: 已修复！现在使用改进的启动脚本，包含错误检查。

### Q2: 后端启动失败
**A**: 检查Python版本（需要3.8+），确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

### Q3: 前端无法连接后端
**A**:
1. 确认后端已启动（访问 http://localhost:8000/docs）
2. 检查防火墙设置
3. 确认后端运行在8000端口

### Q4: 登录失败
**A**:
- 用户名：admin
- 密码：admin123
- 首次登录后会自动创建用户文件

---

## 下一步

1. **登录系统**：访问前端界面，使用默认账号登录
2. **创建雷达模型**：在"雷达模型"页面创建雷达配置
3. **创建场景**：在"场景管理"页面添加目标和环境
4. **运行仿真**：在"仿真控制"页面启动仿真
5. **查看结果**：在"仿真结果"页面查看数据

---

## 技术支持

如遇问题，请检查：
1. 后端日志（backend/logs/目录）
2. 浏览器控制台（F12）
3. 后端健康检查：http://localhost:8000/health

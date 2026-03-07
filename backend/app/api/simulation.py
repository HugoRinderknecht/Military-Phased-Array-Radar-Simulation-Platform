"""
仿真控制 API 路由
"""
from typing import Optional
import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.websockets import WebSocketState

from app.models.simulation import (
    SimulationControl,
    SimulationRequest,
    SimulationStatus,
)
from app.models.user import User
from app.storage.file_manager import get_file_manager
from app.utils.dependencies import get_current_user
from app.utils.logger import get_logger

router = APIRouter(prefix="/sim", tags=["仿真控制"])
logger = get_logger(__name__)

# 仿真运行时存储（生产环境应使用Redis等）
simulations = {}


@router.post("/start")
async def start_simulation(
    request: SimulationRequest,
    current_user: User = Depends(get_current_user),
):
    """
    启动仿真

    Args:
        request: 仿真启动请求
        current_user: 当前用户

    Returns:
        仿真ID和初始状态
    """
    import uuid

    simulation_id = f"sim-{uuid.uuid4().hex[:8]}"

    # 加载场景
    try:
        scene_data = get_file_manager().load_json(
            current_user.username, "scenes", f"{request.scene_id}.json"
        )

        # 加载雷达模型
        radar_data = get_file_manager().load_json(
            current_user.username, "models", f"{scene_data['radar_model_id']}.json"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"场景或雷达模型不存在: {e}",
        )

    # 创建仿真状态
    simulations[simulation_id] = {
        "status": "running",
        "progress": 0.0,
        "current_time": 0.0,
        "total_time": scene_data.get("duration", 100),
        "plots_count": 0,
        "tracks_count": 0,
        "events_count": 0,
        "started_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "scene_data": scene_data,
        "radar_data": radar_data,
        "user_id": current_user.username,
        "task": None,
    }

    logger.info(f"仿真启动: {simulation_id}, 场景: {request.scene_id}")

    # 启动后台任务
    task = asyncio.create_task(
        _run_simulation(
            simulation_id,
            scene_data,
            radar_data,
            request,
        )
    )
    simulations[simulation_id]["task"] = task

    return {
        "simulation_id": simulation_id,
        "status": "running",
        "message": "仿真已启动",
    }


async def _run_simulation(
    simulation_id: str,
    scene_data: dict,
    radar_data: dict,
    request: dict,
):
    """
    仿真运行后台任务

    这是一个简化的仿真框架，完整实现需要集成所有信号处理模块
    """
    try:
        from app.core.signal_processing.waveform import generate_lfm_pulse
        from app.core.signal_processing.antenna import calculate_antenna_pattern
        from app.core.signal_processing.target_echo import simulate_target_echo
        from app.core.signal_processing.clutter import generate_zmnl_clutter
        from app.core.signal_processing.mti_mtd import mti_canceler_1st, mtd_processing
        from app.core.signal_processing.cfar import ca_cfar_1d
        from app.core.data_processing.track_init import LogicTrackInitializer
        from app.core.data_processing.association import NNAassociator
        from app.core.data_processing.filter import KalmanFilter
        import numpy as np

        # 仿真参数
        duration = scene_data.get("duration", 100)
        time_step = scene_data.get("time_step", 0.1)
        num_steps = int(duration / time_step)

        # 雷达参数
        wavelength = 3e8 / radar_data["transmitter"]["frequency"]
        sampling_rate = 2 * radar_data["transmitter"]["bandwidth"]
        prf = radar_data["transmitter"]["prf"]

        # 发射信号
        waveform, _ = generate_lfm_pulse(
            pulse_width=radar_data["transmitter"]["pulse_width"],
            bandwidth=radar_data["transmitter"]["bandwidth"],
            sampling_rate=sampling_rate,
            prf=prf,
            num_pulses=10,
        )

        # 初始化跟踪器
        track_initializer = LogicTrackInitializer(m=3, n=4)
        associator = NNAassociator()
        tracks = {}

        # 运行仿真循环
        for step in range(num_steps):
            if simulation_id not in simulations:
                break

            current_time = step * time_step

            # 更新状态
            simulations[simulation_id]["current_time"] = current_time
            simulations[simulation_id]["progress"] = (step / num_steps) * 100
            simulations[simulation_id]["updated_at"] = datetime.utcnow().isoformat()

            # 这里应该调用完整的信号处理链
            # 1. 生成目标回波
            # 2. 添加杂波
            # 3. MTI/MTD处理
            # 4. CFAR检测
            # 5. 点迹关联
            # 6. Kalman滤波
            # 7. 更新航迹

            # 模拟生成一些检测点迹
            num_plots = np.random.poisson(5)
            plots = []
            for i in range(num_plots):
                plots.append({
                    "plot_id": f"plot-{step}-{i}",
                    "time": current_time,
                    "range": np.random.uniform(5000, 50000),
                    "azimuth": np.random.uniform(-60, 60),
                    "elevation": np.random.uniform(-10, 10),
                    "amplitude": np.random.uniform(0.1, 1.0),
                })

            simulations[simulation_id]["plots_count"] = len(plots)

            # 等待
            await asyncio.sleep(0.1)

        # 仿真完成
        if simulation_id in simulations:
            simulations[simulation_id]["status"] = "completed"
            simulations[simulation_id]["progress"] = 100.0

        logger.info(f"仿真完成: {simulation_id}")

    except Exception as e:
        logger.error(f"仿真错误: {simulation_id}, {e}")
        if simulation_id in simulations:
            simulations[simulation_id]["status"] = "error"
            simulations[simulation_id]["error_message"] = str(e)


@router.post("/control")
async def control_simulation(
    control: SimulationControl,
    current_user: User = Depends(get_current_user),
):
    """
    控制仿真（暂停/恢复/停止/单步）

    Args:
        control: 控制命令
        current_user: 当前用户

    Returns:
        控制结果
    """
    # 这里需要实现仿真控制逻辑
    # 当前是简化版本
    return {
        "status": "success",
        "message": f"命令 '{control.command}' 已执行",
    }


@router.get("/status")
async def get_simulation_status(
    simulation_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    获取仿真状态

    Args:
        simulation_id: 仿真ID
        current_user: 当前用户

    Returns:
        仿真状态
    """
    if simulation_id not in simulations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"仿真不存在: {simulation_id}",
        )

    sim_data = simulations[simulation_id]

    return SimulationStatus(
        simulation_id=simulation_id,
        status=sim_data["status"],
        progress=sim_data["progress"],
        current_time=sim_data["current_time"],
        total_time=sim_data["total_time"],
        plots_count=sim_data["plots_count"],
        tracks_count=sim_data["tracks_count"],
        events_count=sim_data["events_count"],
        started_at=sim_data["started_at"],
        updated_at=sim_data["updated_at"],
        error_message=sim_data.get("error_message"),
    )


@router.get("/list")
async def list_simulations(
    current_user: User = Depends(get_current_user),
):
    """
    获取用户的仿真列表

    Args:
        current_user: 当前用户

    Returns:
        仿真列表
    """
    user_simulations = []

    for sim_id, sim_data in simulations.items():
        if sim_data.get("user_id") == current_user.username:
            user_simulations.append({
                "simulation_id": sim_id,
                "status": sim_data["status"],
                "progress": sim_data["progress"],
                "current_time": sim_data["current_time"],
                "started_at": sim_data["started_at"],
            })

    return user_simulations


@router.delete("/{simulation_id}")
async def delete_simulation(
    simulation_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    删除仿真

    Args:
        simulation_id: 仿真ID
        current_user: 当前用户

    Returns:
        删除结果
    """
    if simulation_id not in simulations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"仿真不存在: {simulation_id}",
        )

    # 取消任务
    task = simulations[simulation_id].get("task")
    if task and not task.done():
        task.cancel()

    del simulations[simulation_id]

    logger.info(f"仿真已删除: {simulation_id} by {current_user.username}")

    return {"status": "success", "message": "仿真已删除"}

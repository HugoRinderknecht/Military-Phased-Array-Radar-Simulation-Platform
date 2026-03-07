"""
雷达模型 API 路由
"""
from datetime import datetime
from typing import List
import json

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.models.radar import RadarModel
from app.models.user import User
from app.storage.file_manager import get_file_manager
from app.utils.dependencies import get_current_user
from app.utils.logger import get_logger

router = APIRouter(prefix="/radars", tags=["雷达模型"])
logger = get_logger(__name__)


@router.get("", response_model=List[RadarModel])
async def list_radars(current_user: User = Depends(get_current_user)):
    """
    获取当前用户的雷达模型列表

    Args:
        current_user: 当前用户

    Returns:
        雷达模型列表
    """
    files = get_file_manager().list_files(current_user.username, "models")

    models = []
    for filename in files:
        try:
            data = get_file_manager().load_json(current_user.username, "models", filename)
            models.append(RadarModel(**data))
        except Exception as e:
            logger.error(f"加载雷达模型失败: {filename}, {e}")

    return models


@router.get("/{model_id}", response_model=RadarModel)
async def get_radar_model(
    model_id: str, current_user: User = Depends(get_current_user)
):
    """
    获取指定的雷达模型

    Args:
        model_id: 模型ID
        current_user: 当前用户

    Returns:
        雷达模型详情

    Raises:
        HTTPException: 模型不存在时
    """
    filename = f"{model_id}.json"

    if not get_file_manager().file_exists(current_user.username, "models", filename):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"雷达模型不存在: {model_id}",
        )

    try:
        data = get_file_manager().load_json(current_user.username, "models", filename)
        return RadarModel(**data)
    except Exception as e:
        logger.error(f"加载雷达模型失败: {model_id}, {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="加载模型失败",
        )


@router.post("", response_model=RadarModel, status_code=status.HTTP_201_CREATED)
async def create_radar_model(
    model: RadarModel, current_user: User = Depends(get_current_user)
):
    """
    创建新的雷达模型

    Args:
        model: 雷达模型数据
        current_user: 当前用户

    Returns:
        创建的雷达模型

    Raises:
        HTTPException: 模型ID已存在时
    """
    import json
    logger.info(f"收到创建雷达模型请求: {json.dumps(model.model_dump(), indent=2, default=str)}")

    filename = f"{model.id}.json"

    if get_file_manager().file_exists(current_user.username, "models", filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"雷达模型ID已存在: {model.id}",
        )

    # 设置创建信息
    model.created_by = current_user.username
    model.created_at = datetime.utcnow().isoformat()
    model.updated_at = datetime.utcnow().isoformat()

    # 保存模型
    data = model.model_dump()
    get_file_manager().save_json(current_user.username, "models", filename, data)

    logger.info(f"创建雷达模型: {model.id} by {current_user.username}")

    return model


@router.put("/{model_id}", response_model=RadarModel)
async def update_radar_model(
    model_id: str,
    model: RadarModel,
    current_user: User = Depends(get_current_user),
):
    """
    更新雷达模型

    Args:
        model_id: 模型ID
        model: 新的雷达模型数据
        current_user: 当前用户

    Returns:
        更新后的雷达模型

    Raises:
        HTTPException: 模型不存在时
    """
    filename = f"{model_id}.json"

    if not get_file_manager().file_exists(current_user.username, "models", filename):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"雷达模型不存在: {model_id}",
        )

    # 确保ID匹配
    if model.id != model_id:
        model.id = model_id

    # 更新时间戳
    model.updated_at = datetime.utcnow().isoformat()

    # 保存模型
    data = model.model_dump()
    get_file_manager().save_json(current_user.username, "models", filename, data)

    logger.info(f"更新雷达模型: {model_id} by {current_user.username}")

    return model


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_radar_model(
    model_id: str, current_user: User = Depends(get_current_user)
):
    """
    删除雷达模型

    Args:
        model_id: 模型ID
        current_user: 当前用户

    Raises:
        HTTPException: 模型不存在时
    """
    filename = f"{model_id}.json"

    if not get_file_manager().file_exists(current_user.username, "models", filename):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"雷达模型不存在: {model_id}",
        )

    get_file_manager().delete_file(current_user.username, "models", filename)

    logger.info(f"删除雷达模型: {model_id} by {current_user.username}")


@router.get("/materials/list")
async def list_materials(current_user: User = Depends(get_current_user)):
    """
    获取材料库列表

    Args:
        current_user: 当前用户

    Returns:
        材料库数据
    """
    materials = get_file_manager().load_materials()
    return materials

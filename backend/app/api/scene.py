"""
场景管理 API 路由
"""
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.models.scene import SceneModel
from app.models.user import User
from app.storage.file_manager import get_file_manager
from app.utils.dependencies import get_current_user
from app.utils.logger import get_logger

router = APIRouter(prefix="/scenes", tags=["场景管理"])
logger = get_logger(__name__)


@router.get("", response_model=List[SceneModel])
async def list_scenes(current_user: User = Depends(get_current_user)):
    """
    获取当前用户的场景列表

    Args:
        current_user: 当前用户

    Returns:
        场景列表
    """
    files = get_file_manager().list_files(current_user.username, "scenes")

    scenes = []
    for filename in files:
        try:
            data = get_file_manager().load_json(current_user.username, "scenes", filename)
            scenes.append(SceneModel(**data))
        except Exception as e:
            logger.error(f"加载场景失败: {filename}, {e}")

    return scenes


@router.get("/{scene_id}", response_model=SceneModel)
async def get_scene(
    scene_id: str, current_user: User = Depends(get_current_user)
):
    """
    获取指定场景

    Args:
        scene_id: 场景ID
        current_user: 当前用户

    Returns:
        场景详情

    Raises:
        HTTPException: 场景不存在时
    """
    filename = f"{scene_id}.json"

    if not get_file_manager().file_exists(current_user.username, "scenes", filename):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"场景不存在: {scene_id}",
        )

    try:
        data = get_file_manager().load_json(current_user.username, "scenes", filename)
        return SceneModel(**data)
    except Exception as e:
        logger.error(f"加载场景失败: {scene_id}, {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="加载场景失败",
        )


@router.post("", response_model=SceneModel, status_code=status.HTTP_201_CREATED)
async def create_scene(
    scene: SceneModel, current_user: User = Depends(get_current_user)
):
    """
    创建新场景

    Args:
        scene: 场景数据
        current_user: 当前用户

    Returns:
        创建的场景

    Raises:
        HTTPException: 场景ID已存在时
    """
    filename = f"{scene.id}.json"

    if get_file_manager().file_exists(current_user.username, "scenes", filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"场景ID已存在: {scene.id}",
        )

    # 验证关联的雷达模型是否存在
    radar_filename = f"{scene.radar_model_id}.json"
    if not get_file_manager().file_exists(current_user.username, "models", radar_filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"关联的雷达模型不存在: {scene.radar_model_id}",
        )

    # 设置创建信息
    scene.created_by = current_user.username
    scene.created_at = datetime.utcnow().isoformat()
    scene.updated_at = datetime.utcnow().isoformat()

    # 保存场景
    data = scene.model_dump()
    get_file_manager().save_json(current_user.username, "scenes", filename, data)

    logger.info(f"创建场景: {scene.id} by {current_user.username}")

    return scene


@router.put("/{scene_id}", response_model=SceneModel)
async def update_scene(
    scene_id: str,
    scene: SceneModel,
    current_user: User = Depends(get_current_user),
):
    """
    更新场景

    Args:
        scene_id: 场景ID
        scene: 新的场景数据
        current_user: 当前用户

    Returns:
        更新后的场景

    Raises:
        HTTPException: 场景不存在时
    """
    filename = f"{scene_id}.json"

    if not get_file_manager().file_exists(current_user.username, "scenes", filename):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"场景不存在: {scene_id}",
        )

    # 确保ID匹配
    if scene.id != scene_id:
        scene.id = scene_id

    # 更新时间戳
    scene.updated_at = datetime.utcnow().isoformat()

    # 保存场景
    data = scene.model_dump()
    get_file_manager().save_json(current_user.username, "scenes", filename, data)

    logger.info(f"更新场景: {scene_id} by {current_user.username}")

    return scene


@router.delete("/{scene_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_scene(
    scene_id: str, current_user: User = Depends(get_current_user)
):
    """
    删除场景

    Args:
        scene_id: 场景ID
        current_user: 当前用户

    Raises:
        HTTPException: 场景不存在时
    """
    filename = f"{scene_id}.json"

    if not get_file_manager().file_exists(current_user.username, "scenes", filename):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"场景不存在: {scene_id}",
        )

    get_file_manager().delete_file(current_user.username, "scenes", filename)

    logger.info(f"删除场景: {scene_id} by {current_user.username}")

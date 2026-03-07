"""
用户认证 API 路由
"""
from datetime import datetime
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status

from app.models.user import LoginRequest, Token, User, UserCreate
from app.storage.file_manager import get_file_manager
from app.utils.crypto import get_password_hash, verify_password
from app.utils.dependencies import create_token_response, get_current_admin, get_current_user
from app.utils.logger import get_logger

router = APIRouter(prefix="/auth", tags=["认证"])
logger = get_logger(__name__)


@router.post("/login", response_model=Token)
async def login(login_data: LoginRequest):
    """
    用户登录

    Args:
        login_data: 登录请求数据

    Returns:
        JWT Token

    Raises:
        HTTPException: 登录失败时
    """
    users_data = get_file_manager().load_users()

    # 查找用户
    user_data = None
    for u in users_data.get("users", []):
        if u["username"] == login_data.username:
            user_data = u
            break

    if not user_data:
        logger.warning(f"登录失败: 用户不存在 - {login_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    # 验证密码
    if not verify_password(login_data.password, user_data["password_hash"]):
        logger.warning(f"登录失败: 密码错误 - {login_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    # 检查用户状态
    if not user_data.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用",
        )

    logger.info(f"用户登录成功: {login_data.username}")

    # 生成 Token
    return create_token_response(
        user_id=user_data["id"],
        username=user_data["username"],
        role=user_data["role"],
    )


@router.post("/register", response_model=User)
async def register(
    user_data: UserCreate,
    current_user: User = Depends(get_current_admin),
):
    """
    注册新用户（仅管理员）

    Args:
        user_data: 用户创建数据
        current_user: 当前管理员用户

    Returns:
        创建的用户信息

    Raises:
        HTTPException: 用户名已存在时
    """
    users_data = get_file_manager().load_users()

    # 检查用户名是否已存在
    for u in users_data.get("users", []):
        if u["username"] == user_data.username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在",
            )

    # 创建新用户
    now = datetime.utcnow().isoformat()
    new_user = {
        "id": f"user-{datetime.utcnow().timestamp()}",
        "username": user_data.username,
        "password_hash": get_password_hash(user_data.password),
        "email": user_data.email,
        "role": user_data.role,
        "created_at": now,
        "updated_at": now,
        "is_active": True,
    }

    users_data["users"].append(new_user)
    get_file_manager().save_users(users_data)

    logger.info(f"新用户注册成功: {user_data.username}")

    return User(
        id=new_user["id"],
        username=new_user["username"],
        email=new_user.get("email"),
        role=new_user["role"],
        created_at=new_user["created_at"],
        is_active=new_user["is_active"],
    )


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    获取当前用户信息

    Args:
        current_user: 当前用户

    Returns:
        用户信息
    """
    return current_user


@router.get("/users", response_model=list[User])
async def list_users(current_user: User = Depends(get_current_admin)):
    """
    获取用户列表（仅管理员）

    Args:
        current_user: 当前管理员用户

    Returns:
        用户列表
    """
    users_data = get_file_manager().load_users()

    return [
        User(
            id=u["id"],
            username=u["username"],
            email=u.get("email"),
            role=u["role"],
            created_at=u["created_at"],
            is_active=u.get("is_active", True),
        )
        for u in users_data.get("users", [])
    ]

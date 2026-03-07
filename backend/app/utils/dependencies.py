"""
认证依赖和工具
"""
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models.user import TokenData, User
from app.storage.file_manager import get_file_manager
from app.utils.crypto import decode_access_token

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """
    获取当前登录用户

    Args:
        credentials: HTTP Bearer 认证凭据

    Returns:
        当前用户信息

    Raises:
        HTTPException: 认证失败时
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    payload = decode_access_token(token)

    if payload is None:
        raise credentials_exception

    username: Optional[str] = payload.get("sub")
    user_id: Optional[str] = payload.get("user_id")
    role: Optional[str] = payload.get("role")

    if username is None or user_id is None:
        raise credentials_exception

    # 从存储加载用户信息
    users_data = get_file_manager().load_users()
    user_data = None
    for u in users_data.get("users", []):
        if u["id"] == user_id and u["username"] == username:
            user_data = u
            break

    if user_data is None:
        raise credentials_exception

    if not user_data.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="用户已被禁用"
        )

    return User(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data.get("email"),
        role=user_data["role"],
        created_at=user_data["created_at"],
        is_active=user_data.get("is_active", True),
    )


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    获取当前管理员用户

    Args:
        current_user: 当前用户

    Returns:
        管理员用户信息

    Raises:
        HTTPException: 非管理员时
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限"
        )
    return current_user


def create_token_response(user_id: str, username: str, role: str) -> dict:
    """
    创建 Token 响应

    Args:
        user_id: 用户ID
        username: 用户名
        role: 用户角色

    Returns:
        Token 响应字典
    """
    from app.config import get_settings
    from app.utils.crypto import create_access_token
    from datetime import timedelta

    settings = get_settings()

    access_token = create_access_token(
        data={
            "sub": username,
            "user_id": user_id,
            "role": role,
        },
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60,
    }


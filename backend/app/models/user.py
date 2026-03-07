"""
用户数据模型
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """用户基础模型"""

    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    role: str = Field(default="user", pattern="^(admin|user)$")


class UserCreate(UserBase):
    """用户创建模型"""

    password: str = Field(..., min_length=6, max_length=100)


class UserUpdate(BaseModel):
    """用户更新模型"""

    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6, max_length=100)


class UserInDB(UserBase):
    """数据库中的用户模型"""

    id: str
    password_hash: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "id": "user-001",
                "username": "admin",
                "email": "admin@example.com",
                "role": "admin",
                "created_at": "2026-02-23T10:00:00",
                "updated_at": "2026-02-23T10:00:00",
                "is_active": True,
            }
        }


class User(UserBase):
    """返回给客户端的用户模型（不含密码）"""

    id: str
    created_at: datetime
    is_active: bool

    class Config:
        json_schema_extra = {
            "example": {
                "id": "user-001",
                "username": "admin",
                "email": "admin@example.com",
                "role": "admin",
                "created_at": "2026-02-23T10:00:00",
                "is_active": True,
            }
        }


class Token(BaseModel):
    """JWT Token 模型"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # 秒

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 86400,
            }
        }


class TokenData(BaseModel):
    """Token 数据模型"""

    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None


class LoginRequest(BaseModel):
    """登录请求模型"""

    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)

    class Config:
        json_schema_extra = {
            "example": {"username": "admin", "password": "admin123"}
        }

"""
加密工具函数
"""
from datetime import datetime, timedelta
from typing import Optional
import hashlib

from jose import JWTError, jwt

from app.config import get_settings

settings = get_settings()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码（SHA-256实现）"""
    password_hash = hashlib.sha256(plain_password.encode()).hexdigest()
    return password_hash == hashed_password


def get_password_hash(password: str) -> str:
    """获取密码哈希（SHA-256）"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建JWT访问令牌

    Args:
        data: 要编码的数据字典
        expires_delta: 过期时间增量

    Returns:
        JWT令牌字符串
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    解码JWT访问令牌

    Args:
        token: JWT令牌字符串

    Returns:
        解码后的数据字典，失败返回None
    """
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        return payload
    except JWTError:
        return None

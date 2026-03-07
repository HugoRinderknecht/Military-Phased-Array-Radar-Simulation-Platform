"""
雷达仿真平台 - 配置管理
"""
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator

# 尝试多种导入方式
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """应用配置"""

    # 应用信息
    app_name: str = "RadarSimulationPlatform"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # 安全配置
    secret_key: str = Field(default="your-secret-key-here-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24小时

    # 数据存储
    data_dir: Path = Field(default=Path("./data"))
    log_dir: Path = Field(default=Path("./logs"))
    max_upload_size: int = 10 * 1024 * 1024  # 10MB

    # CORS
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ]
    )

    # WebSocket
    ws_message_queue_size: int = 100

    # 仿真设置
    max_simultaneous_simulations: int = 10
    simulation_timeout: int = 3600  # 1小时

    # 文件存储
    allowed_extensions: List[str] = Field(default=["json", "h5", "hdf5", "csv"])
    max_file_size_json: int = 10 * 1024 * 1024  # 10MB
    max_file_size_hdf5: int = 1024 * 1024 * 1024  # 1GB

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings

"""
日志工具
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from app.config import get_settings

settings = get_settings()


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        log_format: 日志格式字符串

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除现有处理器
    logger.handlers.clear()

    # 默认格式
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(log_format)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器实例
    """
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}.log"

    return setup_logger(
        name,
        log_file=str(log_file),
        level=logging.DEBUG if settings.debug else logging.INFO,
    )

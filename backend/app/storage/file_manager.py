"""
文件存储管理模块
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from app.config import get_settings
from app.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


class FileManager:
    """文件管理器"""

    def __init__(self):
        self.data_dir = Path(settings.data_dir)
        self._init_directories()
        self._init_materials()
        self._init_users()

    def _init_directories(self):
        """初始化目录结构"""
        directories = [
            self.data_dir,
            self.data_dir / "users",
            self.data_dir / "materials",
            self.data_dir / "logs",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"数据目录初始化完成: {self.data_dir}")

    def _init_materials(self):
        """初始化材料数据库"""
        materials_file = self.data_dir / "materials.json"
        if not materials_file.exists():
            default_materials = {
                "GaAs": {
                    "name": "砷化镓",
                    "max_power_density": 1.0,
                    "efficiency": 0.4,
                    "noise_figure": 2.0,
                    "freq_range": [1e9, 100e9],
                    "thermal_resistance": 10,
                    "description": "传统雷达材料，成熟稳定",
                },
                "GaN": {
                    "name": "氮化镓",
                    "max_power密度": 5.0,
                    "efficiency": 0.6,
                    "noise_figure": 3.0,
                    "freq_range": [1e9, 40e9],
                    "thermal_resistance": 5,
                    "description": "高功率密度，新一代雷达材料",
                },
                "Ga2O3": {
                    "name": "氧化镓",
                    "max_power_density": 8.0,
                    "efficiency": 0.3,
                    "noise_figure": 4.0,
                    "freq_range": [1e9, 10e9],
                    "thermal_resistance": 2,
                    "description": "超高功率密度，新兴材料",
                },
            }
            with open(materials_file, "w", encoding="utf-8") as f:
                json.dump(default_materials, f, ensure_ascii=False, indent=2)
            logger.info("材料数据库初始化完成")

    def _init_users(self):
        """初始化用户文件"""
        users_file = self.data_dir / "users.json"
        if not users_file.exists():
            # 使用短密码哈希避免bcrypt 72字节限制
            import hashlib
            password_hash = hashlib.sha256("admin123".encode()).hexdigest()

            default_users = {
                "users": [
                    {
                        "id": "admin-001",
                        "username": "admin",
                        "password_hash": password_hash,
                        "email": "admin@radarsim.com",
                        "role": "admin",
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                        "is_active": True,
                    }
                ]
            }
            with open(users_file, "w", encoding="utf-8") as f:
                json.dump(default_users, f, ensure_ascii=False, indent=2)
            logger.info("用户数据库初始化完成，默认管理员: admin/admin123")

    def get_user_dir(self, username: str) -> Path:
        """获取用户数据目录"""
        user_dir = self.data_dir / "users" / username
        user_dir.mkdir(parents=True, exist_ok=True)
        # 创建子目录
        (user_dir / "models").mkdir(exist_ok=True)
        (user_dir / "scenes").mkdir(exist_ok=True)
        (user_dir / "results").mkdir(exist_ok=True)
        (user_dir / "uploads").mkdir(exist_ok=True)
        return user_dir

    def save_json(
        self, username: str, category: str, filename: str, data: Dict[str, Any]
    ) -> Path:
        """
        保存JSON文件

        Args:
            username: 用户名
            category: 类别 (models/scenes/results)
            filename: 文件名
            data: 要保存的数据

        Returns:
            文件路径
        """
        user_dir = self.get_user_dir(username)
        category_dir = user_dir / category
        category_dir.mkdir(exist_ok=True)

        file_path = category_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"文件已保存: {file_path}")
        return file_path

    def load_json(self, username: str, category: str, filename: str) -> Dict[str, Any]:
        """
        加载JSON文件

        Args:
            username: 用户名
            category: 类别
            filename: 文件名

        Returns:
            数据字典
        """
        user_dir = self.get_user_dir(username)
        file_path = user_dir / category / filename

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def list_files(self, username: str, category: str) -> List[str]:
        """
        列出用户目录下的文件

        Args:
            username: 用户名
            category: 类别

        Returns:
            文件名列表
        """
        user_dir = self.get_user_dir(username)
        category_dir = user_dir / category

        if not category_dir.exists():
            return []

        return [f.name for f in category_dir.iterdir() if f.is_file()]

    def delete_file(self, username: str, category: str, filename: str) -> bool:
        """
        删除文件

        Args:
            username: 用户名
            category: 类别
            filename: 文件名

        Returns:
            是否删除成功
        """
        user_dir = self.get_user_dir(username)
        file_path = user_dir / category / filename

        if file_path.exists():
            file_path.unlink()
            logger.info(f"文件已删除: {file_path}")
            return True
        return False

    def file_exists(self, username: str, category: str, filename: str) -> bool:
        """检查文件是否存在"""
        user_dir = self.get_user_dir(username)
        file_path = user_dir / category / filename
        return file_path.exists()

    def load_materials(self) -> Dict[str, Any]:
        """加载材料数据库"""
        materials_file = self.data_dir / "materials.json"
        with open(materials_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_users(self) -> Dict[str, Any]:
        """加载用户数据"""
        users_file = self.data_dir / "users.json"
        with open(users_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_users(self, users_data: Dict[str, Any]):
        """保存用户数据"""
        users_file = self.data_dir / "users.json"
        with open(users_file, "w", encoding="utf-8") as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)


# 全局文件管理器实例（延迟初始化）
_file_manager_instance = None


def get_file_manager() -> FileManager:
    """获取文件管理器单例"""
    global _file_manager_instance
    if _file_manager_instance is None:
        _file_manager_instance = FileManager()
    return _file_manager_instance


# 向后兼容的别名（延迟加载）
def __getattr__(name):
    if name == 'file_manager':
        return get_file_manager()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

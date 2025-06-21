import json
import uuid
import os
from datetime import datetime
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ConfigService:
    """配置管理服务"""

    def __init__(self):
        self.config_dir = 'data/configurations'
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        """确保配置目录存在"""
        os.makedirs(self.config_dir, exist_ok=True)

    def save_configuration(self, config_data: Dict[str, Any]) -> str:
        """保存配置"""
        config_id = str(uuid.uuid4())

        config_record = {
            'id': config_id,
            'name': config_data.get('name', f'配置_{config_id[:8]}'),
            'description': config_data.get('description', ''),
            'config': config_data,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        file_path = os.path.join(self.config_dir, f'{config_id}.json')
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_record, f, ensure_ascii=False, indent=2)

        logger.info(f"Configuration saved: {config_id}")
        return config_id

    def load_configuration(self, config_id: str) -> Dict[str, Any]:
        """加载配置"""
        file_path = os.path.join(self.config_dir, f'{config_id}.json')

        if not os.path.exists(file_path):
            raise ValueError(f"Configuration {config_id} not found")

        with open(file_path, 'r', encoding='utf-8') as f:
            config_record = json.load(f)

        return config_record

    def list_configurations(self) -> List[Dict[str, Any]]:
        """列出所有配置"""
        configs = []

        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                try:
                    file_path = os.path.join(self.config_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config_record = json.load(f)

                    # 只返回基本信息，不包含完整配置
                    config_summary = {
                        'id': config_record['id'],
                        'name': config_record['name'],
                        'description': config_record['description'],
                        'created_at': config_record['created_at'],
                        'updated_at': config_record['updated_at']
                    }
                    configs.append(config_summary)
                except Exception as e:
                    logger.error(f"Error loading config {filename}: {str(e)}")

        # 按更新时间倒序排列
        configs.sort(key=lambda x: x['updated_at'], reverse=True)
        return configs

    def update_configuration(self, config_id: str, config_data: Dict[str, Any]):
        """更新配置"""
        file_path = os.path.join(self.config_dir, f'{config_id}.json')

        if not os.path.exists(file_path):
            raise ValueError(f"Configuration {config_id} not found")

        # 读取现有配置
        with open(file_path, 'r', encoding='utf-8') as f:
            config_record = json.load(f)

        # 更新配置
        config_record['config'] = config_data
        config_record['updated_at'] = datetime.now().isoformat()

        if 'name' in config_data:
            config_record['name'] = config_data['name']
        if 'description' in config_data:
            config_record['description'] = config_data['description']

        # 保存更新后的配置
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_record, f, ensure_ascii=False, indent=2)

        logger.info(f"Configuration updated: {config_id}")

    def delete_configuration(self, config_id: str):
        """删除配置"""
        file_path = os.path.join(self.config_dir, f'{config_id}.json')

        if not os.path.exists(file_path):
            raise ValueError(f"Configuration {config_id} not found")

        os.remove(file_path)
        logger.info(f"Configuration deleted: {config_id}")

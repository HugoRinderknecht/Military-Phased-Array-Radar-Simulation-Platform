import json
import uuid
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigService:
    """配置管理服务"""

    def __init__(self):
        self.config_dir = 'data/configurations'
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        """确保配置目录存在"""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create config directory: {str(e)}")
            raise

    def save_configuration(self, config_data: Dict[str, Any]) -> str:
        """保存配置"""
        try:
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

        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise ValueError(f"Failed to save configuration: {str(e)}")

    def load_configuration(self, config_id: str) -> Dict[str, Any]:
        """加载配置"""
        try:
            if not config_id or not config_id.strip():
                raise ValueError("Configuration ID cannot be empty")

            file_path = os.path.join(self.config_dir, f'{config_id}.json')

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Configuration {config_id} not found")

            with open(file_path, 'r', encoding='utf-8') as f:
                config_record = json.load(f)

            return config_record

        except FileNotFoundError:
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file {config_id}: {str(e)}")
            raise ValueError(f"Configuration file {config_id} is corrupted")
        except Exception as e:
            logger.error(f"Failed to load configuration {config_id}: {str(e)}")
            raise ValueError(f"Failed to load configuration: {str(e)}")

    def list_configurations(self, page=1, per_page=20) -> List[Dict[str, Any]]:
        """列出所有配置"""
        # 增强参数验证
        if page < 1 or per_page < 1 or per_page > 100:
            raise ValueError("Invalid pagination parameters")

        configs = []

        try:
            # 获取所有配置文件
            if not os.path.exists(self.config_dir):
                return []

            filenames = os.listdir(self.config_dir)
            config_filenames = [f for f in filenames if f.endswith('.json')]

            # 按文件修改时间排序（最新的在前）
            config_filenames.sort(
                key=lambda x: os.path.getmtime(os.path.join(self.config_dir, x)),
                reverse=True
            )

            # 计算分页
            start = (page - 1) * per_page
            end = start + per_page
            paginated_filenames = config_filenames[start:end]

            for filename in paginated_filenames:
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
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    logger.error(f"Failed to load config {filename}: {str(e)}")
                    # 继续处理其他文件，不中断整个列表操作
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error loading config {filename}: {str(e)}")
                    continue

        except OSError as e:
            logger.error(f"Error accessing config directory: {str(e)}")
            raise ValueError(f"Unable to access configuration directory: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in list_configurations: {str(e)}")
            raise ValueError(f"Failed to list configurations: {str(e)}")

        return configs

    def update_configuration(self, config_id: str, config_data: Dict[str, Any]):
        """更新配置"""
        try:
            if not config_id or not config_id.strip():
                raise ValueError("Configuration ID cannot be empty")

            file_path = os.path.join(self.config_dir, f'{config_id}.json')

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Configuration {config_id} not found")

            # 读取现有配置
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_record = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in config file {config_id}: {str(e)}")
                raise ValueError(f"Configuration file {config_id} is corrupted")

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

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update configuration {config_id}: {str(e)}")
            raise ValueError(f"Failed to update configuration: {str(e)}")

    def delete_configuration(self, config_id: str):
        """删除配置"""
        try:
            if not config_id or not config_id.strip():
                raise ValueError("Configuration ID cannot be empty")

            file_path = os.path.join(self.config_dir, f'{config_id}.json')

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Configuration {config_id} not found")

            os.remove(file_path)
            logger.info(f"Configuration deleted: {config_id}")

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete configuration {config_id}: {str(e)}")
            raise ValueError(f"Failed to delete configuration: {str(e)}")

    def get_configuration_count(self) -> int:
        """获取配置总数"""
        try:
            filenames = [f for f in os.listdir(self.config_dir) if f.endswith('.json')]
            return len(filenames)
        except Exception as e:
            logger.error(f"Failed to get configuration count: {str(e)}")
            return 0

    def validate_configuration(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置数据"""
        try:
            errors = []
            warnings = []

            # 检查必要字段
            if not isinstance(config_data, dict):
                errors.append("Configuration must be a dictionary")
                return {'valid': False, 'errors': errors, 'warnings': warnings}

            # 检查基本字段
            if 'name' in config_data and not isinstance(config_data['name'], str):
                errors.append("Name must be a string")

            if 'description' in config_data and not isinstance(config_data['description'], str):
                errors.append("Description must be a string")

            # 可以根据需要添加更多验证规则
            # 例如：验证雷达参数、环境参数、目标参数等

            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }

        except Exception as e:
            logger.error(f"Failed to validate configuration: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': []
            }

    def search_configurations(self, query: str, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """搜索配置"""
        try:
            if not query or not query.strip():
                return self.list_configurations(page, per_page)

            # 增强参数验证
            if page < 1:
                raise ValueError("Page number must be greater than 0")
            if per_page < 1 or per_page > 100:
                raise ValueError("Per page must be between 1 and 100")

            query = query.lower().strip()
            matching_configs = []

            try:
                filenames = [f for f in os.listdir(self.config_dir) if f.endswith('.json')]
            except OSError as e:
                logger.error(f"Failed to read config directory: {str(e)}")
                raise ValueError("Failed to access configuration directory")

            # 搜索匹配的配置
            for filename in filenames:
                try:
                    file_path = os.path.join(self.config_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        config_record = json.load(f)

                    # 验证配置记录的必要字段
                    if not all(key in config_record for key in ['id', 'name', 'created_at']):
                        continue

                    # 搜索名称和描述
                    name_match = query in config_record['name'].lower()
                    desc_match = query in config_record.get('description', '').lower()

                    if name_match or desc_match:
                        config_summary = {
                            'id': config_record['id'],
                            'name': config_record['name'],
                            'description': config_record.get('description', ''),
                            'created_at': config_record['created_at'],
                            'updated_at': config_record.get('updated_at', config_record['created_at'])
                        }
                        matching_configs.append(config_summary)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in config file {filename}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to search in config {filename}: {str(e)}")
                    continue

            # 按更新时间倒序排列
            try:
                matching_configs.sort(key=lambda x: x['updated_at'], reverse=True)
            except Exception as e:
                logger.warning(f"Failed to sort search results: {str(e)}")

            # 实现分页
            total_count = len(matching_configs)
            start_index = (page - 1) * per_page
            end_index = start_index + per_page

            if start_index >= total_count and total_count > 0:
                raise ValueError(f"Page {page} is out of range")

            paginated_configs = matching_configs[start_index:end_index]
            total_pages = (total_count + per_page - 1) // per_page

            return {
                'configs': paginated_configs,
                'pagination': {
                    'current_page': page,
                    'per_page': per_page,
                    'total_count': total_count,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                },
                'query': query
            }

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to search configurations: {str(e)}")
            raise ValueError(f"Failed to search configurations: {str(e)}")

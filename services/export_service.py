import json
import uuid
import os
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import logging
import threading
import zipfile
from io import StringIO

logger = logging.getLogger(__name__)


class ExportService:
    """数据导出服务"""

    def __init__(self):
        self.export_dir = 'data/exports'
        self.export_tasks = {}  # 导出任务状态
        self._ensure_export_dir()

    def _ensure_export_dir(self):
        """确保导出目录存在"""
        os.makedirs(self.export_dir, exist_ok=True)

    def start_export(self, export_request: Dict[str, Any]) -> str:
        """开始导出任务"""
        export_id = str(uuid.uuid4())

        # 记录导出任务状态
        self.export_tasks[export_id] = {
            'id': export_id,
            'status': 'pending',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'config': export_request
        }

        # 在后台线程中执行导出
        thread = threading.Thread(target=self._execute_export, args=(export_id, export_request))
        thread.daemon = True
        thread.start()

        return export_id

    def _execute_export(self, export_id: str, export_request: Dict[str, Any]):
        """执行导出任务"""
        try:
            self.export_tasks[export_id]['status'] = 'processing'

            simulation_id = export_request.get('simulation_id')
            data_types = export_request.get('data_types', [])
            export_format = export_request.get('format', 'json')
            time_range = export_request.get('time_range', {})

            # 获取仿真数据
            from services.simulation_service import SimulationService
            simulation_service = SimulationService()

            # 更新进度
            self.export_tasks[export_id]['progress'] = 20

            simulation_data = simulation_service.get_simulation_data(
                simulation_id,
                time_range.get('start', 0),
                time_range.get('end'),
                data_types
            )

            # 更新进度
            self.export_tasks[export_id]['progress'] = 50

            # 根据格式导出数据
            file_path = self._export_data(export_id, simulation_data, export_format, export_request)

            # 完成导出
            self.export_tasks[export_id].update({
                'status': 'completed',
                'progress': 100,
                'file_path': file_path,
                'completed_at': datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            self.export_tasks[export_id].update({
                'status': 'failed',
                'error': str(e),
                'completed_at': datetime.now().isoformat()
            })

    def _export_data(self, export_id: str, data: Dict[str, Any],
                     format_type: str, config: Dict[str, Any]) -> str:
        """导出数据到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format_type == 'json':
            filename = f'simulation_export_{export_id[:8]}_{timestamp}.json'
            file_path = os.path.join(self.export_dir, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        elif format_type == 'csv':
            filename = f'simulation_export_{export_id[:8]}_{timestamp}.zip'
            file_path = os.path.join(self.export_dir, filename)

            # 创建ZIP文件包含多个CSV
            with zipfile.ZipFile(file_path, 'w') as zipf:
                for data_type, type_data in data.items():
                    if isinstance(type_data, list) and type_data:
                        csv_content = self._convert_to_csv(type_data)
                        zipf.writestr(f'{data_type}.csv', csv_content)

        elif format_type == 'excel':
            filename = f'simulation_export_{export_id[:8]}_{timestamp}.xlsx'
            file_path = os.path.join(self.export_dir, filename)

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for data_type, type_data in data.items():
                    if isinstance(type_data, list) and type_data:
                        df = pd.DataFrame(type_data)
                        df.to_excel(writer, sheet_name=data_type, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        return file_path

    def _convert_to_csv(self, data: List[Dict]) -> str:
        """转换数据为CSV格式"""
        if not data:
            return ""

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

        return output.getvalue()

    def get_export_status(self, export_id: str) -> Dict[str, Any]:
        """获取导出状态"""
        if export_id not in self.export_tasks:
            raise ValueError(f"Export task {export_id} not found")

        return self.export_tasks[export_id]

    def get_export_file_path(self, export_id: str) -> str:
        """获取导出文件路径"""
        if export_id not in self.export_tasks:
            raise ValueError(f"Export task {export_id} not found")

        task = self.export_tasks[export_id]
        if task['status'] != 'completed':
            raise ValueError(f"Export task {export_id} is not completed")

        return task['file_path']

import json
import uuid
import os
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import threading
import zipfile
from io import BytesIO, StringIO
import time
import gc
import ujson  # 更快的JSON库
import fastparquet  # 高性能Parquet处理
import pyarrow as pa  # 高性能内存处理
import pyarrow.csv as csv_pa  # 高性能CSV处理
import zstandard as zstd  # 高性能压缩

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


class ExportService:
    """高性能数据导出服务"""

    # 导出格式常量
    EXPORT_FORMATS = ['json', 'csv', 'excel', 'parquet', 'jsonl', 'csv_gzip']

    # 导出配置常量
    DEFAULT_CHUNK_SIZE = 10000  # 默认分块大小
    MAX_MEMORY_MB = 1024  # 最大内存使用限制(MB)

    def __init__(self, export_dir: str = 'data/exports'):
        self.export_dir = export_dir
        self.export_tasks = {}  # 导出任务状态
        self._lock = threading.Lock()  # 线程安全锁
        self._ensure_export_dir()

    def _ensure_export_dir(self):
        """确保导出目录存在"""
        os.makedirs(self.export_dir, exist_ok=True)

    def start_export(self, export_request: Dict[str, Any]) -> str:
        """开始导出任务"""
        export_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()

        # 验证导出格式
        export_format = export_request.get('format', 'json').lower()
        if export_format not in self.EXPORT_FORMATS:
            raise ValueError(f"Unsupported export format: {export_format}")

        # 记录导出任务状态
        with self._lock:
            self.export_tasks[export_id] = {
                'id': export_id,
                'status': 'pending',
                'progress': 0,
                'created_at': created_at,
                'started_at': None,
                'completed_at': None,
                'file_path': None,
                'error': None,
                'config': export_request,
                'data_size': 0,
                'memory_usage': 0,
                'time_elapsed': 0.0
            }

        # 在后台线程中执行导出
        thread = threading.Thread(
            target=self._execute_export,
            args=(export_id, export_request),
            name=f"ExportThread-{export_id[:8]}"
        )
        thread.daemon = True
        thread.start()

        return export_id

    def _execute_export(self, export_id: str, export_request: Dict[str, Any]):
        """执行导出任务"""
        start_time = time.time()
        task = self._get_task(export_id)

        try:
            # 更新任务状态
            self._update_task(export_id, {
                'status': 'processing',
                'started_at': datetime.now().isoformat()
            })

            # 获取配置参数
            simulation_id = export_request.get('simulation_id')
            data_types = export_request.get('data_types', [])
            export_format = export_request.get('format', 'json').lower()
            time_range = export_request.get('time_range', {})
            chunk_size = export_request.get('chunk_size', self.DEFAULT_CHUNK_SIZE)

            # 获取仿真数据（模拟）
            # 实际项目中应替换为真实数据获取逻辑
            logger.info(f"Export {export_id}: Fetching simulation data for {simulation_id}")
            self._update_task(export_id, {'progress': 20})

            # 生成模拟数据（代替真实数据获取）
            simulation_data = self._generate_mock_data(data_types, chunk_size)
            data_size = sum(len(v) for v in simulation_data.values())

            self._update_task(export_id, {
                'progress': 40,
                'data_size': data_size
            })

            # 根据格式导出数据
            logger.info(f"Export {export_id}: Exporting {data_size} records in {export_format} format")
            file_path = self._export_data(
                export_id,
                simulation_data,
                export_format,
                export_request
            )

            # 完成导出
            end_time = time.time()
            elapsed = end_time - start_time

            self._update_task(export_id, {
                'status': 'completed',
                'progress': 100,
                'file_path': file_path,
                'completed_at': datetime.now().isoformat(),
                'time_elapsed': round(elapsed, 2),
                'memory_usage': self._get_memory_usage()
            })

            logger.info(f"Export {export_id}: Completed in {elapsed:.2f}s. File: {file_path}")

        except Exception as e:
            logger.error(f"Export {export_id} error: {str(e)}", exc_info=True)
            self._update_task(export_id, {
                'status': 'failed',
                'error': str(e),
                'completed_at': datetime.now().isoformat(),
                'time_elapsed': round(time.time() - start_time, 2)
            })

    def _export_data(self, export_id: str, data: Dict[str, Any],
                     format_type: str, config: Dict[str, Any]) -> str:
        """导出数据到文件 - 使用高性能方法"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_id_short = export_id[:8]

        # 根据格式选择导出方法
        if format_type == 'json':
            return self._export_json(export_id_short, timestamp, data)
        elif format_type == 'jsonl':
            return self._export_jsonl(export_id_short, timestamp, data)
        elif format_type == 'csv':
            return self._export_csv(export_id_short, timestamp, data)
        elif format_type == 'csv_gzip':
            return self._export_csv_gzip(export_id_short, timestamp, data)
        elif format_type == 'excel':
            return self._export_excel(export_id_short, timestamp, data, config)
        elif format_type == 'parquet':
            return self._export_parquet(export_id_short, timestamp, data)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_json(self, export_id: str, timestamp: str, data: Dict[str, Any]) -> str:
        """导出为JSON格式 - 使用ujson优化"""
        filename = f'simulation_export_{export_id}_{timestamp}.json'
        file_path = os.path.join(self.export_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            ujson.dump(data, f, ensure_ascii=False, indent=2)

        return file_path

    def _export_jsonl(self, export_id: str, timestamp: str, data: Dict[str, Any]) -> str:
        """导出为JSON Lines格式 - 高性能流式导出"""
        filename = f'simulation_export_{export_id}_{timestamp}.jsonl'
        file_path = os.path.join(self.export_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            for data_type, records in data.items():
                for record in records:
                    f.write(ujson.dumps(record) + '\n')

        return file_path

    def _export_csv(self, export_id: str, timestamp: str, data: Dict[str, Any]) -> str:
        """导出为CSV格式 - 使用PyArrow优化"""
        filename = f'simulation_export_{export_id}_{timestamp}.zip'
        file_path = os.path.join(self.export_dir, filename)

        with zipfile.ZipFile(file_path, 'w') as zipf:
            for data_type, records in data.items():
                if records:
                    # 使用PyArrow高效转换
                    table = pa.Table.from_pylist(records)
                    csv_buffer = BytesIO()
                    csv_pa.write_csv(table, csv_buffer)

                    # 添加到ZIP
                    zipf.writestr(f'{data_type}.csv', csv_buffer.getvalue())

        return file_path

    def _export_csv_gzip(self, export_id: str, timestamp: str, data: Dict[str, Any]) -> str:
        """导出为Gzip压缩的CSV格式 - 高性能压缩"""
        filename = f'simulation_export_{export_id}_{timestamp}.csv.gz'
        file_path = os.path.join(self.export_dir, filename)

        # 使用PyArrow高效写入压缩CSV
        all_records = []
        for records in data.values():
            all_records.extend(records)

        if all_records:
            table = pa.Table.from_pylist(all_records)
            csv_pa.write_csv(
                table,
                file_path,
                write_options=csv_pa.WriteOptions(include_header=True),
                compression='gzip'
            )

        return file_path

    def _export_excel(self, export_id: str, timestamp: str,
                      data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """导出为Excel格式 - 优化大文件处理"""
        filename = f'simulation_export_{export_id}_{timestamp}.xlsx'
        file_path = os.path.join(self.export_dir, filename)
        chunk_size = config.get('chunk_size', self.DEFAULT_CHUNK_SIZE)

        # 使用分块写入避免内存溢出
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            for data_type, records in data.items():
                if records:
                    # 分块处理大数据
                    for i in range(0, len(records), chunk_size):
                        chunk = records[i:i + chunk_size]
                        df = pd.DataFrame(chunk)

                        # 首次写入包含表头
                        header = (i == 0)
                        df.to_excel(
                            writer,
                            sheet_name=data_type[:31],
                            index=False,
                            startrow=i if i > 0 else 0,
                            header=header
                        )

                        # 更新进度
                        progress = 60 + int(30 * (i + len(chunk)) / len(records))
                        self._update_task(export_id, {'progress': progress})

                        # 手动清理内存
                        del df, chunk
                        gc.collect()

        return file_path

    def _export_parquet(self, export_id: str, timestamp: str, data: Dict[str, Any]) -> str:
        """导出为Parquet格式 - 高性能列式存储"""
        filename = f'simulation_export_{export_id}_{timestamp}.parquet'
        file_path = os.path.join(self.export_dir, filename)

        # 使用PyArrow高效处理
        tables = []
        for data_type, records in data.items():
            if records:
                table = pa.Table.from_pylist(records)
                tables.append(table)

        if tables:
            combined_table = pa.concat_tables(tables)
            pa.parquet.write_table(combined_table, file_path, compression='SNAPPY')

        return file_path

    def _generate_mock_data(self, data_types: List[str], chunk_size: int) -> Dict[str, Any]:
        """生成模拟数据（代替真实数据获取）"""
        mock_data = {}
        for data_type in data_types:
            # 生成模拟数据
            records = []
            for i in range(chunk_size):
                records.append({
                    'id': i,
                    'timestamp': datetime.now().isoformat(),
                    'value': i * 0.5,
                    'status': 'active' if i % 2 == 0 else 'inactive',
                    'data_type': data_type,
                    'metadata': {'key': 'value' * (i % 10)}
                })
            mock_data[data_type] = records
        return mock_data

    def _update_task(self, export_id: str, updates: Dict[str, Any]):
        """更新任务状态（线程安全）"""
        with self._lock:
            if export_id in self.export_tasks:
                self.export_tasks[export_id].update(updates)

    def _get_task(self, export_id: str) -> Dict[str, Any]:
        """获取任务（线程安全）"""
        with self._lock:
            return self.export_tasks.get(export_id, {})

    def _get_memory_usage(self) -> int:
        """获取当前内存使用（MB）"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss // (1024 * 1024)

    def get_export_status(self, export_id: str) -> Dict[str, Any]:
        """获取导出状态"""
        return self._get_task(export_id)

    def get_export_file_path(self, export_id: str) -> Optional[str]:
        """获取导出文件路径"""
        task = self._get_task(export_id)
        if task.get('status') == 'completed':
            return task.get('file_path')
        return None

    def cancel_export(self, export_id: str) -> bool:
        """取消导出任务"""
        task = self._get_task(export_id)
        if task and task['status'] in ['pending', 'processing']:
            self._update_task(export_id, {
                'status': 'cancelled',
                'completed_at': datetime.now().isoformat()
            })
            return True
        return False

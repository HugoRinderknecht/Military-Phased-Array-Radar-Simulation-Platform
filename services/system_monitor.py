import psutil
import time
import threading
import logging
import json
import platform
import os

logger = logging.getLogger(__name__)


class SystemMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """单例模式实现，确保只有一个SystemMonitor实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SystemMonitor, cls).__new__(cls)
        return cls._instance

    def __init__(self, socketio, interval=5.0):
        # 防止重复初始化
        if hasattr(self, '_initialized'):
            logger.warning("SystemMonitor already initialized, updating socketio only")
            self.socketio = socketio
            return

        self.socketio = socketio
        self.interval = max(interval, 5.0)  # 最小间隔5秒
        self.monitor_thread = None
        self.running = False
        self.last_network_io = None
        self.error_count = 0
        self.max_errors = 10
        self._thread_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_emit_time = 0
        self._min_emit_interval = 5.0  # 强制最小发送间隔
        self._initialized = True

        logger.info(f"SystemMonitor initialized with interval {self.interval}s")

    def start(self):
        """启动系统监控"""
        with self._thread_lock:
            if self.running:
                logger.warning("System monitor is already running")
                return False

            if self.monitor_thread and self.monitor_thread.is_alive():
                logger.warning("Previous monitor thread is still alive")
                return False

            # 检查是否在正确的进程中
            if not self._should_start_monitor():
                logger.info("Skipping monitor start in non-main process")
                return False

            self.running = True
            self.error_count = 0
            self._stop_event.clear()
            self._last_emit_time = 0

            # 初始化CPU监控
            try:
                psutil.cpu_percent()
            except Exception as e:
                logger.warning(f"Failed to initialize CPU monitoring: {e}")

            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="SystemMonitor"
            )
            self.monitor_thread.start()
            logger.info(f"System monitor started with {self.interval}s interval")
            return True

    def _should_start_monitor(self):
        """检查是否应该启动监控（避免在重载进程中启动）"""
        # 在Windows调试模式下，只在主进程中启动
        if os.name == 'nt' and os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
            return False
        return True

    def stop(self):
        """停止系统监控"""
        with self._thread_lock:
            if not self.running:
                return

            logger.info("Stopping system monitor...")
            self.running = False
            self._stop_event.set()

        # 等待线程结束
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3.0)
            if self.monitor_thread.is_alive():
                logger.warning("System monitor thread did not stop gracefully")

        logger.info("System monitor stopped")

    def _safe_get_disk_usage(self):
        """安全获取磁盘使用情况"""
        try:
            if platform.system() == 'Windows':
                disk_usage = psutil.disk_usage('C:\\')
            else:
                disk_usage = psutil.disk_usage('/')

            return {
                'total': int(disk_usage.total),
                'used': int(disk_usage.used),
                'free': int(disk_usage.free),
                'percent': round(float(disk_usage.percent), 2)
            }
        except Exception as e:
            logger.debug(f"Error getting disk usage: {e}")
            return {
                'total': 1000000000000,
                'used': 500000000000,
                'free': 500000000000,
                'percent': 50.0
            }

    def _safe_get_network_io(self):
        """安全获取网络IO"""
        try:
            net_io = psutil.net_io_counters()
            if net_io is None:
                raise Exception("No network data available")

            current_data = {
                'sent': int(net_io.bytes_sent),
                'received': int(net_io.bytes_recv)
            }

            if self.last_network_io:
                current_data['sent_rate'] = max(0, current_data['sent'] - self.last_network_io['sent'])
                current_data['received_rate'] = max(0, current_data['received'] - self.last_network_io['received'])
            else:
                current_data['sent_rate'] = 0
                current_data['received_rate'] = 0

            self.last_network_io = {
                'sent': current_data['sent'],
                'received': current_data['received']
            }

            return current_data
        except Exception as e:
            logger.debug(f"Error getting network IO: {e}")
            return {
                'sent': 0,
                'received': 0,
                'sent_rate': 0,
                'received_rate': 0
            }

    def _should_emit(self):
        """检查是否应该发送数据（防止过于频繁）"""
        current_time = time.time()
        if current_time - self._last_emit_time < self._min_emit_interval:
            return False
        self._last_emit_time = current_time
        return True

    def _monitor_loop(self):
        """监控循环"""
        logger.info("System monitor loop started")

        # 初始延迟，避免启动时立即发送
        if self._stop_event.wait(timeout=2.0):
            return

        while self.running and not self._stop_event.is_set():
            loop_start_time = time.time()

            try:
                # 检查是否应该发送数据
                if not self._should_emit():
                    # 等待剩余时间
                    remaining_time = self.interval - (time.time() - loop_start_time)
                    if remaining_time > 0:
                        if self._stop_event.wait(timeout=remaining_time):
                            break
                    continue

                # 获取系统资源信息
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk_data = self._safe_get_disk_usage()
                network_data = self._safe_get_network_io()

                # 构建数据
                resource_data = {
                    'cpu': round(float(cpu_percent), 2),
                    'memory': {
                        'total': int(memory.total),
                        'available': int(memory.available),
                        'used': int(memory.used),
                        'percent': round(float(memory.percent), 2)
                    },
                    'disk': disk_data,
                    'network': network_data,
                    'timestamp': time.time(),
                    'monitor_id': id(self)  # 添加监控器ID用于调试
                }

                # 发送数据
                if self.socketio and self.running:
                    try:
                        json.dumps(resource_data)  # 验证序列化
                        self.socketio.emit('system_resources', resource_data, namespace='/')
                        logger.debug(f"System resources emitted (Monitor ID: {id(self)})")
                    except Exception as emit_error:
                        logger.warning(f"Failed to emit system resources: {emit_error}")
                        self.error_count += 1

                # 重置错误计数
                self.error_count = max(0, self.error_count - 1) if self.error_count > 0 else 0

            except Exception as e:
                self.error_count += 1
                error_msg = f"System monitoring error (attempt {self.error_count}): {str(e)}"

                if self.error_count <= 3:
                    logger.warning(error_msg)
                else:
                    logger.error(error_msg)

                if self.error_count >= self.max_errors:
                    logger.error("Too many consecutive errors, stopping monitor")
                    self.running = False
                    break

            # 计算等待时间
            execution_time = time.time() - loop_start_time
            sleep_time = max(0, self.interval - execution_time)

            if self._stop_event.wait(timeout=sleep_time):
                break

        logger.info("System monitor loop ended")

    def get_current_stats(self):
        """获取当前统计信息"""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk_data = self._safe_get_disk_usage()

            return {
                'cpu': round(float(cpu_percent), 2),
                'memory_percent': round(float(memory.percent), 2),
                'disk_percent': round(float(disk_data['percent']), 2),
                'status': 'active' if self.running else 'inactive',
                'timestamp': time.time(),
                'error_count': self.error_count,
                'monitor_id': id(self)
            }
        except Exception as e:
            logger.error(f"Error getting current stats: {str(e)}")
            return {
                'cpu': 0.0,
                'memory_percent': 0.0,
                'disk_percent': 0.0,
                'status': 'error',
                'timestamp': time.time(),
                'error_count': self.error_count,
                'monitor_id': id(self)
            }

    def is_running(self):
        """检查是否正在运行"""
        return self.running and self.monitor_thread and self.monitor_thread.is_alive()

    def get_monitor_info(self):
        """获取监控器信息"""
        return {
            'running': self.running,
            'interval': self.interval,
            'error_count': self.error_count,
            'thread_alive': self.monitor_thread.is_alive() if self.monitor_thread else False,
            'thread_name': self.monitor_thread.name if self.monitor_thread else None,
            'monitor_id': id(self),
            'last_emit_time': self._last_emit_time,
            'min_emit_interval': self._min_emit_interval
        }

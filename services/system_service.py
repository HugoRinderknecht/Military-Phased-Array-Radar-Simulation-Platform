import psutil
import GPUtil


class SystemService:
    def get_resource_usage(self):
        """获取系统资源使用情况"""
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        gpu_usage = 0.0
        gpu_temp = 0.0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100
                gpu_temp = gpus[0].temperature
        except Exception:
            pass

        return {
            'cpu': {
                'usage': cpu_usage,
                'cores': psutil.cpu_count(logical=False),
                'threads': psutil.cpu_count(logical=True)
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'gpu': {
                'usage': gpu_usage,
                'temperature': gpu_temp
            }
        }

    def get_compute_load(self):
        """获取计算负载情况"""
        load_avg = psutil.getloadavg()
        return {
            'load_1min': load_avg[0],
            'load_5min': load_avg[1],
            'load_15min': load_avg[2],
            'process_count': len(psutil.pids())
        }

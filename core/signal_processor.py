import numpy as np
import pyfftw
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass, field
from enum import Enum
import heapq
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import scipy.signal
from scipy import stats


class TaskType(Enum):
    TARGET_CONFIRMATION = 1
    HIGH_PRIORITY_TRACKING = 2
    LOST_TARGET_SEARCH = 3
    WEAK_TARGET_TRACKING = 4
    NORMAL_TRACKING = 5
    AREA_SEARCH = 6


@dataclass
class Detection:
    """雷达检测结果"""
    range: float
    snr: float
    velocity: float
    angle: Optional[float] = None
    amplitude: Optional[float] = None
    timestamp: Optional[float] = None


@dataclass
class RadarTask:
    task_id: int
    task_type: TaskType
    duration: float
    release_time: float
    due_time: float
    target_id: Optional[int] = None
    beam_position: Optional[Dict[str, float]] = None
    hard_constraint: bool = False
    priority: float = field(init=False)

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError("Task duration must be positive")
        if self.due_time < self.release_time:
            raise ValueError("Due time cannot be earlier than release time")


@dataclass
class ScheduleResult:
    scheduled_tasks: List[RadarTask]
    delayed_tasks: List[RadarTask]
    cancelled_tasks: List[RadarTask]
    total_time: float
    efficiency: float
    doppler_results: Optional[Dict[int, np.ndarray]] = None


@dataclass
class SchedulingConfig:
    """调度配置参数"""
    schedule_interval: float = 60.0
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        'base': 0.5,
        'time': 0.3,
        'environment': 0.2
    })
    time_step: float = 1.0
    max_delay_ratio: float = 0.5


class SignalProcessor:
    """雷达信号处理器"""

    def __init__(self, radar_system, environment):
        self.radar_system = radar_system
        self.environment = environment
        self.c = 299792458.0  # 光速

        # 信号处理参数
        self.sample_rate = 1e6  # 1 MHz
        self.pulse_width = 1e-6  # 1 μs
        self.prf = 1000  # 脉冲重复频率 1 kHz

        # CFAR 参数
        self.cfar_guard_cells = 4
        self.cfar_reference_cells = 16
        self.cfar_threshold_factor = 2.5

        # FFT 设置
        self.fft_size = 1024
        self.doppler_processor = DopplerProcessor(self.fft_size)

        self.logger = logging.getLogger(__name__)

    def process_radar_signal(self, signal: np.ndarray, timestamp: float) -> List[Detection]:
        """处理雷达信号的主要方法"""
        try:
            # 1. 自适应杂波抑制
            clutter_suppressed = self._adaptive_clutter_suppression(signal)

            # 2. 脉冲压缩
            pulse_compressed = self._pulse_compression(clutter_suppressed)

            # 3. CFAR 检测
            detections = self._cfar_detection(pulse_compressed, timestamp)

            return detections

        except Exception as e:
            self.logger.error(f"Signal processing failed: {e}")
            return []

    def _adaptive_clutter_suppression(self, signal: np.ndarray) -> np.ndarray:
        """自适应杂波抑制"""
        # 使用高通滤波器移除低频杂波
        nyquist = self.sample_rate / 2
        cutoff = 1000  # 1 kHz 截止频率
        normalized_cutoff = cutoff / nyquist

        # 设计巴特沃斯高通滤波器
        b, a = scipy.signal.butter(4, normalized_cutoff, btype='high')

        # 应用滤波器
        filtered_signal = scipy.signal.filtfilt(b, a, signal.real) + \
                          1j * scipy.signal.filtfilt(b, a, signal.imag)

        # 自适应权重调整
        noise_power = np.var(filtered_signal[:100])  # 估计噪声功率
        adaptive_gain = 1.0 / (1.0 + noise_power)

        return filtered_signal * adaptive_gain

    def _pulse_compression(self, signal: np.ndarray) -> np.ndarray:
        """脉冲压缩处理"""
        # 生成匹配滤波器（线性调频脉冲）
        pulse_samples = int(self.pulse_width * self.sample_rate)
        t = np.linspace(0, self.pulse_width, pulse_samples)

        # 线性调频参数
        bandwidth = 10e6  # 10 MHz 带宽
        chirp_rate = bandwidth / self.pulse_width

        # 生成参考信号（线性调频）
        reference_pulse = np.exp(1j * np.pi * chirp_rate * t ** 2)

        # 使用 PyFFTW 进行快速卷积
        if len(signal) >= self.fft_size:
            # 分段处理长信号
            compressed_signal = np.zeros_like(signal)
            step_size = self.fft_size - len(reference_pulse) + 1

            for i in range(0, len(signal) - len(reference_pulse) + 1, step_size):
                end_idx = min(i + self.fft_size, len(signal))
                segment = signal[i:end_idx]

                # 使用 scipy 的 correlate 进行匹配滤波
                segment_compressed = scipy.signal.correlate(segment, reference_pulse, mode='same')
                compressed_signal[i:i + len(segment_compressed)] = segment_compressed

            return compressed_signal
        else:
            # 短信号直接处理
            return scipy.signal.correlate(signal, reference_pulse, mode='same')

    def _cfar_detection(self, signal: np.ndarray, timestamp: float) -> List[Detection]:
        """恒虚警率(CFAR)检测"""
        detections = []
        signal_power = np.abs(signal) ** 2

        # CFAR 滑窗检测
        total_cells = 2 * (self.cfar_guard_cells + self.cfar_reference_cells)

        for i in range(total_cells, len(signal_power) - total_cells):
            # 计算保护单元和参考单元的索引
            left_ref_start = i - self.cfar_guard_cells - self.cfar_reference_cells
            left_ref_end = i - self.cfar_guard_cells
            right_ref_start = i + self.cfar_guard_cells + 1
            right_ref_end = i + self.cfar_guard_cells + self.cfar_reference_cells + 1

            # 计算参考单元的平均功率
            left_ref_power = np.mean(signal_power[left_ref_start:left_ref_end])
            right_ref_power = np.mean(signal_power[right_ref_start:right_ref_end])
            reference_power = (left_ref_power + right_ref_power) / 2

            # CFAR 门限
            threshold = self.cfar_threshold_factor * reference_power

            # 检测判决
            if signal_power[i] > threshold:
                # 计算检测参数
                range_val = self._calculate_range(i)
                snr = 10 * np.log10(signal_power[i] / reference_power)
                velocity = self._estimate_velocity(signal, i)

                detection = Detection(
                    range=range_val,
                    snr=snr,
                    velocity=velocity,
                    amplitude=np.abs(signal[i]),
                    timestamp=timestamp
                )
                detections.append(detection)

        return detections

    def _calculate_range(self, sample_index: int) -> float:
        """根据采样索引计算距离"""
        time_delay = sample_index / self.sample_rate
        range_val = self.c * time_delay / 2  # 双程时间
        return range_val

    def _estimate_velocity(self, signal: np.ndarray, center_index: int) -> float:
        """估计多普勒速度"""
        # 提取局部信号段进行多普勒分析
        window_size = min(64, len(signal) - center_index, center_index)
        start_idx = max(0, center_index - window_size // 2)
        end_idx = min(len(signal), center_index + window_size // 2)

        local_signal = signal[start_idx:end_idx]

        # 使用 FFT 估计多普勒频移
        fft_result = np.fft.fft(local_signal)
        fft_power = np.abs(fft_result) ** 2

        # 找到峰值频率
        peak_idx = np.argmax(fft_power)
        doppler_freq = np.fft.fftfreq(len(local_signal), 1 / self.sample_rate)[peak_idx]

        # 将多普勒频移转换为径向速度
        wavelength = self.c / self.radar_system.frequency
        velocity = doppler_freq * wavelength / 2

        return velocity


class PriorityCalculator:
    """统一的优先级计算器"""

    def __init__(self, config: SchedulingConfig):
        self.config = config

    def calculate_dynamic_priority(self, task: RadarTask, current_time: float) -> float:
        base_priority = task.task_type.value

        # 时间因子计算
        time_factor = self._calculate_time_factor(task, current_time)

        # 环境因子计算
        env_factor = self._calculate_environment_factor(task)

        # 综合优先级（数值越大优先级越高）
        weights = self.config.priority_weights
        comprehensive_priority = (
                weights['base'] * (10 - base_priority) +  # 反转基础优先级
                weights['time'] * time_factor +
                weights['environment'] * env_factor
        )

        return comprehensive_priority

    def _calculate_time_factor(self, task: RadarTask, current_time: float) -> float:
        if task.due_time <= current_time:
            return 10.0  # 已过期，最高时间优先级

        time_to_deadline = task.due_time - current_time
        schedule_interval = self.config.schedule_interval

        if time_to_deadline > schedule_interval:
            return 1.0

        return 1.0 + (schedule_interval / time_to_deadline)

    def _calculate_environment_factor(self, task: RadarTask) -> float:
        if task.task_type in [TaskType.TARGET_CONFIRMATION, TaskType.HIGH_PRIORITY_TRACKING]:
            return 1.5
        return 1.0


class SchedulingStrategy(ABC):
    """调度策略抽象基类"""

    @abstractmethod
    def schedule(self, tasks: List[RadarTask], current_time: float,
                 config: SchedulingConfig) -> ScheduleResult:
        pass


class PriorityBasedStrategy(SchedulingStrategy):
    """基于优先级的调度策略"""

    def __init__(self, priority_calculator: PriorityCalculator):
        self.priority_calculator = priority_calculator

    def schedule(self, tasks: List[RadarTask], current_time: float,
                 config: SchedulingConfig) -> ScheduleResult:
        # 计算动态优先级并排序（优先级高的在前）
        for task in tasks:
            task.priority = self.priority_calculator.calculate_dynamic_priority(task, current_time)

        tasks_sorted = sorted(tasks, key=lambda t: t.priority, reverse=True)

        scheduled = []
        delayed = []
        cancelled = []
        time_pointer = current_time

        for task in tasks_sorted:
            if self._can_schedule_task(task, time_pointer, current_time, config):
                scheduled.append(task)
                time_pointer += task.duration
            else:
                if self._should_delay_task(task, time_pointer, config):
                    delayed.append(task)
                else:
                    cancelled.append(task)

        efficiency = len(scheduled) / len(tasks) if tasks else 0.0

        return ScheduleResult(
            scheduled_tasks=scheduled,
            delayed_tasks=delayed,
            cancelled_tasks=cancelled,
            total_time=time_pointer - current_time,
            efficiency=efficiency
        )

    def _can_schedule_task(self, task: RadarTask, time_pointer: float,
                           current_time: float, config: SchedulingConfig) -> bool:
        if task.hard_constraint:
            return time_pointer + task.duration <= task.due_time

        return (time_pointer < current_time + config.schedule_interval and
                time_pointer + task.duration <= task.due_time)

    def _should_delay_task(self, task: RadarTask, time_pointer: float,
                           config: SchedulingConfig) -> bool:
        return task.due_time - time_pointer > task.duration * config.max_delay_ratio


class TimePointerStrategy(SchedulingStrategy):
    """改进的时间指针调度策略"""

    def __init__(self, priority_calculator: PriorityCalculator):
        self.priority_calculator = priority_calculator

    def schedule(self, tasks: List[RadarTask], current_time: float,
                 config: SchedulingConfig) -> ScheduleResult:
        # 使用优先队列优化任务选择
        available_tasks = []
        for task in tasks:
            task.priority = self.priority_calculator.calculate_dynamic_priority(task, current_time)
            heapq.heappush(available_tasks, (-task.priority, task.task_id, task))

        time_pointer = current_time
        scheduled = []
        delayed = []
        cancelled = []

        # 获取所有关键时间点
        time_points = self._get_time_points(tasks, current_time, config)

        for target_time in sorted(time_points):
            if target_time > current_time + config.schedule_interval:
                break

            time_pointer = max(time_pointer, target_time)

            # 找到最佳可调度任务
            best_task = self._find_best_schedulable_task(available_tasks, time_pointer)

            if best_task:
                scheduled.append(best_task)
                time_pointer += best_task.duration
                self._remove_task_from_heap(available_tasks, best_task)

        # 处理剩余任务
        remaining_tasks = [task for _, _, task in available_tasks]
        for task in remaining_tasks:
            if task.due_time > time_pointer:
                delayed.append(task)
            else:
                cancelled.append(task)

        efficiency = len(scheduled) / len(tasks) if tasks else 0.0

        return ScheduleResult(
            scheduled_tasks=scheduled,
            delayed_tasks=delayed,
            cancelled_tasks=cancelled,
            total_time=time_pointer - current_time,
            efficiency=efficiency
        )

    def _get_time_points(self, tasks: List[RadarTask], current_time: float,
                         config: SchedulingConfig) -> List[float]:
        """获取所有关键时间点"""
        time_points = {current_time}
        for task in tasks:
            time_points.add(task.release_time)
            time_points.add(task.due_time)
        return list(time_points)

    def _find_best_schedulable_task(self, available_tasks: List, time_pointer: float) -> Optional[RadarTask]:
        """从堆中找到最佳可调度任务"""
        temp_removed = []
        best_task = None

        while available_tasks:
            neg_priority, task_id, task = heapq.heappop(available_tasks)

            if (task.release_time <= time_pointer and
                    task.due_time >= time_pointer + task.duration):
                best_task = task
                # 将其他任务放回堆中
                for item in temp_removed:
                    heapq.heappush(available_tasks, item)
                break
            else:
                temp_removed.append((neg_priority, task_id, task))

        # 如果没找到合适任务，将所有任务放回堆中
        if not best_task:
            for item in temp_removed:
                heapq.heappush(available_tasks, item)

        return best_task

    def _remove_task_from_heap(self, available_tasks: List, task_to_remove: RadarTask):
        """从堆中移除指定任务"""
        for i, (neg_priority, task_id, task) in enumerate(available_tasks):
            if task.task_id == task_to_remove.task_id:
                available_tasks[i] = available_tasks[-1]
                available_tasks.pop()
                if i < len(available_tasks):
                    heapq.heapify(available_tasks)
                break


class DopplerProcessor:
    """多普勒处理器，使用PyFFTW优化FFT计算"""

    def __init__(self, fft_size: int = 1024, num_threads: int = 4):
        self.fft_size = fft_size
        self.num_threads = num_threads
        self._setup_fftw()
        self.logger = logging.getLogger(__name__)

    def _setup_fftw(self):
        """设置PyFFTW"""
        try:
            # 启用多线程
            pyfftw.config.NUM_THREADS = self.num_threads
            pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

            # 预分配数组和创建FFT对象
            self.input_array = pyfftw.empty_aligned(self.fft_size, dtype='complex128')
            self.output_array = pyfftw.empty_aligned(self.fft_size, dtype='complex128')

            # 创建FFTW对象（会自动优化）
            self.fft_forward = pyfftw.FFTW(
                self.input_array, self.output_array,
                direction='FFTW_FORWARD',
                flags=('FFTW_MEASURE',),
                threads=self.num_threads
            )

            self.fft_backward = pyfftw.FFTW(
                self.output_array, self.input_array,
                direction='FFTW_BACKWARD',
                flags=('FFTW_MEASURE',),
                threads=self.num_threads
            )
        except Exception as e:
            self.logger.warning(f"PyFFTW setup failed, falling back to numpy FFT: {e}")
            self.use_numpy_fft = True
        else:
            self.use_numpy_fft = False

    def process_doppler_batch(self, radar_data: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """批量处理多普勒数据"""
        results = {}

        # 使用线程池并行处理多个目标
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            future_to_target = {
                executor.submit(self._process_single_target, target_id, data): target_id
                for target_id, data in radar_data.items()
            }

            for future in future_to_target:
                target_id = future_to_target[future]
                try:
                    results[target_id] = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing target {target_id}: {e}")
                    results[target_id] = None

        return results

    def _process_single_target(self, target_id: int, data: np.ndarray) -> np.ndarray:
        """处理单个目标的多普勒数据"""
        if len(data) != self.fft_size:
            # 如果数据长度不匹配，进行填充或截断
            if len(data) < self.fft_size:
                padded_data = np.zeros(self.fft_size, dtype=complex)
                padded_data[:len(data)] = data
                data = padded_data
            else:
                data = data[:self.fft_size]

        if self.use_numpy_fft:
            # 使用 numpy FFT 作为后备
            fft_result = np.fft.fft(data)
        else:
            # 将数据复制到预分配的数组中
            self.input_array[:] = data
            # 执行FFT
            self.fft_forward()
            fft_result = self.output_array.copy()

        # 应用窗函数（汉明窗）
        window = np.hamming(self.fft_size)
        windowed_result = fft_result * window

        # 计算功率谱密度
        psd = np.abs(windowed_result) ** 2

        # 多普勒频率范围（假设PRF为1000Hz）
        prf = 1000.0
        doppler_freqs = np.fft.fftfreq(self.fft_size, 1 / prf)

        # 移零频到中心
        psd_shifted = np.fft.fftshift(psd)
        freqs_shifted = np.fft.fftshift(doppler_freqs)

        return np.column_stack([freqs_shifted, psd_shifted])


class ResourceScheduler:
    """优化后的资源调度器"""

    def __init__(self, config: SchedulingConfig = None):
        self.config = config or SchedulingConfig()
        self.current_time = 0.0
        self.task_queue = []
        self.lock = threading.Lock()

        # 初始化组件
        self.priority_calculator = PriorityCalculator(self.config)
        self.strategies = {
            'priority': PriorityBasedStrategy(self.priority_calculator),
            'time_pointer': TimePointerStrategy(self.priority_calculator)
        }

        # 初始化多普勒处理器
        self.doppler_processor = DopplerProcessor()

        self.logger = logging.getLogger(__name__)

    def schedule_resources(self, tasks: List[RadarTask], strategy: str = "priority",
                           radar_data: Dict[int, np.ndarray] = None) -> ScheduleResult:
        """主要的资源调度方法"""
        try:
            if strategy not in self.strategies:
                raise ValueError(f"Unknown strategy: {strategy}")

            # 执行调度
            result = self.strategies[strategy].schedule(tasks, self.current_time, self.config)

            # 如果提供了雷达数据，执行多普勒处理
            if radar_data:
                doppler_results = self.doppler_processor.process_doppler_batch(radar_data)
                result.doppler_results = doppler_results

            self.logger.info(f"Scheduled {len(result.scheduled_tasks)} tasks with {strategy} strategy")
            return result

        except Exception as e:
            self.logger.error(f"Scheduling failed: {e}")
            raise

    def add_task(self, task: RadarTask):
        """线程安全的任务添加"""
        with self.lock:
            task.priority = self.priority_calculator.calculate_dynamic_priority(task, self.current_time)
            heapq.heappush(self.task_queue, (-task.priority, task.task_id, task))

    def get_next_task(self) -> Optional[RadarTask]:
        """获取下一个最高优先级任务"""
        with self.lock:
            if self.task_queue:
                _, _, task = heapq.heappop(self.task_queue)
                return task
            return None

    def update_time(self, new_time: float):
        """更新当前时间"""
        if new_time < self.current_time:
            self.logger.warning("Time going backwards!")

        self.current_time = new_time

        # 重新计算队列中任务的优先级
        with self.lock:
            if self.task_queue:
                tasks = [task for _, _, task in self.task_queue]
                self.task_queue.clear()
                for task in tasks:
                    self.add_task(task)

    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态信息"""
        with self.lock:
            return {
                'current_time': self.current_time,
                'queue_size': len(self.task_queue),
                'config': self.config,
                'fft_size': self.doppler_processor.fft_size
            }


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 创建调度器
    config = SchedulingConfig(
        schedule_interval=120.0,
        priority_weights={'base': 0.4, 'time': 0.4, 'environment': 0.2}
    )
    scheduler = ResourceScheduler(config)

    # 创建测试任务
    tasks = [
        RadarTask(1, TaskType.TARGET_CONFIRMATION, 10.0, 0.0, 30.0, hard_constraint=True),
        RadarTask(2, TaskType.HIGH_PRIORITY_TRACKING, 15.0, 5.0, 50.0),
        RadarTask(3, TaskType.NORMAL_TRACKING, 8.0, 10.0, 60.0)
    ]

    # 模拟雷达数据
    radar_data = {
        1: np.random.random(1024) + 1j * np.random.random(1024),
        2: np.random.random(1024) + 1j * np.random.random(1024)
    }

    # 执行调度
    result = scheduler.schedule_resources(tasks, "priority", radar_data)

    print(f"Scheduled: {len(result.scheduled_tasks)} tasks")
    print(f"Efficiency: {result.efficiency:.2%}")
    if result.doppler_results:
        print(f"Doppler processing completed for {len(result.doppler_results)} targets")

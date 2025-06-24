import logging
from typing import List, Dict, Any, Optional

import numpy as np

from core.data_processor import DataProcessor
from core.rcs_processor import RCSProcessor
from core.resource_scheduler import ResourceScheduler, RadarTask, TaskType
from core.signal_processor import SignalProcessor
from models.environment import Environment, WeatherCondition
from models.radar_system import RadarSystem, SimulationParameters
from models.target import Target, Formation

logger = logging.getLogger(__name__)


class SimulationService:
    def __init__(self):
        # 首先初始化所有必要的属性
        self.signal_processor = None
        self.data_processor = None
        self.resource_scheduler = None
        self.rcs_processor = None
        self.current_time = 0.0
        self.simulation_data = []
        self.targets = []
        self.formations = []
        self.simulation_params = None
        self._active_simulations = {}

        # 然后调用重置方法
        self.reset_simulation()

    def reset_simulation(self):
        """重置仿真状态"""
        self.current_time = 0.0
        self.simulation_data = []
        self.targets = []
        self.formations = []
        self.simulation_params = None

        # 安全地清除处理器状态
        if hasattr(self, 'signal_processor') and self.signal_processor and hasattr(self.signal_processor,
                                                                                   'clear_state'):
            try:
                self.signal_processor.clear_state()
            except Exception as e:
                logger.warning(f"Error clearing signal processor state: {str(e)}")

        if hasattr(self, 'data_processor') and self.data_processor and hasattr(self.data_processor, 'clear_state'):
            try:
                self.data_processor.clear_state()
            except Exception as e:
                logger.warning(f"Error clearing data processor state: {str(e)}")

        if hasattr(self, 'resource_scheduler') and self.resource_scheduler and hasattr(self.resource_scheduler,
                                                                                       'clear_state'):
            try:
                self.resource_scheduler.clear_state()
            except Exception as e:
                logger.warning(f"Error clearing resource scheduler state: {str(e)}")

        if hasattr(self, 'rcs_processor') and self.rcs_processor and hasattr(self.rcs_processor, 'clear_state'):
            try:
                self.rcs_processor.clear_state()
            except Exception as e:
                logger.warning(f"Error clearing RCS processor state: {str(e)}")

        # 重置处理器为 None
        self.signal_processor = None
        self.data_processor = None
        self.resource_scheduler = None
        self.rcs_processor = None

        # 清除活跃仿真跟踪
        if hasattr(self, '_active_simulations'):
            self._active_simulations.clear()

    def initialize_simulation(self, parameters: Dict[str, Any], simulation_id: str = None) -> Dict[str, Any]:
        """初始化仿真"""
        try:
            # 验证必需的参数
            if not parameters:
                return {"status": "error", "message": "Parameters cannot be empty"}

            if 'radar' not in parameters:
                return {"status": "error", "message": "Missing radar configuration"}
            if 'environment' not in parameters:
                return {"status": "error", "message": "Missing environment configuration"}
            if 'targets' not in parameters:
                return {"status": "error", "message": "Missing targets configuration"}

            # 创建雷达系统
            radar_system = self._create_radar_system(parameters['radar'])
            if not radar_system:
                return {"status": "error", "message": "Failed to create radar system"}

            # 创建环境
            environment = self._create_environment(parameters['environment'])
            if not environment:
                return {"status": "error", "message": "Failed to create environment"}

            # 创建目标
            targets = self._create_targets(parameters['targets'])
            if not targets:
                return {"status": "error", "message": "Failed to create targets"}

            # 创建仿真参数
            self.simulation_params = SimulationParameters(
                radar_system=radar_system,
                environment=environment,
                targets=targets,
                simulation_time=max(parameters.get('simulation_time', 10.0), 0.1),
                time_step=max(parameters.get('time_step', 0.06), 0.01),
                monte_carlo_runs=max(parameters.get('monte_carlo_runs', 1), 1)
            )

            # 初始化处理器
            try:
                self.signal_processor = SignalProcessor(radar_system, environment)
                self.data_processor = DataProcessor(radar_system, environment)
                self.resource_scheduler = ResourceScheduler()
                self.rcs_processor = RCSProcessor(radar_system, environment)
            except Exception as e:
                logger.error(f"Failed to initialize processors: {str(e)}")
                return {"status": "error", "message": f"Failed to initialize processors: {str(e)}"}

            return {"status": "success", "message": "Simulation initialized successfully"}

        except Exception as e:
            logger.error(f"Simulation initialization error: {str(e)}")
            return {"status": "error", "message": f"Failed to initialize simulation: {str(e)}"}

    def run_simulation(self) -> Dict[str, Any]:
        """运行仿真"""
        try:
            # 检查是否已初始化
            if not self.simulation_params:
                return {"status": "error", "message": "Simulation not initialized. Call initialize_simulation first."}

            simulation_results = []
            successful_runs = 0

            for run in range(self.simulation_params.monte_carlo_runs):
                try:
                    run_results = self._execute_single_run(run)
                    if run_results and 'steps' in run_results and run_results['steps']:
                        simulation_results.append(run_results)
                        successful_runs += 1
                    else:
                        logger.warning(f"Empty results in simulation run {run}")

                except Exception as e:
                    logger.error(f"Error in simulation run {run}: {str(e)}")
                    # 记录失败的运行
                    simulation_results.append({
                        'run_id': run,
                        'error': str(e),
                        'steps': [],
                        'final_tracks': 0,
                        'total_detections': 0
                    })

            # 检查是否有成功的运行
            if successful_runs == 0:
                return {"status": "error", "message": "All simulation runs failed"}

            # 聚合结果
            try:
                aggregated_results = self._aggregate_results(simulation_results)
            except Exception as e:
                logger.error(f"Failed to aggregate results: {str(e)}")
                return {"status": "error", "message": f"Failed to aggregate results: {str(e)}"}

            return {
                "status": "success",
                "results": aggregated_results,
                "individual_runs": simulation_results,
                "successful_runs": successful_runs,
                "total_runs": self.simulation_params.monte_carlo_runs
            }

        except Exception as e:
            logger.error(f"Simulation execution error: {str(e)}")
            return {"status": "error", "message": f"Simulation failed: {str(e)}"}

    def _execute_single_run(self, run_id: int) -> Dict[str, Any]:
        """执行单次仿真运行"""
        try:
            # 重置仿真状态
            self.current_time = 0.0

            # 复制目标列表（避免修改原始数据）
            targets = [self._copy_target(target) for target in self.simulation_params.targets]
            if not targets:
                raise ValueError("No targets available for simulation")

            # 清空处理器状态
            if self.data_processor and hasattr(self.data_processor, 'tracks'):
                self.data_processor.tracks = []
            if self.data_processor and hasattr(self.data_processor, 'track_id_counter'):
                self.data_processor.track_id_counter = 1

            # 计算时间步数
            time_steps = max(int(self.simulation_params.simulation_time / self.simulation_params.time_step), 1)
            step_results = []

            for step in range(time_steps):
                try:
                    self.current_time = step * self.simulation_params.time_step

                    # 更新目标位置
                    self._update_target_positions(targets, self.simulation_params.time_step)

                    # 生成雷达回波
                    raw_signal = self._generate_radar_returns(targets)

                    # 信号处理
                    detections = []
                    if self.signal_processor:
                        try:
                            detections = self.signal_processor.process_radar_signal(raw_signal, self.current_time)
                        except Exception as e:
                            logger.warning(f"Signal processing error at step {step}: {str(e)}")
                            detections = []

                    # 数据处理
                    tracks = []
                    if self.data_processor:
                        try:
                            tracks = self.data_processor.process_tracking_data(detections)
                        except Exception as e:
                            logger.warning(f"Data processing error at step {step}: {str(e)}")
                            tracks = []

                    # 资源调度
                    schedule_efficiency = 0.0
                    if self.resource_scheduler:
                        try:
                            tasks = self._generate_radar_tasks(tracks)
                            schedule = self.resource_scheduler.schedule_resources(tasks)
                            schedule_efficiency = getattr(schedule, 'efficiency', 0.0)
                        except Exception as e:
                            logger.warning(f"Resource scheduling error at step {step}: {str(e)}")

                    # RCS处理
                    rcs_measurements = 0
                    if self.rcs_processor:
                        try:
                            rcs_results = {}
                            for track in tracks:
                                if hasattr(track, 'detections_history') and len(track.detections_history) > 0:
                                    rcs_results[track.track_id] = self.rcs_processor.process_rcs(track)
                            rcs_measurements = len(rcs_results)
                        except Exception as e:
                            logger.warning(f"RCS processing error at step {step}: {str(e)}")

                    # 记录步骤结果
                    step_result = {
                        'time': self.current_time,
                        'targets': len(targets),
                        'detections': len(detections),
                        'tracks': len(tracks),
                        'true_targets': len([t for t in targets if self._target_in_coverage(t)]),
                        'confirmed_tracks': len([t for t in tracks if getattr(t, 'confirmed', False)]),
                        'scheduling_efficiency': schedule_efficiency,
                        'rcs_measurements': rcs_measurements
                    }
                    step_results.append(step_result)

                except Exception as e:
                    logger.error(f"Error in step {step} of run {run_id}: {str(e)}")
                    # 继续下一步，但记录错误
                    step_result = {
                        'time': self.current_time,
                        'targets': 0,
                        'detections': 0,
                        'tracks': 0,
                        'true_targets': 0,
                        'confirmed_tracks': 0,
                        'scheduling_efficiency': 0.0,
                        'rcs_measurements': 0,
                        'error': str(e)
                    }
                    step_results.append(step_result)

            return {
                'run_id': run_id,
                'steps': step_results,
                'final_tracks': len(self.data_processor.tracks) if self.data_processor and hasattr(self.data_processor,
                                                                                                   'tracks') else 0,
                'total_detections': sum(s.get('detections', 0) for s in step_results)
            }

        except Exception as e:
            logger.error(f"Execution error in run {run_id}: {str(e)}")
            raise

    def _copy_target(self, target: Target) -> Target:
        """创建目标的深拷贝"""
        try:
            # 确保位置和速度是列表或数组
            position = list(target.position) if hasattr(target, 'position') else [0, 0, 0]
            velocity = list(target.velocity) if hasattr(target, 'velocity') else [0, 0, 0]

            new_target = Target(
                target_id=target.target_id,
                position=position,
                velocity=velocity,
                rcs=getattr(target, 'rcs', 0.0),
                altitude=getattr(target, 'altitude', position[2] if len(position) > 2 else 0),
                aspect_angle=getattr(target, 'aspect_angle', 0.0)
            )

            # 复制其他属性
            if hasattr(target, 'is_formation'):
                new_target.is_formation = target.is_formation
            if hasattr(target, 'formation_id'):
                new_target.formation_id = target.formation_id

            # 初始化极坐标参数
            new_target.range = np.sqrt(sum(p ** 2 for p in position))
            new_target.azimuth = np.arctan2(position[1], position[0]) if position[0] != 0 or position[1] != 0 else 0
            new_target.elevation = np.arcsin(position[2] / max(new_target.range, 1e-6)) if new_target.range > 0 else 0

            return new_target
        except Exception as e:
            logger.error(f"Error copying target {target.target_id}: {str(e)}")
            raise

    def _create_radar_system(self, radar_params: Dict[str, Any]) -> Optional[RadarSystem]:
        """创建雷达系统"""
        try:
            return RadarSystem(
                radar_area=max(radar_params.get('radar_area', 50.0), 1.0),
                tr_components=max(radar_params.get('tr_components', 500), 1),
                radar_power=max(radar_params.get('radar_power', 25000), 1000),
                frequency=max(radar_params.get('frequency', 10e9), 1e9),
                antenna_elements=max(radar_params.get('antenna_elements', 64), 1),
                beam_width=max(radar_params.get('beam_width', 2.0), 0.1),
                scan_rate=max(radar_params.get('scan_rate', 6.0), 0.1)
            )
        except Exception as e:
            logger.error(f"Error creating radar system: {str(e)}")
            return None

    def _create_environment(self, env_params: Dict[str, Any]) -> Optional[Environment]:
        """创建环境"""
        try:
            weather = WeatherCondition(
                weather_type=env_params.get('weather_type', 'clear'),
                precipitation_rate=max(env_params.get('precipitation_rate', 0.0), 0.0),
                visibility=max(env_params.get('visibility', 10000.0), 100.0),
                wind_speed=max(env_params.get('wind_speed', 0.0), 0.0),
                wind_direction=env_params.get('wind_direction', 0.0),
                temperature=env_params.get('temperature', 15.0),
                humidity=max(min(env_params.get('humidity', 50.0), 100.0), 0.0)
            )

            return Environment(
                weather=weather,
                clutter_density=max(min(env_params.get('clutter_density', 0.3), 1.0), 0.0),
                interference_level=max(min(env_params.get('interference_level', 0.1), 1.0), 0.0),
                multipath_factor=max(min(env_params.get('multipath_factor', 0.2), 1.0), 0.0),
                electronic_warfare=env_params.get('electronic_warfare', False),
                terrain_type=env_params.get('terrain_type', 'flat')
            )
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            return None

    def _create_targets(self, target_params: Dict[str, Any]) -> List[Target]:
        """创建目标"""
        targets = []
        try:
            num_targets = max(target_params.get('num_targets', 1), 1)
            specific_targets = target_params.get('specific_targets', [])
            target_id = 1

            for i in range(num_targets):
                try:
                    if i < len(specific_targets):
                        spec_target = specific_targets[i]
                        target = Target(
                            target_id=target_id,
                            position=list(spec_target.get('position', [1000, 1000, 1000])),
                            velocity=list(spec_target.get('velocity', [100, 0, 0])),
                            rcs=spec_target.get('rcs', 10.0),
                            altitude=spec_target.get('altitude', 1000),
                            aspect_angle=spec_target.get('aspect_angle', 0.0),
                            is_formation=spec_target.get('is_formation', False),
                            formation_id=spec_target.get('formation_id')
                        )
                    else:
                        target = self._generate_random_target(target_id, target_params)

                    targets.append(target)
                    target_id += 1
                except Exception as e:
                    logger.error(f"Error creating target {target_id}: {str(e)}")

            # 创建编队
            if target_params.get('formations') and targets:
                try:
                    self._create_formations(targets, target_params['formations'])
                except Exception as e:
                    logger.error(f"Error creating formations: {str(e)}")

        except Exception as e:
            logger.error(f"Error creating targets: {str(e)}")

        return targets

    def _generate_random_target(self, target_id: int, params: Dict[str, Any]) -> Target:
        """生成随机目标"""
        try:
            max_range = params.get('max_range', 100000)

            range_val = np.random.uniform(10000, max_range)
            azimuth = np.random.uniform(-np.pi, np.pi)
            elevation = np.random.uniform(-np.pi / 6, np.pi / 6)

            position = [
                range_val * np.cos(azimuth) * np.cos(elevation),
                range_val * np.sin(azimuth) * np.cos(elevation),
                range_val * np.sin(elevation)
            ]

            speed = np.random.uniform(100, 500)
            velocity_azimuth = azimuth + np.random.uniform(-np.pi / 4, np.pi / 4)

            velocity = [
                speed * np.cos(velocity_azimuth),
                speed * np.sin(velocity_azimuth),
                np.random.uniform(-50, 50)
            ]

            return Target(
                target_id=target_id,
                position=position,
                velocity=velocity,
                rcs=np.random.uniform(-10, 20),
                altitude=abs(position[2]),
                aspect_angle=np.random.uniform(0, 2 * np.pi)
            )
        except Exception as e:
            logger.error(f"Error generating random target {target_id}: {str(e)}")
            # 返回默认目标
            return Target(
                target_id=target_id,
                position=[10000, 0, 1000],
                velocity=[100, 0, 0],
                rcs=10.0,
                altitude=1000,
                aspect_angle=0.0
            )

    def _create_formations(self, targets: List[Target], formation_params: List[Dict]):
        """创建编队"""
        try:
            for formation_param in formation_params:
                leader_id = formation_param.get('leader_id')
                member_ids = formation_param.get('member_ids', [])

                leader = next((t for t in targets if t.target_id == leader_id), None)
                members = [t for t in targets if t.target_id in member_ids]

                if leader and members:
                    for member in members:
                        member.is_formation = True
                        member.formation_id = formation_param['formation_id']

                    formation = Formation(
                        formation_id=formation_param['formation_id'],
                        leader_target=leader,
                        members=members,
                        formation_type=formation_param.get('formation_type', 'line'),
                        spacing=formation_param.get('spacing', 100.0)
                    )
                    self.formations.append(formation)
        except Exception as e:
            logger.error(f"Error creating formations: {str(e)}")

    def _update_target_positions(self, targets: List[Target], dt: float):
        """更新目标位置"""
        try:
            # 先更新编队
            for formation in self.formations:
                if hasattr(formation, 'update_formation_positions'):
                    try:
                        formation.update_formation_positions(dt)
                    except Exception as e:
                        logger.warning(f"Error updating formation {formation.formation_id}: {str(e)}")

            # 更新非编队目标
            for target in targets:
                try:
                    if not getattr(target, 'is_formation', False) or getattr(target, 'formation_id', None) is None:
                        # 更新位置
                        target.position[0] += target.velocity[0] * dt
                        target.position[1] += target.velocity[1] * dt
                        target.position[2] += target.velocity[2] * dt

                        # 更新极坐标参数
                        target.range = np.sqrt(sum(p ** 2 for p in target.position))
                        target.azimuth = np.arctan2(target.position[1], target.position[0])
                        target.elevation = np.arcsin(target.position[2] / max(target.range, 1e-6))

                except Exception as e:
                    logger.warning(f"Error updating target {target.target_id}: {str(e)}")

        except Exception as e:
            logger.warning(f"Error updating target positions: {str(e)}")

    def _generate_radar_returns(self, targets: List[Target]) -> np.ndarray:
        """生成雷达回波信号"""
        try:
            signal_length = 2048
            signal = np.zeros(signal_length, dtype=complex)

            for target in targets:
                try:
                    if self._target_in_coverage(target):
                        range_bin = int(target.range * 2048 / 150000)
                        if 0 <= range_bin < signal_length:
                            snr = self._calculate_target_snr(target)
                            signal_amplitude = np.sqrt(max(10 ** (snr / 10), 1e-12))
                            phase = np.random.uniform(0, 2 * np.pi)
                            signal[range_bin] += signal_amplitude * np.exp(1j * phase)
                except Exception as e:
                    logger.warning(f"Error processing target {target.target_id} in radar returns: {str(e)}")

            # 添加噪声和杂波
            try:
                noise = np.random.normal(0, 1, signal_length) + 1j * np.random.normal(0, 1, signal_length)
                clutter = self._generate_clutter(signal_length)
                signal = signal + noise + clutter
            except Exception as e:
                logger.warning(f"Error adding noise and clutter: {str(e)}")

            return signal

        except Exception as e:
            logger.error(f"Error generating radar returns: {str(e)}")
            return np.zeros(2048, dtype=complex)

    def _target_in_coverage(self, target: Target) -> bool:
        """检查目标是否在覆盖范围内"""
        try:
            max_range = 150000
            max_elevation = np.pi / 3

            return (hasattr(target, 'range') and target.range < max_range and
                    hasattr(target, 'elevation') and abs(target.elevation) < max_elevation and
                    hasattr(target, 'azimuth') and abs(target.azimuth) < np.pi)
        except Exception as e:
            logger.warning(f"Error checking target coverage: {str(e)}")
            return False

    def _calculate_target_snr(self, target: Target) -> float:
        """计算目标信噪比"""
        try:
            radar_power = self.simulation_params.radar_system.radar_power
            gain = getattr(self.simulation_params.radar_system, 'gain', 40)  # dB
            wavelength = getattr(self.simulation_params.radar_system, 'wavelength', 0.03)  # m
            rcs_linear = 10 ** (target.rcs / 10)

            range_factor = max(target.range ** 4, 1e12)

            snr_linear = (radar_power * (10 ** (gain / 10)) ** 2 * wavelength ** 2 * rcs_linear) / \
                         ((4 * np.pi) ** 3 * range_factor)

            snr_db = 10 * np.log10(max(snr_linear, 1e-12))

            # 天气损失
            weather_loss = 0
            if (hasattr(self.simulation_params.environment, 'weather') and
                    hasattr(self.simulation_params.environment.weather, 'atmospheric_loss')):
                try:
                    weather_loss = self.simulation_params.environment.weather.atmospheric_loss(
                        self.simulation_params.radar_system.frequency, target.range / 1000
                    )
                except Exception as e:
                    logger.warning(f"Error calculating weather loss: {str(e)}")

            return snr_db - weather_loss

        except Exception as e:
            logger.warning(f"Error calculating SNR for target {target.target_id}: {str(e)}")
            return -10.0  # 默认低SNR

    def _generate_clutter(self, length: int) -> np.ndarray:
        """生成杂波"""
        try:
            clutter_density = self.simulation_params.environment.clutter_density
            clutter_power = -20  # dB

            clutter = np.zeros(length, dtype=complex)
            n_clutter_cells = int(length * clutter_density)

            for i in range(n_clutter_cells):
                cell_idx = np.random.randint(0, length)
                amplitude = np.sqrt(10 ** (clutter_power / 10)) * np.random.rayleigh(1)
                phase = np.random.uniform(0, 2 * np.pi)
                clutter[cell_idx] += amplitude * np.exp(1j * phase)

            return clutter

        except Exception as e:
            logger.warning(f"Error generating clutter: {str(e)}")
            return np.zeros(length, dtype=complex)

    def _generate_radar_tasks(self, tracks: List) -> List[RadarTask]:
        """生成雷达任务"""
        tasks = []
        try:
            task_id = 1

            for track in tracks:
                try:
                    if not getattr(track, 'confirmed', True):
                        task = RadarTask(
                            task_id=task_id,
                            task_type=TaskType.TARGET_CONFIRMATION,
                            duration=5.0,
                            release_time=self.current_time,
                            due_time=self.current_time + 20.0,
                            priority=1.0,
                            target_id=getattr(track, 'track_id', task_id),
                            hard_constraint=True
                        )
                        tasks.append(task)
                        task_id += 1

                    if getattr(track, 'high_mobility', False):
                        task = RadarTask(
                            task_id=task_id,
                            task_type=TaskType.HIGH_PRIORITY_TRACKING,
                            duration=3.0,
                            release_time=self.current_time,
                            due_time=self.current_time + 10.0,
                            priority=2.0,
                            target_id=getattr(track, 'track_id', task_id),
                            hard_constraint=True
                        )
                        tasks.append(task)
                        task_id += 1
                    else:
                        task = RadarTask(
                            task_id=task_id,
                            task_type=TaskType.NORMAL_TRACKING,
                            duration=2.0,
                            release_time=self.current_time,
                            due_time=self.current_time + 30.0,
                            priority=5.0,
                            target_id=getattr(track, 'track_id', task_id)
                        )
                        tasks.append(task)
                        task_id += 1

                except Exception as e:
                    logger.warning(f"Error creating task for track: {str(e)}")

        except Exception as e:
            logger.warning(f"Error generating radar tasks: {str(e)}")

        return tasks

    def _aggregate_results(self, simulation_results: List[Dict]) -> Dict[str, Any]:
        """聚合仿真结果"""
        try:
            if not simulation_results:
                return {'summary': {}, 'time_series': {}}

            # 过滤出成功的运行
            valid_results = [r for r in simulation_results if 'steps' in r and r['steps'] and 'error' not in r]

            if not valid_results:
                return {'summary': {}, 'time_series': {}}

            total_runs = len(valid_results)

            avg_final_tracks = np.mean([r.get('final_tracks', 0) for r in valid_results])
            avg_total_detections = np.mean([r.get('total_detections', 0) for r in valid_results])

            time_series_data = {}
            max_steps = max(len(r['steps']) for r in valid_results) if valid_results else 0

            for step_idx in range(max_steps):
                step_data = []
                for run_result in valid_results:
                    if step_idx < len(run_result['steps']):
                        step_data.append(run_result['steps'][step_idx])

                if step_data:
                    time_series_data[step_idx] = {
                        'time': step_data[0]['time'],
                        'avg_detections': np.mean([s.get('detections', 0) for s in step_data]),
                        'avg_tracks': np.mean([s.get('tracks', 0) for s in step_data]),
                        'avg_confirmed_tracks': np.mean([s.get('confirmed_tracks', 0) for s in step_data]),
                        'avg_scheduling_efficiency': np.mean([s.get('scheduling_efficiency', 0) for s in step_data]),
                        'std_detections': np.std([s.get('detections', 0) for s in step_data]),
                        'std_tracks': np.std([s.get('tracks', 0) for s in step_data])
                    }

            return {
                'summary': {
                    'total_runs': total_runs,
                    'avg_final_tracks': avg_final_tracks,
                    'avg_total_detections': avg_total_detections,
                },
                'time_series': time_series_data
            }

        except Exception as e:
            logger.error(f"Error aggregating results: {str(e)}")
            return {
                'summary': {
                    'total_runs': 0,
                    'avg_final_tracks': 0,
                    'avg_total_detections': 0,
                    'error': str(e)
                },
                'time_series': {}
            }

    # 在 simulation_service.py 中添加这些方法，如果尚未存在

    def get_active_simulation_count(self):
        """返回活跃仿真数量"""
        try:
            if hasattr(self, '_active_simulations') and self._active_simulations:
                return len(self._active_simulations)

            # 检查是否有当前运行的仿真
            if hasattr(self, 'simulation_params') and self.simulation_params is not None:
                return 1

            return 0
        except Exception as e:
            logger.error(f"Error getting active simulation count: {str(e)}")
            return 0

    def get_legacy_status(self):
        """兼容旧版状态接口"""
        try:
            is_running = (hasattr(self, 'simulation_params') and
                          self.simulation_params is not None)

            status_data = {
                'status': 'running' if is_running else 'idle',
                'current_time': getattr(self, 'current_time', 0.0),
                'simulations': [],
                'active_count': self.get_active_simulation_count(),
                'timestamp': time.time()
            }

            # 如果有活跃仿真，添加基本信息
            if is_running:
                status_data['simulations'].append({
                    'id': 'default',
                    'status': 'running',
                    'current_time': self.current_time,
                    'targets': len(getattr(self, 'targets', [])),
                    'formations': len(getattr(self, 'formations', []))
                })

            return status_data
        except Exception as e:
            logger.error(f"Error getting legacy status: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'simulations': [],
                'active_count': 0,
                'timestamp': time.time()
            }

    def get_legacy_status(self):
        """兼容旧版状态接口"""
        try:
            is_running = hasattr(self, 'simulation_params') and self.simulation_params is not None

            status_data = {
                'status': 'running' if is_running else 'idle',
                'current_time': getattr(self, 'current_time', 0.0),
                'simulations': [],
                'active_count': self.get_active_simulation_count()
            }

            # 如果有活跃仿真，添加基本信息
            if is_running:
                status_data['simulations'].append({
                    'id': 'default',
                    'status': 'running',
                    'current_time': self.current_time,
                    'targets': len(getattr(self, 'targets', [])),
                    'formations': len(getattr(self, 'formations', []))
                })

            return status_data
        except Exception as e:
            logger.error(f"Error getting legacy status: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'simulations': [],
                'active_count': 0
            }

    def reset_all_simulations(self):
        """重置所有仿真"""
        try:
            # 重置仿真状态
            self.reset_simulation()

            # 清空活跃仿真跟踪
            if hasattr(self, '_active_simulations'):
                self._active_simulations.clear()

            logger.info("All simulations have been reset successfully")
            return {
                'status': 'success',
                'message': 'All simulations have been reset',
                'active_simulations': 0
            }
        except Exception as e:
            logger.error(f"Error resetting all simulations: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to reset simulations: {str(e)}'
            }

    def start_async_simulation(self, simulation_id: str):
        """启动异步仿真任务"""
        try:
            # 初始化活跃仿真跟踪
            if not hasattr(self, '_active_simulations'):
                self._active_simulations = {}

            # 记录仿真开始
            self._active_simulations[simulation_id] = {
                'status': 'running',
                'start_time': self.current_time,
                'simulation_params': self.simulation_params
            }

            return {
                'simulation_id': simulation_id,
                'status': 'started',
                'message': 'Async simulation task initiated'
            }
        except Exception as e:
            logger.error(f"Error starting async simulation {simulation_id}: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to start async simulation: {str(e)}'
            }

    def pause_simulation(self, simulation_id: str):
        """暂停仿真"""
        try:
            if hasattr(self, '_active_simulations') and simulation_id in self._active_simulations:
                self._active_simulations[simulation_id]['status'] = 'paused'
                return {
                    'simulation_id': simulation_id,
                    'status': 'paused',
                    'message': 'Simulation paused successfully'
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Simulation {simulation_id} not found or not active'
                }
        except Exception as e:
            logger.error(f"Error pausing simulation {simulation_id}: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to pause simulation: {str(e)}'
            }

    def resume_simulation(self, simulation_id: str):
        """恢复仿真"""
        try:
            if hasattr(self, '_active_simulations') and simulation_id in self._active_simulations:
                self._active_simulations[simulation_id]['status'] = 'running'
                return {
                    'simulation_id': simulation_id,
                    'status': 'running',
                    'message': 'Simulation resumed successfully'
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Simulation {simulation_id} not found or not active'
                }
        except Exception as e:
            logger.error(f"Error resuming simulation {simulation_id}: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to resume simulation: {str(e)}'
            }

    def get_simulation_data(self, simulation_id: str, start_time: float = 0.0,
                            end_time: float = None, data_types: List[str] = None):
        """获取仿真数据"""
        try:
            # 检查仿真是否存在
            if hasattr(self, '_active_simulations') and simulation_id not in self._active_simulations:
                raise ValueError(f"Simulation {simulation_id} not found")

            # 获取仿真数据
            if not hasattr(self, 'simulation_data') or not self.simulation_data:
                return {
                    'simulation_id': simulation_id,
                    'data': [],
                    'message': 'No data available'
                }

            filtered_data = []
            for data_point in self.simulation_data:
                data_time = data_point.get('time', 0.0)

                # 时间过滤
                if data_time < start_time:
                    continue
                if end_time is not None and data_time > end_time:
                    continue

                # 数据类型过滤
                if data_types:
                    filtered_point = {'time': data_time}
                    for data_type in data_types:
                        if data_type in data_point:
                            filtered_point[data_type] = data_point[data_type]
                    filtered_data.append(filtered_point)
                else:
                    filtered_data.append(data_point)

            return {
                'simulation_id': simulation_id,
                'data': filtered_data,
                'start_time': start_time,
                'end_time': end_time,
                'data_types': data_types
            }

        except Exception as e:
            logger.error(f"Error getting simulation data for {simulation_id}: {str(e)}")
            return {
                'status': 'error',
                'message': f'Failed to get simulation data: {str(e)}'
            }

    def stop_simulation(self, simulation_id: str = None) -> Dict[str, Any]:
        """停止仿真"""
        try:
            if simulation_id:
                # 停止特定仿真
                if hasattr(self, '_active_simulations') and simulation_id in self._active_simulations:
                    del self._active_simulations[simulation_id]
                    return {
                        "status": "success",
                        "message": f"Simulation {simulation_id} stopped successfully"
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Simulation {simulation_id} not found"
                    }
            else:
                # 停止所有仿真
                self.reset_simulation()
                if hasattr(self, '_active_simulations'):
                    self._active_simulations.clear()
                return {"status": "success", "message": "All simulations stopped successfully"}

        except Exception as e:
            logger.error(f"Error stopping simulation: {str(e)}")
            return {"status": "error", "message": f"Failed to stop simulation: {str(e)}"}

    def get_simulation_status(self, simulation_id: str = None) -> Dict[str, Any]:
        """获取仿真状态"""
        try:
            if simulation_id:
                # 获取特定仿真状态
                if hasattr(self, '_active_simulations') and simulation_id in self._active_simulations:
                    sim_info = self._active_simulations[simulation_id]
                    return {
                        'simulation_id': simulation_id,
                        'status': sim_info.get('status', 'unknown'),
                        'current_time': self.current_time,
                        'start_time': sim_info.get('start_time', 0.0),
                        'is_running': sim_info.get('status') == 'running',
                        'num_targets': len(getattr(self, 'targets', [])),
                        'num_formations': len(getattr(self, 'formations', []))
                    }
                else:
                    raise ValueError(f"Simulation {simulation_id} not found")
            else:
                # 获取通用状态
                return {
                    'current_time': self.current_time,
                    'is_running': hasattr(self, 'simulation_params') and self.simulation_params is not None,
                    'num_targets': len(getattr(self, 'targets', [])),
                    'num_formations': len(getattr(self, 'formations', [])),
                    'active_simulations': self.get_active_simulation_count()
                }
        except Exception as e:
            logger.error(f"Error getting simulation status: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'simulation_id': simulation_id
            }

    def get_simulation_results(self, simulation_id: str = None) -> Dict[str, Any]:
        """获取仿真结果"""
        try:
            if not hasattr(self, 'simulation_data') or not self.simulation_data:
                return {"status": "error", "message": "No simulation results available"}

            return {
                "status": "success",
                "results": self.simulation_data
            }
        except Exception as e:
            logger.error(f"Error getting simulation results: {str(e)}")
            return {"status": "error", "message": f"Failed to get results: {str(e)}"}

    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """验证仿真参数"""
        try:
            errors = []

            # 验证雷达参数
            if 'radar' not in parameters:
                errors.append("Missing radar configuration")
            else:
                radar = parameters['radar']
                if 'radar_area' not in radar or radar['radar_area'] <= 0:
                    errors.append("Invalid radar_area")
                if 'tr_components' not in radar or radar['tr_components'] <= 0:
                    errors.append("Invalid tr_components")
                if 'radar_power' not in radar or radar['radar_power'] <= 0:
                    errors.append("Invalid radar_power")

            # 验证环境参数
            if 'environment' not in parameters:
                errors.append("Missing environment configuration")
            else:
                env = parameters['environment']
                if 'weather_type' not in env:
                    errors.append("Missing weather_type")
                valid_weather_types = ['clear', 'rain', 'snow', 'fog', 'storm']
                if env.get('weather_type') not in valid_weather_types:
                    errors.append(f"Invalid weather_type. Must be one of: {valid_weather_types}")

            # 验证目标参数
            if 'targets' not in parameters:
                errors.append("Missing targets configuration")
            else:
                targets = parameters['targets']
                if 'num_targets' not in targets or targets['num_targets'] <= 0:
                    errors.append("Invalid num_targets")

                # 验证具体目标参数
                if 'specific_targets' in targets:
                    for i, target in enumerate(targets['specific_targets']):
                        if 'position' not in target or len(target['position']) != 3:
                            errors.append(f"Invalid position for target {i}")
                        if 'velocity' not in target or len(target['velocity']) != 3:
                            errors.append(f"Invalid velocity for target {i}")
                        if 'rcs' not in target:
                            errors.append(f"Missing rcs for target {i}")

            # 验证仿真时间参数
            if parameters.get('simulation_time', 0) <= 0:
                errors.append("Invalid simulation_time")
            if parameters.get('time_step', 0) <= 0:
                errors.append("Invalid time_step")
            if parameters.get('monte_carlo_runs', 0) <= 0:
                errors.append("Invalid monte_carlo_runs")

            if errors:
                return {"status": "error", "errors": errors}
            else:
                return {"status": "success", "message": "Parameters are valid"}

        except Exception as e:
            logger.error(f"Error validating parameters: {str(e)}")
            return {"status": "error", "message": f"Parameter validation failed: {str(e)}"}

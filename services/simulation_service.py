import numpy as np
import logging
from typing import List, Dict, Any
from models.radar_system import RadarSystem, SimulationParameters
from models.target import Target, Formation
from models.environment import Environment, WeatherCondition
from models.tracking import Detection
from core.signal_processor import SignalProcessor
from core.data_processor import DataProcessor
from core.resource_scheduler import ResourceScheduler, RadarTask, TaskType
from core.rcs_processor import RCSProcessor

logger = logging.getLogger(__name__)


class SimulationService:
    def __init__(self):
        self.reset_simulation()

    def reset_simulation(self):
        self.current_time = 0.0
        self.simulation_data = []
        self.targets = []
        self.formations = []
        self.simulation_params = None
        self.signal_processor = None
        self.data_processor = None
        self.resource_scheduler = None
        self.rcs_processor = None

    def initialize_simulation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 验证必需的参数
            if 'radar' not in parameters:
                return {"status": "error", "message": "Missing radar configuration"}
            if 'environment' not in parameters:
                return {"status": "error", "message": "Missing environment configuration"}
            if 'targets' not in parameters:
                return {"status": "error", "message": "Missing targets configuration"}

            radar_system = self._create_radar_system(parameters['radar'])
            environment = self._create_environment(parameters['environment'])
            targets = self._create_targets(parameters['targets'])

            self.simulation_params = SimulationParameters(
                radar_system=radar_system,
                environment=environment,
                targets=targets,
                simulation_time=parameters.get('simulation_time', 10.0),
                time_step=parameters.get('time_step', 0.06),
                monte_carlo_runs=parameters.get('monte_carlo_runs', 1)
            )

            # 初始化处理器
            self.signal_processor = SignalProcessor(radar_system, environment)
            self.data_processor = DataProcessor(radar_system, environment)
            self.resource_scheduler = ResourceScheduler()
            self.rcs_processor = RCSProcessor(radar_system, environment)

            return {"status": "success", "message": "Simulation initialized successfully"}

        except Exception as e:
            logger.error(f"Simulation initialization error: {str(e)}")
            return {"status": "error", "message": f"Failed to initialize simulation: {str(e)}"}

    def run_simulation(self) -> Dict[str, Any]:
        try:
            # 检查是否已初始化
            if not hasattr(self, 'simulation_params') or self.simulation_params is None:
                return {"status": "error", "message": "Simulation not initialized. Call initialize_simulation first."}

            simulation_results = []

            for run in range(self.simulation_params.monte_carlo_runs):
                try:
                    run_results = self._execute_single_run(run)
                    simulation_results.append(run_results)
                except Exception as e:
                    logger.error(f"Error in simulation run {run}: {str(e)}")
                    # 继续其他运行，但记录错误
                    simulation_results.append({
                        'run_id': run,
                        'error': str(e),
                        'steps': [],
                        'final_tracks': 0,
                        'total_detections': 0
                    })

            # 如果所有运行都失败了
            if not any('steps' in result and result['steps'] for result in simulation_results):
                return {"status": "error", "message": "All simulation runs failed"}

            aggregated_results = self._aggregate_results(simulation_results)

            return {
                "status": "success",
                "results": aggregated_results,
                "individual_runs": simulation_results
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

            # 清空处理器状态
            if hasattr(self.data_processor, 'tracks'):
                self.data_processor.tracks = []
            if hasattr(self.data_processor, 'track_id_counter'):
                self.data_processor.track_id_counter = 1

            time_steps = int(self.simulation_params.simulation_time / self.simulation_params.time_step)
            step_results = []

            for step in range(time_steps):
                self.current_time = step * self.simulation_params.time_step

                # 更新目标位置
                self._update_target_positions(targets, self.simulation_params.time_step)

                # 生成雷达回波
                raw_signal = self._generate_radar_returns(targets)

                # 信号处理
                try:
                    detections = self.signal_processor.process_radar_signal(raw_signal, self.current_time)
                except Exception as e:
                    logger.warning(f"Signal processing error at step {step}: {str(e)}")
                    detections = []

                # 数据处理
                try:
                    tracks = self.data_processor.process_tracking_data(detections)
                except Exception as e:
                    logger.warning(f"Data processing error at step {step}: {str(e)}")
                    tracks = []

                # 资源调度
                try:
                    tasks = self._generate_radar_tasks(tracks)
                    schedule = self.resource_scheduler.schedule_resources(tasks)
                except Exception as e:
                    logger.warning(f"Resource scheduling error at step {step}: {str(e)}")
                    schedule = type('obj', (object,), {'efficiency': 0.0})()

                # RCS处理
                rcs_results = {}
                try:
                    for track in tracks:
                        if len(track.detections_history) > 0:
                            rcs_results[track.track_id] = self.rcs_processor.process_rcs(track)
                except Exception as e:
                    logger.warning(f"RCS processing error at step {step}: {str(e)}")

                # 记录步骤结果
                step_result = {
                    'time': self.current_time,
                    'targets': len(targets),
                    'detections': len(detections),
                    'tracks': len(tracks),
                    'true_targets': len([t for t in targets if self._target_in_coverage(t)]),
                    'confirmed_tracks': len([t for t in tracks if t.confirmed]),
                    'scheduling_efficiency': getattr(schedule, 'efficiency', 0.0),
                    'rcs_measurements': len(rcs_results)
                }
                step_results.append(step_result)

            return {
                'run_id': run_id,
                'steps': step_results,
                'final_tracks': len(self.data_processor.tracks) if self.data_processor and hasattr(self.data_processor,
                                                                                                   'tracks') else 0,
                'total_detections': sum(s['detections'] for s in step_results)
            }

        except Exception as e:
            logger.error(f"Execution error in run {run_id}: {str(e)}")
            raise

    def _copy_target(self, target: Target) -> Target:
        """创建目标的深拷贝"""
        new_target = Target(
            target_id=target.target_id,
            position=target.position.copy() if hasattr(target.position, 'copy') else list(target.position),
            velocity=target.velocity.copy() if hasattr(target.velocity, 'copy') else list(target.velocity),
            rcs=target.rcs,
            altitude=getattr(target, 'altitude', target.position[2]),
            aspect_angle=getattr(target, 'aspect_angle', 0.0)
        )

        # 复制其他属性
        if hasattr(target, 'is_formation'):
            new_target.is_formation = target.is_formation
        if hasattr(target, 'formation_id'):
            new_target.formation_id = target.formation_id

        return new_target

    def _create_radar_system(self, radar_params: Dict[str, Any]) -> RadarSystem:
        return RadarSystem(
            radar_area=radar_params['radar_area'],
            tr_components=radar_params['tr_components'],
            radar_power=radar_params['radar_power'],
            frequency=radar_params.get('frequency', 10e9),
            antenna_elements=radar_params.get('antenna_elements', 64),
            beam_width=radar_params.get('beam_width', 2.0),
            scan_rate=radar_params.get('scan_rate', 6.0)
        )

    def _create_environment(self, env_params: Dict[str, Any]) -> Environment:
        weather = WeatherCondition(
            weather_type=env_params['weather_type'],
            precipitation_rate=env_params.get('precipitation_rate', 0.0),
            visibility=env_params.get('visibility', 10000.0),
            wind_speed=env_params.get('wind_speed', 0.0),
            wind_direction=env_params.get('wind_direction', 0.0),
            temperature=env_params.get('temperature', 15.0),
            humidity=env_params.get('humidity', 50.0)
        )

        return Environment(
            weather=weather,
            clutter_density=env_params.get('clutter_density', 0.3),
            interference_level=env_params.get('interference_level', 0.1),
            multipath_factor=env_params.get('multipath_factor', 0.2),
            electronic_warfare=env_params.get('electronic_warfare', False),
            terrain_type=env_params.get('terrain_type', 'flat')
        )

    def _create_targets(self, target_params: Dict[str, Any]) -> List[Target]:
        targets = []
        target_id = 1

        num_targets = target_params['num_targets']

        for i in range(num_targets):
            if i < len(target_params.get('specific_targets', [])):
                spec_target = target_params['specific_targets'][i]
                target = Target(
                    target_id=target_id,
                    position=spec_target['position'],
                    velocity=spec_target['velocity'],
                    rcs=spec_target['rcs'],
                    altitude=spec_target['altitude'],
                    aspect_angle=spec_target['aspect_angle'],
                    is_formation=spec_target.get('is_formation', False),
                    formation_id=spec_target.get('formation_id')
                )
            else:
                target = self._generate_random_target(target_id, target_params)

            targets.append(target)
            target_id += 1

        if target_params.get('formations'):
            self._create_formations(targets, target_params['formations'])

        return targets

    def _generate_random_target(self, target_id: int, params: Dict[str, Any]) -> Target:
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

    def _create_formations(self, targets: List[Target], formation_params: List[Dict]):
        for formation_param in formation_params:
            leader_id = formation_param['leader_id']
            member_ids = formation_param['member_ids']

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

    def _update_target_positions(self, targets: List[Target], dt: float):
        """更新目标位置"""
        try:
            # 先更新编队
            for formation in self.formations:
                if hasattr(formation, 'update_formation_positions'):
                    formation.update_formation_positions(dt)

            # 更新非编队目标
            for target in targets:
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
            logger.warning(f"Error updating target positions: {str(e)}")

    def _generate_radar_returns(self, targets: List[Target]) -> np.ndarray:
        """生成雷达回波信号"""
        signal_length = 2048
        signal = np.zeros(signal_length, dtype=complex)

        for target in targets:
            if self._target_in_coverage(target):
                range_bin = int(target.range * 2048 / 150000)
                if 0 <= range_bin < signal_length:
                    snr = self._calculate_target_snr(target)
                    signal_amplitude = np.sqrt(10 ** (snr / 10))
                    phase = np.random.uniform(0, 2 * np.pi)
                    signal[range_bin] += signal_amplitude * np.exp(1j * phase)

        # 添加噪声和杂波
        noise = np.random.normal(0, 1, signal_length) + 1j * np.random.normal(0, 1, signal_length)
        clutter = self._generate_clutter(signal_length)

        return signal + noise + clutter

    def _target_in_coverage(self, target: Target) -> bool:
        max_range = 150000
        max_elevation = np.pi / 3

        return (target.range < max_range and
                abs(target.elevation) < max_elevation and
                abs(target.azimuth) < np.pi)

    def _calculate_target_snr(self, target: Target) -> float:
        radar_power = self.simulation_params.radar_system.radar_power
        gain = self.simulation_params.radar_system.gain
        wavelength = self.simulation_params.radar_system.wavelength
        rcs_linear = 10 ** (target.rcs / 10)

        range_factor = target.range ** 4

        snr_linear = (radar_power * gain ** 2 * wavelength ** 2 * rcs_linear) / \
                     ((4 * np.pi) ** 3 * range_factor)

        snr_db = 10 * np.log10(max(snr_linear, 1e-12))

        # 天气损失
        weather_loss = 0
        if hasattr(self.simulation_params.environment.weather, 'atmospheric_loss'):
            weather_loss = self.simulation_params.environment.weather.atmospheric_loss(
                self.simulation_params.radar_system.frequency, target.range / 1000
            )

        return snr_db - weather_loss

    def _generate_clutter(self, length: int) -> np.ndarray:
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

    def _generate_radar_tasks(self, tracks: List) -> List[RadarTask]:
        tasks = []
        task_id = 1

        for track in tracks:
            if not track.confirmed:
                task = RadarTask(
                    task_id=task_id,
                    task_type=TaskType.TARGET_CONFIRMATION,
                    duration=5.0,
                    release_time=self.current_time,
                    due_time=self.current_time + 20.0,
                    priority=1.0,
                    target_id=track.track_id,
                    hard_constraint=True
                )
                tasks.append(task)
                task_id += 1

            if track.high_mobility:
                task = RadarTask(
                    task_id=task_id,
                    task_type=TaskType.HIGH_PRIORITY_TRACKING,
                    duration=3.0,
                    release_time=self.current_time,
                    due_time=self.current_time + 10.0,
                    priority=2.0,
                    target_id=track.track_id,
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
                    target_id=track.track_id
                )
                tasks.append(task)
                task_id += 1

        return tasks

    def _aggregate_results(self, simulation_results: List[Dict]) -> Dict[str, Any]:
        if not simulation_results:
            return {'summary': {}, 'time_series': {}}

        # 过滤出成功的运行
        valid_results = [r for r in simulation_results if 'steps' in r and r['steps']]

        if not valid_results:
            return {'summary': {}, 'time_series': {}}

        total_runs = len(valid_results)

        avg_final_tracks = np.mean([r.get('final_tracks', 0) for r in valid_results])
        avg_total_detections = np.mean([r.get('total_detections', 0) for r in valid_results])

        time_series_data = {}
        max_steps = max(len(r['steps']) for r in valid_results)

        for step_idx in range(max_steps):
            step_data = []
            for run_result in valid_results:
                if step_idx < len(run_result['steps']):
                    step_data.append(run_result['steps'][step_idx])

            if step_data:
                time_series_data[step_idx] = {
                    'time': step_data[0]['time'],
                    'avg_detections': np.mean([s['detections'] for s in step_data]),
                    'avg_tracks': np.mean([s['tracks'] for s in step_data]),
                    'avg_confirmed_tracks': np.mean([s['confirmed_tracks'] for s in step_data]),
                    'avg_scheduling_efficiency': np.mean([s['scheduling_efficiency'] for s in step_data]),
                    'std_detections': np.std([s['detections'] for s in step_data]),
                    'std_tracks': np.std([s['tracks'] for s in step_data])
                }

        return {
            'summary': {
                'total_runs': total_runs,
                'avg_final_tracks': avg_final_tracks,
                'avg_total_detections': avg_total_detections,
            },
            'time_series': time_series_data
        }

    def get_simulation_status(self) -> Dict[str, Any]:
        return {
            'current_time': self.current_time,
            'is_running': hasattr(self, 'simulation_params') and self.simulation_params is not None,
            'num_targets': len(getattr(self, 'targets', [])),
            'num_formations': len(getattr(self, 'formations', []))
        }

# 配置管理模块

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field


@dataclass
class RadarConfig:
    """雷达系统配置"""
    center_frequency: float = 10e9
    bandwidth: float = 10e6
    sample_rate: float = 20e6
    wavelength: float = 0.03
    max_power: float = 100000.0
    noise_figure: float = 3.0
    system_losses: float = 6.0
    antenna_gain: float = 40.0
    array_type: str = 'planar'
    num_elements_x: int = 32
    num_elements_y: int = 32
    element_spacing_x: float = 0.015
    element_spacing_y: float = 0.015
    waveform_type: str = 'LFM'
    pulse_width: float = 10e-6
    prf: float = 2000.0
    fft_size: int = 1024
    cfar_type: str = 'CA'
    cfar_train_cells: int = 20
    cfar_guard_cells: int = 2
    false_alarm_rate: float = 1e-6
    init_method: str = 'M/N'
    m_value: int = 3
    n_value: int = 3
    association_method: str = 'JPDA'
    filter_model: str = 'IMM'
    max_tracks: int = 100


@dataclass
class NetworkConfig:
    """网络配置"""
    host: str = '0.0.0.0'
    http_port: int = 8000
    websocket_port: int = 8001
    max_connections: int = 10
    heartbeat_interval: float = 1.0


@dataclass
class PerformanceConfig:
    """性能配置"""
    enable_numba_jit: bool = True
    enable_multiprocessing: bool = True
    num_workers: int = 4
    buffer_size_plot: int = 1000
    buffer_size_track: int = 500
    buffer_size_beam: int = 500


@dataclass
class SystemConfig:
    """系统总配置"""
    version: str = '2.0'
    simulation_time_step: float = 0.001
    max_simulation_speed: float = 10.0
    enable_logging: bool = True
    log_level: str = 'INFO'
    log_path: str = './logs'
    radar: RadarConfig = field(default_factory=RadarConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = './configs'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config: Optional[SystemConfig] = None
        self._scenario_config: Optional[Dict] = None
    
    def load_radar_config(self, filename: str = 'radar_config.toml') -> SystemConfig:
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            self._config = SystemConfig()
            return self._config
        
        import toml
        with open(config_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)
        
        self._config = self._parse_config(data)
        return self._config
    
    def _parse_config(self, data: Dict) -> SystemConfig:
        def get_section(section: str, default: Dict = None) -> Dict:
            return data.get(section, default or {})
        
        system = get_section('system', {})
        config = SystemConfig(
            version=system.get('version', '2.0'),
            simulation_time_step=system.get('simulation_time_step', 0.001),
            max_simulation_speed=system.get('max_simulation_speed', 10.0),
            enable_logging=system.get('enable_logging', True),
            log_level=system.get('log_level', 'INFO'),
            log_path=system.get('log_path', './logs'),
        )
        
        radar = get_section('radar', {})
        config.radar = RadarConfig(
            center_frequency=radar.get('center_frequency', 10e9),
            bandwidth=radar.get('bandwidth', 10e6),
            sample_rate=radar.get('sample_rate', 20e6),
            wavelength=radar.get('wavelength', 0.03),
            max_power=radar.get('max_power', 100000.0),
            noise_figure=radar.get('noise_figure', 3.0),
            system_losses=radar.get('system_losses', 6.0),
            antenna_gain=radar.get('antenna_gain', 40.0),
        )
        
        antenna = get_section('antenna', {})
        config.radar.array_type = antenna.get('array_type', 'planar')
        config.radar.num_elements_x = antenna.get('num_elements_x', 32)
        config.radar.num_elements_y = antenna.get('num_elements_y', 32)
        config.radar.element_spacing_x = antenna.get('element_spacing_x', 0.015)
        config.radar.element_spacing_y = antenna.get('element_spacing_y', 0.015)
        
        waveform = get_section('waveform', {})
        config.radar.waveform_type = waveform.get('type', 'LFM')
        config.radar.pulse_width = waveform.get('pulse_width', 10e-6)
        config.radar.prf = waveform.get('prf', 2000.0)
        
        network = get_section('network', {})
        config.network = NetworkConfig(
            host=network.get('host', '0.0.0.0'),
            http_port=network.get('http_port', 8000),
            websocket_port=network.get('websocket_port', 8001),
            max_connections=network.get('max_connections', 10),
            heartbeat_interval=network.get('heartbeat_interval', 1.0),
        )
        
        perf = get_section('performance', {})
        config.performance = PerformanceConfig(
            enable_numba_jit=perf.get('enable_numba_jit', True),
            enable_multiprocessing=perf.get('enable_multiprocessing', True),
            num_workers=perf.get('num_workers', 4),
            buffer_size_plot=perf.get('buffer_size_plot', 1000),
            buffer_size_track=perf.get('buffer_size_track', 500),
            buffer_size_beam=perf.get('buffer_size_beam', 500),
        )
        
        return config
    
    def load_scenario_config(self, filename: str = 'scenario_config.json') -> Dict:
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            self._scenario_config = {}
            return self._scenario_config
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._scenario_config = json.load(f)
        
        return self._scenario_config
    
    def get_config(self) -> Optional[SystemConfig]:
        if self._config is None:
            self.load_radar_config()
        return self._config
    
    def get_scenario_config(self) -> Optional[Dict]:
        if self._scenario_config is None:
            self.load_scenario_config()
        return self._scenario_config


_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_dir: str = './configs') -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


__all__ = [
    'RadarConfig', 'NetworkConfig', 'PerformanceConfig', 'SystemConfig',
    'ConfigManager', 'get_config_manager',
]

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from models.environment import Environment
from models.target import Target


@dataclass
class RadarSystem:
    radar_area: float
    tr_components: int
    radar_power: float
    frequency: float = 10e9
    antenna_elements: int = 64
    beam_width: float = 2.0
    scan_rate: float = 6.0

    def __post_init__(self):
        self.wavelength = 3e8 / self.frequency
        self.element_spacing = self.wavelength / 2
        self.gain = 10 * np.log10(self.tr_components * 0.7)


@dataclass
class SimulationParameters:
    radar_system: RadarSystem
    environment: Environment
    targets: List[Target]
    simulation_time: float = 10.0
    time_step: float = 0.06
    monte_carlo_runs: int = 1

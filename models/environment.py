import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class WeatherCondition:
    weather_type: str  # "clear", "rain", "snow", "fog"
    precipitation_rate: float = 0.0  # mm/h
    visibility: float = 10000.0  # meters
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # radians
    temperature: float = 15.0  # celsius
    humidity: float = 50.0  # percentage

    def atmospheric_loss(self, frequency: float, range_km: float) -> float:
        loss_map = {
            "clear": 0.1,
            "light_rain": 0.3,
            "heavy_rain": 1.2,
            "snow": 0.8,
            "fog": 0.5
        }

        if self.weather_type == "rain":
            if self.precipitation_rate < 2.5:
                weather_key = "light_rain"
            else:
                weather_key = "heavy_rain"
        else:
            weather_key = self.weather_type

        base_loss = loss_map.get(weather_key, 0.1)
        freq_factor = (frequency / 10e9) ** 0.6

        return base_loss * freq_factor * range_km


@dataclass
class Environment:
    weather: WeatherCondition
    clutter_density: float = 0.3
    interference_level: float = 0.1
    multipath_factor: float = 0.2
    electronic_warfare: bool = False
    terrain_type: str = "flat"

    def get_clutter_rcs(self, cell_area: float) -> float:
        clutter_map = {
            "sea": -30,
            "flat": -20,
            "hills": -15,
            "urban": -10,
            "forest": -25
        }
        base_clutter = clutter_map.get(self.terrain_type, -20)
        return base_clutter + 10 * np.log10(cell_area * self.clutter_density)

    def get_noise_figure(self) -> float:
        base_nf = 3.0
        if self.electronic_warfare:
            base_nf += 5.0
        if self.interference_level > 0.5:
            base_nf += 2.0
        return base_nf

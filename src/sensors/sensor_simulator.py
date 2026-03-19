"""
Sensor simulation for AquaTrace AI.

Two sensors:
  PreCondensationSensor  – measures temperature (°C) and relative humidity (%)
                           in the hot vapor zone BEFORE condensation.
  PostCondensationSensor – measures actual water collected (litres) and
                           flow rate (L/min) AFTER condensation.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class SensorReading:
    timestamp: datetime
    sensor_id: str
    temperature: float = 0.0      # °C
    humidity: float = 0.0         # %
    water_collected: float = 0.0  # litres
    flow_rate: float = 0.0        # L/min


class PreCondensationSensor:
    """Simulates the pre-condensation sensor (hot vapor zone)."""

    def __init__(self, sensor_id: str = "PRE-01", rng: np.random.Generator | None = None):
        self.sensor_id = sensor_id
        self._rng = rng or np.random.default_rng()

    def read(self) -> SensorReading:
        temp = float(self._rng.normal(loc=45.0, scale=5.0))     # 35-55 °C typical
        humidity = float(self._rng.uniform(60.0, 95.0))          # high-humidity vapor zone
        return SensorReading(
            timestamp=datetime.now(timezone.utc),
            sensor_id=self.sensor_id,
            temperature=round(temp, 2),
            humidity=round(humidity, 2),
        )


class PostCondensationSensor:
    """Simulates the post-condensation sensor (collection zone)."""

    def __init__(self, sensor_id: str = "POST-01", rng: np.random.Generator | None = None):
        self.sensor_id = sensor_id
        self._rng = rng or np.random.default_rng()

    def read(self, pre_reading: SensorReading) -> SensorReading:
        """
        Actual water collected depends on pre-condensation conditions.
        Higher humidity and lower temperature → more condensation.
        """
        efficiency = (pre_reading.humidity / 100.0) * (1.0 - (pre_reading.temperature - 20.0) / 80.0)
        efficiency = float(np.clip(efficiency, 0.05, 1.0))
        base_volume = 10.0  # litres per cycle
        water = base_volume * efficiency + float(self._rng.normal(0, 0.3))
        water = max(0.0, round(water, 3))
        flow = water / 5.0 + float(self._rng.normal(0, 0.05))
        flow = max(0.0, round(flow, 3))
        return SensorReading(
            timestamp=datetime.now(timezone.utc),
            sensor_id=self.sensor_id,
            temperature=round(float(self._rng.normal(20.0, 2.0)), 2),
            humidity=round(float(self._rng.uniform(40.0, 70.0)), 2),
            water_collected=water,
            flow_rate=flow,
        )

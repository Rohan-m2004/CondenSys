import numpy as np
import pytest
from src.sensors.sensor_simulator import (
    PreCondensationSensor,
    PostCondensationSensor,
    SensorReading,
)


def make_rng(seed=0):
    return np.random.default_rng(seed)


def test_pre_sensor_fields():
    sensor = PreCondensationSensor(rng=make_rng())
    reading = sensor.read()
    assert isinstance(reading, SensorReading)
    assert reading.sensor_id == "PRE-01"
    assert 0 < reading.temperature < 100
    assert 0 <= reading.humidity <= 100


def test_post_sensor_fields():
    pre_sensor = PreCondensationSensor(rng=make_rng())
    post_sensor = PostCondensationSensor(rng=make_rng())
    pre = pre_sensor.read()
    post = post_sensor.read(pre)
    assert isinstance(post, SensorReading)
    assert post.sensor_id == "POST-01"
    assert post.water_collected >= 0
    assert post.flow_rate >= 0


def test_post_sensor_higher_humidity_more_water():
    """Higher humidity pre-readings should produce more water on average."""
    low_reading = SensorReading(
        timestamp=__import__("datetime").datetime.utcnow(),
        sensor_id="PRE-01",
        temperature=45.0,
        humidity=60.0,
    )
    high_reading = SensorReading(
        timestamp=__import__("datetime").datetime.utcnow(),
        sensor_id="PRE-01",
        temperature=45.0,
        humidity=90.0,
    )

    n = 500
    low_rng = np.random.default_rng(0)
    high_rng = np.random.default_rng(0)
    low_waters = [PostCondensationSensor(rng=low_rng).read(low_reading).water_collected for _ in range(n)]
    high_waters = [PostCondensationSensor(rng=high_rng).read(high_reading).water_collected for _ in range(n)]
    assert np.mean(high_waters) > np.mean(low_waters)


def test_sensor_reproducibility():
    """Same seed → same readings."""
    s1 = PreCondensationSensor(rng=np.random.default_rng(99))
    s2 = PreCondensationSensor(rng=np.random.default_rng(99))
    r1 = s1.read()
    r2 = s2.read()
    assert r1.temperature == r2.temperature
    assert r1.humidity == r2.humidity

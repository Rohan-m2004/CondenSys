import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from src.data.data_logger import DataLogger


def make_logger(tmp_path):
    return DataLogger(log_path=tmp_path / "test_log.csv")


def test_logger_creates_file(tmp_path):
    logger = make_logger(tmp_path)
    assert logger.log_path.exists()


def test_log_and_load(tmp_path):
    logger = make_logger(tmp_path)
    logger.log(
        pre_temperature=42.0,
        pre_humidity=75.0,
        post_water_collected=5.2,
        post_flow_rate=1.04,
        predicted_water_recovery=5.0,
        efficiency_score=0.52,
        recommendation="System OK",
    )
    df = logger.load()
    assert len(df) == 1
    assert df.iloc[0]["pre_temperature"] == 42.0
    assert df.iloc[0]["recommendation"] == "System OK"


def test_multiple_rows(tmp_path):
    logger = make_logger(tmp_path)
    for i in range(5):
        logger.log(40.0 + i, 70.0, 4.0, 0.8, 4.1, 0.41, "OK")
    df = logger.load()
    assert len(df) == 5


def test_clear(tmp_path):
    logger = make_logger(tmp_path)
    logger.log(40.0, 70.0, 4.0, 0.8, 4.1, 0.41, "OK")
    logger.clear()
    df = logger.load()
    assert len(df) == 0

import numpy as np
import pytest
from src.models.dnn_model import WaterRecoveryDNN


def _make_dataset(n=200, seed=42):
    rng = np.random.default_rng(seed)
    temperature = rng.normal(45, 5, n)
    humidity = rng.uniform(60, 95, n)
    airflow = rng.uniform(1, 4, n)
    efficiency = (humidity / 100.0) * (1 - (temperature - 20) / 80.0)
    efficiency = np.clip(efficiency, 0.05, 1.0)
    y = 10.0 * efficiency + rng.normal(0, 0.3, n)
    y = np.maximum(y, 0)
    X = np.column_stack([temperature, humidity, airflow])
    return X, y


def test_train_returns_metrics():
    model = WaterRecoveryDNN()
    X, y = _make_dataset()
    metrics = model.train(X, y)
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["r2"] > 0.0


def test_predict_before_train_raises():
    model = WaterRecoveryDNN()
    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict(40.0, 75.0)


def test_predict_returns_expected_keys():
    model = WaterRecoveryDNN()
    X, y = _make_dataset()
    model.train(X, y)
    result = model.predict(40.0, 80.0, 2.5)
    assert "predicted_water_recovery" in result
    assert "efficiency_score" in result
    assert "recommendation" in result
    assert result["predicted_water_recovery"] >= 0


def test_predict_batch_shape():
    model = WaterRecoveryDNN()
    X, y = _make_dataset()
    model.train(X, y)
    X_test = X[:10]
    preds = model.predict_batch(X_test)
    assert preds.shape == (10,)


def test_efficiency_score_in_range():
    model = WaterRecoveryDNN()
    X, y = _make_dataset()
    model.train(X, y)
    for temp, hum in [(35, 85), (50, 90), (30, 70)]:
        res = model.predict(temp, hum)
        assert 0.0 <= res["efficiency_score"] <= 1.0

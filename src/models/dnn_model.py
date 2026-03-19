"""
WaterRecoveryDNN – a Deep Neural Network (MLP) for predicting
water recovery amount from sensor inputs.

Inputs : temperature, humidity, airflow (synthetic feature)
Outputs: predicted_water_recovery (litres), efficiency_score (0-1)
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


_DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "dnn_model.pkl"


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )),
    ])


class WaterRecoveryDNN:
    """Wraps sklearn MLPRegressor with domain-specific helpers."""

    def __init__(self):
        self._pipeline = _build_pipeline()
        self._trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train the model.

        Parameters
        ----------
        X : ndarray, shape (n, 3)
            Columns: [temperature, humidity, airflow]
        y : ndarray, shape (n,)
            Target: actual water collected (litres)

        Returns
        -------
        dict with training metadata
        """
        self._pipeline.fit(X, y)
        self._trained = True
        predictions = self._pipeline.predict(X)
        residuals = y - predictions
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"mse": mse, "rmse": rmse, "r2": round(r2, 4)}

    def predict(self, temperature: float, humidity: float, airflow: float = 2.0) -> dict:
        """
        Predict water recovery for a single observation.

        Returns
        -------
        dict: predicted_water_recovery, efficiency_score, recommendation
        """
        if not self._trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        X = np.array([[temperature, humidity, airflow]])
        water = float(self._pipeline.predict(X)[0])
        water = max(0.0, round(water, 3))
        max_possible = 10.0
        efficiency = round(min(water / max_possible, 1.0), 4)
        recommendation = self._recommendation(temperature, humidity, efficiency)
        return {
            "predicted_water_recovery": water,
            "efficiency_score": efficiency,
            "recommendation": recommendation,
        }

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model has not been trained yet.")
        return self._pipeline.predict(X)

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or _DEFAULT_MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)
        return path

    def load(self, path: Path | None = None) -> None:
        path = Path(path or _DEFAULT_MODEL_PATH)
        self._pipeline = joblib.load(path)
        self._trained = True

    @staticmethod
    def _recommendation(temperature: float, humidity: float, efficiency: float) -> str:
        if efficiency < 0.3:
            return "Low recovery expected. Consider pre-cooling the vapor stream."
        if temperature > 50:
            return "High temperature detected. Increase airflow to boost condensation."
        if humidity < 65:
            return "Low humidity in vapor zone. Check for vapor leaks."
        return "Conditions are optimal for water recovery."

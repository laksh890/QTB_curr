"""Integration: forecasting framework with synthetic market-like frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.forecasting import ForecastingPipeline, ForecastingSettings, get_registry
from iqrp.app.forecasting.models.mock import MockForecastModel
from iqrp.app.forecasting.visualization import plot_forecast


def _synthetic(n: int = 120, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # regime-switching mean
    regime = (t > n // 2).astype(np.int64)
    close = np.cumsum(rng.normal(0.01 * (1 - 2 * regime), 0.05, n)) + 100
    f0 = np.diff(close, prepend=close[0])
    f1 = close - np.convolve(close, np.ones(5) / 5, mode="same")
    target = np.roll(f0, -1)
    target[-1] = target[-2]
    return pl.DataFrame(
        {
            "open_time": t.tolist(),
            "close": close,
            "f0": f0,
            "f1": f1,
            "target": target,
            "regime": regime,
            "sector": ["eq" if i % 3 else "fx" for i in range(n)],
        }
    )


@pytest.mark.integration
def test_end_to_end_forecasting_pipeline(tmp_path: Path) -> None:
    assert "mock" in get_registry().list_names()
    frame = _synthetic()
    settings = ForecastingSettings.from_hydra(
        overrides=[
            "columns.feature_columns=[f0,f1]",
            "columns.target=target",
            "columns.regime_column=regime",
            "preprocessing.feature_selection=correlation",
            "preprocessing.max_features=2",
            "inference.default_horizon=6",
            "training.validation_fraction=0.25",
        ]
    )
    pipe = ForecastingPipeline(settings=settings, model_name="mock")
    result = pipe.run(frame)
    assert result.forecast.values.shape[0] == 6
    assert result.train is not None
    assert result.evaluation is not None
    assert result.evaluation.metrics["rmse"] >= 0
    # persistence via model
    model = pipe.model
    assert isinstance(model, MockForecastModel)
    path = tmp_path / "forecast_model.json"
    model.save(path)
    loaded = MockForecastModel.load(path)
    assert loaded.forecast(frame, horizon=3).values.shape[0] == 3
    plot_forecast(
        frame["target"].to_numpy()[-40:],
        pipe.predict(frame)[-40:],
        tmp_path / "fc.svg",
        settings=settings,
    )
    assert (tmp_path / "fc.svg").is_file()

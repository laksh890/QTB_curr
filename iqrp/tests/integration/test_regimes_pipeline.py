"""Integration tests for the Market Regime Detection Framework."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes import (
    RegimeDetector,
    RegimePredictor,
    RegimeSettings,
    get_registry,
)


def _ohlcv(n: int = 220, seed: int = 11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 3, 1, tzinfo=UTC)
    # Regime-like drift shifts
    rets = np.concatenate(
        [
            rng.normal(-0.002, 0.01, n // 3),
            rng.normal(0.0, 0.006, n // 3),
            rng.normal(0.002, 0.01, n - 2 * (n // 3)),
        ]
    )
    close = 100 * np.cumprod(1 + rets)
    rows = []
    for i in range(n):
        c = float(close[i])
        rows.append(
            {
                "open_time": start + timedelta(hours=i),
                "open": c * 0.999,
                "high": c * 1.01,
                "low": c * 0.99,
                "close": c,
                "volume": float(50 + i % 9),
            }
        )
    return pl.DataFrame(rows)


@pytest.mark.integration
def test_end_to_end_detect_store_forecast_evaluate(tmp_path: Path) -> None:
    frame = _ohlcv()
    settings = RegimeSettings.model_validate(
        {
            **RegimeSettings.default().model_dump(),
            "store_dir": str(tmp_path / "regimes"),
            "duckdb_path": str(tmp_path / "regimes" / "regimes.duckdb"),
            "output_dir": str(tmp_path / "reports"),
        }
    )
    detector = RegimeDetector(settings)
    assert get_registry().list_names()

    model = detector.fit(frame, model_name="mock_regime", n_states=3, window=12)
    predictor = RegimePredictor(settings)
    ids = predictor.predict(model, frame)
    proba = predictor.predict_proba(model, frame)
    tm = predictor.transition_matrix(model)
    fc = predictor.forecast(model, frame, steps=6)
    assert ids.shape[0] == frame.height
    assert proba.shape == (frame.height, 3)
    assert tm.shape == (3, 3)
    assert fc.steps == 6

    result = detector.detect(
        frame,
        model=model,
        persist=True,
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        write_charts=True,
        forecast_steps=6,
    )
    assert result.transitions is not None
    assert result.probabilities is not None
    assert result.persistence is not None
    assert (tmp_path / "reports" / "charts" / "regime_timeline.svg").exists()

    stored = detector.store.read_states(
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        model_name="mock_regime",
    )
    assert stored.height == frame.height

    report = detector.evaluate(model, frame, true_states=ids)
    assert report.log_likelihood <= 0.0 or report.log_likelihood != report.log_likelihood
    assert 0.0 <= report.state_stability <= 1.0

    artifact = detector.save(model, tmp_path / "model.json")
    reloaded = detector.load(artifact)
    assert np.array_equal(reloaded.predict(frame), ids)

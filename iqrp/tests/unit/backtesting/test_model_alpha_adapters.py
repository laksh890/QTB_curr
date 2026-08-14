"""Unit tests for model→alpha adapter layer (wiring only)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from iqrp.app.backtesting.alpha_research.adapters.forecast_adapter import (
    forecast_to_signal_series,
    map_values_to_signal,
    metadata_bundle,
)
from iqrp.app.backtesting.alpha_research.adapters.model_registry import (
    clear_adapters,
    get_adapter,
    register_default_adapters,
)
from iqrp.app.backtesting.alpha_research.adapters.pipeline import align_model_signal_mtf, run_adapter
from iqrp.app.backtesting.alpha_research.adapters.regime_adapter import regime_states_to_signal
from iqrp.app.backtesting.alpha_research.adapters.signal_registration import (
    attach_precomputed_signal,
    clear_model_signal_cache,
    register_model_adapter_signals,
)
from iqrp.app.backtesting.alpha_research.adapters.types import (
    OutputMappingKind,
    SignalMappingConfig,
)
from iqrp.app.backtesting.alpha_research.adapters.validation import (
    AdapterValidationError,
    assert_no_future_columns,
    train_val_oos_slices,
)
from iqrp.app.backtesting.alpha_research.signals import SignalRegistry, get_signal_registry
from iqrp.app.forecasting.base.forecast import Forecast


def _ohlcv(n: int = 240, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=n)
    close = 100 * np.cumprod(1 + rets)
    ts = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "instrument": "BTCUSDT",
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.integers(10, 100, size=n).astype(float),
        }
    )


@pytest.fixture(autouse=True)
def _adapters():
    clear_adapters()
    clear_model_signal_cache()
    register_default_adapters(overwrite=True)
    yield
    clear_adapters()
    clear_model_signal_cache()


def test_adapter_construction_and_serialization():
    a = get_adapter("garch_volatility_v1_1h")
    d = a.to_dict()
    assert d["adapter_id"] == "garch_volatility_v1_1h"
    assert d["causal"] is True
    json.dumps(d)


def test_return_threshold_long_short_flat():
    mapping = SignalMappingConfig(
        kind=OutputMappingKind.RETURN_THRESHOLD, long_threshold=0.01, short_threshold=-0.01
    )
    vals = np.array([0.02, -0.02, 0.0, np.nan])
    out = map_values_to_signal(vals, mapping)
    assert out[0] == 1.0 and out[1] == -1.0 and out[2] == 0.0
    assert np.isnan(out[3])


def test_forecast_object_to_signal():
    fc = Forecast.from_values([0.05], horizon=1, model_name="arima", model_version="1.0.0")
    idx = pd.RangeIndex(5)
    s = forecast_to_signal_series(
        fc, idx, SignalMappingConfig(kind=OutputMappingKind.RETURN_THRESHOLD, long_threshold=0.0)
    )
    assert s.iloc[-1] == 1.0
    assert s.iloc[:-1].isna().all()


def test_regime_label_map():
    mapping = SignalMappingConfig(
        kind=OutputMappingKind.REGIME_LABEL_MAP,
        regime_map={"0": 0.0, "1": 1.0, "2": -1.0},
    )
    out = regime_states_to_signal(np.array([0, 1, 2, 9]), mapping)
    assert list(out) == [0.0, 1.0, -1.0, 0.0]


def test_leakage_rejection_and_oos_slices():
    df = _ohlcv(50)
    df["future_close"] = df["close"].shift(-1)
    with pytest.raises(AdapterValidationError):
        assert_no_future_columns(df)
    sl = train_val_oos_slices(100, train_frac=0.5, validation_frac=0.25)
    assert sl["train"] == slice(0, 50)
    assert sl["oos"] == slice(75, 100)


def test_metadata_bundle_fields():
    m = metadata_bundle(
        source_model="garch",
        model_version="1.0.0",
        forecast_timestamp="t0",
        signal_timestamp="t1",
        source_timeframe="1h",
        execution_timeframe="5m",
        lookback=100,
        horizon=1,
        threshold_config={"kind": "return_threshold"},
        configuration_id="garch_volatility_v1_1h",
    )
    for k in (
        "source_model",
        "model_version",
        "forecast_timestamp",
        "signal_timestamp",
        "source_timeframe",
        "execution_timeframe",
        "lookback",
        "horizon",
        "threshold",
        "configuration_id",
    ):
        assert k in m


def test_mtf_causal_alignment():
    model = _ohlcv(50)
    exec_df = _ohlcv(200)
    exec_df["timestamp"] = pd.date_range("2021-01-01", periods=200, freq="15min", tz="UTC")
    sig = pd.Series(np.linspace(-1, 1, len(model)), index=model.index)
    aligned = align_model_signal_mtf(model, sig, exec_df)
    assert len(aligned) == len(exec_df)


def test_signal_registry_integration_with_precomputed():
    df = _ohlcv(120)
    # synthetic precomputed
    sig = pd.Series(np.sign(df["close"].pct_change().fillna(0)), index=df.index)
    reg = SignalRegistry()
    register_model_adapter_signals(reg, overwrite=True, adapter_ids=["arima_return_v1_1h"])
    framed = attach_precomputed_signal(df, "arima_return_v1_1h", sig)
    got, meta, _ = reg.generate(framed, "arima_return_v1_1h")
    assert len(got) == len(df)
    assert meta["family"] == "model_adapter"


def test_reference_registry_unchanged_by_default():
    ids = {s.signal_id for s in get_signal_registry().list()}
    assert "momentum_signal" in ids
    assert "garch_volatility_v1_1h" not in ids


@pytest.mark.parametrize(
    "adapter_id",
    [
        "garch_volatility_v1_1h",
        "arima_return_v1_1h",
        "xgb_return_v1_1h",
        "mock_regime_v1_1h",
    ],
)
def test_integration_adapters_smoke(adapter_id: str):
    df = _ohlcv(280)
    result = run_adapter(adapter_id, df, train_frac=0.5)
    assert result["status"] in {"PASS", "UNAVAILABLE"}
    if result["status"] == "PASS":
        assert result["signal"] is not None
        assert result["signal"].iloc[: result["slices"]["train"][1]].eq(0).all()
        # deterministic re-run
        result2 = run_adapter(adapter_id, df, train_frac=0.5)
        if result2["status"] == "PASS":
            assert np.allclose(
                result["signal"].to_numpy(),
                result2["signal"].to_numpy(),
                equal_nan=True,
            )


@pytest.mark.parametrize(
    "adapter_id",
    ["lstm_return_v1_1h", "transformer_return_v1_1h", "hmm_regime_v1_1h"],
)
def test_heavy_or_optional_adapters(adapter_id: str):
    df = _ohlcv(320)
    result = run_adapter(adapter_id, df, train_frac=0.5)
    assert result["status"] in {"PASS", "UNAVAILABLE"}
    # Must not crash; UNAVAILABLE is acceptable without implementing missing pieces

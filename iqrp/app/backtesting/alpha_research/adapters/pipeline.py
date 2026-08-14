"""OOS-safe model fit → forecast → signal builders (thin orchestration only).

Does not implement forecasting models. Wiring validation only — not profitability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from iqrp.app.backtesting.alpha_research.adapters.forecast_adapter import (
    map_values_to_signal,
    metadata_bundle,
)
from iqrp.app.backtesting.alpha_research.adapters.model_registry import get_adapter
from iqrp.app.backtesting.alpha_research.adapters.regime_adapter import regime_result_to_signal_series
from iqrp.app.backtesting.alpha_research.adapters.types import ModelAdapterSpec
from iqrp.app.backtesting.alpha_research.adapters.validation import (
    AdapterValidationError,
    assert_no_future_columns,
    assert_timestamps_monotonic,
    train_val_oos_slices,
)
from iqrp.app.backtesting.alpha_research.mtf import align_feature_to_execution

_FEATURE_COLS = ("f_ret1", "f_ret2", "f_ret3", "f_vol")


def _to_polars_ohlcv(frame: pd.DataFrame, *, with_next_return_target: bool = False) -> pl.DataFrame:
    df = frame.copy()
    if "timestamp" in df.columns and "open_time" not in df.columns:
        df = df.rename(columns={"timestamp": "open_time"})
    if "returns" not in df.columns and "close" in df.columns:
        df["returns"] = pd.Series(df["close"]).pct_change().fillna(0.0)
    if "target" not in df.columns:
        if with_next_return_target and "close" in df.columns:
            # Supervised label only — never used as a feature column.
            df["target"] = pd.Series(df["close"]).pct_change().shift(-1)
        elif "returns" in df.columns:
            df["target"] = df["returns"]
    if "close" in df.columns:
        r = pd.Series(df["close"]).pct_change()
        df["f_ret1"] = r.fillna(0.0)
        df["f_ret2"] = r.shift(1).fillna(0.0)
        df["f_ret3"] = r.shift(2).fillna(0.0)
        df["f_vol"] = r.rolling(10, min_periods=3).std().fillna(0.0)
    cols = [c for c in df.columns if c != "instrument"]
    return pl.from_pandas(df[cols])


def _oos_signal_from_scores(
    scores: np.ndarray,
    spec: ModelAdapterSpec,
    index: pd.Index,
    *,
    train_end: int,
) -> pd.Series:
    mapped = map_values_to_signal(scores, spec.signal_mapping)
    out = np.asarray(mapped, dtype=np.float64).copy()
    out[:train_end] = 0.0
    return pd.Series(out, index=index, dtype=np.float64)


def _meta(spec: ModelAdapterSpec, frame: pd.DataFrame, train_end: int) -> dict[str, Any]:
    return metadata_bundle(
        source_model=spec.model_id,
        model_version=spec.model_version,
        forecast_timestamp=frame["timestamp"].iloc[train_end - 1] if train_end else None,
        signal_timestamp=frame["timestamp"].iloc[-1],
        source_timeframe=spec.timeframe,
        execution_timeframe=spec.timeframe,
        lookback=train_end,
        horizon=spec.horizon,
        threshold_config=spec.signal_mapping.to_dict(),
        configuration_id=spec.adapter_id,
    )


def generate_garch_signal(
    frame: pd.DataFrame, spec: ModelAdapterSpec, *, train_frac: float = 0.5
) -> dict[str, Any]:
    from iqrp.app.forecasting.volatility import create_volatility_model

    assert_no_future_columns(frame)
    assert_timestamps_monotonic(frame["timestamp"])
    n = len(frame)
    slices = train_val_oos_slices(n, train_frac=train_frac, validation_frac=0.25)
    train_end = slices["train"].stop
    pl_full = _to_polars_ohlcv(frame)
    model = create_volatility_model(spec.model_id)
    model.fit(pl_full[:train_end], target_column="returns")
    sigma = np.asarray(model.predict(pl_full), dtype=np.float64).reshape(-1)
    signal = _oos_signal_from_scores(sigma, spec, frame.index, train_end=train_end)
    return {
        "status": "PASS",
        "signal": signal,
        "scores": sigma,
        "meta": _meta(spec, frame, train_end),
        "slices": {k: [sl.start, sl.stop] for k, sl in slices.items()},
        "fit_mode": "train_only_params_then_causal_filter",
        "model_exists": True,
        "forecast_generated": True,
        "signal_generated": True,
    }


def generate_arima_signal(
    frame: pd.DataFrame, spec: ModelAdapterSpec, *, train_frac: float = 0.5
) -> dict[str, Any]:
    from iqrp.app.forecasting.statistical import create_statistical_model

    assert_no_future_columns(frame)
    assert_timestamps_monotonic(frame["timestamp"])
    n = len(frame)
    slices = train_val_oos_slices(n, train_frac=train_frac, validation_frac=0.25)
    train_end = slices["train"].stop
    pl_full = _to_polars_ohlcv(frame)
    model = create_statistical_model(spec.model_id, p=1, d=0, q=1)
    model.fit(pl_full[:train_end], target_column="target")
    try:
        pred = np.asarray(model.predict(pl_full), dtype=np.float64).reshape(-1)
        close = frame["close"].to_numpy(dtype=np.float64)
        implied = np.full(n, np.nan)
        implied[1:] = pred[1:] / np.maximum(close[:-1], 1e-12) - 1.0
        scores = implied
    except Exception:  # noqa: BLE001
        scores = np.zeros(n, dtype=np.float64)
        fc = model.forecast(pl_full[:train_end], horizon=1)
        last = float(np.asarray(fc.values).reshape(-1)[0])
        prev = float(frame["close"].iloc[train_end - 1])
        scores[train_end:] = last / max(prev, 1e-12) - 1.0
    signal = _oos_signal_from_scores(scores, spec, frame.index, train_end=train_end)
    return {
        "status": "PASS",
        "signal": signal,
        "scores": scores,
        "meta": _meta(spec, frame, train_end),
        "slices": {k: [sl.start, sl.stop] for k, sl in slices.items()},
        "fit_mode": "train_only",
        "model_exists": True,
        "forecast_generated": True,
        "signal_generated": True,
    }


def generate_tree_signal(
    frame: pd.DataFrame, spec: ModelAdapterSpec, *, train_frac: float = 0.5
) -> dict[str, Any]:
    from iqrp.app.forecasting.tree_models import TreeSettings, create_tree_model

    assert_no_future_columns(frame)
    assert_timestamps_monotonic(frame["timestamp"])
    n = len(frame)
    slices = train_val_oos_slices(n, train_frac=train_frac, validation_frac=0.25)
    train_end = slices["train"].stop
    pl_full = _to_polars_ohlcv(frame, with_next_return_target=True)
    feat_cols = [c for c in _FEATURE_COLS if c in pl_full.columns]
    if not feat_cols:
        raise AdapterValidationError("tree model requires lagged feature columns")
    train = pl_full[:train_end].drop_nulls(subset=["target"])
    settings = TreeSettings.from_mapping(
        {
            "hyperparameters": {"n_estimators": 25, "max_depth": 3},
            "visualization": {"enabled": False},
            "optimization": {"method": "none"},
            "regime": {"enabled": False},
        }
    )
    model_id = spec.model_id
    try:
        model = create_tree_model(model_id, settings=settings)
        model.fit(train, feature_columns=list(feat_cols), target_column="target")
    except Exception as primary_err:  # noqa: BLE001
        try:
            model_id = "hist_gradient_boosting"
            model = create_tree_model(model_id, settings=settings)
            model.fit(train, feature_columns=list(feat_cols), target_column="target")
        except Exception as e:  # noqa: BLE001
            return {
                "status": "UNAVAILABLE",
                "reason": f"tree fit failed: {primary_err}; fallback: {e}"[:400],
                "signal": None,
                "model_exists": True,
                "forecast_generated": False,
                "signal_generated": False,
            }
    pred = np.asarray(model.predict(pl_full), dtype=np.float64).reshape(-1)
    signal = _oos_signal_from_scores(pred, spec, frame.index, train_end=train_end)
    meta = _meta(spec, frame, train_end)
    meta["source_model"] = model_id
    return {
        "status": "PASS",
        "signal": signal,
        "scores": pred,
        "meta": meta,
        "slices": {k: [sl.start, sl.stop] for k, sl in slices.items()},
        "fit_mode": "train_only_next_return_target",
        "warning": (
            "Supervised target is next-bar return for fit only; features are lagged. "
            "Train-region signal zeroed for OOS purity."
        ),
        "model_exists": True,
        "forecast_generated": True,
        "signal_generated": True,
    }


def generate_regime_signal(
    frame: pd.DataFrame,
    spec: ModelAdapterSpec,
    *,
    train_frac: float = 0.5,
    model_name: str | None = None,
) -> dict[str, Any]:
    from iqrp.app.regimes import ensure_regime_models_loaded, get_registry

    name = model_name or spec.model_id
    modules = ["iqrp.app.regimes.models.mock"]
    if name in {"hmm", "markov_chain"}:
        modules.append("iqrp.app.regimes.hmm.model")
    ensure_regime_models_loaded(modules)
    reg = get_registry()
    if name not in reg.list_names():
        return {
            "status": "UNAVAILABLE",
            "reason": f"regime model {name} not in registry ({reg.list_names()})",
            "signal": None,
            "model_exists": False,
            "forecast_generated": False,
            "signal_generated": False,
        }

    assert_no_future_columns(frame)
    assert_timestamps_monotonic(frame["timestamp"])
    n = len(frame)
    slices = train_val_oos_slices(n, train_frac=train_frac, validation_frac=0.25)
    train_end = slices["train"].stop
    pl_full = _to_polars_ohlcv(frame)
    feat_cols = [c for c in ("f_ret1", "f_vol", "returns") if c in pl_full.columns]
    model = reg.create(name)
    try:
        model.fit(pl_full[:train_end], feature_columns=feat_cols or None)
        states = np.asarray(model.predict(pl_full, feature_columns=feat_cols or None)).reshape(-1)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "UNAVAILABLE",
            "reason": str(e)[:300],
            "signal": None,
            "model_exists": True,
            "forecast_generated": False,
            "signal_generated": False,
        }
    signal = regime_result_to_signal_series(states, frame.index, spec.signal_mapping)
    arr = signal.to_numpy(dtype=np.float64).copy()
    arr[:train_end] = 0.0
    signal = pd.Series(arr, index=frame.index)
    return {
        "status": "PASS",
        "signal": signal,
        "states": states,
        "meta": _meta(spec, frame, train_end),
        "slices": {k: [sl.start, sl.stop] for k, sl in slices.items()},
        "fit_mode": "train_only",
        "model_exists": True,
        "forecast_generated": True,
        "signal_generated": True,
    }


def _fast_neural_settings() -> Any:
    from iqrp.app.forecasting.neural import NeuralSettings

    return NeuralSettings.from_mapping(
        {
            "architecture": {
                "lookback": 10,
                "horizon": 1,
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "train": {
                "epochs": 1,
                "batch_size": 32,
                "device": "cpu",
                "early_stopping_patience": 20,
                "seed": 0,
            },
            "scheduler": {"name": "none"},
            "regime": {"enabled": False},
            "visualization": {"enabled": False},
            "optimization": {"method": "none"},
        }
    )


def _fast_transformer_settings() -> Any:
    from iqrp.app.forecasting.transformers import TransformerSettings

    return TransformerSettings.from_mapping(
        {
            "architecture": {
                "lookback": 12,
                "horizon": 1,
                "d_model": 32,
                "n_heads": 4,
                "num_layers": 1,
                "ffn_dim": 64,
                "dropout": 0.0,
                "patch_len": 4,
                "stride": 2,
                "moving_avg": 5,
                "chunk_size": 8,
            },
            "train": {
                "epochs": 1,
                "batch_size": 16,
                "device": "cpu",
                "early_stopping_patience": 20,
                "seed": 0,
            },
            "scheduler": {"name": "none"},
            "regime": {"enabled": False},
            "visualization": {"enabled": False},
            "optimization": {"method": "none"},
        }
    )


def generate_neural_or_transformer_signal(
    frame: pd.DataFrame,
    spec: ModelAdapterSpec,
    *,
    train_frac: float = 0.5,
    kind: str = "neural",
) -> dict[str, Any]:
    try:
        if kind == "neural":
            from iqrp.app.forecasting.neural import create_neural_model

            model = create_neural_model(spec.model_id, settings=_fast_neural_settings())
        else:
            from iqrp.app.forecasting.transformers import create_transformer_model

            model = create_transformer_model(spec.model_id, settings=_fast_transformer_settings())
    except Exception as e:  # noqa: BLE001
        return {
            "status": "UNAVAILABLE",
            "reason": f"create failed: {e}",
            "signal": None,
            "model_exists": False,
            "forecast_generated": False,
            "signal_generated": False,
        }

    assert_no_future_columns(frame)
    assert_timestamps_monotonic(frame["timestamp"])
    n = len(frame)
    slices = train_val_oos_slices(n, train_frac=train_frac, validation_frac=0.25)
    train_end = slices["train"].stop
    pl_full = _to_polars_ohlcv(frame, with_next_return_target=True)
    feat_cols = [c for c in _FEATURE_COLS if c in pl_full.columns]
    train = pl_full[:train_end].drop_nulls(subset=["target"])
    try:
        model.fit(train, feature_columns=list(feat_cols), target_column="target")
        pred = np.asarray(model.predict(pl_full), dtype=np.float64).reshape(-1)
        if pred.size != n:
            # sequence models may return shorter path — right-align
            padded = np.full(n, np.nan, dtype=np.float64)
            padded[-pred.size :] = pred.reshape(-1)[:n]
            pred = padded
        signal = _oos_signal_from_scores(pred, spec, frame.index, train_end=train_end)
        return {
            "status": "PASS",
            "signal": signal,
            "scores": pred,
            "meta": _meta(spec, frame, train_end),
            "slices": {k: [sl.start, sl.stop] for k, sl in slices.items()},
            "fit_mode": "train_only",
            "model_exists": True,
            "forecast_generated": True,
            "signal_generated": True,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "status": "UNAVAILABLE",
            "reason": str(e)[:300],
            "signal": None,
            "model_exists": True,
            "forecast_generated": False,
            "signal_generated": False,
        }


def align_model_signal_mtf(
    model_frame: pd.DataFrame,
    signal: pd.Series,
    execution_frame: pd.DataFrame,
) -> pd.Series:
    aligned = align_feature_to_execution(model_frame, signal, execution_frame["timestamp"])
    aligned.index = execution_frame.index
    return aligned


def run_adapter(adapter_id: str, frame: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    spec = get_adapter(adapter_id)
    family = spec.model_family
    if family == "volatility":
        return generate_garch_signal(frame, spec, **kwargs)
    if family == "statistical":
        return generate_arima_signal(frame, spec, **kwargs)
    if family == "tree_ml":
        return generate_tree_signal(frame, spec, **kwargs)
    if family == "regime":
        return generate_regime_signal(frame, spec, model_name=spec.model_id, **kwargs)
    if family == "neural":
        return generate_neural_or_transformer_signal(frame, spec, kind="neural", **kwargs)
    if family == "transformer":
        return generate_neural_or_transformer_signal(frame, spec, kind="transformer", **kwargs)
    return {
        "status": "UNAVAILABLE",
        "reason": f"unknown family {family}",
        "signal": None,
        "model_exists": False,
        "forecast_generated": False,
        "signal_generated": False,
    }


__all__ = [
    "align_model_signal_mtf",
    "generate_arima_signal",
    "generate_garch_signal",
    "generate_neural_or_transformer_signal",
    "generate_regime_signal",
    "generate_tree_signal",
    "run_adapter",
]

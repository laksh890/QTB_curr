"""Reference RESEARCH features (not profitability claims)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.features import FeatureRegistry, FeatureSpec


def _close(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(frame["close"], dtype=np.float64)


def _high(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(frame["high"], dtype=np.float64)


def _low(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(frame["low"], dtype=np.float64)


def _vol(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(frame["volume"], dtype=np.float64)


def feat_returns(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    lb = max(int(spec.lookback), 1)
    c = _close(frame)
    return c / c.shift(lb) - 1.0


def feat_log_returns(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    lb = max(int(spec.lookback), 1)
    c = _close(frame)
    return np.log(c / c.shift(lb))


def feat_volatility(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    lb = max(int(spec.lookback), 2)
    r = _close(frame).pct_change()
    return r.rolling(lb, min_periods=lb).std(ddof=1)


def feat_atr(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    lb = max(int(spec.lookback), 2)
    h, l, c = _high(frame), _low(frame), _close(frame)
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(lb, min_periods=lb).mean()


def feat_rsi(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    lb = max(int(spec.lookback), 2)
    delta = _close(frame).diff()
    gain = delta.clip(lower=0.0).rolling(lb, min_periods=lb).mean()
    loss = (-delta.clip(upper=0.0)).rolling(lb, min_periods=lb).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def feat_ma(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    lb = max(int(spec.lookback), 1)
    return _close(frame).rolling(lb, min_periods=lb).mean()


def feat_ema(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    lb = max(int(spec.lookback), 1)
    return _close(frame).ewm(span=lb, adjust=False, min_periods=lb).mean()


def feat_macd(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    fast = int(spec.parameters.get("fast", 12))
    slow = int(spec.parameters.get("slow", 26))
    c = _close(frame)
    return c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()


def feat_momentum(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    return feat_returns(frame, spec)


def feat_rolling_zscore(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    lb = max(int(spec.lookback), 2)
    c = _close(frame)
    mu = c.rolling(lb, min_periods=lb).mean()
    sd = c.rolling(lb, min_periods=lb).std(ddof=1)
    return (c - mu) / sd.replace(0.0, np.nan)


def feat_volume_change(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    lb = max(int(spec.lookback), 1)
    v = _vol(frame)
    return v / v.shift(lb) - 1.0


def feat_vwap_distance(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    lb = max(int(spec.lookback), 2)
    c = _close(frame)
    v = _vol(frame)
    pv = (c * v).rolling(lb, min_periods=lb).sum()
    vv = v.rolling(lb, min_periods=lb).sum().replace(0.0, np.nan)
    vwap = pv / vv
    return c / vwap - 1.0


def feat_range(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    return (_high(frame) - _low(frame)) / _close(frame)


def feat_true_range(frame: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    h, l, c = _high(frame), _low(frame), _close(frame)
    prev = c.shift(1)
    return pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)


def register_reference_features(reg: FeatureRegistry) -> None:
    specs = [
        (FeatureSpec("returns", description="Simple N-bar return", lookback=1, family="momentum"), feat_returns),
        (FeatureSpec("log_returns", description="Log N-bar return", lookback=1, family="momentum"), feat_log_returns),
        (FeatureSpec("volatility", description="Rolling return volatility", lookback=20, family="volatility"), feat_volatility),
        (FeatureSpec("ATR", description="Average true range", lookback=14, family="volatility", inputs=("high", "low", "close")), feat_atr),
        (FeatureSpec("RSI", description="Relative strength index", lookback=14, family="mean_reversion"), feat_rsi),
        (FeatureSpec("moving_average", description="Simple moving average", lookback=20, family="trend"), feat_ma),
        (FeatureSpec("EMA", description="Exponential moving average", lookback=20, family="trend"), feat_ema),
        (FeatureSpec("MACD", description="MACD line", lookback=26, family="trend", parameters={"fast": 12, "slow": 26}), feat_macd),
        (FeatureSpec("momentum", description="N-bar momentum (= return)", lookback=20, family="momentum"), feat_momentum),
        (FeatureSpec("rolling_zscore", description="Price rolling z-score", lookback=20, family="mean_reversion"), feat_rolling_zscore),
        (FeatureSpec("volume_change", description="Volume change", lookback=1, family="volume", inputs=("volume",)), feat_volume_change),
        (FeatureSpec("VWAP_distance", description="Distance to rolling VWAP", lookback=20, family="volume", inputs=("close", "volume")), feat_vwap_distance),
        (FeatureSpec("range", description="(high-low)/close", lookback=1, family="price_action", inputs=("high", "low", "close")), feat_range),
        (FeatureSpec("true_range", description="True range", lookback=1, family="price_action", inputs=("high", "low", "close")), feat_true_range),
    ]
    for spec, fn in specs:
        reg.register(spec, fn, overwrite=True)


__all__ = ["register_reference_features"]

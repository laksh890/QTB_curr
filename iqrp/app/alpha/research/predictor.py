"""Simple predictive wrappers for signal → forward return research.

CRITICAL:
- Predictive fit is research evidence, not approval.
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
- Training uses only past observations at each evaluation point (walk-forward).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from iqrp.app.alpha.research.decay import forward_returns
from iqrp.app.features.research._numeric import (
    r_squared,
    ridge_fit_predict,
    safe_nanmean,
)


@dataclass(slots=True)
class PredictionResult:
    predictions: np.ndarray
    r_squared: float
    n_train: int
    n_test: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "r_squared": self.r_squared,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_pred": int(len(self.predictions)),
            "disclaimer": "Predictive R² ≠ alpha.",
        }


class SignalPredictor:
    """Walk-forward univariate ridge from signal to forward returns."""

    def __init__(
        self,
        *,
        horizon: int = 1,
        min_train: int = 60,
        test_size: int = 20,
        step: int = 20,
        ridge_alpha: float = 1.0,
    ) -> None:
        self.horizon = horizon
        self.min_train = min_train
        self.test_size = test_size
        self.step = step
        self.ridge_alpha = ridge_alpha

    def predict(
        self,
        signal: np.ndarray,
        returns: np.ndarray,
    ) -> PredictionResult:
        x = np.asarray(signal, dtype=np.float64)
        r = np.asarray(returns, dtype=np.float64)
        y = forward_returns(r, self.horizon)
        n = len(x)
        preds = np.full(n, np.nan, dtype=np.float64)
        r2s: list[float] = []
        n_train_last = 0
        n_test_last = 0
        start = self.min_train
        while start + self.test_size <= n:
            train_end = start
            test_end = min(n, start + self.test_size)
            # Point-in-time: train on [0, train_end), test on [train_end, test_end)
            x_tr, y_tr = x[:train_end], y[:train_end]
            x_te, y_te = x[train_end:test_end], y[train_end:test_end]
            if np.isfinite(x_tr).sum() < 10 or np.isfinite(y_tr).sum() < 10:
                start += self.step
                continue
            pred = ridge_fit_predict(x_tr, y_tr, x_te, alpha=self.ridge_alpha)
            preds[train_end:test_end] = pred
            r2s.append(r_squared(y_te, pred))
            n_train_last = int(train_end)
            n_test_last = int(test_end - train_end)
            start += self.step
        return PredictionResult(
            predictions=preds,
            r_squared=safe_nanmean(np.asarray(r2s, dtype=np.float64)),
            n_train=n_train_last,
            n_test=n_test_last,
        )


def predict_forward(
    signal: np.ndarray,
    returns: np.ndarray,
    *,
    horizon: int = 1,
    ridge_alpha: float = 1.0,
) -> PredictionResult:
    """Expanding-window convenience predictor."""
    return SignalPredictor(horizon=horizon, ridge_alpha=ridge_alpha).predict(signal, returns)

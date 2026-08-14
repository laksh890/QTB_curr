"""BEKK-GARCH multivariate volatility model."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.volatility.base.volatility_model import VolatilityModel


def _bekk_path(
    eps: np.ndarray, c_mat: np.ndarray, a_mat: np.ndarray, b_mat: np.ndarray
) -> np.ndarray:
    t, k = eps.shape
    h = np.zeros((t, k, k))
    h0 = np.cov(eps, rowvar=False)
    if h0.ndim == 0:
        h0 = np.array([[float(h0)]])
    h0 = 0.5 * (h0 + h0.T) + 1e-6 * np.eye(k)
    h[0] = h0
    ct = c_mat @ c_mat.T
    for i in range(1, t):
        e = eps[i - 1 : i].T  # (k,1)
        h[i] = ct + a_mat @ (e @ e.T) @ a_mat.T + b_mat @ h[i - 1] @ b_mat.T
        h[i] = 0.5 * (h[i] + h[i].T) + 1e-8 * np.eye(k)
    return h


@register_forecast_model
class BEKKModel(VolatilityModel):
    meta = ForecastModelMeta(
        name="bekk",
        version="1.0.0",
        description="Baba-Engle-Kraft-Kroner BEKK-GARCH",
        algorithm_family="volatility",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(self, settings: Any | None = None, **params: Any) -> None:
        super().__init__(settings=settings, **params)
        self._returns_m: np.ndarray | None = None
        self._a_mat: np.ndarray | None = None
        self._b_mat: np.ndarray | None = None
        self._c_mat: np.ndarray | None = None

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> BEKKModel:
        self._maybe_regime(frame, regime_column)
        assets = list(feature_columns or self._vol_settings.columns.assets or [])
        if not assets:
            exclude = {
                self._vol_settings.columns.timestamp,
                self._vol_settings.columns.target,
                self._regime_column,
                "open_time",
                "timestamp",
            }
            assets = [
                c
                for c in frame.columns
                if c not in exclude and getattr(frame[c].dtype, "is_numeric", lambda: False)()
            ]
        if len(assets) < 2:
            tgt = self._resolve_target_name(frame, target_column)
            r0 = frame[tgt].to_numpy().astype(np.float64)
            assets = [tgt, f"{tgt}_b"]
            mat = np.column_stack([r0, np.roll(r0, 1)])
            mat[0, 1] = r0[0]
        else:
            mat = frame.select(assets[:2]).to_numpy().astype(np.float64)  # scalar BEKK on 2 assets
            assets = assets[:2]
        mat = mat - np.mean(mat, axis=0, keepdims=True)
        t, k = mat.shape

        # scalar BEKK parameterization for stability: C lower triangular, A=a*I, B=b*I
        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            c11, c21, c22, a, b = map(float, theta)
            c = np.array([[abs(c11), 0.0], [c21, abs(c22)]])
            am = a * np.eye(k)
            bm = b * np.eye(k)
            return c, am, bm

        def nll(theta: np.ndarray) -> float:
            a, b = float(theta[3]), float(theta[4])
            if a < 0 or b < 0 or a**2 + b**2 >= 0.999:
                return 1e20
            c, am, bm = unpack(theta)
            try:
                h = _bekk_path(mat, c, am, bm)
            except Exception:
                return 1e20
            ll = 0.0
            for i in range(t):
                try:
                    sign, logdet = np.linalg.slogdet(h[i])
                    if sign <= 0:
                        return 1e20
                    inv = np.linalg.inv(h[i])
                    ll += logdet + float(mat[i] @ inv @ mat[i])
                except Exception:
                    return 1e20
            return 0.5 * ll

        s = np.cov(mat, rowvar=False)
        chol = np.linalg.cholesky(s + 1e-6 * np.eye(k))
        x0 = np.array([chol[0, 0], chol[1, 0], chol[1, 1], 0.2, 0.9])
        res = minimize(
            nll,
            x0,
            method="L-BFGS-B",
            bounds=[(1e-6, None), (None, None), (1e-6, None), (1e-6, 0.5), (1e-6, 0.99)],
        )
        c, am, bm = unpack(res.x)
        h = _bekk_path(mat, c, am, bm)
        self._c_mat, self._a_mat, self._b_mat = c, am, bm
        self._cov_series = h
        self._returns_m = mat
        self._assets = assets
        var = h[:, 0, 0]
        params = {
            "a": float(res.x[3]),
            "b": float(res.x[4]),
            "c11": float(c[0, 0]),
            "c21": float(c[1, 0]),
            "c22": float(c[1, 1]),
        }
        ll = -float(nll(res.x))
        self._finalize(
            mat[:, 0],
            var,
            target_column=assets[0],
            params=params,
            loglik=ll,
            aic=-2 * ll + 10,
            bic=-2 * ll + 5 * np.log(max(t, 1)),
            extras={"assets": assets},
        )
        self._feature_columns = assets
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        self._require_fitted()
        assert self._variance is not None
        return np.sqrt(self._variance)

    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        self._require_fitted()
        h = self._default_horizon(horizon)
        assert self._cov_series is not None and self._a_mat is not None
        last = self._cov_series[-1]
        a2 = float(self._params["a"]) ** 2
        b2 = float(self._params["b"]) ** 2
        ct = self._c_mat @ self._c_mat.T
        # unconditional
        persist = a2 + b2
        uncond = ct / max(1 - persist, 1e-6) if persist < 1 else last
        cov_path = np.zeros((h, last.shape[0], last.shape[1]))
        # one-step: C'C + A eps eps' A' + B H B' — use last outer product
        eps = self._returns_m[-1]
        outer = np.outer(eps, eps)
        cov_path[0] = ct + self._a_mat @ outer @ self._a_mat.T + self._b_mat @ last @ self._b_mat.T
        for i in range(1, h):
            cov_path[i] = uncond + (persist**i) * (cov_path[0] - uncond)
            cov_path[i] = 0.5 * (cov_path[i] + cov_path[i].T)
        sigma = np.sqrt(np.clip(cov_path[:, 0, 0], 1e-12, None))
        fc = self._build_vol_forecast(sigma, sigma**2, horizon=h)
        return Forecast.from_values(
            sigma,
            horizon=h,
            model_name=self.meta.name,
            model_version=self.meta.version,
            features_used=tuple(self._feature_columns),
            strategy="recursive",
            intervals=fc.intervals,
            metadata={**fc.metadata, "covariance": cov_path.tolist()},
        )

    def forecast_covariance(self, *, horizon: int | None = None) -> np.ndarray:
        self._require_fitted()
        h = self._default_horizon(horizon)
        fc = self.forecast(
            pl.DataFrame({a: self._returns_m[:, i] for i, a in enumerate(self._assets)}),
            horizon=h,
        )
        return np.asarray(fc.metadata["covariance"], dtype=np.float64)

    def _algorithm_state(self) -> dict[str, Any]:
        state = super()._algorithm_state()
        state["returns_m"] = None if self._returns_m is None else self._returns_m.tolist()
        state["a_mat"] = None if self._a_mat is None else self._a_mat.tolist()
        state["b_mat"] = None if self._b_mat is None else self._b_mat.tolist()
        state["c_mat"] = None if self._c_mat is None else self._c_mat.tolist()
        return state

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        super()._load_algorithm_state(state)
        for key, attr in [
            ("returns_m", "_returns_m"),
            ("a_mat", "_a_mat"),
            ("b_mat", "_b_mat"),
            ("c_mat", "_c_mat"),
        ]:
            val = state.get(key)
            setattr(self, attr, None if val is None else np.asarray(val, dtype=np.float64))

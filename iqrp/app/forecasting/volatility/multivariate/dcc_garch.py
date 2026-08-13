"""Dynamic Conditional Correlation GARCH."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.metadata import ForecastModelMeta
from iqrp.app.forecasting.base.registry import register_forecast_model
from iqrp.app.forecasting.volatility.base.recursion import ewma_variance, garch_variance
from iqrp.app.forecasting.volatility.base.volatility_model import VolatilityModel


def _dcc_q_path(z: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Q_t series and R_t correlation matrices. z: (T,K) standardized."""
    t, k = z.shape
    s = np.cov(z, rowvar=False)
    if s.ndim == 0:
        s = np.array([[float(s)]])
    # ensure PD
    s = 0.5 * (s + s.T) + 1e-6 * np.eye(k)
    q = np.zeros((t, k, k))
    r = np.zeros((t, k, k))
    q[0] = s.copy()
    dinv = np.diag(1.0 / np.sqrt(np.clip(np.diag(q[0]), 1e-12, None)))
    r[0] = dinv @ q[0] @ dinv
    for i in range(1, t):
        outer = np.outer(z[i - 1], z[i - 1])
        q[i] = (1 - a - b) * s + a * outer + b * q[i - 1]
        dinv = np.diag(1.0 / np.sqrt(np.clip(np.diag(q[i]), 1e-12, None)))
        r[i] = dinv @ q[i] @ dinv
        # clip correlations
        r[i] = np.clip(r[i], -0.999, 0.999)
        np.fill_diagonal(r[i], 1.0)
    return q, r


@register_forecast_model
class DCCGARCHModel(VolatilityModel):
    meta = ForecastModelMeta(
        name="dcc_garch",
        version="1.0.0",
        description="Dynamic Conditional Correlation GARCH",
        algorithm_family="volatility",
        task="regression",
        default_horizon=5,
        supports_online=True,
        supports_intervals=True,
    )

    def __init__(self, settings: Any | None = None, **params: Any) -> None:
        super().__init__(settings=settings, **params)
        self._corr_series: np.ndarray | None = None
        self._vol_paths: np.ndarray | None = None  # (T,K)
        self._returns_m: np.ndarray | None = None

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> DCCGARCHModel:
        self._maybe_regime(frame, regime_column)
        assets = list(feature_columns or self._vol_settings.columns.assets or [])
        if not assets:
            # use all numeric except timestamp/target
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
            # synthesize second series from target if needed
            tgt = self._resolve_target_name(frame, target_column)
            r0 = frame[tgt].to_numpy().astype(np.float64)
            assets = [tgt, f"{tgt}_lag"]
            data = {tgt: r0, f"{tgt}_lag": np.roll(r0, 1)}
            data[f"{tgt}_lag"][0] = r0[0]
            mat = np.column_stack([data[a] for a in assets])
        else:
            mat = frame.select(assets).to_numpy().astype(np.float64)
        mat = mat - np.mean(mat, axis=0, keepdims=True)
        t, k = mat.shape
        # univariate GARCH vols
        vols = np.zeros((t, k))
        for j in range(k):
            mean2 = float(np.mean(mat[:, j] ** 2))
            vols[:, j] = np.sqrt(
                garch_variance(mat[:, j], 0.05 * mean2, np.array([0.05]), np.array([0.9]))
            )
            # refine with EWMA if needed
            if not np.all(np.isfinite(vols[:, j])):
                vols[:, j] = np.sqrt(ewma_variance(mat[:, j], 0.94))
        z = mat / np.clip(vols, 1e-12, None)

        def nll(theta: np.ndarray) -> float:
            a, b = float(theta[0]), float(theta[1])
            if a < 0 or b < 0 or a + b >= 0.999:
                return 1e20
            _, rmat = _dcc_q_path(z, a, b)
            ll = 0.0
            for i in range(t):
                ri = rmat[i]
                try:
                    sign, logdet = np.linalg.slogdet(ri)
                    if sign <= 0:
                        return 1e20
                    inv = np.linalg.inv(ri)
                    ll += logdet + float(z[i] @ inv @ z[i])
                except Exception:  # noqa: BLE001
                    return 1e20
            return 0.5 * ll

        res = minimize(nll, np.array([0.05, 0.9]), method="L-BFGS-B", bounds=[(1e-6, 0.5), (1e-6, 0.99)])
        a, b = map(float, res.x)
        if a + b >= 0.999:
            a, b = 0.05, 0.9
        q_path, r_path = _dcc_q_path(z, a, b)
        # covariance H_t = D R D
        cov = np.zeros((t, k, k))
        for i in range(t):
            d = np.diag(vols[i])
            cov[i] = d @ r_path[i] @ d
        # store portfolio / first-asset variance for univariate API
        var = cov[:, 0, 0]
        self._assets = assets
        self._returns_m = mat
        self._vol_paths = vols
        self._corr_series = r_path
        self._cov_series = cov
        params = {"a": a, "b": b, "n_assets": float(k)}
        ll = -float(nll(np.array([a, b])))
        tgt = assets[0]
        self._finalize(
            mat[:, 0],
            var,
            target_column=tgt,
            params=params,
            loglik=ll,
            aic=-2 * ll + 4,
            bic=-2 * ll + 2 * np.log(max(t, 1)),
            extras={"assets": assets},
        )
        self._feature_columns = assets
        return self

    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
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
        assert self._cov_series is not None and self._vol_paths is not None
        last_cov = self._cov_series[-1]
        a = float(self._params["a"])
        b = float(self._params["b"])
        # correlation mean-reverts slowly; vols use last
        cov_path = np.repeat(last_cov[None, :, :], h, axis=0)
        # mild persistence blend toward sample cov
        target = np.mean(self._cov_series, axis=0)
        for i in range(h):
            w = (a + b) ** (i + 1)
            cov_path[i] = w * last_cov + (1 - w) * target
        sigma = np.sqrt(np.clip(np.diagonal(cov_path, axis1=1, axis2=2)[:, 0], 1e-12, None))
        var = sigma**2
        fc = self._build_vol_forecast(sigma, var, horizon=h)
        return Forecast.from_values(
            sigma,
            horizon=h,
            model_name=self.meta.name,
            model_version=self.meta.version,
            features_used=tuple(self._feature_columns),
            strategy="recursive",
            intervals=fc.intervals,
            metadata={
                **fc.metadata,
                "covariance": cov_path.tolist(),
                "correlation": [
                    (c / np.outer(np.sqrt(np.diag(c)), np.sqrt(np.diag(c)))).tolist()
                    for c in cov_path
                ],
            },
        )

    def forecast_covariance(self, *, horizon: int | None = None) -> np.ndarray:
        self._require_fitted()
        h = self._default_horizon(horizon)
        fc = self.forecast(
            pl.DataFrame({a: self._returns_m[:, i] for i, a in enumerate(self._assets)}),
            horizon=h,
        )
        return np.asarray(fc.metadata["covariance"], dtype=np.float64)

    def correlation_path(self) -> np.ndarray:
        self._require_fitted()
        assert self._corr_series is not None
        return self._corr_series.copy()

    def _algorithm_state(self) -> dict[str, Any]:
        state = super()._algorithm_state()
        state["corr_series"] = None if self._corr_series is None else self._corr_series.tolist()
        state["vol_paths"] = None if self._vol_paths is None else self._vol_paths.tolist()
        state["returns_m"] = None if self._returns_m is None else self._returns_m.tolist()
        return state

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        super()._load_algorithm_state(state)
        c = state.get("corr_series")
        self._corr_series = None if c is None else np.asarray(c, dtype=np.float64)
        v = state.get("vol_paths")
        self._vol_paths = None if v is None else np.asarray(v, dtype=np.float64)
        r = state.get("returns_m")
        self._returns_m = None if r is None else np.asarray(r, dtype=np.float64)

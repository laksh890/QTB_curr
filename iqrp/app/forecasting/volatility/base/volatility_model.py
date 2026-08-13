"""Base class for institutional volatility forecasting models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport
from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.forecast_model import ForecastModel
from iqrp.app.forecasting.base.metadata import TrainingMetadata
from iqrp.app.forecasting.base.prediction import PredictionInterval
from iqrp.app.forecasting.postprocessing.intervals import residual_intervals
from iqrp.app.forecasting.volatility.config import VolatilitySettings
from iqrp.app.forecasting.volatility.diagnostics.report import VolatilityDiagnosticReport, run_vol_diagnostics
from iqrp.app.forecasting.volatility.evaluation.metrics import evaluate_volatility


class VolatilityModel(ForecastModel):
    """Common API for univariate / multivariate volatility models."""

    def __init__(self, settings: VolatilitySettings | Any | None = None, **params: Any) -> None:
        if settings is None:
            settings = VolatilitySettings.default()
        elif isinstance(settings, dict):
            settings = VolatilitySettings.from_mapping(settings)
        super().__init__(settings=settings)
        self._vol_settings: VolatilitySettings = settings  # type: ignore[assignment]
        self._params_kw: dict[str, Any] = dict(params)
        self._returns: np.ndarray | None = None
        self._variance: np.ndarray | None = None
        self._params: dict[str, float] = {}
        self._param_vector: np.ndarray | None = None
        self._loglik: float = 0.0
        self._ic: dict[str, float] = {}
        self._regime_params: dict[Any, dict[str, float]] = {}
        self._assets: list[str] = []
        self._cov_series: np.ndarray | None = None  # (T, K, K)

    @property
    def params(self) -> dict[str, float]:
        return dict(self._params)

    @property
    def information_criteria(self) -> dict[str, float]:
        return dict(self._ic)

    def conditional_variance(self) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self._variance, dtype=np.float64).copy()

    def conditional_volatility(self) -> np.ndarray:
        return np.sqrt(self.conditional_variance())

    def annualized_volatility(self) -> np.ndarray:
        ann = float(self._vol_settings.order.annualization)
        return self.conditional_volatility() * np.sqrt(ann)

    def forecast_covariance(self, *, horizon: int | None = None) -> np.ndarray:
        """Return covariance forecast; univariate → shape (H,), multivariate (H,K,K)."""
        self._require_fitted()
        h = self._default_horizon(horizon)
        if self._cov_series is not None and self._cov_series.ndim == 3:
            last = self._cov_series[-1]
            return np.repeat(last[None, :, :], h, axis=0)
        var_path = self.forecast(pl.DataFrame({self._target_column or "returns": self._returns}), horizon=h)
        return np.asarray(var_path.metadata.get("variance", var_path.values**2), dtype=np.float64)

    def diagnostics(self) -> VolatilityDiagnosticReport:
        self._require_fitted()
        assert self._returns is not None and self._variance is not None
        return run_vol_diagnostics(self._returns, self._variance, params=self._params)

    def evaluate(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        probabilities: np.ndarray | None = None,
        realized: np.ndarray | None = None,
    ) -> EvaluationReport:
        self._require_fitted()
        vol = self.predict(frame, feature_columns)
        tgt = target_column or self._target_column or self._vol_settings.columns.target
        rets = frame[tgt].to_numpy().astype(np.float64) if tgt in frame.columns else self._returns
        metrics = evaluate_volatility(rets, vol**2, realized=realized)
        return EvaluationReport(metrics=metrics, method="volatility", n_samples=int(vol.size))

    def forecast_interval(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        level: float | None = None,
        feature_columns: list[str] | None = None,
    ) -> list[PredictionInterval]:
        fc = self.forecast(frame, horizon=horizon, feature_columns=feature_columns)
        if fc.intervals is not None:
            return fc.intervals
        lvl = float(level if level is not None else self._vol_settings.forecast.interval_level)
        return residual_intervals(fc.path(), level=lvl)

    def partial_fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> VolatilityModel:
        if not self._fitted or not self._vol_settings.online.warm_start:
            return self.fit(  # type: ignore[return-value]
                frame, feature_columns, target_column=target_column, regime_column=regime_column
            )
        r_new = self._extract_returns(frame, target_column)
        if self._returns is None:
            return self.fit(  # type: ignore[return-value]
                frame, feature_columns, target_column=target_column, regime_column=regime_column
            )
        mode = self._vol_settings.online.mode
        w = int(self._vol_settings.online.window)
        if mode in {"sliding", "rolling"}:
            r_all = np.concatenate([self._returns, r_new])[-w:]
        else:
            r_all = np.concatenate([self._returns, r_new])
        rebuilt = self._frame_from_returns(r_all, frame, target_column)
        # adaptive warm start: nudge params toward new MLE
        prev = dict(self._params)
        self.fit(
            rebuilt,
            feature_columns or self._feature_columns,
            target_column=target_column or self._target_column,
            regime_column=regime_column or self._regime_column,
        )
        rate = float(self._vol_settings.online.adaptive_rate)
        if prev and self._params:
            for k in self._params:
                if k in prev:
                    self._params[k] = (1 - rate) * prev[k] + rate * self._params[k]
        return self

    def _extract_returns(self, frame: pl.DataFrame, target_column: str | None) -> np.ndarray:
        tgt = target_column or self._target_column or self._vol_settings.columns.target
        if tgt in frame.columns:
            return frame[tgt].to_numpy().astype(np.float64)
        cols = self._resolve_feature_columns(frame, None)
        if cols:
            return frame[cols[0]].to_numpy().astype(np.float64)
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError("Unable to extract returns series", code="VOL_NO_TARGET")

    def _resolve_target_name(self, frame: pl.DataFrame, target_column: str | None) -> str:
        if target_column:
            return target_column
        if self._target_column:
            return self._target_column
        cfg = self._vol_settings.columns.target
        if cfg in frame.columns:
            return cfg
        cols = self._resolve_feature_columns(frame, None)
        if cols:
            return cols[0]
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError("No returns column available", code="VOL_NO_TARGET")

    def _frame_from_returns(
        self, r: np.ndarray, template: pl.DataFrame, target_column: str | None
    ) -> pl.DataFrame:
        tgt = target_column or self._target_column or self._vol_settings.columns.target
        data: dict[str, Any] = {
            self._vol_settings.columns.timestamp: list(range(len(r))),
            tgt: r,
        }
        if self._regime_column and self._regime_column in template.columns:
            reg = template[self._regime_column].to_numpy()
            data[self._regime_column] = reg[-len(r) :] if reg.size >= len(r) else np.resize(reg, len(r))
        return pl.DataFrame(data)

    def _maybe_regime(self, frame: pl.DataFrame, regime_column: str | None) -> np.ndarray | None:
        col = regime_column or (
            self._vol_settings.regime.column if self._vol_settings.regime.enabled else None
        )
        if col and col in frame.columns:
            self._regime_column = col
            return frame[col].to_numpy()
        return None

    def _finalize(
        self,
        returns: np.ndarray,
        variance: np.ndarray,
        *,
        target_column: str,
        params: dict[str, float],
        loglik: float,
        aic: float,
        bic: float,
        extras: dict[str, Any] | None = None,
    ) -> None:
        self._returns = np.asarray(returns, dtype=np.float64).reshape(-1)
        self._variance = np.asarray(variance, dtype=np.float64).reshape(-1)
        self._params = dict(params)
        self._loglik = float(loglik)
        self._ic = {"aic": float(aic), "bic": float(bic), "loglik": float(loglik)}
        self._target_column = target_column
        self._feature_columns = [target_column]
        self._training_meta = TrainingMetadata(
            n_samples=int(self._returns.size),
            n_features=1,
            feature_columns=(target_column,),
            target_column=target_column,
            regime_column=self._regime_column,
            horizon=self._vol_settings.forecast.default_horizon,
            extra={"params": self._params, "ic": self._ic, **(extras or {})},
        )
        self._fitted = True

    def _build_vol_forecast(
        self,
        sigma: np.ndarray,
        variance: np.ndarray,
        *,
        horizon: int,
        regime_used: Any = None,
    ) -> Forecast:
        intervals = residual_intervals(
            sigma,
            residual_std=np.maximum(0.1 * sigma, 1e-6),
            level=self._vol_settings.forecast.interval_level,
        )
        return Forecast.from_values(
            sigma,
            horizon=horizon,
            model_name=self.meta.name,
            model_version=self.meta.version,
            features_used=tuple(self._feature_columns),
            regime_used=regime_used,
            strategy="recursive",
            intervals=intervals,
            metadata={
                "variance": np.asarray(variance, dtype=np.float64).tolist(),
                "params": self._params,
                "ic": self._ic,
                "annualized": (sigma * np.sqrt(self._vol_settings.order.annualization)).tolist(),
            },
        )

    def _default_horizon(self, horizon: int | None) -> int:
        if horizon is not None:
            return max(int(horizon), 1)
        return max(int(self._vol_settings.forecast.default_horizon), 1)

    def _dist_name(self) -> str:
        return str(self._vol_settings.distribution.name)

    def _dist_kwargs(self) -> dict[str, float]:
        d = self._vol_settings.distribution
        return {"df": float(d.df), "skew": float(d.skew), "ged_nu": float(d.ged_nu)}

    @abstractmethod
    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> VolatilityModel:
        ...

    @abstractmethod
    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        """In-sample conditional volatility (σ_t)."""

    @abstractmethod
    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        """N-step ahead conditional volatility path."""

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "params": dict(self._params),
            "param_vector": None if self._param_vector is None else self._param_vector.tolist(),
            "returns": None if self._returns is None else self._returns.tolist(),
            "variance": None if self._variance is None else self._variance.tolist(),
            "loglik": self._loglik,
            "ic": dict(self._ic),
            "regime_params": {str(k): dict(v) for k, v in self._regime_params.items()},
            "assets": list(self._assets),
            "cov_series": None if self._cov_series is None else self._cov_series.tolist(),
            "params_kw": dict(self._params_kw),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._params = dict(state.get("params") or {})
        pv = state.get("param_vector")
        self._param_vector = None if pv is None else np.asarray(pv, dtype=np.float64)
        r = state.get("returns")
        self._returns = None if r is None else np.asarray(r, dtype=np.float64)
        v = state.get("variance")
        self._variance = None if v is None else np.asarray(v, dtype=np.float64)
        self._loglik = float(state.get("loglik", 0.0))
        self._ic = dict(state.get("ic") or {})
        self._regime_params = {k: dict(vv) for k, vv in (state.get("regime_params") or {}).items()}
        self._assets = list(state.get("assets") or [])
        cov = state.get("cov_series")
        self._cov_series = None if cov is None else np.asarray(cov, dtype=np.float64)
        self._params_kw = dict(state.get("params_kw") or {})

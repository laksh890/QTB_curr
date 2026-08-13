"""Institutional Kalman Filtering Engine (State Space + Regime adapters)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import special  # type: ignore[import-untyped]

from iqrp.app.math.probability.likelihood import aic as aic_score, bic as bic_score
from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.base.registry import register_regime_model
from iqrp.app.regimes.kalman.config import KalmanSettings
from iqrp.app.regimes.kalman.diagnostics import KalmanDiagnostics
from iqrp.app.regimes.kalman.evaluator import KalmanEvaluator
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM, build_system
from iqrp.app.regimes.kalman.linear import FilterTrace
from iqrp.app.regimes.kalman.prediction import (
    forecast_observation,
    n_step_predict,
    prediction_intervals,
    predict_state,
)
from iqrp.app.regimes.kalman.serializer import KalmanSerializer
from iqrp.app.regimes.kalman.smoothing import SmoothTrace, rts_smooth
from iqrp.app.regimes.kalman.trainer import KalmanTrainer, run_filter, simulate_lds
from iqrp.app.regimes.kalman.update import update_state
from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.registry import register_state_space_model
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.base.state_space_model import StateSpaceModel, StateSpaceModelMeta

_SOFT_STATE_NAMES = ("bearish", "bullish")


def _soft_trend_proba(means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """Map continuous primary state to soft bullish/bearish probabilities."""
    m = np.asarray(means, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    primary = m[:, 0]
    std = np.sqrt(np.clip(np.array([c[0, 0] for c in covs], dtype=np.float64), 1e-12, None))
    # P(bullish) = Φ(mean / std)
    z = primary / std
    p_bull = 0.5 * (1.0 + special.erf(z / np.sqrt(2.0)))
    p_bull = np.clip(p_bull, 1e-6, 1.0 - 1e-6)
    return np.column_stack([1.0 - p_bull, p_bull])


def _resolve_names(names: tuple[str, ...] | None) -> tuple[str, ...]:
    if names and len(names) >= 2:
        return tuple(names[:2])
    if names:
        return tuple(names) + tuple(_SOFT_STATE_NAMES[len(names) :])
    return _SOFT_STATE_NAMES


@register_state_space_model
class KalmanFilterModel(StateSpaceModel):
    """Production Kalman filter for continuous latent-state estimation."""

    meta = StateSpaceModelMeta(
        name="kalman",
        version="1.0.0",
        description="Institutional Kalman Filtering Engine",
        n_states=2,
        algorithm_family="kalman",
        parameters={},
        state_names=_SOFT_STATE_NAMES,
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        n_obs: int | None = None,
        state_names: tuple[str, ...] | None = None,
        settings: KalmanSettings | None = None,
        random_seed: int | None = None,
        system: LinearGaussianSSM | None = None,
    ) -> None:
        super().__init__()
        self._kf_settings = settings or KalmanSettings.default()
        self._n_latent = int(n_states if n_states is not None else self._kf_settings.n_states)
        self._n_obs = int(n_obs if n_obs is not None else self._kf_settings.n_obs)
        names = state_names if state_names is not None else self._kf_settings.state_names
        self._state_names = _resolve_names(names)
        self._rng = np.random.default_rng(
            random_seed if random_seed is not None else self._kf_settings.random_seed
        )
        self.system: LinearGaussianSSM | None = system
        self._trace: FilterTrace | None = None
        self._smooth: SmoothTrace | None = None
        self._history: list[float] = []
        self._n_iter = 0
        self._converged = False
        self._train_obs: np.ndarray | None = None
        self._controls: np.ndarray | None = None
        self._h_seq: np.ndarray | None = None
        self._online_x: np.ndarray | None = None
        self._online_p: np.ndarray | None = None
        self._update_counter = 0
        self._last_innov: np.ndarray | None = None
        self._last_gain: np.ndarray | None = None
        self.meta = StateSpaceModelMeta(
            name="kalman",
            version="1.0.0",
            description=self.meta.description,
            n_states=2,
            algorithm_family="kalman",
            parameters={
                "n_latent": self._n_latent,
                "n_obs": self._n_obs,
                "filter_type": self._kf_settings.filter_type,
                "application": self._kf_settings.application,
            },
            state_names=self._state_names,
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        return self._state_names

    def fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        controls: np.ndarray | None = None,
        h_seq: np.ndarray | None = None,
    ) -> KalmanFilterModel:
        y = self._extract_obs(observations, observation_columns)
        ctrl = controls if controls is not None else self._extract_controls(observations)
        system = self.system or build_system(
            self._kf_settings,
            n_states=self._n_latent,
            n_obs=1 if self._kf_settings.application == "dynamic_beta" else y.shape[1],
        )
        h = h_seq if h_seq is not None else self._build_h_seq(y, observations)
        # dynamic beta: observation = asset (col0); market (col1) enters via H_t = [1, mkt]
        if (
            (system.application == "dynamic_beta" or self._kf_settings.application == "dynamic_beta")
            and y.shape[1] >= 2
        ):
            mkt = y[:, 1]
            if h is None:
                h = np.stack([np.ones(y.shape[0]), mkt], axis=1).reshape(y.shape[0], 1, 2)
            y = y[:, 0:1]
            system = build_system(self._kf_settings, application="dynamic_beta")
        result = KalmanTrainer(self._kf_settings).fit(
            y, system=system, controls=ctrl, h_seq=h, rng=self._rng
        )
        self._ingest(result.system, result.trace, y, result.history, result.n_iter, result.converged)
        self._controls = ctrl
        self._h_seq = h
        self._online_x = result.trace.means[-1].copy()
        self._online_p = result.trace.covs[-1].copy()
        return self

    def partial_fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> KalmanFilterModel:
        y = self._extract_obs(observations, observation_columns)
        if not self._fitted or self.system is None:
            return self.fit(observations, observation_columns=observation_columns)
        if not self._kf_settings.online.warm_start:
            return self.fit(observations, observation_columns=observation_columns)
        for t in range(y.shape[0]):
            self.update(y[t])
        self._update_counter += y.shape[0]
        every = int(self._kf_settings.online.checkpoint_every)
        if every > 0 and self._update_counter % every == 0:
            # warm re-filter on accumulated train buffer
            if self._train_obs is not None:
                buf = np.vstack([self._train_obs, y])
                self._train_obs = buf
            return self
        if self._train_obs is None:
            self._train_obs = y
        else:
            self._train_obs = np.vstack([self._train_obs, y])
        return self

    def update(self, observation: np.ndarray | float, *, control: np.ndarray | None = None) -> np.ndarray:
        """Single-step online update; returns filtered state mean."""
        self._require_fitted()
        assert self.system is not None
        z = np.asarray(observation, dtype=np.float64).reshape(-1)
        x = self._online_x if self._online_x is not None else self.system.x0.copy()
        p = self._online_p if self._online_p is not None else self.system.p0.copy()
        x_pred, p_pred = predict_state(x, p, self.system.f, self.system.q, b=self.system.b, u=control)
        x_new, p_new, innov, _s, k = update_state(x_pred, p_pred, z, self.system.h, self.system.r)
        self._online_x, self._online_p = x_new, p_new
        self._last_innov, self._last_gain = innov, k
        return x_new

    def filter(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> FilterResult:
        self._require_fitted()
        assert self.system is not None
        raw = self._extract_obs(observations, observation_columns)
        h_seq = None
        if (
            self.system.application == "dynamic_beta"
            and raw.shape[1] >= 2
        ):
            h_seq = np.stack([np.ones(raw.shape[0]), raw[:, 1]], axis=1).reshape(
                raw.shape[0], 1, 2
            )
            y = raw[:, 0:1]
        else:
            y = self._maybe_dynamic_beta_obs(raw)
            if self._h_seq is not None and self._h_seq.shape[0] == y.shape[0]:
                h_seq = self._h_seq
        trace = run_filter(
            y,
            self.system,
            self._kf_settings,
            controls=self._controls if self._controls is not None and self._controls.shape[0] == y.shape[0] else None,
            h_seq=h_seq,
        )
        self._trace = trace
        proba = _soft_trend_proba(trace.means, trace.covs)
        states = np.argmax(proba, axis=1).astype(np.int64)
        scales = np.array(
            [float(np.linalg.det(ensure_s(s))) for s in trace.innovation_covs], dtype=np.float64
        )
        return FilterResult(
            filtered_states=states,
            filtered_probabilities=proba,
            log_likelihood=trace.log_likelihood,
            normalization_constants=np.clip(scales, 1e-300, None),
            metadata={
                "means": trace.means,
                "covs": trace.covs,
                "innovations": trace.innovations,
                "gains": trace.gains,
                "filter": self._kf_settings.filter_type,
            },
        )

    def smooth(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        self._require_fitted()
        assert self.system is not None
        filt = self.filter(observations, observation_columns=observation_columns)
        assert self._trace is not None
        sm = rts_smooth(self._trace, self.system)
        self._smooth = sm
        proba = _soft_trend_proba(sm.means, sm.covs)
        states = np.argmax(proba, axis=1).astype(np.int64)
        return SmootherResult(
            smoothed_states=states,
            smoothed_probabilities=proba,
            backward_messages=proba,
            log_likelihood=filt.log_likelihood,
            metadata={"lag": lag, "means": sm.means, "covs": sm.covs, "gains": sm.gains},
        )

    def predict(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        return self.filter(observations, observation_columns=observation_columns).filtered_states

    def predict_proba(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        return self.filter(
            observations, observation_columns=observation_columns
        ).filtered_probabilities

    def forecast(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        horizon: int | None = None,
    ) -> ForecastResult:
        self._require_fitted()
        assert self.system is not None
        # ensure filtered state is current
        self.filter(observations, observation_columns=observation_columns)
        assert self._trace is not None
        h = int(horizon if horizon is not None else self._kf_settings.forecasting.default_horizon)
        means, covs = n_step_predict(
            self._trace.means[-1],
            self._trace.covs[-1],
            self.system.f,
            self.system.q,
            horizon=h,
            b=self.system.b,
        )
        step_proba = _soft_trend_proba(means, covs)
        terminal = step_proba[-1]
        level = self._kf_settings.forecasting.confidence_level
        lo, hi = prediction_intervals(means[-1], covs[-1], level=level)
        y_hat, s = forecast_observation(means[-1], covs[-1], self.system.h, self.system.r)
        return ForecastResult(
            horizon=h,
            expected_state=int(np.argmax(terminal)),
            probability_distribution=terminal,
            confidence_interval=(float(lo[0]) if lo.size else 0.0, float(hi[0]) if hi.size else 0.0),
            expected_duration={0: float(h) * float(terminal[0]), 1: float(h) * float(terminal[1])},
            step_distributions=step_proba,
            state_names=self._state_names,
            metadata={
                "state_means": means,
                "state_covs": covs,
                "obs_mean": y_hat,
                "obs_cov": s,
                "intervals": (lo, hi),
            },
        )

    def sample(
        self,
        n_steps: int,
        *,
        initial_state: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        assert self.system is not None
        states, obs = simulate_lds(self.system, n_steps, rng=rng or self._rng)
        # discrete soft labels from latent primary state
        std = np.sqrt(np.clip(np.diag(self.system.p0)[0], 1e-6, None))
        proba = _soft_trend_proba(states, np.stack([self.system.p0] * n_steps))
        labels = np.argmax(proba, axis=1).astype(np.int64)
        if initial_state is not None and labels.size:
            labels[0] = int(initial_state)
        _ = std
        return labels, obs

    def log_likelihood(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        return float(
            self.filter(observations, observation_columns=observation_columns).log_likelihood
        )

    def score(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        return self.log_likelihood(observations, observation_columns=observation_columns)

    def aic(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        ll = self.log_likelihood(observations, observation_columns=observation_columns)
        return aic_score(-ll, self._n_params())

    def bic(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        y = self._extract_obs(observations, observation_columns)
        ll = self.log_likelihood(observations, observation_columns=observation_columns)
        return bic_score(-ll, self._n_params(), y.shape[0])

    # --- Kalman-specific API ---

    def state(self) -> np.ndarray:
        self._require_fitted()
        if self._online_x is not None:
            return self._online_x.copy()
        assert self._trace is not None
        return self._trace.means[-1].copy()

    def covariance(self) -> np.ndarray:
        self._require_fitted()
        if self._online_p is not None:
            return self._online_p.copy()
        assert self._trace is not None
        return self._trace.covs[-1].copy()

    def innovation(self) -> np.ndarray:
        self._require_fitted()
        if self._last_innov is not None:
            return self._last_innov.copy()
        assert self._trace is not None
        return self._trace.innovations[-1].copy()

    def kalman_gain(self) -> np.ndarray:
        self._require_fitted()
        if self._last_gain is not None:
            return self._last_gain.copy()
        assert self._trace is not None
        return self._trace.gains[-1].copy()

    def filtered_means(self) -> np.ndarray:
        self._require_fitted()
        assert self._trace is not None
        return self._trace.means.copy()

    def smoothed_means(self) -> np.ndarray:
        self._require_fitted()
        if self._smooth is None and self._trace is not None and self.system is not None:
            self._smooth = rts_smooth(self._trace, self.system)
        assert self._smooth is not None
        return self._smooth.means.copy()

    def evaluate(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        true_states: np.ndarray | None = None,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self.system is not None
        y = self._extract_obs(observations, observation_columns)
        y = self._maybe_dynamic_beta_obs(y)
        filt = self.filter(observations, observation_columns=observation_columns)
        assert self._trace is not None
        sm = rts_smooth(self._trace, self.system) if self.system else None
        self._smooth = sm
        _ = filt
        return KalmanEvaluator().evaluate(
            observations=y,
            trace=self._trace,
            smooth=sm,
            true_states=true_states,
            n_params=self._n_params(),
        )

    def diagnostics(
        self,
        observations: pl.DataFrame | np.ndarray | None = None,
        *,
        observation_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        assert self.system is not None
        if observations is not None:
            self.filter(observations, observation_columns=observation_columns)
        elif self._trace is None and self._train_obs is not None:
            self.filter(self._train_obs)
        if self._trace is None:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No observations for diagnostics", code="KF_NO_OBS")
        if self._smooth is None:
            self._smooth = rts_smooth(self._trace, self.system)
        return KalmanDiagnostics().report(
            self.system, self._trace, smooth=self._smooth, history=self._history
        )

    def save(self, path: Path | str) -> Path:
        return KalmanSerializer().save(self, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> KalmanFilterModel:
        return KalmanSerializer().load(Path(path), model_cls=cls)

    def _ingest(
        self,
        system: LinearGaussianSSM,
        trace: FilterTrace,
        y: np.ndarray,
        history: list[float],
        n_iter: int,
        converged: bool,
    ) -> None:
        self.system = system
        self._trace = trace
        self._smooth = None
        self._history = list(history)
        self._n_iter = int(n_iter)
        self._converged = bool(converged)
        self._train_obs = y
        self._n_latent = system.n_states
        self._n_obs = system.n_obs
        self._fitted = True
        self.meta = StateSpaceModelMeta(
            name="kalman",
            version="1.0.0",
            description=self.meta.description,
            n_states=2,
            algorithm_family="kalman",
            parameters={
                "n_latent": self._n_latent,
                "n_obs": self._n_obs,
                "filter_type": self._kf_settings.filter_type,
                "application": system.application,
                "n_params": self._n_params(),
            },
            state_names=self._state_names,
        )

    def _n_params(self) -> int:
        if self.system is None:
            return self._n_latent * self._n_latent + self._n_obs
        n, m = self.system.n_states, self.system.n_obs
        # count free entries in Q, R (diagonal-ish) + F + H
        return int(n * n + m * n + n * (n + 1) // 2 + m * (m + 1) // 2)

    def _maybe_dynamic_beta_obs(self, y: np.ndarray) -> np.ndarray:
        if self.system is not None and self.system.application == "dynamic_beta" and y.shape[1] >= 2:
            return y[:, 0:1]
        return y

    def _build_h_seq(
        self, y: np.ndarray, observations: pl.DataFrame | np.ndarray
    ) -> np.ndarray | None:
        app = self._kf_settings.application
        if app != "dynamic_beta":
            return None
        if y.shape[1] >= 2:
            mkt = y[:, 1]
            return np.stack([np.ones(y.shape[0]), mkt], axis=1).reshape(y.shape[0], 1, 2)
        return None

    def _extract_controls(self, observations: pl.DataFrame | np.ndarray) -> np.ndarray | None:
        if not isinstance(observations, pl.DataFrame):
            return None
        cols = self._kf_settings.columns.control_columns
        if not cols:
            return None
        present = [c for c in cols if c in observations.columns]
        if not present:
            return None
        return observations.select(present).to_numpy().astype(np.float64)

    def _extract_obs(
        self,
        observations: pl.DataFrame | np.ndarray,
        observation_columns: list[str] | None = None,
    ) -> np.ndarray:
        if isinstance(observations, np.ndarray):
            y = np.asarray(observations, dtype=np.float64)
            return y.reshape(-1, 1) if y.ndim == 1 else y
        cols = observation_columns
        if not cols:
            if self._kf_settings.columns.observation_columns:
                cols = list(self._kf_settings.columns.observation_columns)
            else:
                exclude = {
                    self._kf_settings.columns.timestamp,
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "symbol",
                    "exchange",
                    "timeframe",
                    "state_id",
                }
                cols = [
                    c
                    for c, dt in zip(observations.columns, observations.dtypes, strict=False)
                    if c not in exclude and dt.is_numeric()
                ]
                if not cols and "close" in observations.columns:
                    cols = ["close"]
        if not cols:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No observation columns", code="KF_NO_OBS_COLS")
        return observations.select(cols).to_numpy().astype(np.float64)

    def _algorithm_state(self) -> dict[str, Any]:
        assert self.system is not None
        algo: dict[str, Any] = {
            "system": self.system.to_dict(),
            "history": list(self._history),
            "n_iter": self._n_iter,
            "converged": self._converged,
            "state_names": list(self._state_names),
            "settings": self._kf_settings.model_dump(),
            "n_latent": self._n_latent,
            "n_obs": self._n_obs,
            "online_x": None if self._online_x is None else self._online_x.tolist(),
            "online_p": None if self._online_p is None else self._online_p.tolist(),
        }
        if self._trace is not None:
            algo.update(
                {
                    "means": self._trace.means.tolist(),
                    "covs": self._trace.covs.tolist(),
                    "pred_means": self._trace.pred_means.tolist(),
                    "pred_covs": self._trace.pred_covs.tolist(),
                    "innovations": self._trace.innovations.tolist(),
                    "innovation_covs": self._trace.innovation_covs.tolist(),
                    "gains": self._trace.gains.tolist(),
                    "log_likelihood": self._trace.log_likelihood,
                    "trace_metadata": dict(self._trace.metadata),
                }
            )
        if self._smooth is not None:
            algo["smooth_means"] = self._smooth.means.tolist()
            algo["smooth_covs"] = self._smooth.covs.tolist()
        return algo

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if "settings" in state:
            self._kf_settings = KalmanSettings.from_mapping(state["settings"])
        if "system" in state:
            self.system = LinearGaussianSSM.from_dict(state["system"])
            # restore nonlinear hooks for volatility template
            if self.system.application == "volatility":
                rebuilt = build_system(self._kf_settings, application="volatility")
                self.system = LinearGaussianSSM(
                    f=self.system.f,
                    h=self.system.h,
                    q=self.system.q,
                    r=self.system.r,
                    x0=self.system.x0,
                    p0=self.system.p0,
                    b=self.system.b,
                    application=self.system.application,
                    f_fn=rebuilt.f_fn,
                    h_fn=rebuilt.h_fn,
                    f_jac=rebuilt.f_jac,
                    h_jac=rebuilt.h_jac,
                    metadata=dict(self.system.metadata),
                )
        self._history = list(state.get("history") or [])
        self._n_iter = int(state.get("n_iter", 0))
        self._converged = bool(state.get("converged", False))
        self._n_latent = int(state.get("n_latent", self._n_latent))
        self._n_obs = int(state.get("n_obs", self._n_obs))
        names = state.get("state_names")
        if names:
            self._state_names = tuple(names)
        ox = state.get("online_x")
        op = state.get("online_p")
        self._online_x = None if ox is None else np.asarray(ox, dtype=np.float64)
        self._online_p = None if op is None else np.asarray(op, dtype=np.float64)
        if "means" in state:
            self._trace = FilterTrace(
                means=np.asarray(state["means"], dtype=np.float64),
                covs=np.asarray(state["covs"], dtype=np.float64),
                pred_means=np.asarray(state["pred_means"], dtype=np.float64),
                pred_covs=np.asarray(state["pred_covs"], dtype=np.float64),
                innovations=np.asarray(state["innovations"], dtype=np.float64),
                innovation_covs=np.asarray(state["innovation_covs"], dtype=np.float64),
                gains=np.asarray(state["gains"], dtype=np.float64),
                log_likelihood=float(state.get("log_likelihood", 0.0)),
                metadata=dict(state.get("trace_metadata") or {}),
            )
        if "smooth_means" in state:
            from iqrp.app.regimes.kalman.smoothing import SmoothTrace as ST

            self._smooth = ST(
                means=np.asarray(state["smooth_means"], dtype=np.float64),
                covs=np.asarray(state["smooth_covs"], dtype=np.float64),
                gains=np.zeros(
                    (
                        np.asarray(state["smooth_means"]).shape[0],
                        self._n_latent,
                        self._n_latent,
                    )
                ),
            )
        self.meta = StateSpaceModelMeta(
            name=self.meta.name,
            version=self.meta.version,
            description=self.meta.description,
            n_states=2,
            algorithm_family="kalman",
            parameters={
                "n_latent": self._n_latent,
                "n_obs": self._n_obs,
                "filter_type": self._kf_settings.filter_type,
            },
            state_names=self._state_names,
        )


def ensure_s(s: np.ndarray) -> np.ndarray:
    from iqrp.app.regimes.kalman.covariance import ensure_spd

    return ensure_spd(s)


@register_regime_model
class KalmanRegimeModel(RegimeModel):
    """RegimeModel adapter over :class:`KalmanFilterModel` (soft trend regimes)."""

    meta = RegimeModelMeta(
        name="kalman",
        version="1.0.0",
        description="Kalman soft trend / regime adapter",
        n_states=2,
        algorithm_family="kalman",
        parameters={},
        state_names=_SOFT_STATE_NAMES,
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        settings: KalmanSettings | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._engine = KalmanFilterModel(
            n_states=n_states, settings=settings, random_seed=random_seed
        )
        self.meta = RegimeModelMeta(
            name="kalman",
            version="1.0.0",
            description=self.meta.description,
            n_states=2,
            algorithm_family="kalman",
            parameters=dict(self._engine.meta.parameters),
            state_names=self._engine.state_names,
        )
        self._state_names = self._engine.state_names

    def fit(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> KalmanRegimeModel:
        self._engine.fit(frame, observation_columns=feature_columns)
        self._fitted = True
        self._transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
        self._state_names = self._engine.state_names
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        self._require_fitted()
        return self._engine.predict(frame, observation_columns=feature_columns)

    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        return self._engine.predict_proba(frame, observation_columns=feature_columns)

    def forecast(self, frame: pl.DataFrame, steps: int = 1) -> RegimeForecast:
        self._require_fitted()
        fc = self._engine.forecast(frame, horizon=steps)
        return RegimeForecast.from_probabilities(
            (
                fc.step_distributions
                if fc.step_distributions is not None
                else fc.probability_distribution
            ),
            state_names=self._state_names,
            expected_duration=fc.expected_duration,
        )

    def _algorithm_state(self) -> dict[str, Any]:
        return self._engine.export_state()

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._engine.import_state(state)
        self._fitted = self._engine.is_fitted
        if self._engine.is_fitted:
            self._transition_matrix = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
            self._state_names = self._engine.state_names

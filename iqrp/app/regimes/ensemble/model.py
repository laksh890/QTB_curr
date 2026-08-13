"""Institutional Ensemble Regime Intelligence Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.base.registry import register_regime_model
from iqrp.app.regimes.ensemble.calibration import Calibrator
from iqrp.app.regimes.ensemble.combiner import combine
from iqrp.app.regimes.ensemble.confidence import confidence_report, posterior_confidence
from iqrp.app.regimes.ensemble.config import EnsembleSettings
from iqrp.app.regimes.ensemble.diagnostics import EnsembleDiagnostics
from iqrp.app.regimes.ensemble.disagreement import consensus_score, disagreement_report
from iqrp.app.regimes.ensemble.evaluator import EnsembleEvaluator
from iqrp.app.regimes.ensemble.orchestrator import (
    collect_transition,
    fit_members,
    member_log_likelihoods,
    predict_members,
)
from iqrp.app.regimes.ensemble.registry import EnsembleMember, EnsembleRegistry
from iqrp.app.regimes.ensemble.serializer import EnsembleSerializer
from iqrp.app.regimes.ensemble.trainer import EnsembleTrainer
from iqrp.app.regimes.ensemble.weighting import adaptive_update, compute_weights, equal_weights
from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.registry import register_state_space_model
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.base.state_space_model import StateSpaceModel, StateSpaceModelMeta


def _default_names(n: int, names: tuple[str, ...] | None) -> tuple[str, ...]:
    if names and len(names) >= n:
        return tuple(names[:n])
    if names:
        return tuple(names) + tuple(f"regime_{i}" for i in range(len(names), n))
    return tuple(f"regime_{i}" for i in range(n))


@register_regime_model
class EnsembleRegimeModel(RegimeModel):
    """Unified probabilistic market-regime interface over discovered members."""

    meta = RegimeModelMeta(
        name="ensemble",
        version="1.0.0",
        description="Institutional Ensemble Regime Intelligence Engine",
        n_states=6,
        algorithm_family="ensemble",
        parameters={},
        state_names=(
            "bull",
            "bear",
            "sideways",
            "high_volatility",
            "low_volatility",
            "liquidity_stress",
        ),
    )

    def __init__(
        self,
        *,
        settings: EnsembleSettings | None = None,
        random_seed: int | None = None,
        members: list[EnsembleMember] | None = None,
    ) -> None:
        super().__init__()
        self._ens_settings = settings or EnsembleSettings.default()
        self._rng = np.random.default_rng(
            random_seed if random_seed is not None else self._ens_settings.random_seed
        )
        self._state_names = _default_names(
            self._ens_settings.n_states, self._ens_settings.state_names
        )
        self.members: list[EnsembleMember] = list(members or [])
        self._weights: np.ndarray = equal_weights(max(len(self.members), 1))
        self._calibrator = Calibrator(
            method=(
                self._ens_settings.calibration.method
                if self._ens_settings.calibration.enabled
                else "none"
            ),  # type: ignore[arg-type]
            temperature=self._ens_settings.calibration.temperature,
        )
        self._ensemble_proba: np.ndarray | None = None
        self._member_probas: list[np.ndarray] = []
        self._member_names: list[str] = []
        self._history: list[dict[str, Any]] = []
        self._train_frame: pl.DataFrame | None = None
        self._feature_columns: list[str] | None = None
        self._update_counter = 0
        self.meta = RegimeModelMeta(
            name="ensemble",
            version="1.0.0",
            description=self.meta.description,
            n_states=len(self._state_names),
            algorithm_family="ensemble",
            parameters={
                "combination": self._ens_settings.combination.method,
                "n_states": len(self._state_names),
            },
            state_names=self._state_names,
        )

    def fit(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> EnsembleRegimeModel:
        cols = feature_columns or self._resolve_features(frame)
        result = EnsembleTrainer(self._ens_settings).fit(
            frame, cols, members=self.members or None
        )
        self.members = result.members
        self._weights = result.weights
        self._calibrator = result.calibrator
        self._transition_matrix = result.transition
        self._ensemble_proba = result.ensemble_proba
        self._member_names = list(result.metadata.get("names") or [m.name for m in self.members])
        self._history = list(result.history)
        self._train_frame = frame
        self._feature_columns = cols
        self._fitted = True
        self._state_names = _default_names(self._ens_settings.n_states, self._ens_settings.state_names)
        return self

    def partial_fit(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> EnsembleRegimeModel:
        if not self._fitted or not self.members:
            return self.fit(frame, feature_columns)
        if not self._ens_settings.online.warm_start:
            return self.fit(frame, feature_columns)
        cols = feature_columns or self._feature_columns
        # incremental member updates when supported
        for m in self.members:
            if hasattr(m.model, "partial_fit"):
                try:
                    m.model.partial_fit(frame, cols)  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    m.model.fit(frame, cols)
            else:
                m.model.fit(frame, cols)
            m.metadata["fitted"] = True
        self._predict_cache(frame, cols)
        if self._ens_settings.online.weight_update and self._ensemble_proba is not None:
            # adaptive nudge using member max-proba as instant score
            scores = np.asarray(
                [float(np.mean(p.max(axis=1))) for p in self._member_probas], dtype=np.float64
            )
            self._weights = adaptive_update(
                self._weights,
                scores,
                rate=self._ens_settings.weighting.adaptive_rate,
                min_weight=self._ens_settings.weighting.min_weight,
            )
            for m, w in zip(self.members, self._weights, strict=False):
                m.weight = float(w)
        self._update_counter += 1
        every = int(self._ens_settings.online.recalibrate_every)
        if every > 0 and self._update_counter % every == 0 and self._ensemble_proba is not None:
            y = np.argmax(self._ensemble_proba, axis=1)
            self._calibrator.fit(self._ensemble_proba, y)
        if self._train_frame is None:
            self._train_frame = frame
        else:
            self._train_frame = pl.concat([self._train_frame, frame], how="vertical_relaxed")
        self._history.append({"weights": self.weights()})
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        proba = self.predict_proba(frame, feature_columns)
        return np.argmax(proba, axis=1).astype(np.int64)

    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        cols = feature_columns or self._feature_columns
        return self._predict_cache(frame, cols)

    def forecast(self, frame: pl.DataFrame, steps: int = 1) -> RegimeForecast:
        self._require_fitted()
        proba = self.predict_proba(frame)
        current = proba[-1]
        tm = self.transition_matrix()
        h = max(1, int(steps))
        step = np.empty((h, self.meta.n_states), dtype=np.float64)
        dist = current.copy()
        for t in range(h):
            dist = dist @ tm
            dist = dist / max(float(dist.sum()), 1e-300)
            step[t] = dist
        persist = {
            i: float(1.0 / max(1.0 - tm[i, i], 1e-12)) for i in range(self.meta.n_states)
        }
        return RegimeForecast.from_probabilities(
            step,
            state_names=self._state_names,
            expected_duration=persist,
        )

    def confidence(self, frame: pl.DataFrame | None = None) -> dict[str, Any]:
        self._require_fitted()
        if frame is not None:
            self.predict_proba(frame)
        assert self._ensemble_proba is not None
        return confidence_report(
            self._ensemble_proba,
            self._member_probas,
            transition=self._transition_matrix,
            level=self._ens_settings.forecasting.confidence_level,
        )

    def consensus(self, frame: pl.DataFrame | None = None) -> dict[str, Any]:
        self._require_fitted()
        if frame is not None:
            self.predict_proba(frame)
        return disagreement_report(self._member_probas, names=self._member_names)

    def leaderboard(
        self,
        frame: pl.DataFrame | None = None,
        *,
        true_states: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        self._require_fitted()
        if frame is not None:
            self.predict_proba(frame)
        assert self._ensemble_proba is not None
        hard = {n: np.argmax(p, axis=1) for n, p in zip(self._member_names, self._member_probas, strict=False)}
        proba = {n: p for n, p in zip(self._member_names, self._member_probas, strict=False)}
        ll = {
            n: float(np.sum(np.log(np.clip(p.max(axis=1), 1e-300, None))))
            for n, p in proba.items()
        }
        return EnsembleEvaluator().leaderboard(
            member_probas=proba,
            member_hards=hard,
            log_likes=ll,
            truth=true_states,
            ensemble_proba=self._ensemble_proba,
            ensemble_hard=np.argmax(self._ensemble_proba, axis=1),
        )

    def weights(self) -> dict[str, float]:
        self._require_fitted()
        names = self._member_names or [m.name for m in self.members]
        w = np.asarray(self._weights, dtype=np.float64).reshape(-1)
        if w.size != len(names):
            w = equal_weights(len(names))
        return {n: float(wi) for n, wi in zip(names, w, strict=False)}

    def calibrate(
        self,
        frame: pl.DataFrame,
        true_states: np.ndarray,
        feature_columns: list[str] | None = None,
    ) -> EnsembleRegimeModel:
        self._require_fitted()
        proba = self.predict_proba(frame, feature_columns)
        y = np.asarray(true_states, dtype=np.int64).reshape(-1)
        self._calibrator.fit(proba, y[: proba.shape[0]])
        self._ensemble_proba = self._calibrator.transform(proba)
        return self

    def diagnostics(
        self,
        frame: pl.DataFrame | None = None,
        *,
        true_states: np.ndarray | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        if frame is not None:
            self.predict_proba(frame)
        assert self._ensemble_proba is not None
        board = self.leaderboard(true_states=true_states)
        return EnsembleDiagnostics().report(
            members=self.members,
            weights=self._weights,
            ensemble_proba=self._ensemble_proba,
            member_probas=self._member_probas,
            names=self._member_names,
            history=self._history,
            truth=true_states,
            leaderboard=board,
        )

    def evaluate(
        self,
        frame: pl.DataFrame,
        *,
        true_states: np.ndarray | None = None,
        feature_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        self._require_fitted()
        proba = self.predict_proba(frame, feature_columns)
        hard = np.argmax(proba, axis=1)
        metrics = EnsembleEvaluator().evaluate_member(
            proba=proba,
            hard=hard,
            truth=true_states,
            log_likelihood=float(np.sum(np.log(np.clip(proba.max(axis=1), 1e-300, None)))),
        )
        return {"metrics": metrics, "leaderboard": self.leaderboard(frame, true_states=true_states)}

    def save(self, path: Path | str) -> Path:
        return EnsembleSerializer().save(self, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> EnsembleRegimeModel:
        return EnsembleSerializer().load(Path(path), model_cls=cls)

    def _predict_cache(
        self, frame: pl.DataFrame, feature_columns: list[str] | None
    ) -> np.ndarray:
        mapped, _hards, names = predict_members(
            self.members,
            frame,
            feature_columns,
            n_canonical=self.meta.n_states,
            parallel=True,
        )
        self._member_probas = mapped
        self._member_names = names
        w_map = {m.name: float(m.weight) for m in self.members}
        w = np.asarray([w_map.get(n, 1.0) for n in names], dtype=np.float64)
        w = w / max(float(w.sum()), 1e-300)
        self._weights = w
        ll = member_log_likelihoods(self.members, frame, feature_columns)
        log_ev = np.asarray([ll.get(n, 0.0) for n in names], dtype=np.float64)
        ens = combine(
            mapped,
            w,
            method=self._ens_settings.combination.method,  # type: ignore[arg-type]
            n_states=self.meta.n_states,
            log_evidence=log_ev,
            meta_weights=w,
            scores=w,
        )
        if self._calibrator.fitted:
            ens = self._calibrator.transform(ens)
        self._ensemble_proba = ens
        self._transition_matrix = collect_transition(self.members, self.meta.n_states)
        return ens

    def _resolve_features(self, frame: pl.DataFrame) -> list[str] | None:
        if self._ens_settings.columns.feature_columns:
            return list(self._ens_settings.columns.feature_columns)
        return None

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "settings": self._ens_settings.model_dump(),
            "state_names": list(self._state_names),
            "weights": self._weights.tolist(),
            "member_names": list(self._member_names),
            "member_states": [
                {
                    "name": m.name,
                    "weight": m.weight,
                    "state_map": None if m.state_map is None else m.state_map.tolist(),
                    "model_state": m.model.export_state() if m.model.is_fitted else None,
                    "metadata": dict(m.metadata),
                }
                for m in self.members
            ],
            "calibrator": self._calibrator.to_dict(),
            "transition": None if self._transition_matrix is None else self._transition_matrix.tolist(),
            "ensemble_proba": None if self._ensemble_proba is None else self._ensemble_proba.tolist(),
            "history": list(self._history),
            "feature_columns": self._feature_columns,
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if "settings" in state:
            self._ens_settings = EnsembleSettings.from_mapping(state["settings"])
        names = state.get("state_names")
        if names:
            self._state_names = tuple(names)
        self._weights = np.asarray(state.get("weights") or [1.0], dtype=np.float64)
        self._member_names = list(state.get("member_names") or [])
        self._calibrator = Calibrator.from_dict(state.get("calibrator") or {})
        tm = state.get("transition")
        self._transition_matrix = None if tm is None else np.asarray(tm, dtype=np.float64)
        ep = state.get("ensemble_proba")
        self._ensemble_proba = None if ep is None else np.asarray(ep, dtype=np.float64)
        self._history = list(state.get("history") or [])
        self._feature_columns = state.get("feature_columns")
        # rebuild members via discovery + restore model states
        registry = EnsembleRegistry(self._ens_settings)
        try:
            discovered = registry.create_members()
        except Exception:  # noqa: BLE001
            discovered = []
        by_name = {m.name: m for m in discovered}
        restored: list[EnsembleMember] = []
        for ms in state.get("member_states") or []:
            name = str(ms.get("name"))
            if name in by_name:
                m = by_name[name]
            else:
                continue
            m.weight = float(ms.get("weight", m.weight))
            if ms.get("state_map") is not None:
                m.state_map = np.asarray(ms["state_map"], dtype=np.float64)
            m.metadata = dict(ms.get("metadata") or {})
            payload = ms.get("model_state")
            if payload:
                try:
                    m.model.import_state(payload)
                    m.metadata["fitted"] = m.model.is_fitted
                except Exception:  # noqa: BLE001
                    m.metadata["fitted"] = False
            restored.append(m)
        self.members = restored or discovered
        self.meta = RegimeModelMeta(
            name="ensemble",
            version="1.0.0",
            description=self.meta.description,
            n_states=len(self._state_names),
            algorithm_family="ensemble",
            parameters={"n_members": len(self.members)},
            state_names=self._state_names,
        )


@register_state_space_model
class EnsembleStateSpaceModel(StateSpaceModel):
    """State-space adapter over :class:`EnsembleRegimeModel`."""

    meta = StateSpaceModelMeta(
        name="ensemble",
        version="1.0.0",
        description="Ensemble regime intelligence (state-space adapter)",
        n_states=6,
        algorithm_family="ensemble",
        parameters={},
        state_names=(
            "bull",
            "bear",
            "sideways",
            "high_volatility",
            "low_volatility",
            "liquidity_stress",
        ),
    )

    def __init__(
        self,
        *,
        settings: EnsembleSettings | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self._engine = EnsembleRegimeModel(settings=settings, random_seed=random_seed)
        self._state_names = self._engine._state_names
        self.meta = StateSpaceModelMeta(
            name="ensemble",
            version="1.0.0",
            description=self.meta.description,
            n_states=self._engine.meta.n_states,
            algorithm_family="ensemble",
            parameters=dict(self._engine.meta.parameters),
            state_names=self._state_names,
        )

    def fit(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> EnsembleStateSpaceModel:
        frame = _as_frame(observations, observation_columns)
        self._engine.fit(frame, observation_columns)
        self._fitted = True
        self._state_names = self._engine._state_names
        self.meta = StateSpaceModelMeta(
            name="ensemble",
            version="1.0.0",
            description=self.meta.description,
            n_states=self._engine.meta.n_states,
            algorithm_family="ensemble",
            parameters=dict(self._engine.meta.parameters),
            state_names=self._state_names,
        )
        return self

    def filter(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> FilterResult:
        frame = _as_frame(observations, observation_columns)
        proba = self._engine.predict_proba(frame, observation_columns)
        states = np.argmax(proba, axis=1).astype(np.int64)
        return FilterResult(
            filtered_states=states,
            filtered_probabilities=proba,
            log_likelihood=float(np.sum(np.log(np.clip(proba.max(axis=1), 1e-300, None)))),
            normalization_constants=np.ones(proba.shape[0]),
        )

    def smooth(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        filt = self.filter(observations, observation_columns=observation_columns)
        return SmootherResult(
            smoothed_states=filt.filtered_states,
            smoothed_probabilities=filt.filtered_probabilities,
            backward_messages=filt.filtered_probabilities,
            log_likelihood=filt.log_likelihood,
            metadata={"lag": lag},
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
        frame = _as_frame(observations, observation_columns)
        h = int(horizon if horizon is not None else self._engine._ens_settings.forecasting.default_horizon)
        fc = self._engine.forecast(frame, steps=h)
        probs = np.asarray(fc.probabilities, dtype=np.float64)
        if probs.ndim == 1:
            terminal = probs
            steps = None
        else:
            terminal = probs[-1]
            steps = probs
        return ForecastResult.from_probabilities(
            terminal,
            horizon=h,
            step_distributions=steps,
            state_names=self._state_names,
            expected_duration=fc.expected_duration,
            confidence_level=self._engine._ens_settings.forecasting.confidence_level,
        )

    def sample(
        self,
        n_steps: int,
        *,
        initial_state: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        gen = rng or np.random.default_rng()
        tm = np.asarray(self._engine.transition_matrix(), dtype=np.float64)
        tm = np.clip(tm, 0, None)
        tm = tm / np.clip(tm.sum(axis=1, keepdims=True), 1e-300, None)
        k = tm.shape[0]
        states = np.empty(n_steps, dtype=np.int64)
        states[0] = int(initial_state if initial_state is not None else gen.integers(0, k))
        for t in range(1, n_steps):
            states[t] = int(gen.choice(k, p=tm[states[t - 1]]))
        # dummy observations = state id noise
        obs = states.astype(np.float64).reshape(-1, 1) + gen.normal(0, 0.1, size=(n_steps, 1))
        return states, obs

    def log_likelihood(
        self,
        observations: pl.DataFrame | np.ndarray,
        *,
        observation_columns: list[str] | None = None,
    ) -> float:
        return float(
            self.filter(observations, observation_columns=observation_columns).log_likelihood
        )

    def _algorithm_state(self) -> dict[str, Any]:
        return self._engine.export_state()

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._engine.import_state(state)
        self._fitted = self._engine.is_fitted
        self._state_names = self._engine._state_names


def _as_frame(
    observations: pl.DataFrame | np.ndarray,
    observation_columns: list[str] | None,
) -> pl.DataFrame:
    if isinstance(observations, pl.DataFrame):
        return observations
    y = np.asarray(observations, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    cols = observation_columns or [f"f{i}" for i in range(y.shape[1])]
    return pl.DataFrame({c: y[:, i] for i, c in enumerate(cols)})

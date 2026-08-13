"""Mock regime model for tests and framework plumbing validation."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.regimes.base.probabilities import ProbabilityEngine
from iqrp.app.regimes.base.regime_model import RegimeModel, RegimeModelMeta
from iqrp.app.regimes.base.registry import register_regime_model
from iqrp.app.regimes.config import RegimeSettings


def _select_features(frame: pl.DataFrame, feature_columns: list[str] | None) -> list[str]:
    if feature_columns:
        return feature_columns
    settings = RegimeSettings.default()
    if settings.columns.feature_columns:
        return list(settings.columns.feature_columns)
    exclude = {
        settings.columns.timestamp,
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "exchange",
        "timeframe",
    }
    return [
        c
        for c, dt in zip(frame.columns, frame.dtypes, strict=False)
        if c not in exclude and dt.is_numeric()
    ]


@register_regime_model
class MockRegimeModel(RegimeModel):
    """Threshold / quantile mock detector (not a production algorithm).

    Splits a primary series into ``n_states`` regimes by rolling mean quantiles.
    Exists so the framework can be tested without committing to HMM/Markov yet.
    """

    meta = RegimeModelMeta(
        name="mock_regime",
        version="1.0.0",
        description="Mock quantile/threshold regime detector for framework validation",
        n_states=3,
        algorithm_family="mock",
        parameters={"n_states": 3, "window": 20, "primary_column": "close"},
        state_names=("bear", "sideways", "bull"),
    )

    def __init__(
        self,
        *,
        n_states: int | None = None,
        window: int | None = None,
        primary_column: str | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        params = dict(self.meta.parameters)
        if n_states is not None:
            params["n_states"] = n_states
        if window is not None:
            params["window"] = window
        if primary_column is not None:
            params["primary_column"] = primary_column
        self._params = params
        self._rng = np.random.default_rng(
            random_seed if random_seed is not None else RegimeSettings.default().random_seed
        )
        self._centers: np.ndarray | None = None
        self._edges: np.ndarray | None = None
        k = int(self._params["n_states"])
        self._state_names = self.meta.state_names[:k] or tuple(f"state_{i}" for i in range(k))
        # Rebuild meta with resolved params (immutable -> replace)
        self.meta = RegimeModelMeta(
            name=self.meta.name,
            version=self.meta.version,
            description=self.meta.description,
            n_states=k,
            algorithm_family=self.meta.algorithm_family,
            parameters=params,
            state_names=self._state_names,
        )

    def fit(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> MockRegimeModel:
        cols = _select_features(frame, feature_columns)
        primary = str(self._params.get("primary_column", "close"))
        if primary not in frame.columns:
            if not cols:
                from iqrp.app.core.exceptions import ValidationError

                raise ValidationError(
                    "No feature columns available for mock regime fit",
                    code="REGIME_NO_FEATURES",
                )
            primary = cols[0]
        window = int(self._params["window"])
        series = (
            frame.select(pl.col(primary).pct_change().rolling_mean(window).alias("s"))
            .to_series()
            .to_numpy()
        )
        finite = series[np.isfinite(series)]
        k = int(self._params["n_states"])
        if finite.size < k + 2:
            self._edges = np.linspace(-0.01, 0.01, k + 1)
        else:
            qs = np.linspace(0, 1, k + 1)
            self._edges = np.unique(np.quantile(finite, qs))
            if len(self._edges) < k + 1:
                self._edges = np.linspace(float(np.min(finite)), float(np.max(finite)), k + 1)
        # Soft centers for probability kernels
        self._centers = 0.5 * (self._edges[:-1] + self._edges[1:])
        # Estimate empirical transition matrix from hard labels on train
        hard = self._hard_from_series(series)
        self._transition_matrix = self._count_transitions(hard, k)
        self._fitted = True
        self._feature_columns = cols
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        self._require_fitted()
        series = self._series(frame, feature_columns)
        return self._hard_from_series(series)

    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        series = self._series(frame, feature_columns)
        assert self._centers is not None
        k = len(self._centers)
        proba = np.zeros((len(series), k), dtype=np.float64)
        for i, x in enumerate(series):
            if not np.isfinite(x):
                proba[i] = 1.0 / k
                continue
            dist = np.abs(self._centers - x)
            inv = 1.0 / (dist + 1e-6)
            proba[i] = inv / inv.sum()
        return ProbabilityEngine.normalize_rows(proba)

    def _series(self, frame: pl.DataFrame, feature_columns: list[str] | None) -> np.ndarray:
        primary = str(self._params.get("primary_column", "close"))
        cols = (
            feature_columns
            or getattr(self, "_feature_columns", None)
            or _select_features(frame, None)
        )
        if primary not in frame.columns:
            primary = cols[0] if cols else frame.columns[0]
        window = int(self._params["window"])
        return (
            frame.select(pl.col(primary).pct_change().rolling_mean(window).alias("s"))
            .to_series()
            .to_numpy()
        )

    def _hard_from_series(self, series: np.ndarray) -> np.ndarray:
        assert self._edges is not None
        # digitize into 0..k-1
        idx = np.digitize(series, self._edges[1:-1], right=True)
        idx = np.clip(idx, 0, int(self._params["n_states"]) - 1)
        idx = idx.astype(np.int64)
        idx[~np.isfinite(series)] = int(self._params["n_states"]) // 2
        return np.asarray(idx, dtype=np.int64)

    @staticmethod
    def _count_transitions(states: np.ndarray, k: int) -> np.ndarray:
        tm = np.ones((k, k), dtype=np.float64)  # Dirichlet prior
        for a, b in pairwise(states):
            if 0 <= int(a) < k and 0 <= int(b) < k:
                tm[int(a), int(b)] += 1.0
        return ProbabilityEngine.normalize_rows(tm)

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "params": dict(self._params),
            "edges": None if self._edges is None else self._edges.tolist(),
            "centers": None if self._centers is None else self._centers.tolist(),
            "feature_columns": list(getattr(self, "_feature_columns", [])),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._params = dict(state.get("params") or self._params)
        edges = state.get("edges")
        centers = state.get("centers")
        self._edges = None if edges is None else np.asarray(edges, dtype=np.float64)
        self._centers = None if centers is None else np.asarray(centers, dtype=np.float64)
        self._feature_columns = list(state.get("feature_columns") or [])
        k = int(self._params.get("n_states", self.meta.n_states))
        self._state_names = self.meta.state_names[:k] or tuple(f"state_{i}" for i in range(k))

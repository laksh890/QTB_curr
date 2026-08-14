"""AutoML / hyperparameter optimization for forecast models."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.intelligence.benchmark import benchmark_model
from iqrp.app.forecasting.intelligence.config import IntelligenceSettings


def optimize_model(
    name: str,
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    settings: IntelligenceSettings,
    search_space: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    cfg = settings.automl
    if cfg.method == "none":
        return {}
    space = search_space or _default_space(name)
    if cfg.method == "grid":
        return _grid_search(name, frame, feature_columns, target_column, settings, space)
    if cfg.method in {"random", "hyperband", "successive_halving", "pbt"}:
        return _random_or_bandit(
            name, frame, feature_columns, target_column, settings, space, method=cfg.method
        )
    if cfg.method in {"bayesian", "optuna"}:
        return _optuna_search(name, frame, feature_columns, target_column, settings, space)
    return _random_or_bandit(
        name, frame, feature_columns, target_column, settings, space, method="random"
    )


def _default_space(name: str) -> dict[str, list[Any]]:
    # generic kwargs accepted by mock / many models via settings or kwargs
    return {
        "drift": [0.0, 0.01, -0.01],
    }


def _expand_grid(space: dict[str, list[Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{}]
    for k, vals in space.items():
        nxt = []
        for base in out:
            for v in vals:
                nxt.append({**base, k: v})
        out = nxt
        if len(out) >= limit:
            break
    return out[:limit]


def _eval(
    name: str,
    frame: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    settings: IntelligenceSettings,
    params: dict[str, Any],
) -> float:
    try:
        res = benchmark_model(
            name,
            frame,
            feature_columns=feature_columns,
            target_column=target_column,
            settings=settings,
            model_kwargs=params,
        )
        return float(res.metrics.get("rmse", 1e6))
    except Exception:
        return 1e6


def _grid_search(
    name: str,
    frame: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    settings: IntelligenceSettings,
    space: dict[str, list[Any]],
) -> dict[str, Any]:
    cands = _expand_grid(space, settings.automl.n_trials)
    best_score, best = float("inf"), {}
    for params in cands:
        score = _eval(name, frame, feature_columns, target_column, settings, params)
        if score < best_score:
            best_score, best = score, params
    return best


def _random_or_bandit(
    name: str,
    frame: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    settings: IntelligenceSettings,
    space: dict[str, list[Any]],
    *,
    method: str,
) -> dict[str, Any]:
    rng = np.random.default_rng(settings.seed)
    n_trials = max(int(settings.automl.n_trials), 1)
    # successive halving / hyperband: evaluate all, keep top half repeatedly
    pool = []
    for _ in range(n_trials):
        params = {k: vals[int(rng.integers(0, len(vals)))] for k, vals in space.items()}
        pool.append(params)
    if method in {"successive_halving", "hyperband", "pbt"}:
        survivors = pool
        while len(survivors) > 1:
            scored = [
                (p, _eval(name, frame, feature_columns, target_column, settings, p))
                for p in survivors
            ]
            scored.sort(key=lambda x: x[1])
            keep = max(len(scored) // 2, 1)
            survivors = [p for p, _ in scored[:keep]]
            if method == "pbt" and len(survivors) > 1:
                # perturb best
                best = dict(survivors[0])
                for k, vals in space.items():
                    if rng.random() < 0.3:
                        best[k] = vals[int(rng.integers(0, len(vals)))]
                survivors[0] = best
        return survivors[0]
    best_score, best = float("inf"), {}
    for params in pool:
        score = _eval(name, frame, feature_columns, target_column, settings, params)
        if score < best_score:
            best_score, best = score, params
    return best


def _optuna_search(
    name: str,
    frame: pl.DataFrame,
    feature_columns: list[str],
    target_column: str,
    settings: IntelligenceSettings,
    space: dict[str, list[Any]],
) -> dict[str, Any]:
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float | list[float]:
            params = {}
            for k, vals in space.items():
                params[k] = trial.suggest_categorical(k, list(vals))
            score = _eval(name, frame, feature_columns, target_column, settings, params)
            if settings.automl.multi_objective:
                # second objective: latency proxy via |drift|
                return [score, abs(float(params.get("drift", 0.0)))]
            return score

        if settings.automl.multi_objective:
            study = optuna.create_study(directions=["minimize", "minimize"])
        else:
            study = optuna.create_study(direction="minimize")
        study.optimize(
            objective, n_trials=max(int(settings.automl.n_trials), 1), show_progress_bar=False
        )
        if settings.automl.multi_objective:
            return dict(study.best_trials[0].params) if study.best_trials else {}
        return dict(study.best_params)
    except Exception:
        return _random_or_bandit(
            name, frame, feature_columns, target_column, settings, space, method="random"
        )

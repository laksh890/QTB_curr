"""Hyperparameter optimization: grid, random, Bayesian / Optuna."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

import numpy as np

from iqrp.app.forecasting.tree_models.base.backends import create_estimator, estimator_predict
from iqrp.app.forecasting.tree_models.optimization.cv import make_time_splits

Method = Literal["none", "grid", "random", "bayesian", "optuna"]


def _search_space(base: dict[str, Any]) -> dict[str, list[Any]]:
    return {
        "max_depth": [
            max(2, int(base.get("max_depth", 4)) - 1),
            int(base.get("max_depth", 4)),
            int(base.get("max_depth", 4)) + 1,
        ],
        "learning_rate": [0.03, 0.1, 0.2],
        "n_estimators": [
            max(20, int(base.get("n_estimators", 100)) // 2),
            int(base.get("n_estimators", 100)),
        ],
        "subsample": [0.7, 0.9, 1.0],
        "reg_lambda": [0.1, 1.0, 5.0],
    }


def _score_params(
    backend: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    params: dict[str, Any],
    validation: Any,
) -> float:
    splits = make_time_splits(X.shape[0], validation)
    if not splits:
        # holdout
        n = X.shape[0]
        tr, te = np.arange(int(n * 0.7)), np.arange(int(n * 0.7), n)
        splits = [(tr, te)]
    rmses = []
    for tr, te in splits[:3]:
        try:
            est = create_estimator(backend, task=task, params=params)  # type: ignore[arg-type]
            est.fit(X[tr], y[tr])
            pred = estimator_predict(est, X[te])
            rmses.append(float(np.sqrt(np.mean((y[te] - pred) ** 2))))
        except Exception:
            rmses.append(1e6)
    return float(np.mean(rmses)) if rmses else 1e6


def optimize_hyperparameters(
    backend: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    base_params: dict[str, Any],
    method: Method = "random",
    n_trials: int = 20,
    validation: Any = None,
    pruning: bool = True,
    parallel: bool = True,
) -> tuple[dict[str, Any], list[float]]:
    if method == "none":
        return dict(base_params), []
    if validation is None:
        from iqrp.app.forecasting.tree_models.config import ValidationConfig

        validation = ValidationConfig()
    space = _search_space(base_params)
    if method == "optuna" or method == "bayesian":
        return _optuna_or_bayes(
            backend,
            X,
            y,
            task=task,
            base_params=base_params,
            space=space,
            n_trials=n_trials,
            validation=validation,
            pruning=pruning,
        )
    candidates = _expand_candidates(base_params, space, method=method, n_trials=n_trials)
    scores: list[float] = []
    results: list[tuple[float, dict[str, Any]]] = []

    def _one(p: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        s = _score_params(backend, X, y, task=task, params=p, validation=validation)
        return s, p

    if parallel and len(candidates) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as pool:
            futs = [pool.submit(_one, c) for c in candidates]
            for fut in as_completed(futs):
                try:
                    s, p = fut.result()
                    results.append((s, p))
                    scores.append(s)
                    if pruning and len(scores) >= 3 and s > np.median(scores) * 2:
                        continue
                except Exception:
                    continue
    else:
        for c in candidates:
            s, p = _one(c)
            results.append((s, p))
            scores.append(s)
    if not results:
        return dict(base_params), scores
    results.sort(key=lambda t: t[0])
    return dict(results[0][1]), scores


def _expand_candidates(
    base: dict[str, Any],
    space: dict[str, list[Any]],
    *,
    method: str,
    n_trials: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(base.get("random_state", 42)))
    if method == "grid":
        keys = list(space.keys())
        grids = [space[k] for k in keys]
        # limited cartesian
        out = [dict(base)]
        for i, k in enumerate(keys):
            for v in grids[i]:
                p = dict(base)
                p[k] = v
                out.append(p)
                if len(out) >= n_trials:
                    return out
        return out[:n_trials]
    # random
    out = []
    for _ in range(max(int(n_trials), 1)):
        p = dict(base)
        for k, vals in space.items():
            p[k] = vals[int(rng.integers(0, len(vals)))]
        out.append(p)
    return out


def _optuna_or_bayes(
    backend: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str,
    base_params: dict[str, Any],
    space: dict[str, list[Any]],
    n_trials: int,
    validation: Any,
    pruning: bool,
) -> tuple[dict[str, Any], list[float]]:
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        scores: list[float] = []

        def objective(trial: Any) -> float:
            p = dict(base_params)
            p["max_depth"] = trial.suggest_int("max_depth", 2, 8)
            p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            p["n_estimators"] = trial.suggest_int("n_estimators", 20, 200)
            p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
            p["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
            s = _score_params(backend, X, y, task=task, params=p, validation=validation)
            scores.append(s)
            if pruning:
                trial.report(s, step=len(scores))
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return s

        pruner = optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=max(int(n_trials), 1), show_progress_bar=False)
        best = dict(base_params)
        best.update(study.best_params)
        return best, scores
    except Exception:
        # simple TPE-like random search fallback
        return optimize_hyperparameters(
            backend,
            X,
            y,
            task=task,
            base_params=base_params,
            method="random",
            n_trials=n_trials,
            validation=validation,
            pruning=pruning,
            parallel=False,
        )

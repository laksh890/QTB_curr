"""Hyperparameter optimization for neural models."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.forecasting.neural.base.data import train_val_split
from iqrp.app.forecasting.neural.base.trainer import NeuralTrainer
from iqrp.app.forecasting.neural.config import NeuralSettings


def optimize_neural(
    model: Any,
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    *,
    settings: NeuralSettings,
) -> dict[str, Any]:
    method = settings.optimization.method
    if method == "none":
        return {}
    space_grid = {
        "hidden_size": [32, 64],
        "num_layers": [1, 2],
        "dropout": [0.0, 0.1],
        "learning_rate": [1e-3, 3e-4],
    }
    if method == "optuna" or method == "bayesian":
        return _optuna_search(model, X_seq, y_seq, settings, space_grid)
    candidates = _expand(space_grid, method=method, n_trials=settings.optimization.n_trials, seed=settings.train.seed)
    best_score, best = float("inf"), {}
    for cand in candidates:
        score = _eval_candidate(model, X_seq, y_seq, settings, cand)
        if score < best_score:
            best_score, best = score, cand
    return best


def _expand(space: dict[str, list[Any]], *, method: str, n_trials: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if method == "grid":
        out = [{}]
        for k, vals in space.items():
            nxt = []
            for base in out:
                for v in vals:
                    nxt.append({**base, k: v})
            out = nxt
            if len(out) >= n_trials:
                break
        return out[:n_trials]
    out = []
    for _ in range(max(int(n_trials), 1)):
        out.append({k: vals[int(rng.integers(0, len(vals)))] for k, vals in space.items()})
    return out


def _eval_candidate(
    model: Any, X_seq: np.ndarray, y_seq: np.ndarray, settings: NeuralSettings, cand: dict[str, Any]
) -> float:
    X_tr, y_tr, X_va, y_va = train_val_split(X_seq, y_seq, val_ratio=0.25)
    # shrink epochs for HPO
    s = NeuralSettings.from_mapping(
        {
            **settings.model_dump(),
            "train": {**settings.train.model_dump(), "epochs": min(5, settings.train.epochs), "learning_rate": cand.get("learning_rate", settings.train.learning_rate)},
            "architecture": {
                **settings.architecture.model_dump(),
                "hidden_size": cand.get("hidden_size", settings.architecture.hidden_size),
                "num_layers": cand.get("num_layers", settings.architecture.num_layers),
                "dropout": cand.get("dropout", settings.architecture.dropout),
            },
            "optimization": {"method": "none"},
        }
    )
    try:
        # build fresh module via model factory hooks
        tmp = type(model)(settings=s)
        tmp._lookback = model._lookback
        tmp._horizon = model._horizon
        mod = tmp._build_module(n_features=X_seq.shape[-1], task=s.task.type)
        trainer = NeuralTrainer(s)
        mod, _ = trainer.fit(mod, X_tr, y_tr, X_va, y_va)
        return trainer.evaluate_loss(mod, X_va if X_va.size else X_tr, y_va if y_va.size else y_tr)
    except Exception:  # noqa: BLE001  # pragma: no cover
        return 1e6


def _optuna_search(
    model: Any, X_seq: np.ndarray, y_seq: np.ndarray, settings: NeuralSettings, space: dict[str, list[Any]]
) -> dict[str, Any]:
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: Any) -> float:
            cand = {
                "hidden_size": trial.suggest_categorical("hidden_size", space["hidden_size"]),
                "num_layers": trial.suggest_categorical("num_layers", space["num_layers"]),
                "dropout": trial.suggest_categorical("dropout", space["dropout"]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            }
            return _eval_candidate(model, X_seq, y_seq, settings, cand)

        pruner = optuna.pruners.MedianPruner() if settings.optimization.pruning else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(objective, n_trials=max(int(settings.optimization.n_trials), 1), show_progress_bar=False)
        return dict(study.best_params)
    except Exception:  # noqa: BLE001  # pragma: no cover
        return optimize_neural(model, X_seq, y_seq, settings=NeuralSettings.from_mapping({**settings.model_dump(), "optimization": {"method": "random", "n_trials": settings.optimization.n_trials}}))

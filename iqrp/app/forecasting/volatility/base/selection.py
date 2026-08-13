"""Automatic volatility model selection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import polars as pl

Criterion = Literal["aic", "bic", "loglik", "qlike"]


@dataclass(slots=True)
class VolCandidate:
    name: str
    scores: dict[str, float]
    params: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scores": dict(self.scores),
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class VolSelectionResult:
    best: str
    criterion: str
    leaderboard: list[VolCandidate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": self.best,
            "criterion": self.criterion,
            "leaderboard": [c.to_dict() for c in self.leaderboard],
        }


def select_volatility_models(
    frame: pl.DataFrame,
    *,
    candidates: list[str] | None = None,
    target_column: str = "returns",
    criterion: Criterion = "aic",
    settings: Any = None,
    parallel: bool = True,
) -> VolSelectionResult:
    from iqrp.app.forecasting.volatility import registry as _vol_registry

    names = candidates or ["ewma", "arch", "garch", "gjr_garch", "egarch"]
    board: list[VolCandidate] = []

    def _one(name: str) -> VolCandidate:
        model = _vol_registry.create_volatility_model(name, settings=settings)
        model.fit(frame, target_column=target_column)
        ic = model.information_criteria
        from iqrp.app.forecasting.volatility.evaluation.metrics import qlike

        var = model.conditional_variance()
        r = frame[target_column].to_numpy().astype(np.float64)
        scores = {
            "aic": ic.get("aic", np.inf),
            "bic": ic.get("bic", np.inf),
            "loglik": -ic.get("loglik", -np.inf),  # lower better for sorting unification
            "qlike": qlike(r**2, var),
        }
        # for loglik criterion we want maximize → store negative for min-sort
        return VolCandidate(name=name, scores=scores, params=model.params)

    if parallel and len(names) > 1:
        with ThreadPoolExecutor(max_workers=min(8, len(names))) as pool:
            futs = [pool.submit(_one, n) for n in names]
            for fut in as_completed(futs):
                try:
                    board.append(fut.result())
                except Exception:  # noqa: BLE001
                    continue
    else:
        for n in names:
            try:
                board.append(_one(n))
            except Exception:  # noqa: BLE001
                continue
    key = criterion
    reverse = False
    if key == "loglik":
        # scores store -loglik for min sort; convert display later
        board.sort(key=lambda c: c.scores.get("loglik", np.inf))
    else:
        board.sort(key=lambda c: c.scores.get(key, np.inf))
    best = board[0].name if board else "garch"
    return VolSelectionResult(best=best, criterion=criterion, leaderboard=board)


def rolling_vol_validation(
    returns: np.ndarray,
    fit_forecast_var_fn: Any,
    *,
    train_size: int,
    horizon: int = 1,
    step: int = 5,
) -> dict[str, float]:
    from iqrp.app.forecasting.volatility.evaluation.metrics import qlike, rmse

    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    errs = []
    qlikes = []
    start = 0
    h = max(int(horizon), 1)
    while start + train_size + h <= r.size:
        train = r[start : start + train_size]
        actual = r[start + train_size : start + train_size + h] ** 2
        pred = np.asarray(fit_forecast_var_fn(train, h), dtype=np.float64).reshape(-1)[:h]
        if pred.size < h:
            pred = np.pad(pred, (0, h - pred.size), constant_values=pred[-1] if pred.size else 1e-4)
        errs.extend((actual - pred).tolist())
        qlikes.append(qlike(actual, pred))
        start += step
    if not errs:
        return {"rmse": float("nan"), "qlike": float("nan"), "n": 0}
    e = np.asarray(errs)
    return {"rmse": float(np.sqrt(np.mean(e**2))), "qlike": float(np.mean(qlikes)), "n": int(e.size)}

"""Automatic model order selection via information criteria and CV."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from iqrp.app.forecasting.statistical.base.fitting import (
    fit_ar_ols,
    fit_arma_css,
    fit_var_ols,
    information_criteria,
)

Criterion = Literal["aic", "aicc", "bic", "hqic"]


@dataclass(slots=True)
class CandidateScore:
    order: dict[str, int]
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": dict(self.order),
            "scores": dict(self.scores),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SelectionResult:
    best_order: dict[str, int]
    criterion: str
    leaderboard: list[CandidateScore]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_order": dict(self.best_order),
            "criterion": self.criterion,
            "leaderboard": [c.to_dict() for c in self.leaderboard],
            "metadata": dict(self.metadata),
        }


def select_ar_order(
    y: np.ndarray,
    *,
    max_p: int = 5,
    criterion: Criterion = "aic",
    intercept: bool = True,
) -> SelectionResult:
    board: list[CandidateScore] = []
    for p in range(0, max_p + 1):
        fit = fit_ar_ols(y, p, intercept=intercept)
        ic = information_criteria(fit.loglik, fit.k_params, fit.nobs)
        board.append(CandidateScore(order={"p": p}, scores=ic))
    board.sort(key=lambda c: c.scores.get(criterion, np.inf))
    best = board[0].order if board else {"p": 1}
    return SelectionResult(best_order=best, criterion=criterion, leaderboard=board)


def select_arma_order(
    y: np.ndarray,
    *,
    max_p: int = 3,
    max_q: int = 3,
    criterion: Criterion = "aic",
    intercept: bool = True,
    parallel: bool = True,
) -> SelectionResult:
    candidates = [(p, q) for p in range(0, max_p + 1) for q in range(0, max_q + 1)]

    def _one(pq: tuple[int, int]) -> CandidateScore:
        p, q = pq
        fit = fit_arma_css(y, p, q, intercept=intercept)
        ic = information_criteria(fit.loglik, fit.k_params, fit.nobs)
        return CandidateScore(order={"p": p, "q": q}, scores=ic)

    board: list[CandidateScore] = []
    if parallel and len(candidates) > 2:
        with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as pool:
            futs = [pool.submit(_one, c) for c in candidates]
            for fut in as_completed(futs):
                board.append(fut.result())
    else:
        board = [_one(c) for c in candidates]
    board.sort(key=lambda c: c.scores.get(criterion, np.inf))
    best = board[0].order if board else {"p": 1, "q": 0}
    return SelectionResult(best_order=best, criterion=criterion, leaderboard=board)


def select_arima_order(
    y: np.ndarray,
    *,
    max_p: int = 3,
    max_d: int = 2,
    max_q: int = 3,
    d: int | None = None,
    criterion: Criterion = "aic",
    differencer: Callable[[np.ndarray, int], np.ndarray] | None = None,
    parallel: bool = True,
) -> SelectionResult:
    from iqrp.app.forecasting.statistical.base.stationarity import difference, suggest_differencing

    d_star = int(d if d is not None else suggest_differencing(y, max_d=max_d))
    diff_fn = differencer or (lambda arr, order: difference(arr, order=order))
    z = diff_fn(y, d_star) if d_star else np.asarray(y, dtype=np.float64)
    arma = select_arma_order(z, max_p=max_p, max_q=max_q, criterion=criterion, parallel=parallel)
    best = {**arma.best_order, "d": d_star}
    for c in arma.leaderboard:
        c.order = {**c.order, "d": d_star}
    return SelectionResult(
        best_order=best,
        criterion=criterion,
        leaderboard=arma.leaderboard,
        metadata={"d": d_star},
    )


def select_var_lags(
    Y: np.ndarray,
    *,
    max_lags: int = 5,
    criterion: Criterion = "aic",
) -> SelectionResult:
    board: list[CandidateScore] = []
    for p in range(1, max_lags + 1):
        fit = fit_var_ols(Y, p)
        ic = information_criteria(fit["loglik"], fit["k_params"], fit["nobs"])
        board.append(CandidateScore(order={"p": p}, scores=ic))
    board.sort(key=lambda c: c.scores.get(criterion, np.inf))
    best = board[0].order if board else {"p": 1}
    return SelectionResult(best_order=best, criterion=criterion, leaderboard=board)


def rolling_validation_score(
    y: np.ndarray,
    fit_forecast_fn: Callable[[np.ndarray, int], np.ndarray],
    *,
    train_size: int,
    horizon: int = 1,
    step: int = 1,
) -> dict[str, float]:
    """Walk-forward RMSE / MAE for a fit→forecast callable."""
    x = np.asarray(y, dtype=np.float64).reshape(-1)
    errors: list[float] = []
    abs_errors: list[float] = []
    start = 0
    h = max(int(horizon), 1)
    while start + train_size + h <= x.size:
        train = x[start : start + train_size]
        actual = x[start + train_size : start + train_size + h]
        pred = np.asarray(fit_forecast_fn(train, h), dtype=np.float64).reshape(-1)[:h]
        if pred.size < h:
            pred = np.pad(pred, (0, h - pred.size), constant_values=pred[-1] if pred.size else 0.0)
        err = actual - pred
        errors.extend(err.tolist())
        abs_errors.extend(np.abs(err).tolist())
        start += step
    if not errors:
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0}
    e = np.asarray(errors, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "n": int(e.size),
    }

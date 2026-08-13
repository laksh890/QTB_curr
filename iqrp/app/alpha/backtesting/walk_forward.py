"""Walk-forward splits and backtests (strictly causal in time).

Look-ahead prevention
---------------------
Train indices always precede test indices with an optional ``gap`` (embargo)
between them. Expanding/rolling windows never include future observations in
the training set.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from iqrp.app.alpha.backtesting.signal_backtest import signal_backtest


def walk_forward_splits(
    n: int,
    *,
    train_size: int,
    test_size: int,
    gap: int = 0,
    expanding: bool = False,
    step: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` chronologically with optional embargo gap."""
    n = int(n)
    tr = max(int(train_size), 1)
    te = max(int(test_size), 1)
    g = max(int(gap), 0)
    step_size = int(step) if step is not None else te
    start = 0
    while True:
        if expanding:
            train = np.arange(0, start + tr)
            test_start = start + tr + g
        else:
            train = np.arange(start, start + tr)
            test_start = start + tr + g
        test_end = test_start + te
        if test_end > n or train.size == 0:
            break
        test = np.arange(test_start, test_end)
        if test.size == 0:
            break
        yield train, test
        start += step_size
        if expanding and start + tr + g + te > n:
            # advance expanding origin
            pass
        if not expanding and start + tr + g + te > n:
            break


def walk_forward_backtest(
    signal: Any,
    returns: Any,
    *,
    train_size: int,
    test_size: int,
    gap: int = 0,
    cost_bps: float = 0.0,
    mode: str = "long_short",
    expanding: bool = False,
    score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    returns_are_forward: bool = True,
) -> dict[str, Any]:
    """Walk-forward OOS signal backtest with purge/embargo ``gap``.

    On each fold, optional ``score_fn(train_signal, train_returns)`` may be used
    by callers for model selection; this routine evaluates the raw signal on
    each OOS fold (no refitting) and concatenates OOS net returns.
    """
    sig = np.asarray(signal, dtype=np.float64).reshape(-1)
    ret = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = min(sig.size, ret.size)
    sig, ret = sig[:n], ret[:n]

    oos_gross: list[float] = []
    oos_net: list[float] = []
    fold_stats: list[dict[str, Any]] = []
    scores: list[float] = []

    for tr_idx, te_idx in walk_forward_splits(
        n, train_size=train_size, test_size=test_size, gap=gap, expanding=expanding
    ):
        if score_fn is not None:
            scores.append(float(score_fn(sig[tr_idx], ret[tr_idx])))
        fold = signal_backtest(
            sig[te_idx],
            ret[te_idx],
            cost_bps=cost_bps,
            mode=mode,  # type: ignore[arg-type]
            returns_are_forward=returns_are_forward,
        )
        oos_gross.extend(fold["gross_returns"].tolist())
        oos_net.extend(fold["net_returns"].tolist())
        fold_stats.append(
            {
                "train_start": int(tr_idx[0]),
                "train_end": int(tr_idx[-1]) + 1,
                "test_start": int(te_idx[0]),
                "test_end": int(te_idx[-1]) + 1,
                "net_sharpe": fold["net_sharpe"],
                "n_test": int(te_idx.size),
            }
        )

    net = np.asarray(oos_net, dtype=np.float64)
    gross = np.asarray(oos_gross, dtype=np.float64)
    sd = float(np.std(net, ddof=1)) if net.size > 1 else float("nan")
    net_sharpe = (
        float(np.mean(net) / sd * np.sqrt(252.0)) if net.size > 1 and sd > 1e-15 else float("nan")
    )
    return {
        "oos_gross_returns": gross,
        "oos_net_returns": net,
        "net_sharpe": net_sharpe,
        "n_folds": len(fold_stats),
        "folds": fold_stats,
        "train_scores": scores,
        "gap": int(gap),
        "look_ahead_guard": f"walk_forward_gap={gap}",
    }

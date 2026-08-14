"""Monte Carlo strategy path simulation (bootstrap / residual / regime)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, total_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio

Method = Literal[
    "bootstrap",
    "block_bootstrap",
    "trade_bootstrap",
    "residual_bootstrap",
    "regime_conditioned",
    "correlated",
]

__all__ = [
    "block_bootstrap_paths",
    "bootstrap_paths",
    "correlated_paths",
    "regime_conditioned_paths",
    "residual_bootstrap_paths",
    "run_monte_carlo",
    "trade_bootstrap_paths",
]


def bootstrap_paths(
    returns: Any,
    *,
    n_simulations: int = 500,
    horizon: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """i.i.d. bootstrap of returns → shape (n_simulations, horizon)."""
    r = as_returns(returns)
    if r.size == 0:
        raise ValueError("returns empty")
    h = int(horizon) if horizon is not None else int(r.size)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, r.size, size=(max(int(n_simulations), 1), max(h, 1)))
    return r[idx]


def block_bootstrap_paths(
    returns: Any,
    *,
    n_simulations: int = 500,
    horizon: int | None = None,
    block_size: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Circular block bootstrap preserving short-run dependence."""
    r = as_returns(returns)
    if r.size == 0:
        raise ValueError("returns empty")
    h = int(horizon) if horizon is not None else int(r.size)
    bs = max(int(block_size), 1)
    rng = np.random.default_rng(int(seed))
    n_sim = max(int(n_simulations), 1)
    out = np.empty((n_sim, h), dtype=np.float64)
    n = r.size
    for i in range(n_sim):
        path: list[float] = []
        while len(path) < h:
            start = int(rng.integers(0, n))
            for j in range(bs):
                path.append(float(r[(start + j) % n]))
                if len(path) >= h:
                    break
        out[i] = np.asarray(path[:h], dtype=np.float64)
    return out


def trade_bootstrap_paths(
    trade_pnls: Any,
    *,
    n_simulations: int = 500,
    n_trades: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap trade PnL sequences → (n_simulations, n_trades)."""
    pnl = np.asarray(trade_pnls, dtype=np.float64).reshape(-1)
    pnl = pnl[np.isfinite(pnl)]
    if pnl.size == 0:
        raise ValueError("trade_pnls empty")
    nt = int(n_trades) if n_trades is not None else int(pnl.size)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, pnl.size, size=(max(int(n_simulations), 1), max(nt, 1)))
    return pnl[idx]


def residual_bootstrap_paths(
    returns: Any,
    *,
    fitted: Any | None = None,
    n_simulations: int = 500,
    horizon: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Residual bootstrap around a fitted mean/path.

    If ``fitted`` is None, uses the sample mean as the conditional mean.
    """
    r = as_returns(returns)
    if r.size == 0:
        raise ValueError("returns empty")
    h = int(horizon) if horizon is not None else int(r.size)
    if fitted is None:
        mu = float(np.mean(r))
        mean_path = np.full(h, mu)
        residuals = r - mu
    else:
        f = as_returns(fitted)
        n = min(f.size, r.size)
        mean_path = np.resize(f[:n], h)
        residuals = r[:n] - f[:n]
    rng = np.random.default_rng(int(seed))
    n_sim = max(int(n_simulations), 1)
    idx = rng.integers(0, residuals.size, size=(n_sim, h))
    return mean_path.reshape(1, -1) + residuals[idx]


def regime_conditioned_paths(
    returns: Any,
    regimes: Any,
    *,
    n_simulations: int = 500,
    horizon: int | None = None,
    regime_path: Any | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Bootstrap returns conditioned on regime labels.

    For each simulated bar, sample from historical returns of the same regime.
    ``regime_path`` length ``horizon`` supplies the conditioning sequence;
    otherwise the empirical regime sequence is repeated/truncated.
    """
    r = as_returns(returns)
    labs = np.asarray(regimes).reshape(-1)
    n = min(r.size, labs.size)
    r = r[:n]
    labs = labs[:n]
    h = int(horizon) if horizon is not None else n
    if regime_path is None:
        rp = np.resize(labs, h)
    else:
        rp = np.asarray(regime_path).reshape(-1)
        rp = np.resize(rp, h)

    buckets: dict[str, np.ndarray] = {}
    for lab in {str(x) for x in labs.tolist()}:
        buckets[lab] = r[np.asarray([str(x) == lab for x in labs.tolist()])]

    rng = np.random.default_rng(int(seed))
    n_sim = max(int(n_simulations), 1)
    out = np.empty((n_sim, h), dtype=np.float64)
    fallback = r
    for i in range(n_sim):
        for t in range(h):
            key = str(rp[t])
            pool = buckets.get(key, fallback)
            if pool.size == 0:
                pool = fallback
            out[i, t] = float(pool[int(rng.integers(0, pool.size))])
    return out


def correlated_paths(
    returns: Any,
    *,
    n_simulations: int = 500,
    horizon: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Gaussian correlated Monte Carlo for multivariate returns.

    Returns terminal-period paths shaped (n_simulations, horizon, N) flattened
    to portfolio equal-weight path (n_simulations, horizon) for summarization
    when multivariate; for 1-D, parametric normal draws.
    """
    r = np.asarray(returns, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n_sim = max(int(n_simulations), 1)
    if r.ndim == 1:
        series = as_returns(r)
        h = int(horizon) if horizon is not None else int(series.size)
        mu = float(np.mean(series)) if series.size else 0.0
        sd = float(np.std(series, ddof=1)) if series.size > 1 else 0.0
        return rng.normal(mu, max(sd, 1e-12), size=(n_sim, max(h, 1)))

    h = int(horizon) if horizon is not None else int(r.shape[0])
    mu = np.mean(r, axis=0)
    cov = np.cov(r, rowvar=False)
    # Symmetrize / PSD nudge
    cov = 0.5 * (cov + cov.T)
    jitter = 1e-10 * np.eye(cov.shape[0])
    try:
        draws = rng.multivariate_normal(mu, cov + jitter, size=(n_sim, max(h, 1)))
    except np.linalg.LinAlgError:
        sd = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        draws = rng.normal(mu, sd, size=(n_sim, max(h, 1), r.shape[1]))
    # Equal-weight portfolio path
    w = np.full(r.shape[1], 1.0 / r.shape[1])
    return draws @ w


def run_monte_carlo(
    returns: Any,
    *,
    method: Method = "bootstrap",
    n_simulations: int = 500,
    horizon: int | None = None,
    seed: int = 42,
    block_size: int = 5,
    regimes: Any | None = None,
    trade_pnls: Any | None = None,
    fitted: Any | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Run a Monte Carlo method and summarize path distribution."""
    m = str(method).lower()
    if m == "block_bootstrap":
        paths = block_bootstrap_paths(
            returns,
            n_simulations=n_simulations,
            horizon=horizon,
            block_size=block_size,
            seed=seed,
        )
    elif m == "trade_bootstrap":
        if trade_pnls is None:
            raise ValueError("trade_pnls required for trade_bootstrap")
        paths = trade_bootstrap_paths(
            trade_pnls, n_simulations=n_simulations, n_trades=horizon, seed=seed
        )
    elif m == "residual_bootstrap":
        paths = residual_bootstrap_paths(
            returns,
            fitted=fitted,
            n_simulations=n_simulations,
            horizon=horizon,
            seed=seed,
        )
    elif m == "regime_conditioned":
        if regimes is None:
            raise ValueError("regimes required for regime_conditioned")
        paths = regime_conditioned_paths(
            returns,
            regimes,
            n_simulations=n_simulations,
            horizon=horizon,
            seed=seed,
        )
    elif m == "correlated":
        paths = correlated_paths(returns, n_simulations=n_simulations, horizon=horizon, seed=seed)
    else:
        paths = bootstrap_paths(returns, n_simulations=n_simulations, horizon=horizon, seed=seed)
        m = "bootstrap"

    totals = np.array([total_return(p) for p in paths], dtype=np.float64)
    sharpes = np.array(
        [sharpe_ratio(p, periods_per_year=periods_per_year) for p in paths],
        dtype=np.float64,
    )
    mdds = np.array([max_drawdown(p) for p in paths], dtype=np.float64)

    return {
        "name": "monte_carlo",
        "method": m,
        "n_simulations": int(paths.shape[0]),
        "horizon": int(paths.shape[1]),
        "paths": paths,
        "terminal_returns": totals,
        "mean_terminal": float(np.mean(totals)),
        "p05_terminal": float(np.quantile(totals, 0.05)),
        "p50_terminal": float(np.quantile(totals, 0.50)),
        "p95_terminal": float(np.quantile(totals, 0.95)),
        "mean_sharpe": float(np.mean(sharpes)),
        "mean_max_drawdown": float(np.mean(mdds)),
        "seed": int(seed),
    }

"""Simple DP / heuristic multi-period allocation for small horizons."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from iqrp.app.portfolio.multi_period.rebalancing import apply_drift, turnover
from iqrp.app.portfolio.optimization.projection import (
    as_cov,
    as_vector,
    check_feasibility,
    equal_weights,
    failed_result,
    format_weights,
    infeasible_result,
    make_result,
    parse_constraints,
    project_weights,
)


def _simplex_grid(n: int, levels: int) -> np.ndarray:
    """Nonnegative weights on a discrete simplex grid summing to 1."""
    if n <= 0:
        return np.zeros((0, 0))
    levels = max(int(levels), 1)
    pts = []
    # compositions of levels into n parts
    for comb in product(range(levels + 1), repeat=n):
        if sum(comb) == levels:
            pts.append(np.asarray(comb, dtype=np.float64) / float(levels))
    if not pts:
        pts = [equal_weights(n, 1.0)]
    return np.asarray(pts, dtype=np.float64)


def optimize_dynamic_programming(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    horizons: int = 2,
    mu_path: Any = None,
    transaction_cost: float = 0.001,
    grid_levels: int = 4,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Backward DP on a discrete simplex grid (small n, small horizons).

    Value: E[sum_t  w_t'μ_t - (λ/2) w_t'Σ w_t - tc * turnover(w_{t-1}^drift, w_t)]
    """
    name = "dynamic_programming"
    method = "grid_dp"
    try:
        h = int(horizons)
        if h <= 0:
            raise ValueError("horizons must be positive")
        if cov is None:
            raise ValueError("cov is required")
        c = as_cov(cov)
        n = c.shape[0]
        if n > 6 or h > 4 or (n > 4 and grid_levels > 5):
            # keep combinatorial cost bounded; fall back to greedy heuristic
            method = "greedy_heuristic"

        if mu_path is not None:
            mp = np.asarray(mu_path, dtype=np.float64)
            if mp.ndim == 1:
                mp = np.tile(mp.reshape(1, -1), (h, 1))
        elif mu is not None:
            mp = np.tile(as_vector(mu, n).reshape(1, -1), (h, 1))
        else:
            mp = np.zeros((h, n))

        cstr = parse_constraints(
            constraints,
            n,
            long_only=long_only,
            max_weight=max_weight,
            min_weight=min_weight,
            max_gross=max_gross,
            budget=budget,
        )
        if names is None:
            names = cstr.get("names")
        ok, reason, conflicts = check_feasibility(cstr)
        if not ok:
            return infeasible_result(
                name,
                n,
                method=method,
                reason=reason or "infeasible",
                conflicts=conflicts,
                names=names,
            )
        if cstr["lb"] < 0:
            return infeasible_result(
                name,
                n,
                method=method,
                reason="DP grid currently supports long_only portfolios",
                conflicts=["long_only"],
                names=names,
            )

        lam = max(float(risk_aversion), 1e-8)
        tc = float(transaction_cost)
        w0 = (
            project_weights(as_vector(current_weights, n), cstr)
            if current_weights is not None
            else equal_weights(n, cstr["budget"])
        )

        if method == "greedy_heuristic":
            # forward greedy: myopic MV with TC via turnover module
            from iqrp.app.portfolio.multi_period.optimizer import optimize_multi_period

            res = optimize_multi_period(
                mu=mp[0],
                cov=c,
                current_weights=w0,
                constraints=constraints,
                long_only=long_only,
                max_weight=max_weight,
                risk_aversion=risk_aversion,
                horizons=h,
                mu_path=mp,
                transaction_cost=tc,
                min_weight=min_weight,
                max_gross=max_gross,
                budget=budget,
                names=names,
            )
            out = dict(res)
            out["name"] = name
            out["method"] = "greedy_heuristic"
            return out

        grid = _simplex_grid(n, grid_levels) * cstr["budget"]
        # filter by box
        mask = np.all((grid >= cstr["lb"] - 1e-12) & (grid <= cstr["ub"] + 1e-12), axis=1)
        grid = grid[mask]
        if grid.shape[0] == 0:
            return infeasible_result(
                name,
                n,
                method=method,
                reason="no grid points satisfy box constraints",
                conflicts=["box"],
                names=names,
            )

        g = grid.shape[0]
        # V_{h} = 0; store best action index per state/time
        V = np.zeros(g)
        policy = np.zeros((h, g), dtype=int)

        def stage_util(w: np.ndarray, t: int) -> float:
            return float(w @ mp[t] - 0.5 * lam * float(w @ c @ w))

        for t in range(h - 1, -1, -1):
            V_new = np.full(g, -np.inf)
            for i in range(g):
                # state = holdings before trade at t; for t>0 interpret as drifted from previous choice
                # At backward pass we treat grid nodes as pre-trade states
                best = -np.inf
                best_j = i
                for j in range(g):
                    to = turnover(grid[i], grid[j])
                    val = stage_util(grid[j], t) - tc * 2.0 * to
                    if t < h - 1:
                        # next pre-trade ≈ drift under expected return
                        nxt = apply_drift(grid[j], mp[t])
                        # snap to nearest grid node
                        d2 = np.sum((grid - nxt.reshape(1, -1)) ** 2, axis=1)
                        k = int(np.argmin(d2))
                        val += V[k]
                    if val > best:
                        best = val
                        best_j = j
                V_new[i] = best
                policy[t, i] = best_j
            V = V_new

        # forward simulate from w0 snapped to grid
        d2 = np.sum((grid - w0.reshape(1, -1)) ** 2, axis=1)
        state = int(np.argmin(d2))
        path = []
        for t in range(h):
            action = int(policy[t, state])
            path.append(grid[action].copy())
            if t < h - 1:
                nxt = apply_drift(grid[action], mp[t])
                d2 = np.sum((grid - nxt.reshape(1, -1)) ** 2, axis=1)
                state = int(np.argmin(d2))
            else:
                state = action

        w_final = path[0]  # immediate decision
        return make_result(
            name,
            format_weights(w_final, names),
            success=True,
            status="optimal",
            method=method,
            diagnostics={
                "n_assets": n,
                "horizons": h,
                "grid_size": g,
                "grid_levels": int(grid_levels),
                "weights_path": [[float(x) for x in w.tolist()] for w in path],
                "value": float(V[int(np.argmin(np.sum((grid - w0.reshape(1, -1)) ** 2, axis=1)))]),
            },
            objective_value=float(
                V[int(np.argmin(np.sum((grid - w0.reshape(1, -1)) ** 2, axis=1)))]
            ),
        )
    except Exception as exc:
        n = 0
        try:
            n = int(np.asarray(cov).shape[0]) if cov is not None else 0
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))

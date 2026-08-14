"""Multi-horizon portfolio optimization with transaction costs and drift."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.portfolio.multi_period.rebalancing import apply_drift, rebalance_schedule, turnover
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


def optimize_multi_period(
    mu: Any = None,
    cov: Any = None,
    *,
    current_weights: Any = None,
    constraints: Any = None,
    long_only: bool = True,
    max_weight: float = 0.4,
    risk_aversion: float = 1.0,
    horizons: int = 3,
    mu_path: Any = None,
    cov_path: Any = None,
    return_path: Any = None,
    transaction_cost: float = 0.001,
    rebalance_every: int = 1,
    turnover_threshold: float | None = None,
    min_weight: float | None = None,
    max_gross: float | None = None,
    budget: float = 1.0,
    names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Multi-horizon sequence of weights with TC + drift between rebalances.

    At each rebalance date, solve a single-period MV on that date's (mu, cov),
    penalizing turnover from the drifted current book. Between rebalances,
    weights drift with ``return_path`` (or expected mu).
    """
    name = "multi_period"
    method = "horizon_mv_tc"
    try:
        h = int(horizons)
        if h <= 0:
            raise ValueError("horizons must be positive")

        if mu_path is not None:
            mp = np.asarray(mu_path, dtype=np.float64)
            if mp.ndim == 1:
                mp = np.tile(mp.reshape(1, -1), (h, 1))
            if mp.shape[0] != h:
                raise ValueError("mu_path first dim must equal horizons")
            n = mp.shape[1]
        elif mu is not None:
            m0 = as_vector(mu)
            n = m0.size
            mp = np.tile(m0.reshape(1, -1), (h, 1))
        else:
            if cov is None and cov_path is None:
                raise ValueError("mu/cov or path inputs required")
            n = int(np.asarray(cov if cov is not None else cov_path[0]).shape[0])
            mp = np.zeros((h, n))

        if cov_path is not None:
            cps = [as_cov(c, n) for c in cov_path]
            if len(cps) == 1:
                cps = cps * h
            if len(cps) != h:
                raise ValueError("cov_path length must equal horizons")
        else:
            if cov is None:
                raise ValueError("cov or cov_path required")
            c0 = as_cov(cov, n)
            cps = [c0 for _ in range(h)]

        if return_path is not None:
            rp = np.asarray(return_path, dtype=np.float64)
            if rp.ndim == 1:
                rp = rp.reshape(1, -1)
            if rp.shape[1] != n:
                raise ValueError("return_path width mismatch")
            if rp.shape[0] < h:
                # pad with last / expected
                pad = np.tile(rp[-1], (h - rp.shape[0], 1))
                rp = np.vstack([rp, pad])
            rp = rp[:h]
        else:
            rp = mp.copy()

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

        sched = rebalance_schedule(h, frequency=rebalance_every, threshold=turnover_threshold)
        w = (
            project_weights(as_vector(current_weights, n), cstr)
            if current_weights is not None
            else equal_weights(n, cstr["budget"])
        )

        weights_path: list[list[float]] = []
        traded_path: list[list[float]] = []
        turnover_path: list[float] = []
        tc_total = 0.0
        tc = float(transaction_cost)
        lam = max(float(risk_aversion), 1e-8)

        for t in range(h):
            drifted = w if t == 0 else apply_drift(w, rp[t - 1])
            drifted = (
                project_weights(
                    drifted, {**cstr, "ub": max(cstr["ub"], float(np.max(drifted)) + 1e-9)}
                )
                if cstr["lb"] >= 0
                else drifted
            )
            # keep drifted as book; do not force box on drift (market moves) — trade decision projects
            if not sched["flags"][t]:
                w = drifted
                # normalize to budget after drift for reporting
                s = float(np.sum(w))
                if abs(s) > 1e-14:
                    w = w * (cstr["budget"] / s)
                weights_path.append([float(x) for x in w.tolist()])
                traded_path.append([float(x) for x in w.tolist()])
                turnover_path.append(0.0)
                continue

            # Single-period MV with linear TC against drifted book:
            # max w'μ - (λ/2)w'Σw - tc * ||w - w_drift||_1
            # Implement via mean adjustment / soft L1 using turnover optimizer pattern
            from iqrp.app.portfolio.optimization.turnover import optimize_turnover

            step = optimize_turnover(
                mu=mp[t],
                cov=cps[t],
                current_weights=drifted,
                constraints={
                    "long_only": cstr["long_only"],
                    "max_weight": cstr["ub"],
                    "min_weight": cstr["lb"],
                    "max_gross": cstr.get("max_gross"),
                    "budget": cstr["budget"],
                    "max_turnover": turnover_threshold,
                },
                long_only=cstr["long_only"],
                max_weight=cstr["ub"],
                risk_aversion=lam,
                turnover_penalty=tc,
                max_turnover=turnover_threshold,
                min_weight=cstr["lb"],
                max_gross=cstr.get("max_gross"),
                budget=cstr["budget"],
            )
            if not step.get("success"):
                # hard failure on a period — do not silently relax; keep drifted if only soft TC issue
                if step.get("status") == "infeasible":
                    return make_result(
                        name,
                        format_weights(drifted, names),
                        success=False,
                        status="infeasible",
                        method=method,
                        failure_reason=f"period {t}: {step.get('failure_reason')}",
                        conflicting_constraints=step.get("conflicting_constraints") or [],
                        diagnostics={"period": t, "schedule": sched, "weights_path": weights_path},
                        objective_value=None,
                    )
                # failed → hold drifted projected
                target = project_weights(drifted, cstr)
            else:
                target = as_vector(step["weights"], n)

            to = turnover(drifted, target)
            if turnover_threshold is not None and to > float(turnover_threshold) + 1e-6:
                return make_result(
                    name,
                    format_weights(drifted, names),
                    success=False,
                    status="infeasible",
                    method=method,
                    failure_reason=f"period {t}: turnover {to} exceeds threshold",
                    conflicting_constraints=["max_turnover"],
                    diagnostics={"period": t, "turnover": to},
                    objective_value=None,
                )
            tc_total += tc * 2.0 * to  # round-trip notionally on L1/2 * 2
            w = target
            weights_path.append([float(x) for x in w.tolist()])
            traded_path.append([float(x) for x in target.tolist()])
            turnover_path.append(to)

        # terminal / first-period weight as primary
        w_final = np.asarray(weights_path[-1], dtype=np.float64)
        return make_result(
            name,
            format_weights(w_final, names),
            success=True,
            status="optimal",
            method=method,
            diagnostics={
                "n_assets": n,
                "horizons": h,
                "weights_path": weights_path,
                "traded_path": traded_path,
                "turnover_path": turnover_path,
                "transaction_cost_rate": tc,
                "transaction_cost_total": tc_total,
                "schedule": sched,
            },
            objective_value=-tc_total,
        )
    except Exception as exc:
        n = 0
        try:
            if cov is not None:
                n = int(np.asarray(cov).shape[0])
            elif mu is not None:
                n = int(np.asarray(mu).reshape(-1).size)
        except Exception:
            n = 0
        return failed_result(name, n, method=method, reason=str(exc))

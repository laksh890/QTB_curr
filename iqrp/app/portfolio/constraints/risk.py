"""Risk-metric constraints: VaR / CVaR / drawdown / risk contribution.

Evaluates *precomputed* risk metrics — this module does not recompute VaR/CVaR.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from iqrp.app.portfolio.constraints._types import (
    ConstraintSeverity,
    ConstraintViolation,
    as_weights,
    make_violation,
)


def check_risk_constraints(
    weights: Any | None = None,
    *,
    var: float | None = None,
    cvar: float | None = None,
    expected_shortfall: float | None = None,
    drawdown: float | None = None,
    risk_contribution: Any | None = None,
    max_var: float | None = None,
    max_cvar: float | None = None,
    max_expected_shortfall: float | None = None,
    max_drawdown: float | None = None,
    max_risk_contribution: float | Sequence[float] | None = None,
    risk_metrics: Mapping[str, Any] | None = None,
    severity: ConstraintSeverity | str = ConstraintSeverity.HARD,
) -> list[ConstraintViolation]:
    """Evaluate risk limits against provided metric values.

    Hard risk constraints are never auto-relaxed.
    """
    metrics = dict(risk_metrics or {})
    var_v = float(metrics["var"]) if var is None and "var" in metrics else var
    cvar_v = cvar
    if cvar_v is None:
        if expected_shortfall is not None:
            cvar_v = expected_shortfall
        elif "cvar" in metrics:
            cvar_v = float(metrics["cvar"])
        elif "expected_shortfall" in metrics:
            cvar_v = float(metrics["expected_shortfall"])
    dd_v = float(metrics["drawdown"]) if drawdown is None and "drawdown" in metrics else drawdown
    rc = risk_contribution
    if rc is None and "risk_contribution" in metrics:
        rc = metrics["risk_contribution"]

    out: list[ConstraintViolation] = []

    if max_var is not None and var_v is not None and float(var_v) > float(max_var) + 1e-12:
        out.append(
            make_violation(
                "max_var",
                observed=float(var_v),
                threshold=float(max_var),
                severity=severity,
                reason=f"VaR {float(var_v):.6g} exceeds max_var {float(max_var):.6g}",
            )
        )

    es_cap = max_cvar if max_cvar is not None else max_expected_shortfall
    if es_cap is not None and cvar_v is not None and float(cvar_v) > float(es_cap) + 1e-12:
        out.append(
            make_violation(
                "max_cvar",
                observed=float(cvar_v),
                threshold=float(es_cap),
                severity=severity,
                reason=f"CVaR/ES {float(cvar_v):.6g} exceeds {float(es_cap):.6g}",
            )
        )

    if max_drawdown is not None and dd_v is not None:
        # drawdown often stored as negative; compare on absolute magnitude
        obs = abs(float(dd_v))
        thr = abs(float(max_drawdown))
        if obs > thr + 1e-12:
            out.append(
                make_violation(
                    "max_drawdown",
                    observed=obs,
                    threshold=thr,
                    severity=severity,
                    reason=f"|drawdown|={obs:.6g} exceeds max_drawdown {thr:.6g}",
                    metadata={"drawdown": float(dd_v)},
                )
            )

    if max_risk_contribution is not None and rc is not None:
        arr = np.asarray(rc, dtype=np.float64).reshape(-1)
        thr_arr = np.asarray(max_risk_contribution, dtype=np.float64).reshape(-1)
        if thr_arr.size == 1:
            thr_arr = np.full(arr.size, float(thr_arr[0]))
        for i, (obs, thr) in enumerate(zip(arr, thr_arr, strict=False)):
            if float(obs) > float(thr) + 1e-12:
                out.append(
                    make_violation(
                        "max_risk_contribution",
                        observed=float(obs),
                        threshold=float(thr),
                        severity=severity,
                        reason=(
                            f"risk_contribution[{i}]={float(obs):.6g} exceeds "
                            f"{float(thr):.6g}"
                        ),
                        scope="position",
                        metadata={"index": int(i)},
                    )
                )
    elif max_risk_contribution is not None and weights is not None:
        # no contribution vector provided — skip rather than invent
        pass

    return out

"""Historical execution simulation using volume / spread / impact models.

Integrates ``MarketSimulator`` when available for synthetic paths.
Never invents alpha or exceeds approved residual quantity.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from iqrp.app.execution.slippage.estimator import estimate_slippage
from iqrp.app.execution.slippage.market_impact import market_impact, path_impact
from iqrp.app.execution.transaction_costs.total_cost import pre_trade_cost_estimate


def _ctx(market_context: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(market_context or {})


def simulate_fill_path(
    *,
    side: str,
    quantity: float,
    mid: float,
    spread: float = 0.02,
    adv: float = 1e6,
    volatility: float = 0.02,
    n_slices: int = 5,
    participation: float = 0.05,
    impact_coeff: float = 0.1,
    seed: int | None = 42,
) -> dict[str, Any]:
    """Simulate a backtest-style fill path using volume/spread/impact.

    Returns fills, VWAP, and cost/slippage estimates. Quantity is treated as
    the approved residual — never exceeded.
    """
    qty = abs(float(quantity))
    if qty <= 0.0 or mid <= 0.0:
        return {
            "fills": [],
            "filled_qty": 0.0,
            "exec_vwap": 0.0,
            "arrival_price": float(mid),
            "slippage": {},
            "costs": {},
            "path": [],
        }

    n = max(int(n_slices), 1)
    rng = np.random.default_rng(seed)
    slice_qty = qty / n
    # Cap each slice by participation of ADV
    max_slice = max(float(adv) * max(float(participation), 0.0), slice_qty)
    slice_qty = min(slice_qty, max_slice)
    # Re-scale so total == approved (never exceed)
    planned = [slice_qty] * n
    total_planned = float(sum(planned))
    if total_planned > qty:
        scale = qty / total_planned
        planned = [q * scale for q in planned]

    side_l = str(side).strip().lower()
    is_buy = side_l in {"buy", "b", "cover", "long"}
    half = 0.5 * max(float(spread), 0.0)
    px = float(mid)
    fills: list[dict[str, Any]] = []
    path: list[dict[str, Any]] = []
    remaining = qty

    for i, q in enumerate(planned):
        q = min(float(q), remaining)
        if q <= 0:
            continue
        # Temporary impact + noise (no future info — only current path state)
        impact = market_impact(
            side=side,
            quantity=q,
            mid=px,
            adv=adv,
            volatility=volatility,
            spread=spread,
            impact_coeff=impact_coeff,
        )
        impact_px = float(impact.get("impact", 0.0))
        noise = float(rng.normal(0.0, max(float(volatility) * px * 0.01, 1e-8)))
        if is_buy:
            fill_px = px + half + impact_px + abs(noise) * 0.1
        else:
            fill_px = px - half - impact_px - abs(noise) * 0.1
        fills.append(
            {
                "quantity": q,
                "qty": q,
                "price": float(fill_px),
                "fill_price": float(fill_px),
                "slice": i,
            }
        )
        path.append({"t": i, "mid": px, "fill_price": float(fill_px), "qty": q})
        # Permanent impact moves mid
        permanent = impact_px * 0.5
        px = px + permanent if is_buy else px - permanent
        remaining = max(remaining - q, 0.0)

    filled = float(sum(f["qty"] for f in fills))
    notional = float(sum(f["qty"] * f["price"] for f in fills))
    vwap = notional / filled if filled > 0 else 0.0

    slip = estimate_slippage(
        side=side,
        quantity=qty,
        mid=mid,
        spread=spread,
        adv=adv,
        volatility=volatility,
        impact_coeff=impact_coeff,
    )
    costs = pre_trade_cost_estimate(
        side=side,
        quantity=qty,
        mid=mid,
        spread=spread,
        adv=adv,
        volatility=volatility,
        impact_coeff=impact_coeff,
    )
    mids = [float(mid)] * max(len(fills), 1)
    vols = [float(adv)] * max(len(fills), 1)
    sizes = [float(f["qty"]) for f in fills] or [0.0]
    vols_arr = vols[: len(sizes)]
    mids_arr = mids[: len(sizes)]
    vol_arr = [float(volatility)] * len(sizes)
    try:
        path_imp = path_impact(mids_arr, vols_arr, sizes, vol_arr, impact_coeff=impact_coeff)
        path_imp_out: Any = path_imp.tolist() if hasattr(path_imp, "tolist") else list(path_imp)
    except Exception:  # noqa: BLE001
        path_imp_out = []

    return {
        "fills": fills,
        "filled_qty": filled,
        "exec_vwap": float(vwap),
        "arrival_price": float(mid),
        "residual_qty": max(qty - filled, 0.0),
        "slippage": slip,
        "costs": costs,
        "path_impact": path_imp_out,
        "path": path,
        "source": "volume_spread_impact",
    }


def simulate_with_market_simulator(
    *,
    side: str,
    quantity: float,
    instrument: str = "SIM",
    n_bars: int = 64,
    seed: int | None = 42,
    market_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Try ``MarketSimulator`` for a synthetic mid path; fall back to local model."""
    ctx = _ctx(market_context)
    mid0 = float(ctx.get("mid", ctx.get("price", 100.0)))
    spread = float(ctx.get("spread", 0.02))
    adv = float(ctx.get("adv", 1e6))
    vol = float(ctx.get("volatility", 0.02))

    try:
        from iqrp.app.simulation import MarketSimulator  # type: ignore

        sim = MarketSimulator()
        # Best-effort: different MarketSimulator APIs across versions
        path_mids: list[float] | None = None
        if hasattr(sim, "simulate"):
            out = sim.simulate(n_bars=n_bars, seed=seed)  # type: ignore[call-arg]
            if isinstance(out, Mapping):
                series = out.get("close") or out.get("series") or out.get("mid")
                if series is not None:
                    path_mids = [float(x) for x in list(series)[:n_bars]]
        if path_mids and len(path_mids) >= 2:
            # Execute evenly along simulated path
            qty = abs(float(quantity))
            n = min(len(path_mids), max(int(n_bars // 8), 1))
            slice_qty = qty / n
            fills = []
            for i in range(n):
                px = float(path_mids[min(i * len(path_mids) // n, len(path_mids) - 1)])
                half = 0.5 * spread
                fill_px = px + half if str(side).lower() in {"buy", "b", "cover"} else px - half
                fills.append(
                    {
                        "quantity": slice_qty,
                        "qty": slice_qty,
                        "price": fill_px,
                        "fill_price": fill_px,
                        "slice": i,
                    }
                )
            filled = float(sum(f["qty"] for f in fills))
            vwap = float(sum(f["qty"] * f["price"] for f in fills) / filled) if filled else 0.0
            return {
                "fills": fills,
                "filled_qty": filled,
                "exec_vwap": vwap,
                "arrival_price": float(path_mids[0]),
                "residual_qty": 0.0,
                "path": [{"t": i, "mid": float(m)} for i, m in enumerate(path_mids)],
                "source": "MarketSimulator",
                "instrument": instrument,
            }
    except Exception:  # noqa: BLE001
        pass

    result = simulate_fill_path(
        side=side,
        quantity=quantity,
        mid=mid0,
        spread=spread,
        adv=adv,
        volatility=vol,
        seed=seed,
    )
    result["instrument"] = instrument
    return result


def simulate_execution(
    *,
    orders: Sequence[Mapping[str, Any]] | None = None,
    side: str | None = None,
    quantity: float | None = None,
    market_context: Mapping[str, Any] | None = None,
    use_market_simulator: bool = True,
    seed: int | None = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Backtest-style execution simulation for one or more child orders."""
    ctx = _ctx(market_context)
    results: list[dict[str, Any]] = []

    if orders:
        for o in orders:
            o_side = str(o.get("side", side or "buy"))
            o_qty = float(o.get("quantity", o.get("qty", quantity or 0.0)))
            inst = str(o.get("instrument", "SIM"))
            o_ctx = dict(ctx)
            if isinstance(ctx.get(inst), Mapping):
                o_ctx.update(dict(ctx[inst]))  # type: ignore[index]
            if use_market_simulator:
                results.append(
                    simulate_with_market_simulator(
                        side=o_side,
                        quantity=o_qty,
                        instrument=inst,
                        seed=seed,
                        market_context=o_ctx,
                    )
                )
            else:
                results.append(
                    simulate_fill_path(
                        side=o_side,
                        quantity=o_qty,
                        mid=float(o_ctx.get("mid", o_ctx.get("price", 100.0))),
                        spread=float(o_ctx.get("spread", 0.02)),
                        adv=float(o_ctx.get("adv", 1e6)),
                        volatility=float(o_ctx.get("volatility", 0.02)),
                        seed=seed,
                        **{k: v for k, v in kwargs.items() if k in {"n_slices", "participation", "impact_coeff"}},
                    )
                )
        return {"orders": results, "n": len(results)}

    side_f = str(side or "buy")
    qty_f = float(quantity or 0.0)
    if use_market_simulator:
        return simulate_with_market_simulator(
            side=side_f,
            quantity=qty_f,
            market_context=ctx,
            seed=seed,
        )
    return simulate_fill_path(
        side=side_f,
        quantity=qty_f,
        mid=float(ctx.get("mid", ctx.get("price", 100.0))),
        spread=float(ctx.get("spread", 0.02)),
        adv=float(ctx.get("adv", 1e6)),
        volatility=float(ctx.get("volatility", 0.02)),
        seed=seed,
        **{k: v for k, v in kwargs.items() if k in {"n_slices", "participation", "impact_coeff"}},
    )


__all__ = [
    "simulate_execution",
    "simulate_fill_path",
    "simulate_with_market_simulator",
]

"""Production-parity portfolio / execution adapters with named isolated fallbacks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class IsolatedPortfolioFallback:
    """Deterministic local portfolio construction when production modules fail."""

    name = "IsolatedPortfolioFallback"

    @staticmethod
    def signals_to_raw_weights(
        signals: Mapping[str, float] | Sequence[float],
        *,
        names: Sequence[str] | None = None,
        budget: float = 1.0,
        long_only: bool = True,
    ) -> dict[str, Any]:
        if isinstance(signals, Mapping):
            name_list = list(names) if names is not None else list(signals.keys())
            vals = [float(signals.get(n, 0.0)) for n in name_list]
        else:
            vals = [float(x) for x in signals]
            name_list = list(names) if names is not None else [f"a{i}" for i in range(len(vals))]
        if not vals:
            return {"names": [], "weights": [], "backend": IsolatedPortfolioFallback.name}
        if long_only:
            pos = [max(v, 0.0) for v in vals]
            s = sum(pos)
            if s <= 1e-18:
                w = [float(budget) / len(vals)] * len(vals)
            else:
                w = [float(budget) * v / s for v in pos]
        else:
            abs_s = sum(abs(v) for v in vals)
            if abs_s <= 1e-18:
                w = [0.0] * len(vals)
            else:
                w = [float(budget) * v / abs_s for v in vals]
        return {
            "names": list(name_list),
            "weights": w,
            "backend": IsolatedPortfolioFallback.name,
        }

    @staticmethod
    def build_target_weights(
        weights: Mapping[str, float] | Sequence[float],
        *,
        names: Sequence[str] | None = None,
    ) -> dict[str, float]:
        if isinstance(weights, Mapping):
            return {str(k): float(v) for k, v in weights.items()}
        name_list = list(names) if names is not None else [f"a{i}" for i in range(len(list(weights)))]
        return {str(n): float(w) for n, w in zip(name_list, weights)}


class PortfolioConstructionAdapter:
    """Try ``iqrp.app.portfolio``; fall back to :class:`IsolatedPortfolioFallback`."""

    def __init__(self) -> None:
        self.backend = IsolatedPortfolioFallback.name
        self._prod = None
        try:
            from iqrp.app.portfolio import (  # noqa: F401
                PortfolioConstructionEngine,
                build_target_weights,
                signals_to_raw_weights,
            )

            self._prod = {
                "signals_to_raw_weights": signals_to_raw_weights,
                "build_target_weights": build_target_weights,
                "PortfolioConstructionEngine": PortfolioConstructionEngine,
            }
            self.backend = "iqrp.app.portfolio"
        except Exception:  # noqa: BLE001
            self._prod = None
            self.backend = IsolatedPortfolioFallback.name

    def targets_from_signals(
        self,
        signals: Mapping[str, float],
        *,
        budget: float = 1.0,
        long_only: bool = True,
        method: str = "proportional",
    ) -> dict[str, float]:
        if not signals:
            return {}
        names = list(signals.keys())
        vals = [float(signals[n]) for n in names]
        try:
            if self._prod is not None:
                raw = self._prod["signals_to_raw_weights"](
                    vals,
                    method=method,
                    long_only=long_only,
                    budget=budget,
                    names=names,
                )
                tw = self._prod["build_target_weights"](
                    raw.get("weights", []),
                    names=raw.get("names", names),
                    method=method,
                    source="backtest_runner",
                    long_only=long_only,
                )
                if hasattr(tw, "as_dict"):
                    return {str(k): float(v) for k, v in tw.as_dict().items()}
                if hasattr(tw, "weights") and hasattr(tw, "names"):  # pragma: no cover
                    return {str(n): float(w) for n, w in zip(tw.names, tw.weights)}
                if isinstance(tw, Mapping):  # pragma: no cover
                    return {str(k): float(v) for k, v in tw.items()}
        except Exception:  # noqa: BLE001  # pragma: no cover
            self.backend = IsolatedPortfolioFallback.name
        raw = IsolatedPortfolioFallback.signals_to_raw_weights(
            signals, budget=budget, long_only=long_only
        )
        return IsolatedPortfolioFallback.build_target_weights(
            raw["weights"], names=raw["names"]
        )

    def targets_from_weights(self, weights: Mapping[str, float]) -> dict[str, float]:
        try:
            if self._prod is not None:
                tw = self._prod["build_target_weights"](
                    dict(weights), source="backtest_runner"
                )
                if hasattr(tw, "as_dict"):
                    return {str(k): float(v) for k, v in tw.as_dict().items()}
                if hasattr(tw, "weights") and hasattr(tw, "names"):
                    return {str(n): float(w) for n, w in zip(tw.names, tw.weights)}
        except Exception:  # noqa: BLE001
            self.backend = IsolatedPortfolioFallback.name
        return IsolatedPortfolioFallback.build_target_weights(weights)


class IsolatedExecutionFallback:
    """Simple mid ± half-spread fills when production execution is unavailable."""

    name = "IsolatedExecutionFallback"

    @staticmethod
    def simulate_execution(
        orders: Sequence[Mapping[str, Any]],
        *,
        market_context: Mapping[str, Any] | None = None,
        spread_bps: float = 1.0,
        slippage_bps: float = 0.0,
        commission_bps: float = 0.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        del seed  # deterministic given inputs
        ctx = dict(market_context or {})
        results: list[dict[str, Any]] = []
        for o in orders:
            inst = str(o.get("instrument", o.get("symbol", "")))
            side = str(o.get("side", "buy")).lower()
            qty = abs(float(o.get("quantity", o.get("qty", 0.0))))
            inst_ctx = ctx.get(inst) if isinstance(ctx.get(inst), Mapping) else ctx
            mid = float(
                (inst_ctx or {}).get(
                    "mid",
                    (inst_ctx or {}).get("price", o.get("price", 0.0)),
                )
                or 0.0
            )
            if qty <= 0 or mid <= 0:
                continue
            half = mid * (float(spread_bps) + float(slippage_bps)) / 10_000.0 / 2.0
            px = mid + half if side in {"buy", "b", "cover", "long"} else mid - half
            notional = qty * px
            fee = notional * float(commission_bps) / 10_000.0
            results.append(
                {
                    "instrument": inst,
                    "side": side,
                    "quantity": qty,
                    "filled_qty": qty,
                    "exec_vwap": px,
                    "price": px,
                    "fee": fee,
                    "slippage": half,
                    "fills": [{"quantity": qty, "price": px, "fee": fee}],
                    "backend": IsolatedExecutionFallback.name,
                }
            )
        return {"orders": results, "n": len(results), "backend": IsolatedExecutionFallback.name}

    @staticmethod
    def plan_from_targets(
        current: Mapping[str, float],
        target: Mapping[str, float],
        *,
        equity: float,
        prices: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        instruments = sorted(set(current) | set(target) | set(prices))
        for inst in instruments:
            px = float(prices.get(inst, 0.0) or 0.0)
            if px <= 0:
                continue
            cur_qty = float(current.get(inst, 0.0))
            tgt_w = float(target.get(inst, 0.0))
            tgt_qty = (tgt_w * float(equity)) / px
            delta = tgt_qty - cur_qty
            if abs(delta) * px < 1e-6:
                continue
            side = "buy" if delta > 0 else "sell"
            orders.append(
                {
                    "instrument": inst,
                    "side": side,
                    "quantity": abs(delta),
                    "order_type": "market",
                    "price": px,
                }
            )
        return orders

    @staticmethod
    def estimate_costs(
        orders: Sequence[Mapping[str, Any]],
        *,
        commission_bps: float = 0.0,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> dict[str, Any]:
        total = 0.0
        rows = []
        bps = float(commission_bps) + float(spread_bps) + float(slippage_bps)
        for o in orders:
            qty = abs(float(o.get("quantity", 0.0) or 0.0))
            px = float(o.get("price", 0.0) or 0.0)
            cost = qty * px * bps / 10_000.0
            total += cost
            rows.append({"instrument": o.get("instrument"), "total_cost": cost})
        return {"orders": rows, "total_cost": total, "backend": IsolatedExecutionFallback.name}


class ExecutionSimulationAdapter:
    """Try ``iqrp.app.execution.ExecutionEngine``; fall back to isolated simulation."""

    def __init__(self) -> None:
        self.backend = IsolatedExecutionFallback.name
        self._engine = None
        try:
            from iqrp.app.execution import ExecutionEngine

            self._engine = ExecutionEngine()
            self.backend = "iqrp.app.execution.ExecutionEngine"
        except Exception:  # noqa: BLE001
            self._engine = None
            self.backend = IsolatedExecutionFallback.name

    def plan_from_targets(
        self,
        current: Mapping[str, float],
        target_weights: Mapping[str, float],
        *,
        equity: float,
        prices: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        # Convert weights → target quantities
        target_qty: dict[str, float] = {}
        for inst, w in target_weights.items():
            px = float(prices.get(inst, 0.0) or 0.0)
            if px <= 0:
                continue
            target_qty[str(inst)] = float(w) * float(equity) / px
        for inst in current:
            target_qty.setdefault(str(inst), 0.0)

        try:
            if self._engine is not None:
                orders = self._engine.plan_from_targets(dict(current), target_qty)
                out = []
                for o in orders:
                    if hasattr(o, "to_dict"):
                        d = dict(o.to_dict())
                    else:
                        d = {
                            "instrument": getattr(o, "instrument", ""),
                            "side": getattr(getattr(o, "side", None), "value", getattr(o, "side", "buy")),
                            "quantity": float(getattr(o, "quantity", 0.0) or 0.0),
                            "order_type": "market",
                            "order_id": getattr(o, "order_id", None),
                        }
                    inst = str(d.get("instrument", getattr(o, "instrument", "")))
                    d["instrument"] = inst
                    if d.get("price") is None or float(d.get("price") or 0.0) <= 0:
                        d["price"] = float(prices.get(inst, 0.0) or 0.0)
                    if not d.get("side"):  # pragma: no cover
                        d["side"] = getattr(
                            getattr(o, "side", None), "value", getattr(o, "side", "buy")
                        )
                    out.append(d)
                if out:
                    return out
        except Exception:  # noqa: BLE001  # pragma: no cover
            self.backend = IsolatedExecutionFallback.name

        return IsolatedExecutionFallback.plan_from_targets(
            current, target_weights, equity=equity, prices=prices
        )

    def estimate_costs(
        self,
        orders: Sequence[Mapping[str, Any]],
        *,
        market_context: Mapping[str, Any] | None = None,
        commission_bps: float = 0.0,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ) -> dict[str, Any]:
        # Prefer isolated estimator for plain dict orders from the runner;
        # production engine expects Order objects and may reject dict specs.
        try:
            if self._engine is not None and orders and hasattr(orders[0], "instrument"):
                return dict(self._engine.estimate_costs(list(orders), market_context=market_context))
        except Exception:  # noqa: BLE001  # pragma: no cover
            self.backend = IsolatedExecutionFallback.name
        return IsolatedExecutionFallback.estimate_costs(
            orders,
            commission_bps=commission_bps,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
        )

    def simulate_execution(
        self,
        orders: Sequence[Mapping[str, Any]],
        *,
        market_context: Mapping[str, Any] | None = None,
        spread_bps: float = 1.0,
        slippage_bps: float = 0.0,
        commission_bps: float = 0.0,
        seed: int | None = 42,
    ) -> dict[str, Any]:
        try:
            if self._engine is not None:
                result = self._engine.simulate_execution(
                    orders=list(orders),
                    market_context=market_context,
                    seed=seed,
                )
                # Normalize + attach fees from bps if missing
                normalized = []
                for row in list(result.get("orders") or [result]):
                    if not isinstance(row, Mapping):  # pragma: no cover
                        continue
                    inst = str(row.get("instrument", ""))
                    # Nested per-order simulation results may omit instrument
                    fills = list(row.get("fills") or [])
                    qty = float(row.get("filled_qty", row.get("quantity", 0.0)) or 0.0)
                    if qty <= 0 and fills:
                        qty = float(sum(float(f.get("quantity", f.get("qty", 0.0)) or 0.0) for f in fills))
                    px = float(row.get("exec_vwap", row.get("price", 0.0)) or 0.0)
                    if px <= 0 and fills:
                        px = float(fills[0].get("price", fills[0].get("fill_price", 0.0)) or 0.0)
                    fee = row.get("fee", 0.0)
                    if isinstance(fee, Mapping):
                        fee = float(fee.get("total_cost", fee.get("fee", 0.0)) or 0.0)
                    else:
                        fee = float(fee or 0.0)
                    if fee <= 0 and px > 0 and qty > 0:
                        fee = abs(qty * px) * float(commission_bps) / 10_000.0
                    side = "buy"
                    for o in orders:
                        if not inst or str(o.get("instrument")) == inst:
                            side = str(o.get("side", "buy"))
                            if not inst:
                                inst = str(o.get("instrument", ""))
                            break
                    slip = row.get("slippage", 0.0)
                    if isinstance(slip, Mapping):
                        slip = float(slip.get("slippage", slip.get("cost", 0.0)) or 0.0)
                    else:  # pragma: no cover
                        slip = float(slip or 0.0)
                    normalized.append(
                        {
                            "instrument": inst,
                            "side": side,
                            "quantity": qty,
                            "filled_qty": qty,
                            "price": px,
                            "exec_vwap": px,
                            "fee": fee,
                            "slippage": slip,
                            "fills": fills,
                            "backend": self.backend,
                        }
                    )
                if normalized:
                    return {"orders": normalized, "n": len(normalized), "backend": self.backend}
        except Exception:  # noqa: BLE001
            self.backend = IsolatedExecutionFallback.name

        return IsolatedExecutionFallback.simulate_execution(
            orders,
            market_context=market_context,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            seed=seed,
        )


__all__ = [
    "ExecutionSimulationAdapter",
    "IsolatedExecutionFallback",
    "IsolatedPortfolioFallback",
    "PortfolioConstructionAdapter",
]

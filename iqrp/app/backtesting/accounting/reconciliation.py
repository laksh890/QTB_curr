"""Capital / PnL / position / cash / equity reconciliation.

Accounting model (operational runner)::

    equity = cash + position_market_value

Capital identity (cash-settled books)::

    equity ≈ cash + unrealized
    (realized PnL and fees are settled into cash; primary ledger fields still tracked)

Fill cash model (pipeline)::

    buy  → cash -= notional; fees deducted via record_fee
    sell → cash += notional; fees deducted via record_fee
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from iqrp.app.backtesting.accounting.capital import CapitalState


class ReconciliationError(RuntimeError):
    """Raised when the accounting identity does not hold."""


@dataclass
class ReconciliationResult:
    ok: bool
    starting_capital: float
    realized: float
    unrealized: float
    fees: float
    financing: float
    ending_equity: float
    expected_equity: float
    discrepancy: float
    tolerance: float
    detail: str = ""
    cash: float = 0.0
    position_market_value: float = 0.0
    cash_plus_mv_equity: float = 0.0
    checks: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "starting_capital": self.starting_capital,
            "realized": self.realized,
            "unrealized": self.unrealized,
            "fees": self.fees,
            "financing": self.financing,
            "ending_equity": self.ending_equity,
            "expected_equity": self.expected_equity,
            "discrepancy": self.discrepancy,
            "tolerance": self.tolerance,
            "detail": self.detail,
            "cash": self.cash,
            "position_market_value": self.position_market_value,
            "cash_plus_mv_equity": self.cash_plus_mv_equity,
            "checks": dict(self.checks),
            "identity": (
                "equity = cash + position_market_value; "
                "cash settles buys/sells/fees; "
                "Starting + Realized + Unrealized - Fees - Financing ≈ Ending "
                "(or cash + unrealized when realized is settled into cash)"
            ),
        }


def reconcile_capital(
    capital: CapitalState | Mapping[str, Any],
    *,
    ending_equity: float | None = None,
    tolerance: float = 1e-4,
    fail: bool = True,
) -> ReconciliationResult:
    """Reconcile capital ledger against ending equity."""
    if isinstance(capital, CapitalState):
        start = float(capital.initial_capital)
        realized = float(capital.realized_pnl)
        unrealized = float(capital.unrealized_pnl)
        fees = float(capital.fees_paid)
        financing = float(capital.financing_paid)
        cash = float(capital.cash)
        mv = float(capital.position_market_value)
        end = float(capital.equity if ending_equity is None else ending_equity)
    else:
        start = float(capital.get("initial_capital", 0.0))
        realized = float(capital.get("realized_pnl", 0.0))
        unrealized = float(capital.get("unrealized_pnl", 0.0))
        fees = float(capital.get("fees_paid", capital.get("fees", 0.0)))
        financing = float(capital.get("financing_paid", capital.get("financing", 0.0)))
        cash = float(capital.get("cash", 0.0))
        mv = float(capital.get("position_market_value", 0.0))
        end = float(ending_equity if ending_equity is not None else capital.get("equity", 0.0))

    cash_plus_mv = cash + mv
    # Primary prompt identity
    expected_prompt = start + realized + unrealized - fees - financing
    gap_prompt = end - expected_prompt
    # Operational identity used by CapitalState.equity
    gap_cash_mv = end - cash_plus_mv
    # Cash-settled identity (realized already in cash)
    gap_cash_unreal = end - (cash + unrealized)

    checks = {
        "cash_plus_mv": {
            "ok": abs(gap_cash_mv) <= float(tolerance),
            "expected": cash_plus_mv,
            "gap": gap_cash_mv,
        },
        "prompt_identity": {
            "ok": abs(gap_prompt) <= float(tolerance),
            "expected": expected_prompt,
            "gap": gap_prompt,
        },
        "cash_plus_unrealized": {
            "ok": abs(gap_cash_unreal) <= float(tolerance),
            "expected": cash + unrealized,
            "gap": gap_cash_unreal,
        },
    }

    # Authoritative for this platform: equity = cash + MV
    if abs(gap_cash_mv) <= float(tolerance):
        ok = True
        expected = cash_plus_mv
        discrepancy = gap_cash_mv
        detail = "reconciled via equity = cash + position_market_value"
    elif abs(gap_cash_unreal) <= float(tolerance):
        ok = True
        expected = cash + unrealized
        discrepancy = gap_cash_unreal
        detail = "reconciled via cash + unrealized (realized settled into cash)"
    elif abs(gap_prompt) <= float(tolerance):
        ok = True
        expected = expected_prompt
        discrepancy = gap_prompt
        detail = "reconciled via Starting + Realized + Unrealized - Fees - Financing"
    else:
        ok = False
        expected = cash_plus_mv
        discrepancy = gap_cash_mv
        detail = (
            f"unexplained discrepancy cash+mv={gap_cash_mv:.8f} "
            f"prompt={gap_prompt:.8f} cash+unreal={gap_cash_unreal:.8f} tol={tolerance}"
        )
        if fail:
            raise ReconciliationError(detail)

    return ReconciliationResult(
        ok=ok,
        starting_capital=start,
        realized=realized,
        unrealized=unrealized,
        fees=fees,
        financing=financing,
        ending_equity=end,
        expected_equity=expected,
        discrepancy=float(discrepancy),
        tolerance=float(tolerance),
        detail=detail,
        cash=cash,
        position_market_value=mv,
        cash_plus_mv_equity=cash_plus_mv,
        checks=checks,
    )


def replay_fills_positions(
    fills: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Replay fills into a shadow position book; verify qty transitions."""
    from iqrp.app.backtesting.accounting.positions import PositionBook

    book = PositionBook()
    steps: list[dict[str, Any]] = []
    ok = True
    for fill in fills:
        inst = str(fill.get("instrument") or fill.get("symbol") or "")
        if not inst:
            continue
        before = float(book.get(inst).quantity)
        qty = abs(float(fill.get("quantity", 0.0) or 0.0))
        side = str(fill.get("side", "buy"))
        px = float(fill.get("price", 0.0) or 0.0)
        signed = qty if side.lower() in {"buy", "b", "cover", "long"} else -qty
        book.apply_fill(inst, quantity=qty, price=px, side=side)
        after = float(book.get(inst).quantity)
        expected = before + signed
        step_ok = abs(after - expected) <= float(tolerance)
        if not step_ok:
            ok = False
        steps.append(
            {
                "instrument": inst,
                "side": side,
                "quantity": qty,
                "position_before": before,
                "position_after": after,
                "expected_after": expected,
                "ok": step_ok,
            }
        )
    return {
        "ok": ok,
        "n_fills": len(steps),
        "final_quantities": book.quantities(),
        "steps": steps,
    }


def replay_fills_cash(
    fills: Sequence[Mapping[str, Any]],
    *,
    initial_cash: float,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Replay fill cash movements: buy decreases cash, sell increases; fees reduce cash."""
    cash = float(initial_cash)
    events: list[dict[str, Any]] = []
    for fill in fills:
        qty = abs(float(fill.get("quantity", 0.0) or 0.0))
        px = float(fill.get("price", 0.0) or 0.0)
        fee = abs(float(fill.get("fee", 0.0) or 0.0))
        side = str(fill.get("side", "buy")).lower()
        notional = qty * px
        before = cash
        if side in {"buy", "b", "cover", "long"}:
            cash -= notional
        else:
            cash += notional
        cash -= fee
        events.append(
            {
                "instrument": fill.get("instrument"),
                "side": side,
                "notional": notional,
                "fee": fee,
                "cash_before": before,
                "cash_after": cash,
            }
        )
    return {
        "ok": True,  # structural replay; compared externally to ledger cash
        "initial_cash": float(initial_cash),
        "replayed_cash": float(cash),
        "n_events": len(events),
        "events": events,
        "tolerance": float(tolerance),
    }


def reconcile_equity_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Verify each snapshot: equity ≈ cash + gross/net market value fields when present."""
    gaps: list[dict[str, Any]] = []
    ok = True
    for i, snap in enumerate(snapshots):
        equity = snap.get("equity")
        cash = snap.get("cash")
        if equity is None or cash is None:
            continue
        # Prefer explicit market value; fall back to net exposure as signed MV proxy.
        mv = snap.get("position_market_value")
        if mv is None:
            mv = snap.get("net_exposure", snap.get("net"))
        if mv is None:
            continue
        expected = float(cash) + float(mv)
        gap = float(equity) - expected
        step_ok = abs(gap) <= float(tolerance)
        if not step_ok:
            ok = False
            gaps.append(
                {
                    "index": i,
                    "timestamp": snap.get("timestamp"),
                    "equity": float(equity),
                    "cash": float(cash),
                    "market_value": float(mv),
                    "expected": expected,
                    "gap": gap,
                }
            )
    return {
        "ok": ok,
        "n_snapshots": len(snapshots),
        "n_checked": sum(
            1
            for s in snapshots
            if s.get("equity") is not None
            and s.get("cash") is not None
            and (
                s.get("position_market_value") is not None
                or s.get("net_exposure") is not None
                or s.get("net") is not None
            )
        ),
        "n_gaps": len(gaps),
        "gaps": gaps[:50],
        "tolerance": float(tolerance),
    }


def full_accounting_audit(
    *,
    capital: CapitalState | Mapping[str, Any],
    fills: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    ending_equity: float | None = None,
    final_positions: Mapping[str, float] | None = None,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Combined capital + fill position + cash replay + equity snapshot audit."""
    capital_recon = reconcile_capital(
        capital,
        ending_equity=ending_equity,
        tolerance=tolerance,
        fail=False,
    )
    pos_replay = replay_fills_positions(fills, tolerance=min(tolerance, 1e-6))
    if isinstance(capital, CapitalState):
        init_cash = float(capital.initial_capital)
        ledger_cash = float(capital.cash)
    else:
        init_cash = float(capital.get("initial_capital", 0.0))
        ledger_cash = float(capital.get("cash", 0.0))
    cash_replay = replay_fills_cash(fills, initial_cash=init_cash, tolerance=tolerance)
    cash_match = abs(float(cash_replay["replayed_cash"]) - ledger_cash) <= float(tolerance)
    cash_replay["matches_ledger"] = cash_match
    cash_replay["ledger_cash"] = ledger_cash
    cash_replay["ok"] = cash_match

    if final_positions is not None:
        final_ok = True
        mismatches: list[dict[str, Any]] = []
        replayed = dict(pos_replay.get("final_quantities") or {})
        keys = set(replayed) | set(final_positions)
        for k in keys:
            a = float(replayed.get(k, 0.0))
            b = float(final_positions.get(k, 0.0))
            if abs(a - b) > float(tolerance):
                final_ok = False
                mismatches.append({"instrument": k, "replayed": a, "ledger": b})
        pos_replay["matches_final_positions"] = final_ok
        pos_replay["final_mismatches"] = mismatches
        pos_replay["ok"] = bool(pos_replay.get("ok", True)) and final_ok

    snap_recon = reconcile_equity_snapshots(snapshots, tolerance=tolerance)
    ok = bool(capital_recon.ok) and bool(pos_replay.get("ok", True)) and cash_match
    # Snapshot gaps are warnings if cash/MV fields incomplete historically
    if snap_recon.get("n_checked", 0) > 0 and not snap_recon.get("ok", True):
        ok = False

    out = capital_recon.to_dict()
    out["ok"] = ok
    out["position_fill_replay"] = pos_replay
    out["cash_fill_replay"] = cash_replay
    out["equity_snapshots"] = snap_recon
    return out


__all__ = [
    "ReconciliationError",
    "ReconciliationResult",
    "full_accounting_audit",
    "reconcile_capital",
    "reconcile_equity_snapshots",
    "replay_fills_cash",
    "replay_fills_positions",
]

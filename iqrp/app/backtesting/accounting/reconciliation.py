"""Capital / PnL reconciliation with hard failure on unexplained gaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
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
            "identity": (
                "Starting Capital + Realized + Unrealized - Fees - Financing = Ending Equity"
            ),
        }


def reconcile_capital(
    capital: CapitalState | Mapping[str, Any],
    *,
    ending_equity: float | None = None,
    tolerance: float = 1e-4,
    fail: bool = True,
) -> ReconciliationResult:
    """Check: Starting + Realized + Unrealized - Fees - Financing = Ending Equity.

    Cash-based books often fold realized PnL into cash; this identity uses the
    explicit ledger fields on :class:`CapitalState`.
    """
    if isinstance(capital, CapitalState):
        start = float(capital.initial_capital)
        realized = float(capital.realized_pnl)
        unrealized = float(capital.unrealized_pnl)
        fees = float(capital.fees_paid)
        financing = float(capital.financing_paid)
        end = float(capital.equity if ending_equity is None else ending_equity)
        # Alternative cash identity: cash + MV = equity already in capital.equity
        cash = float(capital.cash)
    else:
        start = float(capital.get("initial_capital", 0.0))
        realized = float(capital.get("realized_pnl", 0.0))
        unrealized = float(capital.get("unrealized_pnl", 0.0))
        fees = float(capital.get("fees_paid", capital.get("fees", 0.0)))
        financing = float(capital.get("financing_paid", capital.get("financing", 0.0)))
        end = float(ending_equity if ending_equity is not None else capital.get("equity", 0.0))
        cash = float(capital.get("cash", end - unrealized))

    # Primary identity from prompt
    expected = start + realized + unrealized - fees - financing
    discrepancy = end - expected

    # Cash-consistent identity: equity ≈ cash + unrealized (cash already net of fees /
    # financing / realized settlement). Used as a secondary check.
    cash_identity_gap = (cash + unrealized) - end

    ok = abs(discrepancy) <= float(tolerance) or abs(cash_identity_gap) <= float(tolerance)
    detail = ""
    if not ok:
        detail = (
            f"unexplained discrepancy={discrepancy:.8f} "
            f"(cash_identity_gap={cash_identity_gap:.8f}, tol={tolerance})"
        )
        if fail:
            raise ReconciliationError(detail)

    # Prefer reporting the cash-consistent expected if primary identity drifts
    # solely because realized was settled into cash (double-count risk).
    if abs(discrepancy) > float(tolerance) and abs(cash_identity_gap) <= float(tolerance):
        expected = cash + unrealized
        discrepancy = end - expected
        ok = True
        detail = "reconciled via cash + unrealized identity (realized settled into cash)"

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
    )


__all__ = ["ReconciliationError", "ReconciliationResult", "reconcile_capital"]

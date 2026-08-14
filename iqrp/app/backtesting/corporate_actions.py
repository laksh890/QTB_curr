"""Corporate-action adjustments for point-in-time backtests.

Supports splits, dividends, mergers, delistings, and symbol changes.

CRITICAL: Only actions with ``ex_date <= asof`` (or effective timestamp
``<= event.timestamp``) may be applied. Future corporate actions are
look-ahead and must be excluded via :func:`actions_asof`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from iqrp.app.backtesting.pit import LookaheadViolation, assert_no_lookahead


class CorporateActionType(str, Enum):
    SPLIT = "SPLIT"
    DIVIDEND = "DIVIDEND"
    MERGER = "MERGER"
    DELISTING = "DELISTING"
    SYMBOL_CHANGE = "SYMBOL_CHANGE"
    OTHER = "OTHER"


@dataclass(slots=True)
class CorporateAction:
    """A single corporate action with an effective / ex-date."""

    action_type: CorporateActionType
    symbol: str
    ex_date: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    action_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action_type, CorporateActionType):
            object.__setattr__(
                self,
                "action_type",
                CorporateActionType(str(self.action_type)),
            )
        if self.ex_date.tzinfo is None:
            raise ValueError(f"CorporateAction.ex_date must be timezone-aware ({self.symbol})")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "symbol", str(self.symbol))


def actions_asof(
    actions: Sequence[CorporateAction],
    asof: datetime,
) -> list[CorporateAction]:
    """Return corporate actions with ``ex_date <= asof`` (PIT filter)."""
    assert_no_lookahead(asof, asof)  # validates tz-awareness
    return [a for a in actions if a.ex_date <= asof]


def adjust_price_for_split(price: float, ratio: float) -> float:
    """Adjust a pre-split price to post-split terms.

    ``ratio`` is new/old shares (e.g. 2.0 for a 2-for-1 split).
    Historical prices are divided by the ratio so series stay continuous.
    """
    if ratio <= 0:
        raise ValueError(f"split ratio must be positive, got {ratio}")
    return float(price) / float(ratio)


def adjust_quantity_for_split(quantity: float, ratio: float) -> float:
    """Adjust a share quantity across a split (multiply by ratio)."""
    if ratio <= 0:
        raise ValueError(f"split ratio must be positive, got {ratio}")
    return float(quantity) * float(ratio)


def adjust_price_for_dividend(
    price: float,
    dividend: float,
    *,
    method: str = "subtract",
) -> float:
    """Cash-dividend price adjustment.

    Parameters
    ----------
    method:
        ``subtract`` — classic close-to-close adjustment (price - dividend).
        ``factor`` — multiplicative ``price * (price - dividend) / price``.
    """
    px = float(price)
    div = float(dividend)
    if method == "subtract":
        return px - div
    if method == "factor":
        if px == 0:
            raise ValueError("cannot apply factor dividend adjustment to zero price")
        return px * ((px - div) / px)
    raise ValueError(f"unsupported dividend method: {method!r}")


@dataclass(slots=True)
class PositionState:
    """Minimal position state mutated by corporate-action application."""

    symbol: str
    quantity: float
    cash: float = 0.0
    cost_basis: float | None = None


@dataclass(slots=True)
class AdjustmentResult:
    """Outcome of applying corporate actions to a position book."""

    positions: dict[str, PositionState]
    cash_delta: float
    applied: list[CorporateAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def apply_corporate_actions(
    positions: Mapping[str, PositionState] | Mapping[str, float],
    actions: Sequence[CorporateAction],
    *,
    asof: datetime,
    cash: float = 0.0,
    enforce_pit: bool = True,
) -> AdjustmentResult:
    """Apply all PIT-eligible corporate actions to positions.

    Handles
    -------
    * **SPLIT** — payload ``ratio`` (new/old); adjusts quantity (and optional
      cost basis).
    * **DIVIDEND** — payload ``amount`` per share; credits cash.
    * **MERGER** — payload ``new_symbol``, optional ``exchange_ratio``
      (default 1.0); rolls quantity into the survivor.
    * **DELISTING** — payload optional ``liquidation_price``; closes position
      to cash when provided, otherwise zeroes quantity.
    * **SYMBOL_CHANGE** — payload ``new_symbol``; renames the position key.
    """
    book: dict[str, PositionState] = {}
    for sym, pos in positions.items():
        if isinstance(pos, PositionState):
            book[sym] = PositionState(
                symbol=pos.symbol,
                quantity=float(pos.quantity),
                cash=float(pos.cash),
                cost_basis=pos.cost_basis,
            )
        else:
            book[str(sym)] = PositionState(symbol=str(sym), quantity=float(pos))

    eligible = actions_asof(list(actions), asof)
    # Sort for determinism: ex_date, action_type value, symbol, action_id
    eligible.sort(key=lambda a: (a.ex_date, a.action_type.value, a.symbol, a.action_id))

    cash_delta = 0.0
    notes: list[str] = []
    applied: list[CorporateAction] = []

    for action in eligible:
        if enforce_pit:
            assert_no_lookahead(action.ex_date, asof, context=f"ca:{action.action_type.value}")

        pos = book.get(action.symbol)
        if pos is None and action.action_type not in {
            CorporateActionType.SYMBOL_CHANGE,
            CorporateActionType.MERGER,
        }:
            # No position in this symbol — still record for audit when relevant.
            notes.append(f"skip {action.action_type.value} {action.symbol}: no position")
            applied.append(action)
            continue

        if action.action_type is CorporateActionType.SPLIT:
            ratio = float(action.payload.get("ratio", action.payload.get("split_ratio", 0)))
            if ratio <= 0:
                raise ValueError(f"SPLIT requires positive ratio: {action}")
            if pos is not None:
                pos.quantity = adjust_quantity_for_split(pos.quantity, ratio)
                if pos.cost_basis is not None:
                    pos.cost_basis = adjust_price_for_split(pos.cost_basis, ratio)
            notes.append(f"split {action.symbol} ratio={ratio}")

        elif action.action_type is CorporateActionType.DIVIDEND:
            amount = float(action.payload.get("amount", action.payload.get("dividend", 0.0)))
            qty = 0.0 if pos is None else pos.quantity
            credit = amount * qty
            cash_delta += credit
            notes.append(f"dividend {action.symbol} amount={amount} credit={credit}")

        elif action.action_type is CorporateActionType.MERGER:
            if pos is None:
                notes.append(f"skip merger {action.symbol}: no position")
                applied.append(action)
                continue
            new_symbol = str(action.payload["new_symbol"])
            ratio = float(action.payload.get("exchange_ratio", 1.0))
            converted = pos.quantity * ratio
            survivor = book.get(new_symbol)
            if survivor is None:
                book[new_symbol] = PositionState(
                    symbol=new_symbol,
                    quantity=converted,
                    cost_basis=pos.cost_basis,
                )
            else:
                survivor.quantity += converted
            del book[action.symbol]
            notes.append(f"merger {action.symbol}->{new_symbol} ratio={ratio} qty={converted}")

        elif action.action_type is CorporateActionType.DELISTING:
            if pos is None:
                notes.append(f"skip delisting {action.symbol}: no position")
                applied.append(action)
                continue
            liq = action.payload.get("liquidation_price")
            if liq is not None:
                proceeds = float(liq) * pos.quantity
                cash_delta += proceeds
                notes.append(f"delist {action.symbol} liquidated @ {liq} proceeds={proceeds}")
            else:
                notes.append(f"delist {action.symbol} quantity zeroed (no liq price)")
            del book[action.symbol]

        elif action.action_type is CorporateActionType.SYMBOL_CHANGE:
            if pos is None:
                notes.append(f"skip symbol_change {action.symbol}: no position")
                applied.append(action)
                continue
            new_symbol = str(action.payload["new_symbol"])
            if new_symbol in book and new_symbol != action.symbol:
                book[new_symbol].quantity += pos.quantity
            else:
                pos.symbol = new_symbol
                book[new_symbol] = pos
            if new_symbol != action.symbol:
                del book[action.symbol]
            notes.append(f"symbol_change {action.symbol}->{new_symbol}")

        elif action.action_type is CorporateActionType.OTHER:
            notes.append(f"other action on {action.symbol}: {action.payload}")

        else:  # pragma: no cover
            raise LookaheadViolation(f"unhandled corporate action: {action.action_type}")

        applied.append(action)

    # Attach cash to a synthetic ledger note; caller owns cash account.
    _ = cash  # reserved for future cash-book integration
    return AdjustmentResult(
        positions=book,
        cash_delta=cash_delta,
        applied=applied,
        notes=notes,
    )


def build_action(
    action_type: CorporateActionType | str,
    symbol: str,
    ex_date: datetime,
    **payload: Any,
) -> CorporateAction:
    """Convenience constructor for :class:`CorporateAction`."""
    return CorporateAction(
        action_type=CorporateActionType(str(action_type)),
        symbol=symbol,
        ex_date=ex_date,
        payload=payload,
    )


def cumulative_split_factor(
    actions: Sequence[CorporateAction],
    symbol: str,
    *,
    asof: datetime,
) -> float:
    """Product of split ratios for ``symbol`` with ``ex_date <= asof``."""
    factor = 1.0
    for action in actions_asof(actions, asof):
        if action.symbol != symbol:
            continue
        if action.action_type is CorporateActionType.SPLIT:
            factor *= float(action.payload.get("ratio", action.payload.get("split_ratio", 1.0)))
    return factor


__all__ = [
    "AdjustmentResult",
    "CorporateAction",
    "CorporateActionType",
    "PositionState",
    "actions_asof",
    "adjust_price_for_dividend",
    "adjust_price_for_split",
    "adjust_quantity_for_split",
    "apply_corporate_actions",
    "build_action",
    "cumulative_split_factor",
]

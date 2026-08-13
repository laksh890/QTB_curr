"""Reference buy-and-hold strategy (not a performance claim)."""

from __future__ import annotations

from typing import Any, Mapping

from iqrp.app.backtesting.strategy.base import Strategy


class BuyAndHoldStrategy(Strategy):
    """Target full investment on the first bar; hold thereafter.

    Modes
    -----
    ``equal_weight``
        Equal weight across the current universe / observed instruments.
    ``first_instrument``
        100% weight in the first instrument (sorted name).

    This is a reference configuration for pipeline smoke tests. It does not
    imply profitability.
    """

    strategy_id = "buy_and_hold"
    strategy_version = "1.0.0"

    def __init__(self, *, mode: str = "equal_weight") -> None:
        mode_l = str(mode or "equal_weight").strip().lower()
        if mode_l not in {"equal_weight", "first_instrument"}:
            raise ValueError("mode must be 'equal_weight' or 'first_instrument'")
        self.mode = mode_l
        self._entered = False
        self._target_weights: dict[str, float] = {}

    def initialize(self, context: Any) -> None:
        self._entered = False
        self._target_weights = {}
        context.strategy_state = {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "mode": self.mode,
            "entered": False,
        }

    def _instruments(self, context: Any) -> list[str]:
        universe = list(getattr(context, "universe", None) or [])
        if universe:
            return [str(x) for x in universe]
        prices = dict(getattr(context, "latest_prices", {}) or {})
        return sorted(str(k) for k in prices.keys())

    def _build_targets(self, instruments: list[str]) -> dict[str, float]:
        if not instruments:
            return {}
        if self.mode == "first_instrument":
            return {instruments[0]: 1.0}
        w = 1.0 / float(len(instruments))
        return {inst: w for inst in instruments}

    def on_market_data(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        # Eager entry on first market observation so orders/fills occur early.
        if self._entered:
            return {"target_weights": dict(self._target_weights), "rebalance": False}
        instruments = self._instruments(context)
        if not instruments:
            bars = dict((event.payload or {}).get("bars") or {})
            instruments = sorted(str(k) for k in bars.keys())
            if not instruments:
                sym = (event.payload or {}).get("instrument") or (event.payload or {}).get("symbol")
                if sym:
                    instruments = [str(sym)]
        targets = self._build_targets(instruments)
        if not targets:
            return None
        self._entered = True
        self._target_weights = dict(targets)
        context.strategy_state = {
            **dict(getattr(context, "strategy_state", {}) or {}),
            "entered": True,
            "target_weights": dict(targets),
        }
        return {
            "signals": dict(targets),
            "target_weights": dict(targets),
            "rebalance": True,
            "reason": "buy_and_hold_entry",
        }

    def on_features(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        if not self._entered:
            return self.on_market_data(event, context)
        # Keep rebalance True until the book actually holds the target (first fill).
        holding = False
        positions = getattr(context, "positions", None)
        if positions is not None and hasattr(positions, "open_instruments"):
            holding = bool(positions.open_instruments())
        return {
            "signals": dict(self._target_weights),
            "target_weights": dict(self._target_weights),
            "rebalance": (not holding),
        }

    def on_signal(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        if self._target_weights:
            return {"signals": dict(self._target_weights), "target_weights": dict(self._target_weights)}
        return None

    def on_end(self, context: Any) -> Mapping[str, Any] | None:
        return {
            "strategy_id": self.strategy_id,
            "entered": self._entered,
            "final_targets": dict(self._target_weights),
        }


__all__ = ["BuyAndHoldStrategy"]

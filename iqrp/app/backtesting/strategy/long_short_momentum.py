"""Long/short research strategy for short-horizon multi-trade sessions.

Transitions LONG / SHORT / FLAT pass through the existing runner cascade
(risk → portfolio → order → execution → fill). This is not a parallel
execution system.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from iqrp.app.backtesting.strategy.base import Strategy


class LongShortMomentumStrategy(Strategy):
    """Repeated long/short/flat opportunities within a session.

    Parameters
    ----------
    lookback:
        Bars for momentum sign.
    holding_bars:
        Hold each signal for N bars before allowing a new entry.
    allow_short:
        If False, short signals become FLAT.
    cooldown_bars:
        Optional bars to remain flat after an exit (explicit re-entry rule).
    position_size:
        Absolute target weight magnitude.
    """

    strategy_id = "long_short_momentum"
    strategy_version = "1.0.0"

    def __init__(
        self,
        *,
        lookback: int = 1,
        holding_bars: int = 1,
        allow_short: bool = True,
        cooldown_bars: int = 0,
        position_size: float = 1.0,
        instrument: str | None = None,
    ) -> None:
        self.lookback = max(int(lookback), 1)
        self.holding_bars = max(int(holding_bars), 1)
        self.allow_short = bool(allow_short)
        self.cooldown_bars = max(int(cooldown_bars), 0)
        self.position_size = float(position_size)
        self.instrument = instrument
        self._history: list[float] = []
        self._side = 0  # -1 short, 0 flat, +1 long
        self._held = 0
        self._cooldown = 0
        self._trade_log: list[dict[str, Any]] = []

    def initialize(self, context: Any) -> None:
        self._history = []
        self._side = 0
        self._held = 0
        self._cooldown = 0
        self._trade_log = []
        context.strategy_state = {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "side": "FLAT",
            "allow_short": self.allow_short,
        }

    def _pick_instrument(self, context: Any, event: Any) -> str | None:
        if self.instrument:
            return self.instrument
        universe = list(getattr(context, "universe", None) or [])
        if universe:
            return str(universe[0])
        prices = dict(getattr(context, "latest_prices", {}) or {})
        if prices:
            return sorted(prices)[0]
        bars = dict((event.payload or {}).get("bars") or {})
        if bars:
            return sorted(str(k) for k in bars)[0]
        return None

    def _desired_side(self) -> int:
        if len(self._history) <= self.lookback:
            return 0
        past = self._history[-(self.lookback + 1)]
        now = self._history[-1]
        if past <= 0:
            return 0
        sgn = 1 if now / past - 1.0 > 0 else (-1 if now / past - 1.0 < 0 else 0)
        if sgn < 0 and not self.allow_short:
            return 0
        return int(sgn)

    def on_features(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        inst = self._pick_instrument(context, event)
        if not inst:
            return None
        px = float(getattr(context, "latest_prices", {}).get(inst, 0.0) or 0.0)
        if px <= 0:
            bars = dict((event.payload or {}).get("bars") or {})
            bar = bars.get(inst) or {}
            px = float(bar.get("close", 0.0) or 0.0)
        if px <= 0:
            return None
        self._history.append(px)

        if self._cooldown > 0:
            self._cooldown -= 1
            self._side = 0
            self._held = 0
        elif self._held > 0:
            self._held -= 1
            if self._held == 0:
                # exit to flat then optional cooldown
                prev = self._side
                self._side = 0
                if self.cooldown_bars:
                    self._cooldown = self.cooldown_bars
                self._trade_log.append({"action": "exit", "from": prev})
        else:
            desired = self._desired_side()
            if desired != self._side:
                # enter / reverse
                self._trade_log.append(
                    {
                        "action": "reverse" if self._side and desired else "enter",
                        "from": self._side,
                        "to": desired,
                    }
                )
                self._side = desired
                self._held = self.holding_bars if desired != 0 else 0

        weight = float(self._side) * self.position_size
        side_name = "LONG" if self._side > 0 else ("SHORT" if self._side < 0 else "FLAT")
        context.strategy_state = {
            **dict(getattr(context, "strategy_state", {}) or {}),
            "side": side_name,
            "instrument": inst,
            "n_transitions": len(self._trade_log),
        }
        targets = {inst: weight} if abs(weight) > 1e-15 else {inst: 0.0}
        return {
            "signals": {inst: float(self._side)},
            "target_weights": targets,
            "rebalance": True,
            "side": side_name,
            "reason": "long_short_momentum",
        }

    def on_signal(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        st = dict(getattr(context, "strategy_state", {}) or {})
        inst = st.get("instrument")
        if not inst:
            return None
        side = st.get("side", "FLAT")
        w = self.position_size if side == "LONG" else (-self.position_size if side == "SHORT" else 0.0)
        return {"signals": {inst: w}, "target_weights": {inst: w}, "side": side}

    def on_end(self, context: Any) -> Mapping[str, Any] | None:
        return {
            "strategy_id": self.strategy_id,
            "transitions": list(self._trade_log),
            "final_side": dict(getattr(context, "strategy_state", {}) or {}).get("side"),
        }


__all__ = ["LongShortMomentumStrategy"]

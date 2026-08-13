"""Simple cross-sectional momentum reference strategy (demo only)."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping

from iqrp.app.backtesting.strategy.base import Strategy


class CrossSectionalMomentumStrategy(Strategy):
    """Lookback return rank → long top names (equal-weight among selected).

    Reference / demonstration configuration only. No profitability claim.
    """

    strategy_id = "cross_sectional_momentum"
    strategy_version = "1.0.0"

    def __init__(
        self,
        *,
        lookback: int = 20,
        top_n: int | None = None,
        long_only: bool = True,
    ) -> None:
        self.lookback = max(int(lookback), 2)
        self.top_n = None if top_n is None else max(int(top_n), 1)
        self.long_only = bool(long_only)
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.lookback + 1))
        self._last_targets: dict[str, float] = {}

    def initialize(self, context: Any) -> None:
        self._history.clear()
        self._last_targets = {}
        context.strategy_state = {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "lookback": self.lookback,
            "top_n": self.top_n,
        }

    def _update_history(self, prices: Mapping[str, float]) -> None:
        for inst, px in prices.items():
            if px is None:
                continue
            p = float(px)
            if p > 0:
                self._history[str(inst)].append(p)

    def _scores(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for inst, series in self._history.items():
            if len(series) < self.lookback:
                continue
            start = float(series[0])
            end = float(series[-1])
            if start <= 0:
                continue
            out[inst] = end / start - 1.0
        return out

    def _targets_from_scores(self, scores: Mapping[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        n = self.top_n if self.top_n is not None else max(1, len(ranked) // 2) or 1
        selected = ranked[: min(n, len(ranked))]
        if not selected:
            return {}
        if self.long_only:
            w = 1.0 / float(len(selected))
            return {inst: w for inst, _ in selected}
        # Long top half / short bottom half among scored names
        mid = max(1, len(ranked) // 2)
        longs = ranked[:mid]
        shorts = ranked[-mid:]
        lw = 0.5 / float(len(longs))
        sw = -0.5 / float(len(shorts))
        targets = {inst: lw for inst, _ in longs}
        for inst, _ in shorts:
            targets[inst] = targets.get(inst, 0.0) + sw
        return targets

    def on_features(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        prices = dict(getattr(context, "latest_prices", {}) or {})
        bars = dict((event.payload or {}).get("bars") or {})
        for inst, bar in bars.items():
            if isinstance(bar, Mapping) and bar.get("close") is not None:
                prices[str(inst)] = float(bar["close"])
        self._update_history(prices)
        scores = self._scores()
        if not scores:
            # Warm-up: stay flat until lookback is available
            return {"signals": {}, "target_weights": {}, "rebalance": False, "warmup": True}
        targets = self._targets_from_scores(scores)
        self._last_targets = dict(targets)
        return {
            "signals": dict(scores),
            "target_weights": dict(targets),
            "rebalance": True,
            "momentum_scores": dict(scores),
        }

    def on_signal(self, event: Any, context: Any) -> Mapping[str, Any] | None:
        if self._last_targets:
            return {"target_weights": dict(self._last_targets)}
        return None

    def on_end(self, context: Any) -> Mapping[str, Any] | None:
        return {"final_targets": dict(self._last_targets), "n_history": len(self._history)}


__all__ = ["CrossSectionalMomentumStrategy"]

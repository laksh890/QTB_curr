"""Streaming risk monitor with snapshot export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.risk.base import LimitBreach, RiskMeasure, RiskState
from iqrp.app.risk.tail.drawdown import drawdown_state, max_drawdown
from iqrp.app.risk.tail.var import historical_var


@dataclass
class RiskMonitor:
    """Accumulate returns and risk metrics; emit to_dict-friendly snapshots."""

    caution: float = 0.05
    reduced_risk: float = 0.10
    capital_preservation: float = 0.15
    trading_halt: float = 0.20
    var_confidence: float = 0.95
    _returns: list[float] = field(default_factory=list, repr=False)
    _measures: dict[str, Any] = field(default_factory=dict, repr=False)
    _breaches: list[LimitBreach] = field(default_factory=list, repr=False)
    risk_state: RiskState = RiskState.NORMAL

    def update(
        self,
        ret: float | None = None,
        *,
        measures: dict[str, Any] | None = None,
        breaches: list[LimitBreach] | None = None,
    ) -> RiskState:
        """Ingest a new return and/or external measures/breaches."""
        if ret is not None and np.isfinite(ret):
            self._returns.append(float(ret))

        if measures:
            for k, v in measures.items():
                if isinstance(v, RiskMeasure):
                    self._measures[k] = v.to_dict()
                elif isinstance(v, dict):
                    self._measures[k] = dict(v)
                else:
                    self._measures[k] = {"name": k, "value": float(v)}

        if breaches:
            self._breaches.extend(breaches)

        if self._returns:
            dd = drawdown_state(
                self._returns,
                caution=self.caution,
                reduced_risk=self.reduced_risk,
                capital_preservation=self.capital_preservation,
                trading_halt=self.trading_halt,
            )
            self.risk_state = RiskState(dd["risk_state"])
            self._measures["drawdown"] = dd
            self._measures["var"] = historical_var(
                self._returns, confidence=self.var_confidence
            ).to_dict()
            self._measures["max_drawdown"] = max_drawdown(self._returns).to_dict()

        # Escalate state on hard breaches
        for b in self._breaches:
            if b.severity.value == "HARD":
                if self.risk_state == RiskState.NORMAL:
                    self.risk_state = RiskState.CAUTION
                if "drawdown" in b.limit_name.lower() or "loss" in b.limit_name.lower():
                    self.risk_state = RiskState.CAPITAL_PRESERVATION

        return self.risk_state

    def snapshot(self) -> dict[str, Any]:
        """Current monitor snapshot."""
        return {
            "name": "risk_monitor_snapshot",
            "risk_state": self.risk_state.value,
            "n_obs": len(self._returns),
            "last_return": self._returns[-1] if self._returns else None,
            "measures": dict(self._measures),
            "breaches": [b.to_dict() for b in self._breaches],
            "thresholds": {
                "caution": self.caution,
                "reduced_risk": self.reduced_risk,
                "capital_preservation": self.capital_preservation,
                "trading_halt": self.trading_halt,
            },
        }

    def reset(self) -> None:
        self._returns.clear()
        self._measures.clear()
        self._breaches.clear()
        self.risk_state = RiskState.NORMAL

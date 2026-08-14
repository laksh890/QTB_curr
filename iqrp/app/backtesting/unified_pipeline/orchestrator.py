"""Unified Trading Orchestrator — AlphaCandidate → Risk → Portfolio → Execution.

Deterministic research/backtest cascade. Does not optimize or claim profitability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.backtesting.accounting import (
    CapitalState,
    FillLog,
    OrderLog,
    PositionBook,
    SnapshotBook,
    TradeLedger,
)
from iqrp.app.backtesting.accounting.snapshots import PortfolioSnapshot
from iqrp.app.backtesting.unified_pipeline.candidate import validate_candidate
from iqrp.app.backtesting.unified_pipeline.execution_bridge import (
    apply_fills_to_ledgers,
    build_execution_engine,
    plan_and_execute,
    reconcile_pipeline_state,
)
from iqrp.app.backtesting.unified_pipeline.portfolio_gate import (
    apply_portfolio_constraints,
    weight_to_quantity,
    weights_to_target_object,
)
from iqrp.app.backtesting.unified_pipeline.risk_gate import (
    default_risk_engine,
    evaluate_candidate_risk,
    size_approved_exposure,
)
from iqrp.app.backtesting.unified_pipeline.types import (
    AlphaCandidate,
    LineageRecord,
    StageOutcome,
)
from iqrp.app.execution import ExecutionEngine
from iqrp.app.portfolio.config import PortfolioSettings
from iqrp.app.risk import RiskIntelligenceEngine


DISCLAIMER = (
    "UNIFIED TRADING PIPELINE — integration validation only. "
    "PIPELINE WORKS ≠ STRATEGY WORKS ≠ PROFITABLE ≠ ROBUST ≠ PRODUCTION READY."
)


@dataclass
class UnifiedPipelineState:
    """Mutable portfolio/risk state visible to subsequent candidates."""

    capital: CapitalState
    positions: PositionBook
    order_log: OrderLog = field(default_factory=OrderLog)
    fill_log: FillLog = field(default_factory=FillLog)
    trade_ledger: TradeLedger = field(default_factory=TradeLedger)
    snapshots: SnapshotBook = field(default_factory=SnapshotBook)
    target_weights: dict[str, float] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    seen_candidate_ids: set[str] = field(default_factory=set)
    stage_log: list[dict[str, Any]] = field(default_factory=list)
    lineage: list[dict[str, Any]] = field(default_factory=list)

    def weight_map(self) -> dict[str, float]:
        """Current weights from position market values / equity."""
        eq = max(float(self.capital.equity), 1e-12)
        out: dict[str, float] = {}
        for inst, qty in self.positions.quantities().items():
            px = float(self.prices.get(inst, 0.0) or 0.0)
            out[inst] = float(qty) * px / eq
        # overlay explicit targets when flat book but pending
        for k, v in self.target_weights.items():
            out.setdefault(k, float(v))
        return out

    def mark(self, prices: dict[str, float]) -> None:
        self.prices.update({k: float(v) for k, v in prices.items()})
        self.positions.mark_all(self.prices)
        self.capital.mark_unrealized(
            self.positions.total_unrealized(),
            market_value=self.positions.total_market_value(),
        )


class UnifiedTradingOrchestrator:
    """Process one or many AlphaCandidates against shared portfolio state."""

    def __init__(
        self,
        *,
        initial_capital: float = 1_000_000.0,
        risk_engine: RiskIntelligenceEngine | None = None,
        execution_engine: ExecutionEngine | None = None,
        portfolio_settings: PortfolioSettings | None = None,
        base_returns: np.ndarray | None = None,
        long_only: bool = False,
        max_position: float = 0.10,
        max_gross: float = 1.5,
    ) -> None:
        self.risk = risk_engine or default_risk_engine(max_position=max_position)
        self.execution = execution_engine or build_execution_engine()
        base_ps = portfolio_settings or PortfolioSettings.default()
        self.portfolio_settings = base_ps.model_copy(update={"long_only": bool(long_only)})
        self.long_only = bool(long_only)
        self.max_position = float(max_position)
        self.max_gross = float(max_gross)
        self.returns = (
            np.asarray(base_returns, dtype=np.float64)
            if base_returns is not None
            else np.random.default_rng(0).normal(0, 0.01, 120)
        )
        self.state = UnifiedPipelineState(
            capital=CapitalState(initial_capital=float(initial_capital)),
            positions=PositionBook(),
        )

    def process_candidates(
        self,
        candidates: list[AlphaCandidate],
        *,
        asof: str,
        prices: dict[str, float],
        returns: np.ndarray | None = None,
        simulation_mode: str = "fill",
    ) -> dict[str, Any]:
        """Deterministic per-timestamp cascade for multiple candidates.

        Order: validate → risk → sizing → portfolio → orders → execute → account → feedback.
        Candidates sorted by candidate_id for determinism.
        """
        rets = np.asarray(returns if returns is not None else self.returns, dtype=np.float64)
        self.state.mark(prices)
        ordered = sorted(candidates, key=lambda c: c.candidate_id)
        results: list[dict[str, Any]] = []

        for cand in ordered:
            step = self._process_one(
                cand,
                asof=asof,
                prices=prices,
                returns=rets,
                simulation_mode=simulation_mode,
            )
            results.append(step)
            self.state.stage_log.append(step)

        # Snapshot after batch
        wm = self.state.weight_map()
        gross = sum(abs(v) for v in wm.values())
        net = sum(wm.values())
        snap = PortfolioSnapshot(
            timestamp=asof,
            equity=self.state.capital.equity,
            cash=self.state.capital.cash,
            gross_exposure=gross,
            net_exposure=net,
            leverage=gross,
            positions=dict(self.state.positions.quantities()),
            meta={"n_candidates": len(ordered)},
        )
        self.state.snapshots.add(snap)

        recon = reconcile_pipeline_state(
            capital=self.state.capital,
            positions=self.state.positions,
            fill_log=self.state.fill_log,
            snapshots=[],
        )
        return {
            "disclaimer": DISCLAIMER,
            "asof": asof,
            "n_candidates": len(ordered),
            "results": results,
            "positions": self.state.positions.quantities(),
            "target_weights": dict(self.state.target_weights),
            "equity": self.state.capital.equity,
            "cash": self.state.capital.cash,
            "gross_exposure_weights": sum(abs(v) for v in self.state.weight_map().values()),
            "net_exposure_weights": sum(self.state.weight_map().values()),
            "reconciliation": recon,
            "lineage": list(self.state.lineage),
        }

    def _process_one(
        self,
        candidate: AlphaCandidate,
        *,
        asof: str,
        prices: dict[str, float],
        returns: np.ndarray,
        simulation_mode: str,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "candidate": candidate.to_dict(),
        }
        ok, codes = validate_candidate(
            candidate,
            asof=asof,
            seen_ids=self.state.seen_candidate_ids,
            require_model_version=True,
            require_dataset=True,
        )
        if not ok:
            out["outcome"] = StageOutcome.CANDIDATE_REJECTED.value
            out["rejection_codes"] = codes
            return out
        self.state.seen_candidate_ids.add(candidate.candidate_id)

        if abs(candidate.direction) < 1e-15 and abs(candidate.signal_value) < 1e-15:
            # Explicit FLAT — may close via requested_weight 0
            if candidate.requested_weight is None or abs(float(candidate.requested_weight)) < 1e-15:
                out["outcome"] = StageOutcome.SKIPPED_FLAT.value
                return out

        # RISK against shared current weights
        risk_res = evaluate_candidate_risk(
            candidate,
            risk_engine=self.risk,
            current_weights=self.state.weight_map(),
            returns=returns,
        )
        out["risk"] = risk_res.to_dict()
        if risk_res.outcome == StageOutcome.RISK_REJECTED:
            out["outcome"] = StageOutcome.RISK_REJECTED.value
            return out

        sizing = size_approved_exposure(
            risk_engine=self.risk,
            approved_exposure=risk_res.approved_exposure,
            returns=returns,
            equity=self.state.capital.equity,
            confidence=float(candidate.confidence or 0.5),
        )
        out["sizing"] = sizing.to_dict()

        port = apply_portfolio_constraints(
            instrument=candidate.instrument,
            proposed_weight=sizing.final_size,
            current_weights=self.state.weight_map(),
            settings=self.portfolio_settings,
            max_gross=self.max_gross,
            max_position=self.max_position,
            long_only=self.long_only,
        )
        out["portfolio"] = port.to_dict()
        if port.outcome == StageOutcome.PORTFOLIO_REJECTED:
            out["outcome"] = StageOutcome.PORTFOLIO_REJECTED.value
            return out

        # Update shared target weights before execution so next candidate sees aggregate intent
        tw = dict(self.state.weight_map())
        tw[candidate.instrument] = port.target_position_weight
        self.state.target_weights = tw
        out["target_weights_object"] = weights_to_target_object(tw, long_only=self.long_only).to_dict()

        px = float(prices.get(candidate.instrument, 0.0) or 0.0)
        cur_qty = dict(self.state.positions.quantities())
        tgt_qty = dict(cur_qty)
        tgt_qty[candidate.instrument] = weight_to_quantity(
            port.target_position_weight, equity=self.state.capital.equity, price=px
        )

        lineage = LineageRecord(
            candidate_id=candidate.candidate_id,
            signal_id=candidate.signal_id,
            model_id=candidate.source_model,
            model_version=candidate.source_model_version,
            dataset_id=candidate.data_version,
            dataset_checksum=candidate.dataset_checksum,
            risk_decision_id=risk_res.risk_decision_id,
            portfolio_decision_id=port.portfolio_decision_id,
            order_id="",
            extra={
                "expected_horizon": candidate.expected_horizon,
                "signal_timeframe": candidate.signal_timeframe,
                "execution_timeframe": candidate.execution_timeframe,
            },
        )
        exec_res = plan_and_execute(
            engine=self.execution,
            current_qty=cur_qty,
            target_qty=tgt_qty,
            prices=prices,
            instrument=candidate.instrument,
            lineage=lineage,
            simulation_mode=simulation_mode,
        )
        out["execution"] = exec_res
        acct = apply_fills_to_ledgers(
            capital=self.state.capital,
            positions=self.state.positions,
            order_log=self.state.order_log,
            fill_log=self.state.fill_log,
            trade_ledger=self.state.trade_ledger,
            exec_result=exec_res,
            timestamp=asof,
            lineage=lineage,
        )
        out["accounting"] = acct
        self.state.mark(prices)
        self.state.lineage.append(lineage.to_dict())
        out["outcome"] = exec_res.get("outcome", StageOutcome.FILL_COMPLETE.value)
        out["post_state"] = {
            "positions": self.state.positions.quantities(),
            "weights": self.state.weight_map(),
            "equity": self.state.capital.equity,
        }
        return out


__all__ = ["DISCLAIMER", "UnifiedPipelineState", "UnifiedTradingOrchestrator"]

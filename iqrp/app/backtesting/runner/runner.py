"""BacktestRunner — operational lifecycle for Phase 13 executable runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.backtesting.runner.checkpoint import checkpoint_path, write_checkpoint
from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
from iqrp.app.backtesting.runner.executor import PipelineExecutor, load_market_frame
from iqrp.app.backtesting.runner.lifecycle import Lifecycle, RunnerLifecycleState
from iqrp.app.backtesting.runner.persistence import persist_result
from iqrp.app.backtesting.runner.recovery import restore_context
from iqrp.app.backtesting.runner.reports import write_reports
from iqrp.app.backtesting.runner.result import OperationalBacktestResult
from iqrp.app.backtesting.runner.validation import (
    ValidationReport,
    integrity_validate,
    preflight_validate,
)
from iqrp.app.backtesting.strategy import StrategyRegistry
from iqrp.app.backtesting.strategy.base import Strategy


class BacktestRunner:
    """Create → validate → prepare → run (pause/resume/cancel) → result/report."""

    def __init__(
        self,
        config: BacktestRunConfig | Mapping[str, Any] | str | Path,
        *,
        strategy: Strategy | None = None,
    ) -> None:
        if isinstance(config, BacktestRunConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = BacktestRunConfig.from_yaml(config)
        else:
            self.config = BacktestRunConfig.from_dict(config)
        self._strategy_override = strategy
        self.strategy: Strategy | None = strategy
        self.lifecycle = Lifecycle()
        self._executor: PipelineExecutor | None = None
        self._frame = None
        self._data_detail: dict[str, Any] = {}
        self._result: OperationalBacktestResult | None = None
        self._validation: ValidationReport | None = None
        self._persisted_root: Path | None = None
        self._report_paths: dict[str, str] = {}
        self._pause = False
        self._cancel = False

    # ------------------------------------------------------------------ status
    def status(self) -> RunnerLifecycleState:
        return self.lifecycle.state

    def create(self) -> RunnerLifecycleState:
        return self.lifecycle.transition(
            RunnerLifecycleState.CREATED, reason="create", allow_same=True
        )

    # --------------------------------------------------------------- validate
    def validate(self) -> ValidationReport:
        self.lifecycle.transition(RunnerLifecycleState.VALIDATING, reason="validate")
        strategy_ok = False
        try:
            if self._strategy_override is not None:
                strategy_ok = True
            else:
                StrategyRegistry.get(self.config.strategy_id, self.config.strategy_version)
                strategy_ok = True
        except Exception:
            strategy_ok = False

        dataset_ok = False
        detail = ""
        try:
            if self.config.dataset_path:
                frame, data_detail = load_market_frame(self.config)
                self._frame = frame
                self._data_detail = data_detail
                dataset_ok = bool(data_detail.get("ok", True))
                detail = str(data_detail.get("critical_failures") or "")
            elif self.config.dataset_id:
                detail = "dataset_id provided without resolvable local path"
                dataset_ok = False
        except Exception as exc:
            detail = str(exc)
            dataset_ok = False

        report = preflight_validate(
            self.config,
            strategy_registered=strategy_ok,
            dataset_ok=dataset_ok,
            dataset_detail=detail,
        )
        self._validation = report
        if not report.ok:
            self.lifecycle.transition(RunnerLifecycleState.FAILED, reason="preflight_failed")
            raise ValueError(f"preflight validation failed: {report.to_dict()}")
        return report

    # ---------------------------------------------------------------- prepare
    def prepare(self) -> None:
        if self.lifecycle.state is RunnerLifecycleState.FAILED:
            raise RuntimeError("cannot prepare a failed runner")
        self.lifecycle.transition(RunnerLifecycleState.PREPARING, reason="prepare")
        if self.strategy is None:
            if self._strategy_override is not None:  # pragma: no cover
                self.strategy = self._strategy_override
            else:
                self.strategy = StrategyRegistry.create(
                    self.config.strategy_id,
                    self.config.strategy_version,
                    **dict(self.config.strategy_params or {}),
                )
        if self._frame is None:  # pragma: no cover
            self._frame, self._data_detail = load_market_frame(self.config)

        self._executor = PipelineExecutor(
            self.config,
            self.strategy,
            frame=self._frame,
            data_detail=self._data_detail,
        )
        self._executor.prepare()

        if self.config.resume_from:
            restore_context(self._executor.context, self.config.resume_from)

    # ------------------------------------------------------------------- run
    def run(self) -> OperationalBacktestResult:
        if self._executor is None or self._executor.context is None:  # pragma: no cover
            self.prepare()
        assert self._executor is not None and self._executor.context is not None

        self.lifecycle.transition(RunnerLifecycleState.RUNNING, reason="run")
        self._pause = False
        self._cancel = False
        ctx = self._executor.context
        ctx.pause_requested = False
        ctx.cancel_requested = False

        try:
            resume_after = ctx.current_time
            self._executor.run(resume_after=resume_after if self.config.resume_from else None)
            if ctx.invalidated:
                self.lifecycle.transition(
                    RunnerLifecycleState.INVALIDATED,
                    reason=ctx.invalidation_reason or "invalidated",
                )
            elif self._cancel:  # pragma: no cover
                self.lifecycle.transition(RunnerLifecycleState.CANCELLED, reason="cancelled")
            else:
                self._result = self._build_result()
                self._persisted_root = persist_result(self._result, self.config.output_dir)
                integrity = integrity_validate(
                    ctx,
                    self._result,
                    results_persisted=self._persisted_root is not None,
                )
                self._validation = integrity
                self._result.diagnostics["integrity"] = integrity.to_dict()
                if not integrity.ok and ctx.invalidated:  # pragma: no cover
                    self.lifecycle.transition(
                        RunnerLifecycleState.INVALIDATED,
                        reason="integrity_failed",
                    )
                elif not integrity.ok:
                    # Soft-fail warnings allowed; critical reconciliation still fails run
                    critical = [
                        i
                        for i in integrity.issues
                        if i.severity == "critical" and i.code != "results_persisted"
                    ]
                    if critical:
                        self.lifecycle.transition(
                            RunnerLifecycleState.FAILED,
                            reason="integrity_failed",
                        )
                        raise RuntimeError(f"integrity validation failed: {integrity.to_dict()}")
                    self.lifecycle.transition(  # pragma: no cover
                        RunnerLifecycleState.COMPLETED, reason="completed_with_warnings"
                    )
                else:
                    self.lifecycle.transition(RunnerLifecycleState.COMPLETED, reason="completed")
                # Persist lifecycle status on the result before writing reports.
                self._result.status = self.lifecycle.state.value
                self._report_paths = write_reports(self._result, self.config.output_dir)
                # Optional extensions
                if self.config.walk_forward_config:
                    self._result.walk_forward = self.walk_forward()
                if self.config.scenario_config:
                    self._result.scenarios = self.scenarios()
        except Exception:
            if self.lifecycle.state not in {
                RunnerLifecycleState.FAILED,
                RunnerLifecycleState.INVALIDATED,
                RunnerLifecycleState.CANCELLED,
            }:  # pragma: no cover
                self.lifecycle.transition(RunnerLifecycleState.FAILED, reason="exception")
            raise

        assert self._result is not None
        self._result.status = self.lifecycle.state.value
        return self._result

    def pause(self) -> RunnerLifecycleState:
        self._pause = True
        if self._executor and self._executor.context:
            self._executor.context.pause_requested = True
            cp_root = self.config.checkpoint_dir or self.config.output_dir
            write_checkpoint(
                self._executor.context,
                checkpoint_path(cp_root, self.config.backtest_id),
            )
        return self.lifecycle.transition(RunnerLifecycleState.PAUSED, reason="pause")

    def resume(self) -> OperationalBacktestResult:
        if self.lifecycle.state is not RunnerLifecycleState.PAUSED:
            # Allow resume from checkpoint path
            if self.config.resume_from or self.config.checkpoint_dir:
                pass
            else:
                raise RuntimeError("resume requires PAUSED state or resume_from checkpoint")
        self.config = self.config.with_updates(
            resume_from=str(
                checkpoint_path(
                    self.config.checkpoint_dir or self.config.output_dir,
                    self.config.backtest_id,
                )
            )
        )
        self.prepare()
        return self.run()

    def cancel(self) -> RunnerLifecycleState:
        self._cancel = True
        if self._executor and self._executor.context:
            self._executor.context.cancel_requested = True
        return self.lifecycle.transition(RunnerLifecycleState.CANCELLED, reason="cancel")

    # --------------------------------------------------------- result/report
    def result(self) -> OperationalBacktestResult:
        if self._result is None:
            if self._executor and self._executor.context:
                self._result = self._build_result()
            else:
                raise RuntimeError("no result available; call run() first")
        return self._result

    def report(self) -> str:
        if not self._report_paths:
            res = self.result()
            self._report_paths = write_reports(res, self.config.output_dir)
        return self._report_paths.get("markdown") or self._report_paths.get("json") or ""

    # ---------------------------------------------- optional research modes
    def walk_forward(self) -> dict[str, Any]:
        """Run configured walk-forward using existing WalkForwardEngine (no redesign)."""
        cfg = dict(self.config.walk_forward_config or {})
        if not cfg:
            return {}
        try:
            from iqrp.app.backtesting.walk_forward import WalkForwardEngine

            engine = WalkForwardEngine()
            n = int(len(self._frame)) if self._frame is not None else int(cfg.get("n", 100))
            train_size = int(cfg.get("train_periods", cfg.get("train_size", 50)))
            test_size = int(cfg.get("test_periods", cfg.get("test_size", 10)))
            windows = engine.windows(
                max(n, 1),
                train_size=train_size,
                test_size=test_size,
                mode=str(cfg.get("mode", "rolling")),
                purge=int(cfg.get("purge_periods", cfg.get("purge", 0))),
                embargo=int(cfg.get("embargo_periods", cfg.get("embargo", 0))),
            )
            return {
                "engine": type(engine).__name__,
                "n_windows": len(windows),
                "config": cfg,
            }
        except Exception as exc:
            return {"error": str(exc), "config": cfg}

    def retrain(self) -> dict[str, Any]:
        cfg = dict(self.config.model_config or {})
        if not cfg.get("enabled"):
            return {"skipped": True}
        try:
            from iqrp.app.backtesting.rolling_retraining import RollingRetrainer

            return {"retrainer": RollingRetrainer.__name__, "config": cfg}
        except Exception as exc:
            return {"error": str(exc), "config": cfg}

    def scenarios(self) -> dict[str, Any]:
        cfg = dict(self.config.scenario_config or {})
        if not cfg:
            return {}
        try:
            from iqrp.app.backtesting.scenarios import ScenarioEngine

            eng = ScenarioEngine()
            return {"engine": type(eng).__name__, "config": cfg}
        except Exception as exc:
            return {"error": str(exc), "config": cfg}

    def parameter_sweep(
        self,
        grid: Sequence[Mapping[str, Any]],
        *,
        max_workers: int | None = None,
    ) -> list[dict[str, Any]]:
        from iqrp.app.backtesting.runner.parallel import parameter_sweep_parallel

        return parameter_sweep_parallel(
            self.config.to_dict(),
            grid,
            max_workers=max_workers,
            seed0=int(self.config.seed),
        )

    # -------------------------------------------------------------- internals
    def _build_result(self) -> OperationalBacktestResult:
        assert self._executor is not None and self._executor.context is not None
        ctx = self._executor.context
        eq = list(ctx.equity_curve)
        rets = list(ctx.returns)
        perf: dict[str, Any] = {}
        risk: dict[str, Any] = {}
        try:
            from iqrp.app.backtesting.performance import (
                max_drawdown,
                sharpe_ratio,
                summarize_returns,
            )

            perf = summarize_returns(rets) if rets else {}
            if not isinstance(perf, dict):  # pragma: no cover
                perf = {"summary": perf}
            perf["sharpe"] = float(sharpe_ratio(rets)) if rets else 0.0
            risk["max_drawdown"] = float(max_drawdown(rets)) if rets else 0.0
        except Exception:
            if eq:
                arr = np.asarray(eq, dtype=np.float64)
                perf = {
                    "total_return": float(arr[-1] / arr[0] - 1.0) if arr[0] else 0.0,
                    "ending_equity": float(arr[-1]),
                }
                peak = np.maximum.accumulate(arr)
                dd = 1.0 - arr / np.maximum(peak, 1e-12)
                risk["max_drawdown"] = float(np.max(dd)) if dd.size else 0.0

        risk.update(dict(ctx.risk_state))
        positions_log = []
        for snap in ctx.snapshots.snapshots:
            positions_log.append(
                {
                    "timestamp": snap.timestamp,
                    "positions": dict(snap.positions),
                    "equity": snap.equity,
                }
            )

        return OperationalBacktestResult(
            backtest_id=self.config.backtest_id,
            status=self.lifecycle.state.value,
            equity_curve=eq,
            returns=rets,
            timestamps=list(ctx.timestamps),
            orders=ctx.orders.to_list(),
            fills=ctx.fills.to_list(),
            trades=ctx.trades.to_list(),
            positions_log=positions_log,
            snapshots=ctx.snapshots.to_list(),
            capital=ctx.capital.to_dict(),
            performance=perf,
            risk=risk,
            execution={
                "backend": ctx.execution_adapter.backend,
                "n_orders": len(ctx.orders),
                "n_fills": len(ctx.fills),
            },
            diagnostics={
                **dict(ctx.diagnostics),
                "portfolio_backend": ctx.portfolio_adapter.backend,
                "execution_backend": ctx.execution_adapter.backend,
                "event_count": ctx.event_count,
                "bar_count": ctx.bar_count,
                "lifecycle": self.lifecycle.to_dict(),
            },
            initial_capital=float(self.config.initial_capital),
            seed=int(self.config.seed),
            config=self.config.to_dict(),
        )


__all__ = ["BacktestRunner"]

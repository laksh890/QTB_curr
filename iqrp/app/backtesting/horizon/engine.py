"""Horizon research engine — sweep data/signal/holding without fabricating bars."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from iqrp.app.backtesting.horizon.availability import (
    detect_native_frequency,
    filter_available_timeframes,
)
from iqrp.app.backtesting.horizon.capacity import capacity_scenario_report
from iqrp.app.backtesting.horizon.config import HorizonResearchConfig
from iqrp.app.backtesting.horizon.costs import apply_cost_drag, gross_vs_net_sharpe
from iqrp.app.backtesting.horizon.half_life import signal_half_life_report
from iqrp.app.backtesting.horizon.metrics import horizon_performance_metrics
from iqrp.app.backtesting.horizon.multiple_testing import multiple_testing_record
from iqrp.app.backtesting.horizon.neighborhood import neighborhood_robustness
from iqrp.app.backtesting.horizon.overtrading import overtrading_diagnostics, position_path_sides
from iqrp.app.backtesting.horizon.parse import parse_holding, parse_timeframe
from iqrp.app.backtesting.horizon.ranking import (
    classify_horizon,
    compute_horizon_research_score,
    select_best_robust_horizon,
)
from iqrp.app.backtesting.horizon.report import build_horizon_matrix, build_horizon_report
from iqrp.app.backtesting.horizon.resampling import UnavailableFrequencyError, resample_ohlcv
from iqrp.app.backtesting.horizon.simulate import SignalFn, simulate_positions
from iqrp.app.backtesting.horizon.trade_analytics import trade_frequency_report
from iqrp.app.backtesting.horizon.turnover import turnover_report
from iqrp.app.backtesting.horizon.types import (
    HoldingPeriod,
    HorizonResult,
    HorizonSpec,
    HorizonStatus,
    Timeframe,
)
from iqrp.app.backtesting.horizon.walk_forward import evaluate_oos


class HorizonResearchEngine:
    """Discover economically viable horizons as a research capability.

    Does not redesign portfolio/execution/risk. Vectorized sweeps reuse the
    existing performance / capacity / statistical-validation libraries.
    Long/short transitions for live cascade runs go through BacktestRunner.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        config: HorizonResearchConfig | Mapping[str, Any] | None = None,
        signal_fn: SignalFn | None = None,
        regimes: Any | None = None,
    ) -> None:
        self.frame = frame.copy()
        self.config = (
            config
            if isinstance(config, HorizonResearchConfig)
            else HorizonResearchConfig.from_dict(dict(config or {}))
        )
        self.signal_fn = signal_fn
        self.regimes = regimes
        self.native = detect_native_frequency(self.frame)
        self.results: list[HorizonResult] = []
        self._availability: dict[str, Any] = {}

    def availability(self) -> dict[str, Any]:
        self._availability = filter_available_timeframes(
            self.native, self.config.data_timeframes
        )
        return dict(self._availability)

    def _instrument_frame(self, instrument: str | None = None) -> pd.DataFrame:
        df = self.frame
        inst = instrument or self.config.instrument
        if not inst:
            if "instrument" in df.columns:
                counts = df["instrument"].value_counts()
                inst = str(counts.index[0])
            else:
                raise ValueError("instrument required")
        self.config.instrument = str(inst)
        return df.loc[df["instrument"] == inst].copy()

    def evaluate_spec(
        self,
        *,
        data_timeframe: str | Timeframe,
        signal_timeframe: str | Timeframe | None = None,
        holding: str | int | HoldingPeriod = 1,
        instrument: str | None = None,
    ) -> HorizonResult:
        cfg = self.config
        dtf = parse_timeframe(data_timeframe)
        stf = parse_timeframe(signal_timeframe or dtf)
        hold = parse_holding(holding, bar_seconds=stf.seconds)
        spec = HorizonSpec(
            data_timeframe=dtf,
            signal_timeframe=stf,
            holding=hold,
            instrument=instrument or cfg.instrument,
            strategy_id=cfg.strategy_id,
        )

        # Availability: data timeframe must be derivable from native
        if dtf.seconds + 1e-9 < self.native.seconds:
            return HorizonResult(
                spec=spec,
                status=HorizonStatus.UNAVAILABLE,
                reason=(
                    f"data timeframe {dtf} finer than native {self.native}; "
                    "no fabricated intraday/finer bars"
                ),
            )
        # Signal cannot be finer than data
        if stf.seconds + 1e-9 < dtf.seconds:
            return HorizonResult(
                spec=spec,
                status=HorizonStatus.UNAVAILABLE,
                reason=f"signal timeframe {stf} finer than data timeframe {dtf}",
            )

        base = self._instrument_frame(spec.instrument)
        try:
            data_bars = resample_ohlcv(base, dtf, native=self.native)
            # Further downsample for signal if needed
            signal_bars = (
                data_bars
                if abs(stf.seconds - dtf.seconds) < 1e-9
                else resample_ohlcv(data_bars, stf, native=dtf)
            )
        except UnavailableFrequencyError as exc:
            return HorizonResult(
                spec=spec,
                status=HorizonStatus.UNAVAILABLE,
                reason=str(exc),
            )

        if len(signal_bars) < 10:
            return HorizonResult(
                spec=spec,
                status=HorizonStatus.INSUFFICIENT_DATA,
                reason=f"only {len(signal_bars)} bars after resampling",
            )

        holding_bars = int(hold.bars or 1)
        sim = simulate_positions(
            signal_bars,
            signal_fn=self.signal_fn,
            params=cfg.signal_params,
            holding_bars=holding_bars,
            allow_short=cfg.allow_short,
        )

        # Periods/year from signal timeframe
        bars_per_day = max(86400.0 / stf.seconds, 1e-12)
        ppy = float(cfg.periods_per_year)
        if stf.seconds < 86400:
            ppy = bars_per_day * 252.0

        cost = apply_cost_drag(
            sim["gross_returns"],
            commission_bps=cfg.commission_bps,
            spread_bps=cfg.spread_bps,
            slippage_bps=cfg.slippage_bps,
            turnover_per_period=sim["turnover_per_period"],
            financing_bps_per_period=cfg.financing_bps_per_period,
            impact_bps_per_period=cfg.impact_bps_per_period,
        )
        gn = gross_vs_net_sharpe(
            cost["gross_returns"], cost["net_returns"], periods_per_year=ppy
        )
        cost.update(gn)

        freq = trade_frequency_report(sim["trades"], timestamps=list(sim["timestamps"]))
        to = turnover_report(
            sim["positions"],
            periods_per_day=bars_per_day if stf.seconds < 86400 else 1.0,
            net_alpha=float(cost["net_alpha"]),
            net_pnl=float(cost["net_pnl"]),
        )
        metrics = horizon_performance_metrics(
            cost["gross_returns"],
            cost["net_returns"],
            trades=sim["trades"],
            periods_per_year=ppy,
            turnover=to.get("annualized_turnover"),
        )
        half = signal_half_life_report(sim["signal"], signal_bars["close"].to_numpy())
        cap = capacity_scenario_report(
            cost["net_returns"],
            capital_levels=cfg.capital_levels,
            turnover=float(to.get("annualized_turnover") or 1.0),
            adv=cfg.capacity_adv,
            impact_coef=cfg.capacity_impact_coef,
            impact_exp=cfg.capacity_impact_exp,
            periods_per_year=ppy,
        )
        oos_pack = evaluate_oos(
            cost["gross_returns"],
            cost["net_returns"],
            timestamps=list(sim["timestamps"]),
            train_end=cfg.train_end,
            validation_end=cfg.validation_end,
            train_frac=cfg.train_frac,
            validation_frac=cfg.validation_frac,
            periods_per_year=ppy,
        )
        oos_flat = dict(oos_pack.get("oos") or {})
        oos_flat["train"] = oos_pack.get("train")
        oos_flat["validation"] = oos_pack.get("validation")
        oos_flat["splits"] = {
            "train_frac": cfg.train_frac,
            "validation_frac": cfg.validation_frac,
            "train_end": cfg.train_end,
            "validation_end": cfg.validation_end,
        }

        over = overtrading_diagnostics(position_path_sides(sim["positions"]))

        regime_rep: dict[str, Any] = {}
        if self.regimes is not None:
            try:
                from iqrp.app.alpha.regime.regime_performance import regime_returns

                regime_rep = regime_returns(cost["net_returns"], self.regimes)
            except Exception as exc:  # noqa: BLE001
                try:
                    from iqrp.app.backtesting.scenarios.regime import (
                        classify_simple_regimes,
                        run_regime_scenario,
                    )

                    labels = classify_simple_regimes(cost["net_returns"])
                    regime_rep = run_regime_scenario(cost["net_returns"], labels)
                except Exception as exc2:  # noqa: BLE001
                    regime_rep = {"error": str(exc), "fallback_error": str(exc2)}

        status, reason = classify_horizon(
            metrics,
            oos=oos_flat,
            costs=cost,
            neighborhood={},
            available=True,
            insufficient=len(sim["trades"]) < int(cfg.robust_gates.get("min_trades", 5)),
            gates=cfg.robust_gates,
        )
        # cost inefficient override already in classify

        result = HorizonResult(
            spec=spec,
            status=status,
            reason=reason,
            metrics=metrics,
            trade_frequency=freq,
            costs={
                k: v
                for k, v in cost.items()
                if k not in {"gross_returns", "net_returns"}
            },
            turnover=to,
            capacity=cap,
            half_life=half,
            oos=oos_flat,
            regime=regime_rep if isinstance(regime_rep, dict) else {"data": regime_rep},
            overtrading=over,
        )
        # stash arrays for later neighborhood / MT (not serialized heavily)
        result.metrics["_net_returns_ref"] = "stored_separately"
        result._net_returns = cost["net_returns"]  # type: ignore[attr-defined]
        result._gross_returns = cost["gross_returns"]  # type: ignore[attr-defined]
        return result

    def sweep(self) -> list[HorizonResult]:
        """Run full configured grid; mark unavailable frequencies explicitly."""
        cfg = self.config
        avail = self.availability()
        signal_tfs = list(cfg.signal_timeframes or cfg.data_timeframes)
        results: list[HorizonResult] = []

        for d in cfg.data_timeframes:
            dtf = parse_timeframe(d)
            if dtf.seconds + 1e-9 < self.native.seconds:
                for s in signal_tfs:
                    for h in cfg.holding_bars:
                        spec = HorizonSpec(
                            data_timeframe=dtf,
                            signal_timeframe=parse_timeframe(s),
                            holding=parse_holding(h),
                            instrument=cfg.instrument or "",
                            strategy_id=cfg.strategy_id,
                        )
                        results.append(
                            HorizonResult(
                                spec=spec,
                                status=HorizonStatus.UNAVAILABLE,
                                reason=(
                                    f"native dataset frequency is {self.native}; "
                                    f"cannot provide {dtf} without fabricating bars"
                                ),
                            )
                        )
                continue

            for s in signal_tfs:
                stf = parse_timeframe(s)
                if stf.seconds + 1e-9 < dtf.seconds:
                    continue
                for h in cfg.holding_bars:
                    results.append(
                        self.evaluate_spec(
                            data_timeframe=dtf,
                            signal_timeframe=stf,
                            holding=h,
                        )
                    )

        # Neighborhood across available data timeframes
        evaluated = [
            {
                "spec": r.spec.to_dict(),
                "metrics": r.metrics,
                "key": r.spec.key,
            }
            for r in results
            if r.status not in {HorizonStatus.UNAVAILABLE, HorizonStatus.INSUFFICIENT_DATA}
        ]
        neigh = neighborhood_robustness(
            evaluated,
            metric_key="net_sharpe",
            max_ratio=cfg.neighborhood_max_ratio,
        )
        for r in results:
            label = str(r.spec.data_timeframe)
            if label in neigh:
                r.neighborhood = neigh[label]
                # reclassify with neighborhood
                status, reason = classify_horizon(
                    r.metrics,
                    oos=r.oos,
                    costs=r.costs,
                    neighborhood=r.neighborhood,
                    available=r.status != HorizonStatus.UNAVAILABLE,
                    insufficient=r.status == HorizonStatus.INSUFFICIENT_DATA,
                    gates=cfg.robust_gates,
                )
                if r.status != HorizonStatus.UNAVAILABLE:
                    r.status = status
                    r.reason = reason

        # Scores + multiple testing
        sharpes = [
            float(r.metrics.get("net_sharpe", 0.0) or 0.0)
            for r in results
            if r.status != HorizonStatus.UNAVAILABLE
        ]
        n_cfg = len([r for r in results if r.status != HorizonStatus.UNAVAILABLE])
        n_horizons = len({str(r.spec.data_timeframe) for r in results})
        best_rets = None
        best_score = -1e18
        for r in results:
            if r.status == HorizonStatus.UNAVAILABLE:
                r.robustness_score = None
                continue
            mt = multiple_testing_record(
                n_configurations=max(n_cfg, 1),
                n_strategies=1,
                n_horizons=n_horizons,
                n_parameter_combinations=len(cfg.holding_bars) * len(signal_tfs),
                observed_sharpes=sharpes,
                returns_for_best=getattr(r, "_net_returns", None),
            )
            r.multiple_testing = mt
            scored = compute_horizon_research_score(
                r.metrics,
                oos=r.oos,
                costs=r.costs,
                turnover=r.turnover,
                capacity=r.capacity,
                neighborhood=r.neighborhood,
                multiple_testing=mt,
                weights=cfg.score_weights,
            )
            r.robustness_score = float(scored["score"])
            r.metrics["horizon_research_score_components"] = scored["components"]
            if r.robustness_score > best_score and hasattr(r, "_net_returns"):
                best_score = r.robustness_score
                best_rets = getattr(r, "_net_returns", None)

        # Attach global MT summary with best returns
        global_mt = multiple_testing_record(
            n_configurations=max(n_cfg, 1),
            n_strategies=1,
            n_horizons=n_horizons,
            n_parameter_combinations=len(cfg.holding_bars) * max(len(signal_tfs), 1),
            observed_sharpes=sharpes,
            returns_for_best=best_rets,
        )
        self._global_mt = global_mt
        self._availability = avail
        self.results = results
        return results

    def selection(self) -> dict[str, Any]:
        return select_best_robust_horizon(self.results, gates=self.config.robust_gates)

    def matrix(self) -> list[dict[str, Any]]:
        return build_horizon_matrix(self.results)

    def report(self) -> dict[str, Any]:
        return build_horizon_report(
            self.results,
            config=self.config,
            native=self.native,
            availability=self._availability or self.availability(),
            selection=self.selection(),
            multiple_testing=getattr(self, "_global_mt", {}),
        )


__all__ = ["HorizonResearchEngine"]

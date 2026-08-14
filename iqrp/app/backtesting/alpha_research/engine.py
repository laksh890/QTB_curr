"""Alpha research engine — FEATURE → SIGNAL → POSITION → COST → PERFORMANCE → STATS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.analytics import (
    decay_curve,
    evaluate_cost_aware,
    incremental_alpha,
    parameter_stability,
    permutation_importance,
    positions_from_signal,
    signal_correlation_matrix,
    time_of_day_report,
    timeseries_ic_report,
)
from iqrp.app.backtesting.alpha_research.experiments import (
    ExperimentRegistry,
    ExperimentSpec,
    now_iso,
)
from iqrp.app.backtesting.alpha_research.features import get_feature_registry
from iqrp.app.backtesting.alpha_research.leakage import run_leakage_suite
from iqrp.app.backtesting.alpha_research.mtf import align_feature_to_execution
from iqrp.app.backtesting.alpha_research.multiple_testing import research_breadth_record
from iqrp.app.backtesting.alpha_research.ranking import classify_alpha, compute_alpha_research_score
from iqrp.app.backtesting.alpha_research.signals import get_signal_registry
from iqrp.app.backtesting.alpha_research.types import (
    DEFAULT_ALPHA_GATES,
    SAMPLE_TOO_SHORT_DISCLAIMER,
    TimeframeContext,
    bars_per_day,
    holding_clock_minutes,
    map_alpha_to_research_status,
)
from iqrp.app.backtesting.horizon.walk_forward import evaluate_oos
from iqrp.app.backtesting.performance.drawdown import max_drawdown


class AlphaSignalResearchEngine:
    """Generic alpha research / signal discovery for canonical OHLCV datasets.

    Separates FEATURE / SIGNAL / POSITION / EXECUTION assumptions / PERFORMANCE.
    Does not claim profitability. Short samples are marked SAMPLE TOO SHORT.
    """

    def __init__(
        self,
        *,
        experiment_registry: ExperimentRegistry | None = None,
        cost_model: Mapping[str, float] | None = None,
        gates: Mapping[str, Any] | None = None,
        score_weights: Mapping[str, float] | None = None,
        market_type: str = "equity",
        timezone: str = "Asia/Kolkata",
    ) -> None:
        self.features = get_feature_registry()
        self.signals = get_signal_registry()
        self.experiments = experiment_registry or ExperimentRegistry()
        self.cost_model = dict(
            cost_model
            or {"commission_bps": 1.0, "spread_bps": 2.0, "slippage_bps": 2.0}
        )
        self.gates = {**DEFAULT_ALPHA_GATES, **dict(gates or {})}
        self.score_weights = dict(score_weights or {})
        self.market_type = str(market_type)
        self.timezone = str(timezone)
        self.results: list[dict[str, Any]] = []

    def evaluate_candidate(
        self,
        frame: pd.DataFrame,
        *,
        signal_id: str,
        timeframe: str,
        holding_bars: int = 5,
        parameters: Mapping[str, Any] | None = None,
        dataset_id: str = "",
        dataset_checksum: str = "",
        dataset_kind: str = "",
        n_sessions: int | None = None,
        feature_frame: pd.DataFrame | None = None,
        feature_timeframe: str | None = None,
        purge_bars: int | None = None,
        embargo_bars: int | None = None,
        cost_scenario: str = "BASE",
        cost_model: Mapping[str, float] | None = None,
        run_leakage: bool = True,
        run_importance: bool = True,
        run_regime: bool = True,
        persist_experiment: bool = True,
        train_frac: float = 0.5,
        validation_frac: float = 0.25,
        precomputed_signal: pd.Series | None = None,
        precomputed_feats: Mapping[str, pd.Series] | None = None,
        precomputed_sig_meta: Mapping[str, Any] | None = None,
        precomputed_leakage: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = {"holding_bars": holding_bars, **dict(parameters or {})}
        costs_cfg = dict(cost_model or self.cost_model)
        hb = int(holding_bars)
        purge = int(purge_bars if purge_bars is not None else hb)
        embargo = int(embargo_bars if embargo_bars is not None else hb)
        # Optional MTF: compute features on coarser frame, align to execution
        mtf_meta = None
        sig_frame = frame
        if precomputed_signal is not None:
            signal = precomputed_signal
            feats = dict(precomputed_feats or {})
            sig_meta = dict(precomputed_sig_meta or {"feature_ids": [], "signal_id": signal_id})
        elif feature_frame is not None and feature_timeframe:
            raw_sig, sig_meta, feats = self.signals.generate(
                feature_frame, signal_id, parameters=params, feature_registry=self.features
            )
            aligned = align_feature_to_execution(feature_frame, raw_sig, frame["timestamp"])
            aligned.index = frame.index
            signal = aligned
            mtf_meta = TimeframeContext(
                feature_timeframe=feature_timeframe,
                signal_timeframe=feature_timeframe,
                execution_timeframe=timeframe,
            ).to_dict()
            sig_meta = {**sig_meta, "mtf": mtf_meta}
            feats = {
                k: align_feature_to_execution(feature_frame, v, frame["timestamp"]) for k, v in feats.items()
            }
            for k in feats:
                feats[k].index = frame.index
        else:
            signal, sig_meta, feats = self.signals.generate(
                sig_frame, signal_id, parameters=params, feature_registry=self.features
            )

        primary_feat = next(iter(feats.values())) if feats else signal
        lookback = int(params.get("lookback", 20))
        feat_ids = list(sig_meta.get("feature_ids") or [])
        primary_id = feat_ids[0] if feat_ids else "momentum"

        if precomputed_leakage is not None:
            leakage = dict(precomputed_leakage)
        elif run_leakage:

            def _recompute(f: pd.DataFrame):
                return self.features.compute(
                    f, primary_id, parameters={"lookback": lookback}
                )[0]

            leakage = run_leakage_suite(
                frame,
                primary_feat,
                lookback=lookback,
                compute_fn=_recompute,
            )
        else:
            leakage = {"ok": True, "skipped": True, "note": "leakage suite run at campaign level"}

        positions = positions_from_signal(signal.fillna(0.0), hb)
        rets = frame["close"].pct_change().fillna(0.0)
        bpd = bars_per_day(timeframe, market_type=self.market_type)
        ppy = 252.0 * float(bpd)
        sessions = n_sessions
        if sessions is None:
            ts = pd.to_datetime(frame["timestamp"], utc=True)
            if str(self.market_type).lower() in {"crypto", "cryptocurrency"}:
                sessions = int(ts.dt.tz_convert("UTC").dt.date.nunique())
            else:
                sessions = int(ts.dt.tz_convert("Asia/Kolkata").dt.date.nunique())
        cost = evaluate_cost_aware(
            positions,
            rets,
            commission_bps=float(costs_cfg["commission_bps"]),
            spread_bps=float(costs_cfg["spread_bps"]),
            slippage_bps=float(costs_cfg["slippage_bps"]),
            periods_per_year=ppy,
            timestamps=frame["timestamp"],
            n_calendar_days=sessions,
        )
        ic = timeseries_ic_report(signal, frame["close"].to_numpy())
        decay = decay_curve(signal, frame["close"].to_numpy())
        tod = time_of_day_report(
            frame["timestamp"], signal, cost["net_returns"], timezone=self.timezone
        )
        oos = evaluate_oos(
            cost["gross_returns"],
            cost["net_returns"],
            timestamps=frame["timestamp"],
            train_frac=train_frac,
            validation_frac=validation_frac,
            periods_per_year=ppy,
            purge_bars=purge,
            embargo_bars=embargo,
        )
        oos["purge_bars"] = purge
        oos["embargo_bars"] = embargo
        oos["purge_embargo_reasoning"] = (
            f"Forward-return targets of length {hb} bars require purge={purge} and "
            f"embargo={embargo} between chronological TRAIN→VALIDATION→OOS windows "
            "to reduce label leakage. Chronological order alone is not sufficient."
        )

        regime_rep: dict[str, Any] = {}
        if run_regime:
            try:
                from iqrp.app.backtesting.scenarios.regime import (
                    classify_simple_regimes,
                    evaluate_regime_robustness,
                    run_regime_scenario,
                )

                labels = classify_simple_regimes(cost["net_returns"])
                regime_rep = run_regime_scenario(
                    cost["net_returns"], labels, periods_per_year=ppy
                )
                rob = evaluate_regime_robustness(
                    cost["net_returns"], labels, periods_per_year=ppy
                )
                regime_rep["robustness"] = rob
            except Exception as exc:  # noqa: BLE001
                regime_rep = {"unavailable": str(exc)}
        else:
            regime_rep = {"skipped": True}

        if run_importance:
            fr1 = np.asarray(frame["close"].pct_change().shift(-1), dtype=np.float64)
            imp = permutation_importance(signal.to_numpy(), fr1, n_perm=10, seed=0)
        else:
            imp = {"skipped": True}

        regime_stab = 0.5
        if isinstance(regime_rep.get("robustness"), dict):
            regime_stab = float(regime_rep["robustness"].get("score") or 0.5)

        metrics = {
            "net_sharpe": cost["net_sharpe"],
            "gross_sharpe": cost["gross_sharpe"],
            "net_alpha": cost["net_alpha"],
            "expectancy": float(np.mean([t["pnl"] for t in cost["trades"]])) if cost["trades"] else 0.0,
            "max_drawdown": float(max_drawdown(cost["net_returns"])),
            "trade_count": int(cost["trade_frequency"].get("total_trades", 0)),
            "annualized_turnover": float((cost["turnover"] or {}).get("annualized_turnover") or 0.0),
            "alpha_survives_costs": cost["alpha_survives_costs"],
            "alpha_collapses_after_costs": cost["alpha_collapses_after_costs"],
            "mean_ic": ic.get("mean_ic"),
            "ic_stability": ic.get("ic_ir") if ic.get("ic_ir") is not None else 0.5,
            "oos_sharpe": float((oos.get("oos") or {}).get("net_sharpe") or 0.0),
            "oos_evaluated": bool((oos.get("oos") or {}).get("evaluated")),
            "parameter_stability": 0.5,
            "regime_stability": regime_stab,
            "fragile": False,
        }
        scored = compute_alpha_research_score(metrics, weights=self.score_weights)
        classification, reason = classify_alpha(metrics, gates=self.gates, n_sessions=sessions)
        research_status = map_alpha_to_research_status(classification.value, metrics)

        side = cost.get("side_counts") or {}
        clock_min = holding_clock_minutes(timeframe, hb)
        train_deg = None
        val_deg = None
        try:
            tr_s = float((oos.get("train") or {}).get("net_sharpe") or 0.0)
            va_s = float((oos.get("validation") or {}).get("net_sharpe") or 0.0)
            oo_s = float((oos.get("oos") or {}).get("net_sharpe") or 0.0)
            train_deg = oo_s - tr_s
            val_deg = oo_s - va_s
        except Exception:  # noqa: BLE001
            pass

        row = {
            "feature": list(sig_meta.get("feature_ids") or []),
            "signal": signal_id,
            "instrument": str(frame["instrument"].iloc[0]) if "instrument" in frame.columns else "",
            "timeframe": timeframe,
            "dataset_kind": dataset_kind or ("SOURCE" if timeframe == "1m" else "DERIVED"),
            "holding_period_bars": hb,
            "holding_period_minutes": clock_min,
            "trades": metrics["trade_count"],
            "trades_per_day": cost["trade_frequency"].get("trades_per_day"),
            "signals_per_day": cost["trade_frequency"].get("signals_per_day")
            or cost["trade_frequency"].get("trades_per_day"),
            "long_trades": side.get("long_trades"),
            "short_trades": side.get("short_trades"),
            "long_observations": side.get("long_observations"),
            "short_observations": side.get("short_observations"),
            "flat_observations": side.get("flat_observations"),
            "gross_return": cost["gross_pnl"],
            "net_return": cost["net_pnl"],
            "gross_Sharpe": cost["gross_sharpe"],
            "Sharpe": cost["net_sharpe"],
            "gross_edge_per_trade": cost.get("gross_edge_per_trade"),
            "net_edge_per_trade": cost.get("net_edge_per_trade"),
            "drawdown": metrics["max_drawdown"],
            "IC": ic.get("mean_ic"),
            "IC_type": "time_series_IC",
            "turnover": metrics["annualized_turnover"],
            "cost": cost["transaction_costs"],
            "cost_scenario": cost_scenario,
            "OOS_performance": (oos.get("oos") or {}).get("net_sharpe"),
            "train_to_oos_degradation": train_deg,
            "validation_to_oos_degradation": val_deg,
            "robustness": scored["score"],
            "classification": classification.value,
            "research_status": research_status,
            "result_class": "research_simulated",
            "sample_flag": "SAMPLE TOO SHORT" if classification.value == "SAMPLE_TOO_SHORT" else None,
            "disclaimer": SAMPLE_TOO_SHORT_DISCLAIMER
            if sessions is not None and sessions < int(self.gates["min_sessions_for_significance"])
            else (
                "Research evidence is not a profitability guarantee. "
                "Research / simulated result only. Not live performance."
            ),
        }

        eid = ExperimentRegistry.new_id()
        payload = {
            "signal_meta": sig_meta,
            "leakage": leakage,
            "costs": {k: v for k, v in cost.items() if k not in {"gross_returns", "net_returns", "positions", "trades"}},
            "ic": ic,
            "decay": {k: v for k, v in decay.items() if k != "by_horizon"} | {"by_horizon": decay.get("by_horizon")},
            "time_of_day": tod,
            "oos": oos,
            "regime": regime_rep if isinstance(regime_rep, dict) else {"data": regime_rep},
            "importance": imp,
            "score": scored,
            "matrix_row": row,
            "mtf": mtf_meta,
            "n_sessions": sessions,
            "classification": classification.value,
            "classification_reason": reason,
            "research_status": research_status,
            "cost_scenario": cost_scenario,
            "bars_per_day": bpd,
            "periods_per_year": ppy,
            "purge_bars": purge,
            "embargo_bars": embargo,
        }
        r_checksum = ExperimentRegistry.result_checksum(row)
        spec = ExperimentSpec(
            experiment_id=eid,
            timestamp=now_iso(),
            dataset_id=dataset_id,
            dataset_checksum=dataset_checksum,
            feature_versions={fid: "1.0.0" for fid in (sig_meta.get("feature_ids") or [])},
            signal_id=signal_id,
            signal_version="1.0.0",
            parameters={
                **dict(params),
                "cost_scenario": cost_scenario,
                "purge_bars": purge,
                "embargo_bars": embargo,
                "market_type": self.market_type,
                "timezone": self.timezone,
                "holding_minutes": clock_min,
                "dataset_kind": row["dataset_kind"],
            },
            timeframe=timeframe,
            holding_period=hb,
            cost_model=dict(costs_cfg),
            risk_configuration={"note": "risk/execution authoritative when run via BacktestRunner"},
            random_seed=0,
            result_checksum=r_checksum,
            classification=classification.value,
            matrix_row=row,
            notes=[
                "Research evidence is not a profitability guarantee.",
                SAMPLE_TOO_SHORT_DISCLAIMER if row.get("sample_flag") else "",
            ],
        )
        self.experiments.register(spec, persist=persist_experiment)
        out = {
            "experiment_id": eid,
            **payload,
            "experiment": spec.to_dict(),
            "signal_series": signal,
            "positions": positions,
        }
        self.results.append({k: v for k, v in out.items() if k not in {"signal_series", "positions"}})
        return out

    def run_matrix(
        self,
        frames_by_tf: Mapping[str, pd.DataFrame],
        *,
        signal_ids: Sequence[str],
        timeframes: Sequence[str],
        holding_bars: Sequence[int] = (3, 5),
        lookbacks: Sequence[int] = (10, 20),
        dataset_checksums: Mapping[str, str] | None = None,
        dataset_prefix: str = "nifty50_intraday_",
        dataset_ids: Mapping[str, str] | None = None,
        dataset_kinds: Mapping[str, str] | None = None,
        persist_experiment: bool = True,
    ) -> dict[str, Any]:
        checksums = dict(dataset_checksums or {})
        ids = dict(dataset_ids or {})
        kinds = dict(dataset_kinds or {})
        breadth = {
            "n_features_tested": len(self.features.list()),
            "n_signals_tested": len(signal_ids),
            "n_parameter_combinations": len(lookbacks) * len(holding_bars),
            "n_horizons": len(timeframes),
            "n_datasets": len(timeframes),
        }
        for tf in timeframes:
            frame = frames_by_tf[tf]
            ds_id = ids.get(tf, f"{dataset_prefix}{tf}")
            for sid in signal_ids:
                for lb in lookbacks:
                    for hold in holding_bars:
                        self.evaluate_candidate(
                            frame,
                            signal_id=sid,
                            timeframe=tf,
                            holding_bars=int(hold),
                            parameters={"lookback": int(lb)},
                            dataset_id=ds_id,
                            dataset_checksum=checksums.get(tf, checksums.get(ds_id, "")),
                            dataset_kind=kinds.get(tf, "SOURCE" if tf == "1m" else "DERIVED"),
                            persist_experiment=persist_experiment,
                        )
        by_sig: dict[str, list[float]] = {}
        for r in self.results:
            sid = r["experiment"]["signal_id"]
            by_sig.setdefault(sid, [])
        corr = {"note": "see pairwise if multiple series stored"}
        incr = None
        if len(self.results) >= 2:
            incr = incremental_alpha(
                {
                    "net_pnl": self.results[0]["costs"]["net_pnl"],
                    "net_sharpe": self.results[0]["costs"]["net_sharpe"],
                    "turnover": self.results[0]["costs"]["turnover"],
                },
                {
                    "net_pnl": self.results[1]["costs"]["net_pnl"],
                    "net_sharpe": self.results[1]["costs"]["net_sharpe"],
                    "turnover": self.results[1]["costs"]["turnover"],
                },
            )

        stab_scores = {}
        for r in self.results:
            if r["experiment"]["signal_id"] == signal_ids[0] and r["experiment"]["timeframe"] == timeframes[0]:
                key = f"lb{r['experiment']['parameters'].get('lookback')}_h{r['experiment']['holding_period']}"
                stab_scores[key] = float(r["score"]["score"])
        center = next(iter(stab_scores), None)
        stability = parameter_stability(stab_scores, center_key=center) if center else {}

        mt = research_breadth_record(**breadth, n_experiments=len(self.results))
        matrix = [r["matrix_row"] for r in self.results]
        return {
            "title": "Alpha Research Matrix",
            "matrix": matrix,
            "n_experiments": len(self.results),
            "multiple_testing": mt,
            "incremental_example": incr,
            "parameter_stability_example": stability,
            "signal_correlation_note": corr,
            "disclaimers": [
                SAMPLE_TOO_SHORT_DISCLAIMER,
                "Research evidence is not a profitability guarantee.",
                "Research / simulated results only — not live performance.",
                "Reference signals are NOT claimed profitable.",
                "Single-instrument IC is time-series correlation, not cross-sectional IC.",
            ],
        }


__all__ = ["AlphaSignalResearchEngine"]

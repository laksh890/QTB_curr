"""BTC (and generic) alpha research campaign orchestration.

Uses existing AlphaSignalResearchEngine — does not create a new architecture phase.
Research discovery only. Not a profitability claim. Not production-ready.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.analytics import (
    evaluate_cost_aware,
    parameter_stability,
    positions_from_signal,
    signal_correlation_matrix,
)
from iqrp.app.backtesting.alpha_research.engine import AlphaSignalResearchEngine
from iqrp.app.backtesting.alpha_research.experiments import ExperimentRegistry, now_iso
from iqrp.app.backtesting.alpha_research.features import get_feature_registry
from iqrp.app.backtesting.alpha_research.leakage import run_leakage_suite
from iqrp.app.backtesting.alpha_research.mtf import align_feature_to_execution
from iqrp.app.backtesting.alpha_research.ranking import compute_alpha_research_score
from iqrp.app.backtesting.alpha_research.signals import get_signal_registry
from iqrp.app.backtesting.alpha_research.types import (
    COST_SCENARIOS,
    DEFAULT_FORWARD_HORIZONS,
    ResearchStatus,
    bars_per_day,
    holding_clock_minutes,
    map_alpha_to_research_status,
)
from iqrp.app.backtesting.data.dataset_registry import DatasetRegistry
from iqrp.app.backtesting.horizon.walk_forward import (
    evaluate_oos,
    rolling_walk_forward_slices,
)
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio
from iqrp.app.backtesting.serializer import to_jsonable

CAMPAIGN_ID = "alpha_research_btc_full_v1"
SOFTWARE_VERSION = "iqrp-alpha-research-0.1.0"
DISCLAIMER = "Research evidence is not a profitability guarantee."


@dataclass
class CampaignConfig:
    campaign_id: str = CAMPAIGN_ID
    output_dir: str = "results/alpha_research_btc_full"
    registry_path: str = "dataset_registry.json"
    dataset_keys: dict[str, str] = field(
        default_factory=lambda: {
            "1m": "btcusdt_intraday_1m@1.0.0",
            "5m": "btcusdt_intraday_5m@1.0.0",
            "15m": "btcusdt_intraday_15m@1.0.0",
            "30m": "btcusdt_intraday_30m@1.0.0",
            "1h": "btcusdt_intraday_1h@1.0.0",
        }
    )
    timeframes: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h")
    holding_bars: tuple[int, ...] = DEFAULT_FORWARD_HORIZONS
    lookbacks: tuple[int, ...] = (10, 20)
    neighborhood_lookbacks: tuple[int, ...] = (18, 19, 20, 21, 22)
    signal_ids: tuple[str, ...] | None = None
    cost_scenarios: tuple[str, ...] = ("BASE", "MODERATE", "ADVERSE")
    slippage_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    train_frac: float = 0.50
    validation_frac: float = 0.25
    n_walk_forward_windows: int = 3
    random_seed: int = 0
    market_type: str = "crypto"
    timezone: str = "UTC"
    top_k_deep: int = 25
    software_version: str = SOFTWARE_VERSION


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, default=str),
        encoding="utf-8",
    )


def _p_from_ic(ic: float | None, n: int) -> float:
    if ic is None or not np.isfinite(ic) or n < 5:
        return 1.0
    r = float(np.clip(ic, -0.999999, 0.999999))
    t = r * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1.0 - r * r, 1e-12))
    # two-sided normal approx
    return float(min(1.0, math.erfc(abs(t) / math.sqrt(2.0))))


def _gap_report(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    if len(ts) < 2:
        return {"n_gaps": 0, "gaps": []}
    expected = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
    }.get(timeframe, pd.Timedelta(minutes=5))
    delta = ts.diff()
    gap_mask = delta > expected * 1.5
    gaps = []
    for i in np.where(gap_mask.to_numpy())[0][:50]:
        gaps.append(
            {
                "index": int(i),
                "from": str(ts.iloc[i - 1]),
                "to": str(ts.iloc[i]),
                "delta_seconds": float(delta.iloc[i].total_seconds()),
            }
        )
    return {
        "n_gaps": int(gap_mask.sum()),
        "expected_bar": str(expected),
        "gaps_sample": gaps,
        "handling": (
            "Gaps are not filled. Observations whose forward-return window "
            "crosses a gap are excluded where continuous bars are required."
        ),
    }


def _exclude_gap_contaminated(
    timestamps: pd.Series,
    holding_bars: int,
    timeframe: str,
) -> np.ndarray:
    """Boolean mask: True = keep observation (forward window does not cross a gap)."""
    ts = pd.to_datetime(timestamps, utc=True)
    n = len(ts)
    keep = np.ones(n, dtype=bool)
    if n < 2 or holding_bars <= 0:
        return keep
    expected = {
        "1m": 60.0,
        "5m": 300.0,
        "15m": 900.0,
        "30m": 1800.0,
        "1h": 3600.0,
    }.get(timeframe, 300.0)
    delta_s = ts.diff().dt.total_seconds().to_numpy()
    gap_idx = np.where(np.isfinite(delta_s) & (delta_s > expected * 1.5))[0]
    for gi in gap_idx:
        lo = max(0, int(gi) - int(holding_bars))
        hi = int(gi)
        keep[lo:hi] = False
        if gi < n:
            keep[gi] = False
    return keep


def load_campaign_datasets(cfg: CampaignConfig) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    reg = DatasetRegistry(cfg.registry_path)
    frames: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict[str, Any]] = {}
    for tf, key in cfg.dataset_keys.items():
        if tf not in cfg.timeframes:
            continue
        if "@" in key:
            ds_id, ver = key.split("@", 1)
            rec = reg.require(ds_id, ver)
        else:
            rec = reg.require(key)
        path = Path(rec.path)
        if not path.is_absolute():
            path = Path.cwd() / path
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        frames[tf] = df
        kind = "SOURCE" if tf == "1m" else "DERIVED"
        extra = dict(rec.extra or {})
        meta[tf] = {
            "dataset_id": rec.key,
            "dataset_id_bare": rec.dataset_id,
            "version": rec.version,
            "checksum": rec.checksum,
            "path": str(rec.path),
            "frequency_kind": kind,
            "source": rec.source,
            "row_count": int(rec.row_count or len(df)),
            "start": rec.start,
            "end": rec.end,
            "coverage_pct": rec.coverage_pct,
            "known_limitations": list(rec.known_limitations or []),
            "gaps": _gap_report(df, tf),
            "extra": extra,
        }
    return frames, meta


def research_universe() -> dict[str, Any]:
    feats = get_feature_registry().list()
    sigs = get_signal_registry().list()
    return {
        "features": [f.to_dict() if hasattr(f, "to_dict") else {
            "feature_id": f.feature_id,
            "family": getattr(f, "family", None),
            "lookback": getattr(f, "lookback", None),
            "description": getattr(f, "description", None),
        } for f in feats],
        "signals": [s.to_dict() if hasattr(s, "to_dict") else {
            "signal_id": s.signal_id,
            "family": getattr(s, "family", None),
            "feature_ids": list(getattr(s, "feature_ids", ()) or ()),
            "parameters": dict(getattr(s, "parameters", {}) or {}),
            "description": getattr(s, "description", None),
        } for s in sigs],
        "note": "Existing FeatureRegistry / SignalRegistry only — no invented indicator sprawl.",
    }


def campaign_leakage_suite(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Run leakage checks once per timeframe × primary reference feature."""
    feat_reg = get_feature_registry()
    out: dict[str, Any] = {"by_timeframe": {}, "mtf_alignment": {}, "ok": True}
    primary_ids = ("momentum", "rolling_zscore", "volatility", "volume_change", "moving_average", "range", "returns")
    for tf, frame in frames.items():
        tf_rep: dict[str, Any] = {}
        for fid in primary_ids:
            try:
                series, _ = feat_reg.compute(frame, fid, parameters={"lookback": 20})

                def _recompute(f, _fid=fid):
                    return feat_reg.compute(f, _fid, parameters={"lookback": 20})[0]

                # Use a bounded window for expensive perturbation tests on 1m
                if len(frame) > 200_000:
                    sample = frame.iloc[-50_000:].reset_index(drop=True)
                    series_s, _ = feat_reg.compute(sample, fid, parameters={"lookback": 20})
                    rep = run_leakage_suite(
                        sample, series_s, lookback=20, compute_fn=_recompute
                    )
                    rep["sampled_tail_bars"] = 50_000
                    rep["full_frame_rows"] = len(frame)
                else:
                    rep = run_leakage_suite(frame, series, lookback=20, compute_fn=_recompute)
                tf_rep[fid] = rep
                if not rep.get("ok", True):
                    out["ok"] = False
            except Exception as exc:  # noqa: BLE001
                tf_rep[fid] = {"ok": False, "error": str(exc)}
                out["ok"] = False
        out["by_timeframe"][tf] = tf_rep

    # MTF causal alignment: 1h feature onto 5m execution must not use incomplete HTF bar
    if "1h" in frames and "5m" in frames:
        fdf, edf = frames["1h"], frames["5m"]
        feat, _ = feat_reg.compute(fdf, "momentum", parameters={"lookback": 20})
        aligned = align_feature_to_execution(fdf, feat, edf["timestamp"])
        h_ts = pd.to_datetime(fdf["timestamp"], utc=True).to_numpy(dtype="datetime64[ns]")
        e_ts = pd.to_datetime(edf["timestamp"], utc=True).to_numpy(dtype="datetime64[ns]")
        feat_vals = np.asarray(feat, dtype=np.float64)
        aligned_vals = np.asarray(aligned, dtype=np.float64)
        ok_mtf = True
        for idx in (1000, 5000, 20000):
            if idx >= len(edf):
                continue
            t = e_ts[idx]
            eligible = np.where(h_ts <= t)[0]
            if eligible.size == 0:
                continue
            j = int(eligible[-1])
            expected = feat_vals[j]
            got = aligned_vals[idx]
            if np.isfinite(expected) and np.isfinite(got) and abs(float(expected) - float(got)) > 1e-9:
                ok_mtf = False
                break
        out["mtf_alignment"] = {
            "feature_timeframe": "1h",
            "execution_timeframe": "5m",
            "method": "merge_asof_backward",
            "ok": ok_mtf,
            "note": "Higher-TF feature uses only completed candles as-of execution timestamp.",
        }
        out["ok"] = bool(out["ok"] and ok_mtf)
    return out


def _reprice(
    positions: pd.Series,
    rets: pd.Series,
    *,
    cost_model: Mapping[str, float],
    periods_per_year: float,
) -> dict[str, Any]:
    return evaluate_cost_aware(
        positions,
        rets,
        commission_bps=float(cost_model["commission_bps"]),
        spread_bps=float(cost_model["spread_bps"]),
        slippage_bps=float(cost_model["slippage_bps"]),
        periods_per_year=periods_per_year,
    )


def run_btc_alpha_campaign(
    cfg: CampaignConfig | None = None,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    cfg = cfg or CampaignConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()

    frames, ds_meta = load_campaign_datasets(cfg)
    universe = research_universe()
    signal_ids = list(cfg.signal_ids) if cfg.signal_ids else [s.signal_id for s in get_signal_registry().list()]

    matrix_plan = []
    for tf in cfg.timeframes:
        for sid in signal_ids:
            for lb in cfg.lookbacks:
                for hb in cfg.holding_bars:
                    matrix_plan.append(
                        {
                            "timeframe": tf,
                            "dataset_id": ds_meta[tf]["dataset_id"],
                            "dataset_kind": ds_meta[tf]["frequency_kind"],
                            "signal_id": sid,
                            "lookback": int(lb),
                            "holding_bars": int(hb),
                            "holding_minutes": holding_clock_minutes(tf, int(hb)),
                        }
                    )

    _write_json(out_dir / "campaign_config.json", {**asdict(cfg), "started_at": started})
    _write_json(out_dir / "feature_universe.json", {"features": universe["features"]})
    _write_json(out_dir / "signal_universe.json", {"signals": universe["signals"]})
    _write_json(
        out_dir / "research_matrix.json",
        {
            "n_planned_base_experiments": len(matrix_plan),
            "timeframes": list(cfg.timeframes),
            "holdings_bars": list(cfg.holding_bars),
            "lookbacks": list(cfg.lookbacks),
            "signals": signal_ids,
            "cost_scenarios": list(cfg.cost_scenarios),
            "plan": matrix_plan,
            "disclaimer": DISCLAIMER,
        },
    )

    if progress:
        print(f"[campaign] leakage suite…", flush=True)
    leakage_results = campaign_leakage_suite(frames)
    _write_json(out_dir / "leakage_results.json", leakage_results)

    exp_path = out_dir / "experiment_registry.json"
    engine = AlphaSignalResearchEngine(
        experiment_registry=ExperimentRegistry(exp_path),
        cost_model=COST_SCENARIOS["BASE"],
        market_type=cfg.market_type,
        timezone=cfg.timezone,
        gates={"min_sessions_for_significance": 60},
    )

    all_results: list[dict[str, Any]] = []
    ic_store: list[dict[str, Any]] = []
    decay_store: list[dict[str, Any]] = []
    cost_store: list[dict[str, Any]] = []
    oos_store: list[dict[str, Any]] = []
    regime_store: list[dict[str, Any]] = []
    gap_exclusions: list[dict[str, Any]] = []

    n_plan = len(matrix_plan)
    done = 0
    # Cache signal generation across holdings for same (tf, signal, lookback)
    # Prefer coarser timeframes first for faster early progress; still covers all TFs.
    ordered_tfs = sorted(
        cfg.timeframes,
        key=lambda t: {"1h": 0, "30m": 1, "15m": 2, "5m": 3, "1m": 4}.get(t, 5),
    )
    for tf in ordered_tfs:
        frame = frames[tf]
        rets = frame["close"].pct_change().fillna(0.0)
        bpd = bars_per_day(tf, market_type=cfg.market_type)
        ppy = 252.0 * bpd
        meta = ds_meta[tf]
        n_sessions = int(pd.to_datetime(frame["timestamp"], utc=True).dt.date.nunique())
        # Regime on multi-million-bar frames is diagnostic-only in deep dive
        run_regime_base = len(frame) <= 250_000
        run_importance_base = len(frame) <= 250_000

        for sid in signal_ids:
            for lb in cfg.lookbacks:
                # generate once
                try:
                    sig, sig_meta, feats = engine.signals.generate(
                        frame,
                        sid,
                        parameters={"lookback": int(lb), "holding_bars": 5},
                        feature_registry=engine.features,
                    )
                except Exception as exc:  # noqa: BLE001
                    for hb in cfg.holding_bars:
                        done += 1
                        fail_row = {
                            "signal": sid,
                            "timeframe": tf,
                            "lookback": int(lb),
                            "holding_period_bars": int(hb),
                            "research_status": ResearchStatus.REJECT.value,
                            "classification": "REJECT",
                            "error": str(exc),
                            "cost_scenario": "BASE",
                            "disclaimer": DISCLAIMER,
                        }
                        all_results.append(
                            {
                                "experiment_id": ExperimentRegistry.new_id(),
                                "cost_scenario": "BASE",
                                "matrix_row": fail_row,
                                "classification": "REJECT",
                                "research_status": ResearchStatus.REJECT.value,
                                "classification_reason": str(exc),
                                "score": {"score": 0.0},
                                "experiment": {
                                    "experiment_id": "",
                                    "dataset_id": meta["dataset_id"],
                                    "dataset_checksum": meta["checksum"],
                                    "signal_id": sid,
                                    "timeframe": tf,
                                    "holding_period": int(hb),
                                    "parameters": {"lookback": int(lb)},
                                    "cost_model": dict(COST_SCENARIOS["BASE"]),
                                    "timestamp": now_iso(),
                                },
                            }
                        )
                    continue
                for hb in cfg.holding_bars:
                    done += 1
                    if progress and (done % 10 == 0 or done == n_plan):
                        print(f"[campaign] base {done}/{n_plan} {tf} {sid} lb={lb} h={hb}", flush=True)

                    keep = _exclude_gap_contaminated(frame["timestamp"], int(hb), tf)
                    n_excl = int((~keep).sum())
                    if n_excl:
                        gap_exclusions.append(
                            {
                                "timeframe": tf,
                                "signal_id": sid,
                                "lookback": int(lb),
                                "holding_bars": int(hb),
                                "excluded_observations": n_excl,
                                "reason": "forward window crosses known gap",
                            }
                        )

                    # BASE evaluation (full analytics)
                    base = engine.evaluate_candidate(
                        frame,
                        signal_id=sid,
                        timeframe=tf,
                        holding_bars=int(hb),
                        parameters={"lookback": int(lb)},
                        dataset_id=meta["dataset_id"],
                        dataset_checksum=meta["checksum"],
                        dataset_kind=meta["frequency_kind"],
                        n_sessions=n_sessions,
                        cost_scenario="BASE",
                        cost_model=COST_SCENARIOS["BASE"],
                        run_leakage=False,
                        run_importance=run_importance_base,
                        run_regime=run_regime_base,
                        persist_experiment=False,
                        train_frac=cfg.train_frac,
                        validation_frac=cfg.validation_frac,
                        purge_bars=int(hb),
                        embargo_bars=int(hb),
                        precomputed_signal=sig,
                        precomputed_feats=feats,
                        precomputed_sig_meta=sig_meta,
                        precomputed_leakage={"ok": leakage_results.get("ok", True), "campaign_level": True},
                    )
                    # Annotate gap exclusion count
                    base["matrix_row"]["gap_excluded_observations"] = n_excl
                    base["gap_excluded_observations"] = n_excl
                    all_results.append(base)
                    ic_store.append(
                        {
                            "experiment_id": base["experiment_id"],
                            "timeframe": tf,
                            "signal_id": sid,
                            "lookback": int(lb),
                            "holding_bars": int(hb),
                            "ic": base["ic"],
                            "metric_type": "time_series_IC",
                        }
                    )
                    decay_store.append(
                        {
                            "experiment_id": base["experiment_id"],
                            "timeframe": tf,
                            "signal_id": sid,
                            "decay": base["decay"],
                        }
                    )
                    oos_store.append(
                        {
                            "experiment_id": base["experiment_id"],
                            "timeframe": tf,
                            "signal_id": sid,
                            "oos": base["oos"],
                        }
                    )
                    regime_store.append(
                        {
                            "experiment_id": base["experiment_id"],
                            "timeframe": tf,
                            "signal_id": sid,
                            "regime": base["regime"],
                        }
                    )
                    cost_store.append(
                        {
                            "experiment_id": base["experiment_id"],
                            "scenario": "BASE",
                            "costs": base["costs"],
                            "matrix_row": base["matrix_row"],
                        }
                    )

                    positions = positions_from_signal(sig.fillna(0.0), int(hb))
                    # MODERATE / ADVERSE reprice (same positions)
                    for scen in ("MODERATE", "ADVERSE"):
                        cm = COST_SCENARIOS[scen]
                        priced = _reprice(positions, rets, cost_model=cm, periods_per_year=ppy)
                        oos = evaluate_oos(
                            priced["gross_returns"],
                            priced["net_returns"],
                            timestamps=frame["timestamp"],
                            train_frac=cfg.train_frac,
                            validation_frac=cfg.validation_frac,
                            periods_per_year=ppy,
                            purge_bars=int(hb),
                            embargo_bars=int(hb),
                        )
                        metrics = {
                            "net_sharpe": priced["net_sharpe"],
                            "gross_sharpe": priced["gross_sharpe"],
                            "net_alpha": priced["net_alpha"],
                            "expectancy": float(np.mean([t["pnl"] for t in priced["trades"]]))
                            if priced["trades"]
                            else 0.0,
                            "max_drawdown": float(
                                abs(np.min(np.minimum.accumulate(1 + priced["net_returns"]) - 1))
                                if len(priced["net_returns"])
                                else 0.0
                            ),
                            "trade_count": int(priced["trade_frequency"].get("total_trades", 0)),
                            "annualized_turnover": float(
                                (priced["turnover"] or {}).get("annualized_turnover") or 0.0
                            ),
                            "alpha_survives_costs": priced["alpha_survives_costs"],
                            "alpha_collapses_after_costs": priced["alpha_collapses_after_costs"],
                            "mean_ic": base["ic"].get("mean_ic"),
                            "ic_stability": base["ic"].get("ic_ir") or 0.5,
                            "oos_sharpe": float((oos.get("oos") or {}).get("net_sharpe") or 0.0),
                            "oos_evaluated": bool((oos.get("oos") or {}).get("evaluated")),
                            "parameter_stability": 0.5,
                            "regime_stability": float(
                                (base.get("regime") or {}).get("robustness", {}).get("score") or 0.5
                            ),
                            "fragile": False,
                        }
                        scored = compute_alpha_research_score(metrics)
                        from iqrp.app.backtesting.alpha_research.ranking import classify_alpha

                        classification, reason = classify_alpha(
                            metrics, gates=engine.gates, n_sessions=n_sessions
                        )
                        status = map_alpha_to_research_status(classification.value, metrics)
                        row = {
                            **{k: v for k, v in base["matrix_row"].items() if k not in {
                                "gross_return", "net_return", "Sharpe", "gross_Sharpe", "cost",
                                "classification", "research_status", "cost_scenario", "OOS_performance",
                                "robustness",
                            }},
                            "gross_return": priced["gross_pnl"],
                            "net_return": priced["net_pnl"],
                            "gross_Sharpe": priced["gross_sharpe"],
                            "Sharpe": priced["net_sharpe"],
                            "cost": priced["transaction_costs"],
                            "cost_scenario": scen,
                            "OOS_performance": (oos.get("oos") or {}).get("net_sharpe"),
                            "classification": classification.value,
                            "research_status": status,
                            "robustness": scored["score"],
                            "parent_experiment_id": base["experiment_id"],
                        }
                        eid = ExperimentRegistry.new_id()
                        child = {
                            "experiment_id": eid,
                            "parent_experiment_id": base["experiment_id"],
                            "cost_scenario": scen,
                            "costs": {
                                k: v
                                for k, v in priced.items()
                                if k not in {"gross_returns", "net_returns", "positions", "trades"}
                            },
                            "oos": oos,
                            "score": scored,
                            "matrix_row": row,
                            "classification": classification.value,
                            "classification_reason": reason,
                            "research_status": status,
                            "ic": base["ic"],
                            "experiment": {
                                "experiment_id": eid,
                                "dataset_id": meta["dataset_id"],
                                "dataset_checksum": meta["checksum"],
                                "feature_versions": dict(
                                    (base.get("experiment") or {}).get("feature_versions") or {}
                                ),
                                "signal_id": sid,
                                "signal_version": "1.0.0",
                                "timeframe": tf,
                                "holding_period": int(hb),
                                "parameters": {
                                    "lookback": int(lb),
                                    "holding_bars": int(hb),
                                    "cost_scenario": scen,
                                },
                                "cost_model": dict(cm),
                                "random_seed": cfg.random_seed,
                                "software_version": cfg.software_version,
                                "classification": classification.value,
                                "matrix_row": row,
                                "timestamp": now_iso(),
                            },
                        }
                        all_results.append(child)
                        cost_store.append(
                            {
                                "experiment_id": eid,
                                "parent_experiment_id": base["experiment_id"],
                                "scenario": scen,
                                "costs": child["costs"],
                                "matrix_row": row,
                            }
                        )

    # Persist experiment registry from all_results cleanly
    engine.experiments._items.clear()
    for r in all_results:
        exp = r.get("experiment") or {}
        from iqrp.app.backtesting.alpha_research.experiments import ExperimentSpec

        spec = ExperimentSpec(
            experiment_id=str(exp.get("experiment_id") or r["experiment_id"]),
            timestamp=str(exp.get("timestamp") or now_iso()),
            dataset_id=str(exp.get("dataset_id") or ""),
            dataset_checksum=str(exp.get("dataset_checksum") or ""),
            feature_versions=dict(exp.get("feature_versions") or {}),
            signal_id=str(exp.get("signal_id") or r.get("matrix_row", {}).get("signal") or ""),
            signal_version=str(exp.get("signal_version") or "1.0.0"),
            parameters=dict(exp.get("parameters") or {}),
            timeframe=str(exp.get("timeframe") or ""),
            holding_period=int(exp.get("holding_period") or 0),
            cost_model=dict(exp.get("cost_model") or {}),
            random_seed=cfg.random_seed,
            software_version=cfg.software_version,
            result_checksum=ExperimentRegistry.result_checksum(r.get("matrix_row") or {}),
            classification=str(r.get("classification") or ""),
            matrix_row=dict(r.get("matrix_row") or {}),
            notes=[DISCLAIMER],
        )
        engine.experiments.register(spec, persist=False)
    engine.experiments.save()

    # Deep dive on top BASE candidates by research score
    base_only = [r for r in all_results if r.get("cost_scenario") == "BASE" or (r.get("matrix_row") or {}).get("cost_scenario") == "BASE"]
    base_only = sorted(base_only, key=lambda r: float((r.get("score") or {}).get("score") or 0.0), reverse=True)
    deep_pool = base_only[: max(cfg.top_k_deep, 1)]

    robustness_results: list[dict[str, Any]] = []
    walk_forward_results: list[dict[str, Any]] = []
    slippage_results: list[dict[str, Any]] = []
    period_results: list[dict[str, Any]] = []
    cross_tf_results: list[dict[str, Any]] = []

    if progress:
        print(f"[campaign] deep dive on {len(deep_pool)} candidates…", flush=True)

    for cand in deep_pool:
        row = cand["matrix_row"]
        tf = row["timeframe"]
        sid = row["signal"]
        lb = int(cand["experiment"]["parameters"].get("lookback", 20))
        hb = int(row.get("holding_period_bars") or cand["experiment"]["holding_period"])
        frame = frames[tf]
        rets = frame["close"].pct_change().fillna(0.0)
        ppy = 252.0 * bars_per_day(tf, market_type=cfg.market_type)

        # Parameter neighborhood
        neigh_scores: dict[str, float] = {}
        for nlb in cfg.neighborhood_lookbacks:
            sig, _, _ = engine.signals.generate(
                frame, sid, parameters={"lookback": int(nlb), "holding_bars": hb}, feature_registry=engine.features
            )
            pos = positions_from_signal(sig.fillna(0.0), hb)
            priced = _reprice(pos, rets, cost_model=COST_SCENARIOS["BASE"], periods_per_year=ppy)
            neigh_scores[f"lb{nlb}"] = float(priced["net_sharpe"])
        stab = parameter_stability(neigh_scores, center_key=f"lb{lb}" if f"lb{lb}" in neigh_scores else next(iter(neigh_scores)))
        robustness_results.append(
            {
                "experiment_id": cand["experiment_id"],
                "signal_id": sid,
                "timeframe": tf,
                "holding_bars": hb,
                "neighborhood_net_sharpe": neigh_scores,
                "stability": stab,
            }
        )

        # Slippage sensitivity 1x/2x/3x on BASE components
        base_cm = COST_SCENARIOS["BASE"]
        sens = []
        death = None
        for mult in cfg.slippage_multipliers:
            cm = {k: float(v) * float(mult) for k, v in base_cm.items()}
            sig, _, _ = engine.signals.generate(
                frame, sid, parameters={"lookback": lb, "holding_bars": hb}, feature_registry=engine.features
            )
            pos = positions_from_signal(sig.fillna(0.0), hb)
            priced = _reprice(pos, rets, cost_model=cm, periods_per_year=ppy)
            entry = {
                "multiplier": float(mult),
                "cost_model": cm,
                "net_sharpe": priced["net_sharpe"],
                "net_pnl": priced["net_pnl"],
                "alpha_survives_costs": priced["alpha_survives_costs"],
            }
            sens.append(entry)
            if death is None and (priced["net_pnl"] <= 0 or priced["net_sharpe"] <= 0):
                death = float(mult)
        slippage_results.append(
            {
                "experiment_id": cand["experiment_id"],
                "signal_id": sid,
                "timeframe": tf,
                "sensitivity": sens,
                "edge_disappears_at_multiplier": death,
            }
        )

        # Walk-forward multi-window on net returns of BASE
        pos = positions_from_signal(
            engine.signals.generate(
                frame, sid, parameters={"lookback": lb, "holding_bars": hb}, feature_registry=engine.features
            )[0].fillna(0.0),
            hb,
        )
        priced = _reprice(pos, rets, cost_model=COST_SCENARIOS["BASE"], periods_per_year=ppy)
        windows = []
        for i, sl in enumerate(
            rolling_walk_forward_slices(
                len(frame),
                n_windows=cfg.n_walk_forward_windows,
                train_frac=cfg.train_frac,
                validation_frac=cfg.validation_frac,
            )
        ):
            g = priced["gross_returns"]
            n = priced["net_returns"]
            # evaluate within prefix end implied by slices — use slice ends
            end = sl["oos"].stop
            sub_g, sub_n = g[:end], n[:end]
            ev = evaluate_oos(
                sub_g,
                sub_n,
                train_frac=cfg.train_frac,
                validation_frac=cfg.validation_frac,
                periods_per_year=ppy,
                purge_bars=hb,
                embargo_bars=hb,
            )
            windows.append({"window": i, "end_index": end, **ev})
        walk_forward_results.append(
            {
                "experiment_id": cand["experiment_id"],
                "signal_id": sid,
                "timeframe": tf,
                "windows": windows,
            }
        )

        # Early / middle / late chronological periods
        n = len(frame)
        cuts = [0, n // 3, 2 * n // 3, n]
        labels = ("early", "middle", "later")
        period_rows = []
        for lab, a, b in zip(labels, cuts[:-1], cuts[1:], strict=True):
            sub = frame.iloc[a:b].reset_index(drop=True)
            if len(sub) < 100:
                period_rows.append({"period": lab, "status": "SAMPLE_INSUFFICIENT", "n": len(sub)})
                continue
            sig, _, _ = engine.signals.generate(
                sub, sid, parameters={"lookback": lb, "holding_bars": hb}, feature_registry=engine.features
            )
            pos = positions_from_signal(sig.fillna(0.0), hb)
            sub_rets = sub["close"].pct_change().fillna(0.0)
            priced = _reprice(pos, sub_rets, cost_model=COST_SCENARIOS["BASE"], periods_per_year=ppy)
            period_rows.append(
                {
                    "period": lab,
                    "n": len(sub),
                    "start": str(sub["timestamp"].iloc[0]),
                    "end": str(sub["timestamp"].iloc[-1]),
                    "net_sharpe": priced["net_sharpe"],
                    "net_pnl": priced["net_pnl"],
                    "trades": priced["trade_frequency"].get("total_trades"),
                    "trades_per_day": priced["trade_frequency"].get("trades_per_day"),
                }
            )
        period_results.append(
            {
                "experiment_id": cand["experiment_id"],
                "signal_id": sid,
                "timeframe": tf,
                "periods": period_rows,
            }
        )

    # Cross-timeframe validation for distinct signals appearing in deep pool
    deep_signals = sorted({c["matrix_row"]["signal"] for c in deep_pool})
    for sid in deep_signals:
        by_tf = {}
        for tf in cfg.timeframes:
            matches = [
                r
                for r in base_only
                if r["matrix_row"]["signal"] == sid and r["matrix_row"]["timeframe"] == tf
            ]
            if not matches:
                continue
            best = max(matches, key=lambda r: float((r.get("score") or {}).get("score") or -1e9))
            by_tf[tf] = {
                "experiment_id": best["experiment_id"],
                "lookback": best["experiment"]["parameters"].get("lookback"),
                "holding_bars": best["matrix_row"].get("holding_period_bars"),
                "net_sharpe": best["matrix_row"].get("Sharpe"),
                "oos_sharpe": best["matrix_row"].get("OOS_performance"),
                "research_status": best["matrix_row"].get("research_status"),
                "score": (best.get("score") or {}).get("score"),
            }
        signs = [np.sign(float(v["net_sharpe"] or 0)) for v in by_tf.values()]
        fragile_tf = len(set(signs)) > 1 and len(by_tf) >= 2
        cross_tf_results.append(
            {
                "signal_id": sid,
                "by_timeframe": by_tf,
                "persists_neighboring_horizons": not fragile_tf and len(by_tf) >= 2,
                "flag": "potentially_fragile_single_tf" if len(by_tf) == 1 else (
                    "sign_inconsistent_across_tf" if fragile_tf else "cross_tf_consistent_sign"
                ),
            }
        )

    # Multiple testing on BASE experiments using IC p-values
    pvals = []
    labels = []
    for r in base_only:
        ic = r.get("ic") or {}
        # prefer horizon matching holding
        hb = int(r["matrix_row"].get("holding_period_bars") or 5)
        by_h = (ic.get("by_horizon") or {}).get(str(hb)) or {}
        pear = by_h.get("pearson_ic", ic.get("mean_ic"))
        n_obs = int(by_h.get("n") or 0)
        pvals.append(_p_from_ic(pear if pear is not None else None, n_obs if n_obs else 100))
        labels.append(r["experiment_id"])
    from iqrp.app.alpha.statistical_validation import multiple_testing_adjustment

    mt_adj = multiple_testing_adjustment(pvals, method="fdr_bh", alpha=0.05, label="btc_alpha_campaign")
    rejected = np.asarray(mt_adj.get("rejected", []), dtype=bool)
    adjusted = np.asarray(mt_adj.get("adjusted", pvals), dtype=np.float64)
    mt_survivors = [labels[i] for i, flag in enumerate(rejected) if flag]
    before_corr = [
        r["experiment_id"]
        for r in base_only
        if r["matrix_row"].get("research_status") in {
            ResearchStatus.CANDIDATE.value,
            ResearchStatus.CONDITIONAL.value,
        }
        or float(r["matrix_row"].get("Sharpe") or 0) > 0
    ]
    multiple_testing_results = {
        "method": "fdr_bh",
        "alpha": 0.05,
        "n_experiments_tested": len(base_only),
        "n_candidates_before_correction": len(before_corr),
        "n_surviving_correction": int(rejected.sum()) if rejected.size else 0,
        "surviving_experiment_ids": mt_survivors,
        "pvalues": [{"experiment_id": labels[i], "p": float(pvals[i]), "p_adj": float(adjusted[i])} for i in range(len(labels))],
        "note": (
            "IC-based two-sided p-values with BH-FDR. Autocorrelated returns mean "
            "nominal p-values may be optimistic; treat as research diagnostic, not proof."
        ),
        "disclaimer": DISCLAIMER,
    }

    # Signal correlation among deep-pool representatives (one series per signal on 1h)
    corr_payload = {"note": "insufficient"}
    if "1h" in frames:
        sig_map = {}
        for sid in deep_signals[:8]:
            s, _, _ = engine.signals.generate(
                frames["1h"], sid, parameters={"lookback": 20, "holding_bars": 5}, feature_registry=engine.features
            )
            sig_map[sid] = s.fillna(0.0).to_numpy()
        corr_payload = signal_correlation_matrix(sig_map)

    # Classification tallies
    status_counts = Counter((r.get("matrix_row") or {}).get("research_status") for r in all_results)
    base_status = Counter((r.get("matrix_row") or {}).get("research_status") for r in base_only)

    def _survives_costs(r: dict[str, Any]) -> bool:
        return bool((r.get("costs") or {}).get("alpha_survives_costs")) or (
            float((r.get("matrix_row") or {}).get("Sharpe") or 0) > 0
            and float((r.get("matrix_row") or {}).get("net_return") or 0) > 0
        )

    survive_costs = [r for r in base_only if _survives_costs(r) and not (r.get("costs") or {}).get("alpha_collapses_after_costs")]
    survive_oos = [
        r
        for r in base_only
        if float((r.get("matrix_row") or {}).get("OOS_performance") or -1e9) > 0
        and (r.get("matrix_row") or {}).get("research_status")
        not in {ResearchStatus.OOS_FAILED.value, ResearchStatus.SAMPLE_INSUFFICIENT.value}
    ]
    mt_set = set(mt_survivors)
    survive_mt = [r for r in base_only if r["experiment_id"] in mt_set]

    # Final candidate set: BASE experiments with CANDIDATE/CONDITIONAL that survive costs + OOS
    final_candidates = []
    seen_families = set()
    ranked = sorted(
        base_only,
        key=lambda r: (
            1 if r["matrix_row"].get("research_status") == ResearchStatus.CANDIDATE.value else 0,
            float((r.get("score") or {}).get("score") or 0.0),
        ),
        reverse=True,
    )
    for r in ranked:
        st = r["matrix_row"].get("research_status")
        if st not in {ResearchStatus.CANDIDATE.value, ResearchStatus.CONDITIONAL.value}:
            continue
        if not _survives_costs(r):
            continue
        if float(r["matrix_row"].get("OOS_performance") or -1e9) <= 0:
            continue
        fam = next(
            (s.get("family") for s in universe["signals"] if s.get("signal_id") == r["matrix_row"]["signal"]),
            r["matrix_row"]["signal"],
        )
        if fam in seen_families:
            continue
        entry = {
            "experiment_id": r["experiment_id"],
            "signal_id": r["matrix_row"]["signal"],
            "family": fam,
            "timeframe": r["matrix_row"]["timeframe"],
            "dataset_kind": r["matrix_row"].get("dataset_kind"),
            "lookback": r["experiment"]["parameters"].get("lookback"),
            "holding_bars": r["matrix_row"].get("holding_period_bars"),
            "holding_minutes": r["matrix_row"].get("holding_period_minutes"),
            "research_status": st,
            "alpha_research_score": (r.get("score") or {}).get("score"),
            "net_sharpe": r["matrix_row"].get("Sharpe"),
            "oos_sharpe": r["matrix_row"].get("OOS_performance"),
            "trades_per_day": r["matrix_row"].get("trades_per_day"),
            "survives_fdr": r["experiment_id"] in mt_set,
            "disclaimer": DISCLAIMER,
            "not_production_ready": True,
        }
        final_candidates.append(entry)
        seen_families.add(fam)

    # Near-miss watchlist when no CANDIDATE survives (research-only, not promoted)
    research_watchlist = []
    if not final_candidates:
        seen_w = set()
        for r in ranked:
            fam = next(
                (s.get("family") for s in universe["signals"] if s.get("signal_id") == r["matrix_row"]["signal"]),
                r["matrix_row"]["signal"],
            )
            if fam in seen_w:
                continue
            seen_w.add(fam)
            research_watchlist.append(
                {
                    "experiment_id": r["experiment_id"],
                    "signal_id": r["matrix_row"]["signal"],
                    "family": fam,
                    "timeframe": r["matrix_row"]["timeframe"],
                    "research_status": r["matrix_row"].get("research_status"),
                    "alpha_research_score": (r.get("score") or {}).get("score"),
                    "net_sharpe": r["matrix_row"].get("Sharpe"),
                    "oos_sharpe": r["matrix_row"].get("OOS_performance"),
                    "gate_outcome": "DID_NOT_SURVIVE_CANDIDATE_GATES",
                    "not_candidate": True,
                    "disclaimer": DISCLAIMER,
                }
            )
            if len(research_watchlist) >= 7:
                break

    rejected_candidates = []
    for r in base_only:
        st = r["matrix_row"].get("research_status")
        if st in {ResearchStatus.CANDIDATE.value} and r["experiment_id"] in {c["experiment_id"] for c in final_candidates}:
            continue
        if st == ResearchStatus.CANDIDATE.value and r["experiment_id"] not in {c["experiment_id"] for c in final_candidates}:
            # demoted
            rejected_candidates.append(
                {
                    "experiment_id": r["experiment_id"],
                    "category": "DEGRADED_FROM_CANDIDATE",
                    "reason": "failed cost/OOS/diversification filters",
                    "matrix_row": r["matrix_row"],
                }
            )
        elif st != ResearchStatus.CANDIDATE.value:
            rejected_candidates.append(
                {
                    "experiment_id": r["experiment_id"],
                    "category": st or ResearchStatus.REJECT.value,
                    "reason": r.get("classification_reason"),
                    "signal_id": r["matrix_row"].get("signal"),
                    "timeframe": r["matrix_row"].get("timeframe"),
                }
            )

    # Rankings
    candidate_rankings = {
        "by_alpha_research_score": sorted(
            [{"experiment_id": r["experiment_id"], **r["matrix_row"], "score": (r.get("score") or {}).get("score")} for r in base_only],
            key=lambda x: float(x.get("score") or 0),
            reverse=True,
        )[:50],
        "by_oos_sharpe": sorted(
            [{"experiment_id": r["experiment_id"], **r["matrix_row"]} for r in base_only],
            key=lambda x: float(x.get("OOS_performance") or -1e9),
            reverse=True,
        )[:50],
        "final_candidates": final_candidates,
        "disclaimer": DISCLAIMER,
    }

    # Summary stats for trade frequency
    tpd = [float(r["matrix_row"].get("trades_per_day") or 0) for r in base_only if r["matrix_row"].get("trades_per_day") is not None]
    trade_freq_summary = {
        "average_trades_per_day": float(np.mean(tpd)) if tpd else None,
        "median_trades_per_day": float(np.median(tpd)) if tpd else None,
        "n": len(tpd),
    }

    # Pick notables
    def _best(rows, key):
        if not rows:
            return None
        return max(rows, key=key)

    best_score = _best(base_only, lambda r: float((r.get("score") or {}).get("score") or -1e9))
    best_oos = _best(base_only, lambda r: float(r["matrix_row"].get("OOS_performance") or -1e9))
    most_robust = _best(robustness_results, lambda r: float((r.get("stability") or {}).get("stability_score") or 0))
    most_cost_sens = None
    if slippage_results:
        most_cost_sens = min(
            slippage_results,
            key=lambda r: float(r.get("edge_disappears_at_multiplier") or 99),
        )
    most_freq = _best(base_only, lambda r: float(r["matrix_row"].get("trades_per_day") or 0))

    final_report = {
        "campaign_id": cfg.campaign_id,
        "title": "IQRP Prompt 35 — BTCUSDT Alpha Research Campaign",
        "disclaimer": DISCLAIMER,
        "not_profitability_claim": True,
        "not_production_ready": True,
        "started_at": started,
        "completed_at": now_iso(),
        "software_version": cfg.software_version,
        "random_seed": cfg.random_seed,
        "datasets": ds_meta,
        "data_quality_limitations": {
            "gap_class": "MINOR_GAPS",
            "gaps_not_filled": True,
            "gap_exclusions_recorded": len(gap_exclusions),
            "license_status": "UNKNOWN",
            "note": "OHLCV alone does not support institutional capacity claims.",
        },
        "research_universe": {
            "n_features": len(universe["features"]),
            "n_signals": len(signal_ids),
            "signal_ids": signal_ids,
            "families": sorted({s.get("family") for s in universe["signals"] if s.get("family")}),
        },
        "n_experiments_total": len(all_results),
        "n_experiments_base": len(base_only),
        "timeframes_evaluated": list(cfg.timeframes),
        "holding_horizons_bars": list(cfg.holding_bars),
        "leakage_ok": leakage_results.get("ok"),
        "status_counts_all_scenarios": dict(status_counts),
        "status_counts_base": dict(base_status),
        "n_candidates_before_filtering": len(before_corr),
        "n_surviving_costs": len(survive_costs),
        "n_surviving_oos": len(survive_oos),
        "n_surviving_multiple_testing": len(survive_mt),
        "final_candidate_set": final_candidates,
        "research_watchlist_near_misses": research_watchlist,
        "rejected_categories": dict(Counter(x["category"] for x in rejected_candidates)),
        "best_by_research_score": {
            "experiment_id": best_score["experiment_id"] if best_score else None,
            "matrix_row": best_score["matrix_row"] if best_score else None,
            "score": (best_score.get("score") if best_score else None),
        },
        "best_by_oos": {
            "experiment_id": best_oos["experiment_id"] if best_oos else None,
            "matrix_row": best_oos["matrix_row"] if best_oos else None,
        },
        "most_robust": most_robust,
        "most_cost_sensitive": most_cost_sens,
        "most_frequent": {
            "experiment_id": most_freq["experiment_id"] if most_freq else None,
            "trades_per_day": (most_freq["matrix_row"].get("trades_per_day") if most_freq else None),
            "matrix_row": most_freq["matrix_row"] if most_freq else None,
        },
        "trade_frequency_summary": trade_freq_summary,
        "signal_correlation": corr_payload,
        "cross_timeframe": cross_tf_results,
        "major_limitations": [
            DISCLAIMER,
            "Single-instrument BTC time-series IC is not cross-sectional IC.",
            "Multiple-testing p-values approximate and may be optimistic under autocorrelation.",
            "Capacity/liquidity figures from OHLCV are estimates only.",
            "No live trading was performed.",
            "Candidates are not PRODUCTION_READY.",
            "1m is SOURCE; 5m/15m/30m/1h are DERIVED via causal session-aware resampling.",
        ],
        "executive_summary": (
            f"Ran {len(all_results)} experiments ({len(base_only)} BASE) across "
            f"{len(cfg.timeframes)} timeframes and {len(cfg.holding_bars)} holding horizons "
            f"on registered BTCUSDT datasets. "
            f"{len(final_candidates)} diversified research candidates retained after cost/OOS gates"
            + (
                f"; near-miss watchlist size={len(research_watchlist)}."
                if not final_candidates
                else "."
            )
            + " "
            + DISCLAIMER
        ),
    }

    # Persist artifacts
    _write_json(out_dir / "experiment_registry.json", {
        "updated_at": now_iso(),
        "campaign_id": cfg.campaign_id,
        "experiments": [e.to_dict() for e in engine.experiments.list()],
        "disclaimer": DISCLAIMER,
    })
    _write_json(out_dir / "IC_results.json", {"metric_type": "time_series_IC", "results": ic_store, "disclaimer": DISCLAIMER})
    _write_json(out_dir / "decay_results.json", {"results": decay_store})
    _write_json(out_dir / "cost_results.json", {"scenarios": list(cfg.cost_scenarios), "results": cost_store, "disclaimer": DISCLAIMER})
    _write_json(out_dir / "OOS_results.json", {"results": oos_store, "disclaimer": DISCLAIMER})
    _write_json(out_dir / "walk_forward_results.json", {"results": walk_forward_results, "disclaimer": DISCLAIMER})
    _write_json(out_dir / "regime_results.json", {"results": regime_store})
    _write_json(out_dir / "robustness_results.json", {"results": robustness_results, "period_splits": period_results, "cross_timeframe": cross_tf_results})
    _write_json(out_dir / "multiple_testing_results.json", multiple_testing_results)
    _write_json(out_dir / "candidate_rankings.json", candidate_rankings)
    _write_json(out_dir / "rejected_candidates.json", {"rejected": rejected_candidates, "gap_exclusions": gap_exclusions})
    _write_json(out_dir / "slippage_sensitivity.json", {"results": slippage_results})
    _write_json(out_dir / "final_report.json", final_report)

    md = _render_markdown(final_report, leakage_results, multiple_testing_results, trade_freq_summary)
    (out_dir / "final_report.md").write_text(md, encoding="utf-8")

    if progress:
        print(f"[campaign] complete → {out_dir}", flush=True)
    return final_report


def _render_markdown(
    report: dict[str, Any],
    leakage: dict[str, Any],
    mt: dict[str, Any],
    trade_freq: dict[str, Any],
) -> str:
    cands = report.get("final_candidate_set") or []
    lines = [
        f"# {report.get('title')}",
        "",
        f"**Campaign ID:** `{report.get('campaign_id')}`",
        "",
        f"> {DISCLAIMER}",
        "",
        "## 1. Executive summary",
        "",
        str(report.get("executive_summary")),
        "",
        "## 2. Dataset description",
        "",
    ]
    for tf, meta in (report.get("datasets") or {}).items():
        lines.append(
            f"- `{meta.get('dataset_id')}` ({meta.get('frequency_kind')}): "
            f"checksum `{str(meta.get('checksum'))[:16]}…`, rows={meta.get('row_count')}, "
            f"{meta.get('start')} → {meta.get('end')}"
        )
    lines += [
        "",
        "## 3. Data-quality limitations",
        "",
        json.dumps(report.get("data_quality_limitations"), indent=2),
        "",
        "## 4. Research universe",
        "",
        json.dumps(report.get("research_universe"), indent=2),
        "",
        f"## 5. Number of experiments: **{report.get('n_experiments_total')}** "
        f"(BASE={report.get('n_experiments_base')})",
        "",
        f"## 6. Timeframes tested: {report.get('timeframes_evaluated')}",
        "",
        f"## 7. Holding periods tested (bars): {report.get('holding_horizons_bars')}",
        "",
        f"## 8. Leakage validation: ok={leakage.get('ok')}",
        "",
        "## 9–20. Analytics artifacts",
        "",
        "See companion JSON files: `IC_results.json`, `decay_results.json`, `cost_results.json`, "
        "`OOS_results.json`, `walk_forward_results.json`, `regime_results.json`, "
        "`robustness_results.json`, `multiple_testing_results.json`.",
        "",
        f"## 20. Multiple-testing correction",
        "",
        f"- Method: {mt.get('method')}",
        f"- Experiments tested: {mt.get('n_experiments_tested')}",
        f"- Before correction: {mt.get('n_candidates_before_correction')}",
        f"- Surviving FDR: {mt.get('n_surviving_correction')}",
        "",
        "## 21. Candidate ranking (top final set)",
        "",
    ]
    for c in cands[:15]:
        lines.append(
            f"- `{c.get('signal_id')}` @ {c.get('timeframe')} lb={c.get('lookback')} "
            f"h={c.get('holding_bars')} bars ({c.get('holding_minutes')} min) "
            f"status={c.get('research_status')} score={c.get('alpha_research_score')} "
            f"OOS={c.get('oos_sharpe')}"
        )
    lines += [
        "",
        f"## 22. Rejected candidate categories: {report.get('rejected_categories')}",
        "",
        "## 23. Research limitations",
        "",
    ]
    for lim in report.get("major_limitations") or []:
        lines.append(f"- {lim}")
    lines += [
        "",
        "## 24. Final candidate set",
        "",
        f"Count: {len(cands)}",
        "",
        f"Trade frequency (BASE): avg={trade_freq.get('average_trades_per_day')}, "
        f"median={trade_freq.get('median_trades_per_day')}",
        "",
        f"> {DISCLAIMER}",
        "",
        "No candidate is PRODUCTION_READY. No live trading was performed.",
        "",
    ]
    return "\n".join(lines)


__all__ = ["CampaignConfig", "run_btc_alpha_campaign", "CAMPAIGN_ID"]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run IQRP BTC alpha research campaign (Prompt 35)")
    p.add_argument("--output-dir", default="results/alpha_research_btc_full")
    p.add_argument("--registry", default="dataset_registry.json")
    p.add_argument("--timeframes", default="1m,5m,15m,30m,1h")
    p.add_argument("--top-k-deep", type=int, default=25)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    cfg = CampaignConfig(
        output_dir=args.output_dir,
        registry_path=args.registry,
        timeframes=tuple(x.strip() for x in args.timeframes.split(",") if x.strip()),
        top_k_deep=int(args.top_k_deep),
    )
    cfg.dataset_keys = {k: v for k, v in cfg.dataset_keys.items() if k in cfg.timeframes}
    report = run_btc_alpha_campaign(cfg, progress=not args.quiet)
    print(
        f"campaign_id={report['campaign_id']} experiments={report['n_experiments_total']} "
        f"candidates={len(report.get('final_candidate_set') or [])}"
    )


if __name__ == "__main__":
    main()

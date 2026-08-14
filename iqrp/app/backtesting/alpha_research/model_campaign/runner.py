"""Prompt 39 model-driven alpha research campaign runner.

Uses existing adapters + AlphaSignalResearchEngine. Predeclared protocol only.
"""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.adapters.model_registry import (
    register_adapter,
    register_default_adapters,
)
from iqrp.app.backtesting.alpha_research.adapters.pipeline import align_model_signal_mtf, run_adapter
from iqrp.app.backtesting.alpha_research.adapters.types import (
    ModelAdapterSpec,
    OutputMappingKind,
    SignalMappingConfig,
)
from iqrp.app.backtesting.alpha_research.analytics import positions_from_signal
from iqrp.app.backtesting.alpha_research.campaign import load_campaign_datasets
from iqrp.app.backtesting.alpha_research.campaign import CampaignConfig as _LegacyCfg
from iqrp.app.backtesting.alpha_research.engine import AlphaSignalResearchEngine
from iqrp.app.backtesting.alpha_research.experiments import ExperimentRegistry, now_iso
from iqrp.app.backtesting.alpha_research.model_campaign.protocol import (
    COMBINATIONS,
    DISCLAIMER,
    ENSEMBLES,
    MODEL_SPECS,
    MTF_PAIRS,
    REFERENCE_SIGNALS,
    ModelCampaignConfig,
    apply_direction_mask,
    combine_and_agree,
)
from iqrp.app.backtesting.alpha_research.mtf import align_feature_to_execution
from iqrp.app.backtesting.alpha_research.signals import get_signal_registry
from iqrp.app.backtesting.alpha_research.types import (
    COST_SCENARIOS,
    ResearchStatus,
    bars_per_day,
    holding_clock_minutes,
    map_alpha_to_research_status,
)
from iqrp.app.backtesting.serializer import to_jsonable


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str), encoding="utf-8")


def _p_from_ic(ic: float | None, n: int) -> float:
    if ic is None or not np.isfinite(ic) or n < 5:
        return 1.0
    r = float(np.clip(ic, -0.999999, 0.999999))
    t = r * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1.0 - r * r, 1e-12))
    return float(min(1.0, math.erfc(abs(t) / math.sqrt(2.0))))


def _trade_stats(positions: pd.Series, timeframe: str, market_type: str = "crypto") -> dict[str, Any]:
    pos = positions.fillna(0.0).to_numpy(dtype=float)
    changes = np.diff(pos, prepend=0.0)
    entries = np.where(np.abs(changes) > 1e-12)[0]
    n_trades = int(np.sum(np.abs(changes) > 1e-12))
    long_entries = int(np.sum(changes > 1e-12))
    short_entries = int(np.sum(changes < -1e-12))
    # holding lengths between flips
    holds: list[int] = []
    i = 0
    n = len(pos)
    while i < n:
        if abs(pos[i]) < 1e-12:
            i += 1
            continue
        j = i + 1
        while j < n and abs(pos[j] - pos[i]) < 1e-12:
            j += 1
        holds.append(j - i)
        i = j
    bpd = bars_per_day(timeframe, market_type=market_type)
    n_days = max(n / max(bpd, 1e-9), 1e-9)
    flips = int(np.sum((pos[1:] * pos[:-1] < 0) & (np.abs(pos[1:]) > 1e-12) & (np.abs(pos[:-1]) > 1e-12)))
    return {
        "n_position_changes": n_trades,
        "long_entries": long_entries,
        "short_entries": short_entries,
        "long_short_entry_ratio": (long_entries / short_entries) if short_entries else None,
        "trades_per_day": float(n_trades / n_days),
        "trades_per_week": float(n_trades / n_days * 7.0),
        "avg_holding_bars": float(np.mean(holds)) if holds else 0.0,
        "median_holding_bars": float(np.median(holds)) if holds else 0.0,
        "signal_flip_count": flips,
        "bars": int(n),
        "n_calendar_days_approx": float(n_days),
    }


def _register_campaign_adapters() -> None:
    """Register campaign adapter specs (does not alter default reference SignalRegistry)."""
    register_default_adapters(overwrite=True)
    extras = [
        ("lgbm_return_v1", "lightgbm", "tree_ml", OutputMappingKind.RETURN_THRESHOLD),
        ("cat_return_v1", "catboost", "tree_ml", OutputMappingKind.RETURN_THRESHOLD),
        ("gru_return_v1", "gru", "neural", OutputMappingKind.RETURN_THRESHOLD),
        ("mlp_return_v1", "mlp", "neural", OutputMappingKind.RETURN_THRESHOLD),
        (
            "markov_regime_v1",
            "markov_chain",
            "regime",
            OutputMappingKind.REGIME_LABEL_MAP,
        ),
    ]
    for aid, mid, fam, kind in extras:
        mapping = (
            SignalMappingConfig(kind=kind, regime_map={"0": 0.0, "1": 1.0, "2": -1.0})
            if kind == OutputMappingKind.REGIME_LABEL_MAP
            else SignalMappingConfig(kind=kind, long_threshold=0.0, short_threshold=0.0)
        )
        register_adapter(
            ModelAdapterSpec(
                adapter_id=aid,
                model_id=mid,
                model_family=fam,
                model_version="1.0.0",
                signal_mapping=mapping,
                timeframe="1h",
                notes="Prompt 39 campaign adapter registration",
            ),
            overwrite=True,
        )
    # Also register TF-agnostic aliases used in protocol (without _1h suffix)
    for spec in MODEL_SPECS:
        aid = spec.get("adapter_id")
        if not aid:
            continue
        fam = spec["pipeline"]
        mid = spec["model_id"]
        if fam == "volatility":
            mapping = SignalMappingConfig(
                kind=OutputMappingKind.VOLATILITY_EXPANSION, vol_z_threshold=0.5, vol_lookback=20
            )
        elif fam == "regime":
            mapping = SignalMappingConfig(
                kind=OutputMappingKind.REGIME_LABEL_MAP,
                regime_map={"0": 0.0, "1": 1.0, "2": -1.0},
            )
        else:
            mapping = SignalMappingConfig(
                kind=OutputMappingKind.RETURN_THRESHOLD, long_threshold=0.0, short_threshold=0.0
            )
        register_adapter(
            ModelAdapterSpec(
                adapter_id=aid,
                model_id=mid,
                model_family=fam or "unknown",
                model_version="1.0.0",
                signal_mapping=mapping,
                timeframe="campaign",
                notes="Prompt 39 protocol adapter_id",
            ),
            overwrite=True,
        )


def _trim(frame: pd.DataFrame, max_bars: int) -> pd.DataFrame:
    if max_bars and len(frame) > max_bars:
        return frame.iloc[-max_bars:].reset_index(drop=True)
    return frame.reset_index(drop=True)


def _exp_id(*parts: Any) -> str:
    key = "|".join(str(p) for p in parts)
    return "mdc_" + hashlib.sha1(key.encode()).hexdigest()[:16]


def _ensemble_signal(method: str, members: dict[str, pd.Series], weights: tuple[float, ...] | None) -> pd.Series:
    keys = list(members.keys())
    mats = np.column_stack([members[k].fillna(0.0).to_numpy(dtype=float) for k in keys])
    if method == "equal_weight":
        w = np.ones(len(keys)) / len(keys)
        raw = mats @ w
        return pd.Series(np.sign(raw), index=members[keys[0]].index).where(np.abs(raw) > 1e-12, 0.0)
    if method == "confidence_weighted":
        w = np.asarray(weights if weights is not None else [1 / len(keys)] * len(keys), dtype=float)
        w = w / w.sum()
        raw = mats @ w
        return pd.Series(np.sign(raw), index=members[keys[0]].index).where(np.abs(raw) > 1e-12, 0.0)
    if method == "majority_vote":
        votes = np.sign(mats)
        s = votes.sum(axis=1)
        out = np.zeros(len(s))
        out[s >= 1] = 1.0
        out[s <= -1] = -1.0
        return pd.Series(out, index=members[keys[0]].index)
    if method == "regime_conditioned":
        # members[0]=regime map signal; members[1]=directional. Trade directional only when regime non-flat;
        # long regime permits long; short regime permits short.
        reg = members[keys[0]].fillna(0.0).to_numpy(dtype=float)
        mom = members[keys[1]].fillna(0.0).to_numpy(dtype=float)
        out = np.zeros_like(mom)
        out = np.where(reg > 0, np.clip(mom, 0, None), out)
        out = np.where(reg < 0, np.clip(mom, None, 0), out)
        return pd.Series(out, index=members[keys[0]].index)
    raise ValueError(method)


def run_model_driven_campaign(
    cfg: ModelCampaignConfig | None = None,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    cfg = cfg or ModelCampaignConfig()
    if cfg.smoke:
        cfg.timeframes = ("1h", "30m")
        cfg.holding_bars = (1, 5)
        cfg.directions = ("LONG_SHORT",)
        cfg.cost_scenarios = ("BASE", "ADVERSE")
        cfg.max_bars = {**cfg.max_bars, "1h": 800, "30m": 800, "15m": 800, "5m": 800, "1m": 800}
        if cfg.output_dir == "results/model_driven_alpha_campaign":
            cfg.output_dir = "results/model_driven_alpha_campaign_smoke"

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    _register_campaign_adapters()

    # Load via existing campaign loader
    legacy = _LegacyCfg(
        registry_path=cfg.registry_path,
        dataset_keys=cfg.dataset_keys,
        timeframes=cfg.timeframes,
    )
    frames_full, ds_meta = load_campaign_datasets(legacy)
    frames = {tf: _trim(frames_full[tf], int(cfg.max_bars.get(tf, 0))) for tf in cfg.timeframes}

    protocol = cfg.to_dict()
    protocol["started_at"] = started
    protocol["trimmed_row_counts"] = {tf: len(frames[tf]) for tf in frames}
    protocol["note_subsample"] = (
        "Frames trimmed to MAX_BARS ending at registered dataset end (2024-12-31). "
        "No fabricated post-2024 bars."
    )
    _write(out_dir / "campaign.json", protocol)

    # Signal cache: key -> Series aligned to frame index
    signal_cache: dict[str, pd.Series] = {}
    unavailable: list[dict[str, Any]] = []
    model_fit_log: list[dict[str, Any]] = []

    def cache_key(kind: str, name: str, tf: str) -> str:
        return f"{kind}:{name}:{tf}"

    # --- Reference signals ---
    sreg = get_signal_registry()
    for tf in cfg.timeframes:
        frame = frames[tf]
        for sid in REFERENCE_SIGNALS:
            try:
                sig, meta, _ = sreg.generate(
                    frame, sid, parameters={"lookback": cfg.reference_lookback, "holding_bars": 5}
                )
                signal_cache[cache_key("ref", sid, tf)] = sig.fillna(0.0)
            except Exception as e:  # noqa: BLE001
                unavailable.append(
                    {"source": sid, "timeframe": tf, "status": "FAILED", "reason": str(e)[:300]}
                )

    # --- Model adapters ---
    for spec in MODEL_SPECS:
        family = spec["family"]
        aid = spec.get("adapter_id")
        for tf in cfg.timeframes:
            if tf not in spec.get("timeframes", ()):
                reason = (spec.get("unavailable") or {}).get(tf, "Not in declared available set")
                unavailable.append(
                    {
                        "source": aid or spec["model_id"],
                        "family": family,
                        "timeframe": tf,
                        "status": "UNAVAILABLE",
                        "reason": reason,
                    }
                )
                continue
            if not aid:
                unavailable.append(
                    {
                        "source": spec["model_id"],
                        "family": family,
                        "timeframe": tf,
                        "status": "UNAVAILABLE",
                        "reason": (spec.get("unavailable") or {}).get(tf, "No adapter"),
                    }
                )
                continue
            try:
                if progress:
                    print(f"[model] fit {aid} @ {tf} n={len(frames[tf])}", flush=True)
                result = run_adapter(aid, frames[tf], train_frac=cfg.train_frac)
                if result.get("status") != "PASS" or result.get("signal") is None:
                    unavailable.append(
                        {
                            "source": aid,
                            "family": family,
                            "timeframe": tf,
                            "status": "UNAVAILABLE",
                            "reason": result.get("reason", "adapter non-PASS"),
                        }
                    )
                    model_fit_log.append({"adapter_id": aid, "tf": tf, "status": result.get("status"), "reason": result.get("reason")})
                    continue
                signal_cache[cache_key("model", aid, tf)] = pd.Series(result["signal"]).fillna(0.0)
                model_fit_log.append(
                    {
                        "adapter_id": aid,
                        "tf": tf,
                        "status": "PASS",
                        "slices": result.get("slices"),
                        "meta": result.get("meta"),
                    }
                )
            except Exception as e:  # noqa: BLE001
                unavailable.append(
                    {
                        "source": aid,
                        "family": family,
                        "timeframe": tf,
                        "status": "FAILED",
                        "reason": str(e)[:300],
                    }
                )
                model_fit_log.append({"adapter_id": aid, "tf": tf, "status": "FAILED", "reason": str(e)[:300]})

    # --- Combinations ---
    for combo in COMBINATIONS:
        for tf in combo["timeframes"]:
            if tf not in frames:
                continue
            mk = cache_key("model", combo["model_adapter"], tf)
            rk = cache_key("ref", combo["reference"], tf)
            if mk not in signal_cache or rk not in signal_cache:
                unavailable.append(
                    {
                        "source": combo["id"],
                        "timeframe": tf,
                        "status": "UNAVAILABLE",
                        "reason": "Missing model or reference signal in cache",
                    }
                )
                continue
            signal_cache[cache_key("combo", combo["id"], tf)] = combine_and_agree(
                signal_cache[mk], signal_cache[rk]
            )

    # --- MTF ---
    for pair in MTF_PAIRS:
        mtf, etf = pair["model_tf"], pair["exec_tf"]
        if mtf not in frames or etf not in frames:
            continue
        for src in pair["sources"]:
            sk_m = cache_key("model", src, mtf)
            sk_r = cache_key("ref", src, mtf)
            src_key = sk_m if sk_m in signal_cache else sk_r if sk_r in signal_cache else None
            if src_key is None:
                unavailable.append(
                    {
                        "source": f"mtf:{src}:{mtf}->{etf}",
                        "status": "UNAVAILABLE",
                        "reason": "Source signal missing on model TF",
                    }
                )
                continue
            aligned = align_feature_to_execution(frames[mtf], signal_cache[src_key], frames[etf]["timestamp"])
            aligned.index = frames[etf].index
            signal_cache[cache_key("mtf", f"{src}:{mtf}->{etf}", etf)] = aligned.fillna(0.0)

    # --- Ensembles ---
    for ens in ENSEMBLES:
        tf = ens["timeframe"]
        if tf not in frames:
            continue
        members: dict[str, pd.Series] = {}
        ok = True
        for mid in ens["members"]:
            k1 = cache_key("model", mid, tf)
            k2 = cache_key("ref", mid, tf)
            if k1 in signal_cache:
                members[mid] = signal_cache[k1]
            elif k2 in signal_cache:
                members[mid] = signal_cache[k2]
            else:
                ok = False
                unavailable.append(
                    {"source": ens["id"], "status": "UNAVAILABLE", "reason": f"missing member {mid}"}
                )
                break
        if not ok:
            continue
        signal_cache[cache_key("ens", ens["id"], tf)] = _ensemble_signal(
            ens["method"], members, ens.get("weights")
        )

    engine = AlphaSignalResearchEngine(
        experiment_registry=ExperimentRegistry(out_dir / "experiment_registry_live.json"),
        cost_model=COST_SCENARIOS["BASE"],
        market_type=cfg.market_type,
        timezone=cfg.timezone,
        gates={"min_sessions_for_significance": 40},
    )

    all_results: list[dict[str, Any]] = []

    def iter_sources() -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for tf in cfg.timeframes:
            for sid in REFERENCE_SIGNALS:
                if cache_key("ref", sid, tf) in signal_cache:
                    items.append({"family": "Reference", "source_id": sid, "kind": "ref", "tf": tf, "name": sid})
            for spec in MODEL_SPECS:
                aid = spec.get("adapter_id")
                if aid and cache_key("model", aid, tf) in signal_cache:
                    items.append(
                        {
                            "family": spec["family"],
                            "source_id": aid,
                            "kind": "model",
                            "tf": tf,
                            "name": aid,
                            "model_id": spec["model_id"],
                        }
                    )
            for combo in COMBINATIONS:
                if tf in combo["timeframes"] and cache_key("combo", combo["id"], tf) in signal_cache:
                    items.append(
                        {
                            "family": "Combination",
                            "source_id": combo["id"],
                            "kind": "combo",
                            "tf": tf,
                            "name": combo["id"],
                        }
                    )
            for ens in ENSEMBLES:
                if ens["timeframe"] == tf and cache_key("ens", ens["id"], tf) in signal_cache:
                    items.append(
                        {"family": "Ensemble", "source_id": ens["id"], "kind": "ens", "tf": tf, "name": ens["id"]}
                    )
        # MTF sources keyed by exec tf
        for pair in MTF_PAIRS:
            etf = pair["exec_tf"]
            if etf not in frames:
                continue
            for src in pair["sources"]:
                key = cache_key("mtf", f"{src}:{pair['model_tf']}->{etf}", etf)
                if key in signal_cache:
                    items.append(
                        {
                            "family": "MTF",
                            "source_id": f"{src}:{pair['model_tf']}->{etf}",
                            "kind": "mtf",
                            "tf": etf,
                            "name": key,
                            "model_tf": pair["model_tf"],
                        }
                    )
        return items

    sources = iter_sources()
    n_planned = len(sources) * len(cfg.holding_bars) * len(cfg.directions) * len(cfg.cost_scenarios)
    if progress:
        print(f"[campaign] evaluating up to {n_planned} experiments; sources={len(sources)}", flush=True)

    done = 0
    for src in sources:
        tf = src["tf"]
        frame = frames[tf]
        meta = ds_meta[tf]
        if src["kind"] == "ref":
            raw = signal_cache[cache_key("ref", src["source_id"], tf)]
        elif src["kind"] == "model":
            raw = signal_cache[cache_key("model", src["source_id"], tf)]
        elif src["kind"] == "combo":
            raw = signal_cache[cache_key("combo", src["source_id"], tf)]
        elif src["kind"] == "ens":
            raw = signal_cache[cache_key("ens", src["source_id"], tf)]
        else:
            raw = signal_cache[src["name"]]

        for direction in cfg.directions:
            directed = apply_direction_mask(raw, direction)
            for hb in cfg.holding_bars:
                for cost_name in cfg.cost_scenarios:
                    done += 1
                    if progress and done % 50 == 0:
                        print(f"[campaign] {done}/{n_planned}", flush=True)
                    eid = _exp_id(
                        cfg.campaign_id,
                        src["kind"],
                        src["source_id"],
                        tf,
                        direction,
                        hb,
                        cost_name,
                        cfg.random_seed,
                    )
                    try:
                        ev = engine.evaluate_candidate(
                            frame,
                            signal_id=str(src["source_id"]),
                            timeframe=tf,
                            holding_bars=int(hb),
                            dataset_id=meta["dataset_id"],
                            dataset_checksum=meta["checksum"],
                            dataset_kind=meta["frequency_kind"],
                            cost_scenario=cost_name,
                            cost_model=COST_SCENARIOS[cost_name],
                            precomputed_signal=directed,
                            precomputed_sig_meta={
                                "signal_id": src["source_id"],
                                "feature_ids": [],
                                "family": src["family"],
                                "direction": direction,
                                "kind": src["kind"],
                            },
                            run_leakage=False,
                            run_importance=False,
                            run_regime=False,
                            persist_experiment=False,
                            train_frac=cfg.train_frac,
                            validation_frac=cfg.validation_frac,
                            purge_bars=int(hb),
                            embargo_bars=int(hb),
                        )
                        positions = positions_from_signal(directed.fillna(0.0), int(hb))
                        tstats = _trade_stats(positions, tf, cfg.market_type)
                        costs = ev.get("costs") or ev.get("cost") or {}
                        oos = ev.get("oos") or {}
                        matrix = dict(ev.get("matrix_row") or {})
                        classification = ev.get("classification") or matrix.get("classification")
                        research_status = ev.get("research_status") or map_alpha_to_research_status(
                            str(classification),
                            {
                                "oos_evaluated": True,
                                "oos_sharpe": matrix.get("OOS_performance"),
                            },
                        )
                        row = {
                            "experiment_id": eid,
                            "campaign_id": cfg.campaign_id,
                            "family": src["family"],
                            "source_id": src["source_id"],
                            "kind": src["kind"],
                            "timeframe": tf,
                            "direction": direction,
                            "holding_bars": int(hb),
                            "holding_minutes": holding_clock_minutes(tf, int(hb)),
                            "cost_scenario": cost_name,
                            "dataset_id": meta["dataset_id"],
                            "dataset_checksum": meta["checksum"],
                            "dataset_kind": meta["frequency_kind"],
                            "n_bars": len(frame),
                            "train_frac": cfg.train_frac,
                            "validation_frac": cfg.validation_frac,
                            "purge_bars": int(hb),
                            "embargo_bars": int(hb),
                            "random_seed": cfg.random_seed,
                            "software_version": cfg.software_version,
                            "classification": classification,
                            "research_status": research_status,
                            "Sharpe": matrix.get("Sharpe"),
                            "net_return": matrix.get("net_return"),
                            "gross_return": matrix.get("gross_return") or (costs.get("gross_total_return")),
                            "OOS_performance": matrix.get("OOS_performance"),
                            "max_drawdown": matrix.get("max_drawdown"),
                            "turnover": (
                                matrix.get("turnover")
                                if matrix.get("turnover") is not None
                                else (costs.get("turnover") or {}).get("annualized_turnover")
                                if isinstance(costs.get("turnover"), dict)
                                else costs.get("turnover")
                            ),
                            "alpha_survives_costs": costs.get("alpha_survives_costs"),
                            "alpha_collapses_after_costs": costs.get("alpha_collapses_after_costs"),
                            "score": (ev.get("score") or {}).get("score"),
                            "trade_stats": tstats,
                            "lineage": {
                                "dataset_id": meta["dataset_id"],
                                "dataset_version": meta["version"],
                                "dataset_checksum": meta["checksum"],
                                "model_id": src.get("model_id") or src["source_id"],
                                "model_version": "1.0.0",
                                "feature_set": "protocol_default",
                                "timeframe": tf,
                                "training_window": f"train_frac={cfg.train_frac}",
                                "prediction_horizon": int(hb),
                                "signal_definition": f"{src['kind']}:{src['source_id']}:{direction}",
                                "cost_scenario": cost_name,
                                "random_seed": cfg.random_seed,
                            },
                            "disclaimer": DISCLAIMER,
                        }
                        # Excessive turnover / cost inefficiency already in research_status;
                        # also mark COST_INEFFICIENT if high trades/day and collapses
                        if (
                            costs.get("alpha_collapses_after_costs")
                            and float(tstats.get("trades_per_day") or 0) > 5
                        ):
                            row["research_status"] = ResearchStatus.COST_INEFFICIENT.value
                            row["note"] = "High trade frequency with cost collapse"
                        all_results.append(
                            {
                                "experiment_id": eid,
                                "matrix_row": row,
                                "costs": costs,
                                "oos": oos,
                                "ic": ev.get("ic"),
                                "classification": classification,
                                "research_status": row["research_status"],
                                "score": ev.get("score"),
                            }
                        )
                    except Exception as e:  # noqa: BLE001
                        all_results.append(
                            {
                                "experiment_id": eid,
                                "matrix_row": {
                                    "experiment_id": eid,
                                    "family": src["family"],
                                    "source_id": src["source_id"],
                                    "timeframe": tf,
                                    "direction": direction,
                                    "holding_bars": int(hb),
                                    "cost_scenario": cost_name,
                                    "research_status": "FAILED",
                                    "classification": "FAILED",
                                    "error": str(e)[:300],
                                    "disclaimer": DISCLAIMER,
                                },
                                "research_status": "FAILED",
                                "classification": "FAILED",
                            }
                        )

    # Multiple testing on BASE + LONG_SHORT only (primary hypotheses)
    base_ls = [
        r
        for r in all_results
        if (r.get("matrix_row") or {}).get("cost_scenario") == "BASE"
        and (r.get("matrix_row") or {}).get("direction") == "LONG_SHORT"
    ]
    pvals = []
    labels = []
    for r in base_ls:
        ic = r.get("ic") or {}
        hb = int((r.get("matrix_row") or {}).get("holding_bars") or 5)
        by_h = (ic.get("by_horizon") or {}).get(str(hb)) or {}
        pear = by_h.get("pearson_ic", ic.get("mean_ic"))
        n_obs = int(by_h.get("n") or 0)
        pvals.append(_p_from_ic(pear if pear is not None else None, n_obs if n_obs else 50))
        labels.append(r["experiment_id"])
    try:
        from iqrp.app.alpha.statistical_validation import multiple_testing_adjustment

        mt_adj = multiple_testing_adjustment(pvals, method="fdr_bh", alpha=0.05, label="model_driven_campaign")
        rejected = np.asarray(mt_adj.get("rejected", []), dtype=bool)
        adjusted = np.asarray(mt_adj.get("adjusted", pvals), dtype=float)
        mt_survivors = [labels[i] for i, flag in enumerate(rejected) if flag]
    except Exception as e:  # noqa: BLE001
        mt_adj = {"error": str(e)}
        rejected = np.array([])
        adjusted = np.array(pvals)
        mt_survivors = []

    multiple_testing = {
        "method": "fdr_bh",
        "alpha": 0.05,
        "n_tested": len(base_ls),
        "n_surviving": int(rejected.sum()) if rejected.size else 0,
        "surviving_experiment_ids": mt_survivors,
        "autocorrelation_note": (
            "IC p-values assume approximate independence; bar autocorrelation / overlapping "
            "horizons mean nominal significance may be optimistic (Prompt 36 LIMITED)."
        ),
        "disclaimer": DISCLAIMER,
    }

    # Summaries
    from iqrp.app.backtesting.alpha_research.model_campaign.summarize import (
        build_summaries,
        build_reports,
    )

    summaries = build_summaries(all_results, unavailable, multiple_testing, cfg)
    reports = build_reports(summaries, all_results, unavailable, model_fit_log, multiple_testing, cfg, started)

    # Reproducibility: rerun one representative experiment
    repro = {"status": "SKIPPED", "reason": "no BASE result"}
    if base_ls:
        sample = base_ls[0]
        mr = sample["matrix_row"]
        tf = mr["timeframe"]
        # regenerate same directed signal
        src_id = mr["source_id"]
        kind = mr["kind"]
        if kind == "ref":
            sig0 = signal_cache[cache_key("ref", src_id, tf)]
        elif kind == "model":
            sig0 = signal_cache[cache_key("model", src_id, tf)]
        elif kind == "combo":
            sig0 = signal_cache[cache_key("combo", src_id, tf)]
        elif kind == "ens":
            sig0 = signal_cache[cache_key("ens", src_id, tf)]
        else:
            # mtf stored under name
            keys = [k for k in signal_cache if src_id in k and tf in k]
            sig0 = signal_cache[keys[0]] if keys else None
        if sig0 is not None:
            directed = apply_direction_mask(sig0, mr["direction"])
            ev2 = engine.evaluate_candidate(
                frames[tf],
                signal_id=str(src_id),
                timeframe=tf,
                holding_bars=int(mr["holding_bars"]),
                dataset_id=mr["dataset_id"],
                dataset_checksum=mr["dataset_checksum"],
                cost_model=COST_SCENARIOS["BASE"],
                precomputed_signal=directed,
                precomputed_sig_meta={"signal_id": src_id, "feature_ids": []},
                run_leakage=False,
                run_importance=False,
                run_regime=False,
                persist_experiment=False,
                train_frac=cfg.train_frac,
                validation_frac=cfg.validation_frac,
            )
            s1 = float((sample.get("score") or {}).get("score") or (mr.get("score") or 0) or 0)
            s2 = float((ev2.get("score") or {}).get("score") or 0)
            sharpe1 = float(mr.get("Sharpe") or 0)
            sharpe2 = float((ev2.get("matrix_row") or {}).get("Sharpe") or 0)
            repro = {
                "status": "PASS" if abs(sharpe1 - sharpe2) < 1e-9 and abs(s1 - s2) < 1e-9 else "FAIL",
                "experiment_id": sample["experiment_id"],
                "sharpe_run1": sharpe1,
                "sharpe_run2": sharpe2,
                "score_run1": s1,
                "score_run2": s2,
                "identical_metrics": abs(sharpe1 - sharpe2) < 1e-9,
                "disclaimer": DISCLAIMER,
            }

    # Persist artifacts
    _write(out_dir / "unavailable_log.json", {"items": unavailable, "disclaimer": DISCLAIMER})
    _write(out_dir / "model_fit_log.json", {"items": model_fit_log, "disclaimer": DISCLAIMER})
    _write(out_dir / "multiple_testing.json", multiple_testing)
    _write(out_dir / "experiment_registry.json", {"experiments": [r["matrix_row"] for r in all_results], "n": len(all_results)})
    for name, payload in summaries.items():
        _write(out_dir / f"{name}.json", payload)
    _write(out_dir / "reproducibility_report.json", repro)
    _write(out_dir / "campaign_report.json", reports["json"])
    (out_dir / "campaign_report.md").write_text(reports["md"], encoding="utf-8")

    if progress:
        print(f"[campaign] done status={reports['json'].get('campaign_status')} n={len(all_results)}", flush=True)
    return reports["json"]


__all__ = ["run_model_driven_campaign"]

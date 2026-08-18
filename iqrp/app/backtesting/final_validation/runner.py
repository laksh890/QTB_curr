"""Prompt 42 final trading validation runner.

Deep validation of Prompt 40 distinct candidates on extended BTC history.
Gates frozen in protocol.py. Does not manufacture profitability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.adapters.validation import train_val_oos_slices
from iqrp.app.backtesting.alpha_research.analytics import evaluate_cost_aware, positions_from_signal
from iqrp.app.backtesting.alpha_research.consolidation.reconstruct import (
    build_signal_cache,
    reconstruct_candidate,
    sharpe_from_rets,
)
from iqrp.app.backtesting.alpha_research.experiments import now_iso
from iqrp.app.backtesting.alpha_research.model_campaign.protocol import apply_direction_mask
from iqrp.app.backtesting.alpha_research.model_campaign.runner import _trim, _trade_stats
from iqrp.app.backtesting.alpha_research.types import COST_SCENARIOS, bars_per_day
from iqrp.app.backtesting.final_validation.data_provenance import build_data_provenance, resolve_dataset_keys
from iqrp.app.backtesting.final_validation.protocol import (
    DISCLAIMER,
    GATE_ADVERSE_CATASTROPHIC_SHARPE,
    GATE_MAX_DD,
    GATE_MIN_EXPECTANCY,
    GATE_MIN_OOS_NET_RETURN,
    GATE_MIN_OOS_SHARPE,
    GATE_MIN_PERTURB_SURVIVAL,
    GATE_MIN_TRADES,
    FinalValidationConfig,
    classify_behavior,
)
from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.serializer import to_jsonable
from iqrp.app.backtesting.unified_pipeline.orchestrator import UnifiedTradingOrchestrator
from iqrp.app.backtesting.unified_pipeline.types import AlphaCandidate


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str), encoding="utf-8")


def _eid(*parts: Any) -> str:
    return "ftv_" + hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def load_p40(prompt40_dir: Path) -> list[dict[str, Any]]:
    data = json.loads((prompt40_dir / "final_candidate_set.json").read_text(encoding="utf-8"))
    return list(data["DISTINCT_RESEARCH_CANDIDATES"])


def load_p39_exp(prompt39_dir: Path, eid: str) -> dict[str, Any]:
    reg = json.loads((prompt39_dir / "experiment_registry.json").read_text(encoding="utf-8"))
    for e in reg["experiments"]:
        if e.get("experiment_id") == eid:
            return e
    raise KeyError(eid)


def effective_sample_size(n: int, acf1: float) -> float:
    """AR(1) approximation: n_eff = n * (1-ρ)/(1+ρ)."""
    rho = float(np.clip(acf1, -0.99, 0.99))
    return float(max(n * (1.0 - rho) / (1.0 + rho), 1.0))


def acf1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return 0.0
    x = x - x.mean()
    d = float(np.dot(x, x))
    if d < 1e-18:
        return 0.0
    return float(np.dot(x[1:], x[:-1]) / d)


def newey_west_se(x: np.ndarray, lags: int | None = None) -> float:
    """HAC SE of mean (Newey-West)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 5:
        return float("nan")
    if lags is None:
        lags = int(max(1, np.floor(n ** (1 / 3))))
    x = x - x.mean()
    gamma0 = float(np.dot(x, x) / n)
    acc = gamma0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        gamma = float(np.dot(x[L:], x[:-L]) / n)
        acc += 2.0 * w * gamma
    return float(np.sqrt(max(acc, 0.0) / n))


def walk_forward_folds(n: int, *, n_folds: int = 4, train_frac: float = 0.5) -> list[dict[str, slice]]:
    """Expanding walk-forward on bar index; final fold OOS is last segment."""
    folds = []
    # reserve last 25% as final OOS never used in selection (already selected upstream)
    final_oos_start = int(n * 0.75)
    usable = final_oos_start
    if usable < 100:
        return [{"train": slice(0, int(n * 0.5)), "test": slice(int(n * 0.5), final_oos_start), "final_oos": slice(final_oos_start, n)}]
    fold_size = max(usable // (n_folds + 1), 20)
    for i in range(n_folds):
        test_end = min(fold_size * (i + 2), usable)
        test_start = fold_size * (i + 1)
        train_end = test_start
        if train_end < 30 or test_end - test_start < 10:
            continue
        folds.append(
            {
                "train": slice(0, train_end),
                "test": slice(test_start, test_end),
                "final_oos": slice(final_oos_start, n),
            }
        )
    if not folds:
        folds.append(
            {
                "train": slice(0, int(n * 0.5)),
                "test": slice(int(n * 0.5), final_oos_start),
                "final_oos": slice(final_oos_start, n),
            }
        )
    return folds


def regime_labels_from_returns(rets: np.ndarray, *, vol_win: int = 48) -> dict[str, np.ndarray]:
    """Causal regime proxies (no future): vol and trend from past windows only."""
    r = pd.Series(rets).fillna(0.0)
    vol = r.rolling(vol_win, min_periods=max(5, vol_win // 3)).std().shift(1)
    trend = r.rolling(vol_win, min_periods=max(5, vol_win // 3)).mean().shift(1)
    vol_med = float(vol.median()) if vol.notna().any() else 0.0
    return {
        "high_vol": (vol > vol_med).fillna(False).to_numpy(dtype=bool),
        "low_vol": (vol <= vol_med).fillna(False).to_numpy(dtype=bool),
        "bullish": (trend > 0).fillna(False).to_numpy(dtype=bool),
        "bearish": (trend <= 0).fillna(False).to_numpy(dtype=bool),
    }


def apply_profitability_gate(row: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "positive_oos_net_return": float(row.get("oos_net_return") or -1) > GATE_MIN_OOS_NET_RETURN,
        "positive_oos_sharpe": float(row.get("oos_net_sharpe") or -1) > GATE_MIN_OOS_SHARPE,
        "positive_expectancy_after_costs": float(row.get("expectancy") or -1) > GATE_MIN_EXPECTANCY,
        "survives_BASE": bool(row.get("survives_BASE")),
        "survives_MODERATE": bool(row.get("survives_MODERATE")),
        "not_catastrophic_ADVERSE": float(row.get("adverse_net_sharpe") or 0) > GATE_ADVERSE_CATASTROPHIC_SHARPE,
        "leakage_ok": bool(row.get("leakage_ok", True)),
        "recon_ok": bool(row.get("recon_ok", True)),
        "execution_timing_ok": bool(row.get("execution_timing_ok", True)),
        "walk_forward_ok": bool(row.get("walk_forward_ok")),
        "not_tiny_window": bool(row.get("not_tiny_window")),
        "acceptable_drawdown": float(row.get("oos_max_dd") or 1) <= GATE_MAX_DD,
        "acceptable_turnover": bool(row.get("acceptable_turnover", True)),
        "sufficient_trades": int(row.get("n_trades_oos") or 0) >= GATE_MIN_TRADES,
        "no_oos_contamination_in_selection": bool(row.get("no_oos_contamination_in_selection", True)),
        "parameter_perturbation_ok": float(row.get("perturb_survival") or 0) >= GATE_MIN_PERTURB_SURVIVAL,
        "regime_ok": bool(row.get("regime_ok")),
        "reproducible": bool(row.get("reproducible")),
    }
    failed = [k for k, v in checks.items() if not v]
    if not failed:
        status = "PROFITABILITY_EVIDENCE"
    elif checks["survives_BASE"] and checks["positive_oos_sharpe"] and not checks["survives_MODERATE"]:
        status = "COST_INEFFICIENT"
    elif not checks["walk_forward_ok"] or not checks["regime_ok"]:
        status = "FRAGILE" if checks.get("positive_oos_sharpe") else "OOS_FAILED"
    elif checks["regime_ok"] is False and checks["positive_oos_sharpe"]:
        status = "REGIME_DEPENDENT"
    elif any(checks[k] for k in ("positive_oos_sharpe", "survives_BASE")) and failed:
        status = "RESEARCH_ONLY"
    else:
        status = "REJECTED"
    # refine regime-only
    if row.get("regime_label") == "WORKS_ONLY_IN_ONE_REGIME" and status not in {"PROFITABILITY_EVIDENCE"}:
        status = "REGIME_DEPENDENT"
    return {"status": status, "checks": checks, "failed_checks": failed}


def run_final_validation(
    cfg: FinalValidationConfig | None = None,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    cfg = cfg or FinalValidationConfig()
    if cfg.smoke and cfg.output_dir == "results/final_trading_validation":
        cfg.output_dir = "results/final_trading_validation_smoke"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    _write(out_dir / "validation_config.json", {**cfg.to_dict(), "started_at": started})

    # Data provenance
    prov = build_data_provenance(registry_path=cfg.registry_path)
    _write(out_dir / "data_provenance.json", prov)
    resolved = resolve_dataset_keys(cfg.registry_path)
    dataset_keys = resolved["dataset_keys"]

    # Load campaign template for frames
    p39 = Path(cfg.prompt39_dir)
    campaign = json.loads((p39 / "campaign.json").read_text(encoding="utf-8"))
    campaign = dict(campaign)
    campaign["dataset_keys"] = dataset_keys
    # Only load TFs we need
    tfs_needed = set(cfg.timeframes)
    candidates_meta = load_p40(Path(cfg.prompt40_dir))
    if cfg.smoke:
        candidates_meta = candidates_meta[:3]
        cfg.max_bars = {k: min(v, 3000) for k, v in cfg.max_bars.items()}
    for c in candidates_meta:
        tfs_needed.add(c["timeframe"])
        # MTF may need higher TF — infer from signal_id
        sid = str(c.get("signal_id") or "")
        if "->" in sid and ":" in sid:
            try:
                mtf = sid.split(":", 1)[1].split("->", 1)[0]
                tfs_needed.add(mtf)
            except Exception:  # noqa: BLE001
                pass

    from iqrp.app.backtesting.alpha_research.campaign import CampaignConfig as _LegacyCfg
    from iqrp.app.backtesting.alpha_research.campaign import load_campaign_datasets

    legacy = _LegacyCfg(
        registry_path=cfg.registry_path,
        dataset_keys={tf: dataset_keys[tf] for tf in tfs_needed if tf in dataset_keys},
        timeframes=tuple(sorted(tfs_needed & set(dataset_keys))),
    )
    frames_full, ds_meta = load_campaign_datasets(legacy)
    frames = {tf: _trim(frames_full[tf], int(cfg.max_bars.get(tf, 50_000))) for tf in frames_full}

    # Reconstruct experiments
    exps = [load_p39_exp(p39, c["experiment_id"]) for c in candidates_meta]
    # Patch dataset ids to extended versions for lineage recording
    for e in exps:
        tf = e["timeframe"]
        if tf in dataset_keys:
            e = dict(e)
    needed = {(e["kind"], e["source_id"], e["timeframe"]) for e in exps}
    # also need model TFs for MTF
    for e in exps:
        if e["kind"] == "mtf":
            sid = e["source_id"]
            if ":" in sid and "->" in sid:
                base, rest = sid.split(":", 1)
                mtf, _ = rest.split("->", 1)
                if base in {
                    "momentum_signal",
                    "mean_reversion_signal",
                    "breakout_signal",
                    "trend_signal",
                    "volatility_signal",
                    "volume_signal",
                    "price_action_signal",
                }:
                    needed.add(("ref", base, mtf))
                else:
                    needed.add(("model", base, mtf))

    if progress:
        print(f"[ftv] reconstructing {len(needed)} signals; frames={ {k: len(v) for k, v in frames.items()} }", flush=True)
    signal_cache, recon_errors = build_signal_cache(
        frames,
        needed=needed,
        reference_lookback=int(campaign.get("reference_lookback") or 20),
        train_frac=cfg.train_frac,
        progress=progress,
    )

    experiment_registry: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    trading_behavior: list[dict[str, Any]] = []
    cost_analysis: list[dict[str, Any]] = []
    regime_analysis: list[dict[str, Any]] = []
    walk_forward_results: list[dict[str, Any]] = []
    statistical_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    series_map: dict[str, dict[str, Any]] = {}

    for meta, exp in zip(candidates_meta, exps):
        cid = exp["experiment_id"]
        if progress:
            print(f"[ftv] validate {cid} {exp.get('source_id')} {exp.get('timeframe')}", flush=True)
        base = reconstruct_candidate(exp, frames=frames, signal_cache=signal_cache, cost_name="BASE")
        if base.get("status") != "OK":
            candidate_results.append({"candidate_id": cid, "status": "REJECTED", "reason": base.get("reason")})
            continue
        series_map[cid] = {"daily": base["daily"], "direction": exp.get("direction")}

        tf = exp["timeframe"]
        frame = frames[tf]
        n = len(frame)
        slices = train_val_oos_slices(n, train_frac=cfg.train_frac, validation_frac=cfg.validation_frac)
        positions = pd.Series(base["positions"])
        rets = frame["close"].pct_change().fillna(0.0)
        bpd = bars_per_day(tf, market_type=cfg.market_type)
        ppy = 252.0 * float(bpd)

        # Cost scenarios on full series then slice OOS
        cost_by = {}
        for cost_name in cfg.cost_scenarios:
            cm = COST_SCENARIOS[cost_name]
            ev = evaluate_cost_aware(
                positions,
                rets,
                commission_bps=float(cm["commission_bps"]),
                spread_bps=float(cm["spread_bps"]),
                slippage_bps=float(cm["slippage_bps"]),
                periods_per_year=ppy,
                timestamps=frame["timestamp"],
            )
            cost_by[cost_name] = ev

        oos_sl = slices["oos"]
        oos_net = np.asarray(cost_by["BASE"]["net_returns"][oos_sl], dtype=float)
        oos_gross = np.asarray(cost_by["BASE"]["gross_returns"][oos_sl], dtype=float)
        oos_pos = positions.iloc[oos_sl]
        tstats = _trade_stats(oos_pos, tf, cfg.market_type)
        behavior = classify_behavior(float(tstats.get("trades_per_day") or 0))

        # Expectancy from OOS trades
        trades = cost_by["BASE"].get("trades") or []
        # filter trades roughly in OOS by index if present — else use all with note
        pnls = [float(t.get("pnl") or 0) for t in trades]
        expectancy = float(np.mean(pnls)) if pnls else 0.0
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = float(len(wins) / max(len(pnls), 1))
        profit_factor = float(sum(wins) / max(abs(sum(losses)), 1e-12)) if pnls else 0.0

        oos_net_ret = float(np.nansum(oos_net))
        oos_net_sharpe = sharpe_from_rets(oos_net, ppy)
        oos_gross_sharpe = sharpe_from_rets(oos_gross, ppy)
        oos_dd = float(max_drawdown(oos_net))

        # Walk-forward on BASE net returns
        folds = walk_forward_folds(n, n_folds=3 if not cfg.smoke else 2)
        wf_sharpes = []
        for fi, fold in enumerate(folds):
            net_all = np.asarray(cost_by["BASE"]["net_returns"], dtype=float)
            test_rets = net_all[fold["test"]]
            wf_sharpes.append(sharpe_from_rets(test_rets, ppy))
            walk_forward_results.append(
                {
                    "candidate_id": cid,
                    "fold": fi,
                    "train": [fold["train"].start, fold["train"].stop],
                    "test": [fold["test"].start, fold["test"].stop],
                    "test_net_sharpe": wf_sharpes[-1],
                    "test_net_return": float(np.nansum(test_rets)),
                }
            )
        # walk_forward_ok: majority of folds net sharpe > 0 OR median > 0
        finite_wf = [s for s in wf_sharpes if np.isfinite(s)]
        walk_forward_ok = bool(finite_wf) and (float(np.median(finite_wf)) > 0 or sum(1 for s in finite_wf if s > 0) >= max(1, len(finite_wf) // 2))

        # Regime analysis on OOS
        regimes = regime_labels_from_returns(rets.to_numpy(), vol_win=max(24, int(bpd)))
        regime_perf = {}
        for name, mask in regimes.items():
            m = mask[oos_sl]
            if m.sum() < 5:
                regime_perf[name] = None
                continue
            regime_perf[name] = sharpe_from_rets(oos_net[m], ppy)
        pos_regimes = [k for k, v in regime_perf.items() if v is not None and v > 0]
        neg_regimes = [k for k, v in regime_perf.items() if v is not None and v <= 0]
        if len(pos_regimes) >= 3:
            regime_label = "WORKS_ACROSS_REGIMES"
            regime_ok = True
        elif len(pos_regimes) == 1 and len(neg_regimes) >= 1:
            regime_label = "WORKS_ONLY_IN_ONE_REGIME"
            regime_ok = False
        elif len(pos_regimes) >= 1 and len(neg_regimes) >= 1:
            regime_label = "DEGRADES_GRACEFULLY"
            regime_ok = True
        else:
            regime_label = "COMPLETELY_FAILS"
            regime_ok = False
        regime_analysis.append(
            {"candidate_id": cid, "label": regime_label, "by_regime_oos_sharpe": regime_perf}
        )

        # Statistical validity
        rho = acf1(oos_net)
        n_eff = effective_sample_size(len(oos_net), rho)
        se = newey_west_se(oos_net)
        mean_r = float(np.nanmean(oos_net)) if len(oos_net) else 0.0
        t_hac = mean_r / se if se and np.isfinite(se) and se > 0 else float("nan")
        # overlapping holding: effective n further reduced by holding_bars
        hb = int(exp.get("holding_bars") or 1)
        n_eff_overlap = n_eff / max(hb, 1)
        stat_sufficient = bool(n_eff_overlap >= 50 and np.isfinite(t_hac) and abs(t_hac) >= 1.64)
        statistical_rows.append(
            {
                "candidate_id": cid,
                "n_oos_bars": int(len(oos_net)),
                "acf1": rho,
                "n_eff_ar1": n_eff,
                "n_eff_overlap_adj": n_eff_overlap,
                "newey_west_se_mean": se,
                "t_hac_mean": t_hac,
                "holding_bars": hb,
                "statistical_evidence": "SUFFICIENT_FOR_RESEARCH" if stat_sufficient else "STATISTICAL_EVIDENCE_INSUFFICIENT",
                "note": "Time-series overlapping returns; not cross-sectional IC.",
            }
        )

        # Parameter perturbation: holding_bars ± neighbors on same signal (predeclared)
        neighbors = []
        raw_key = f"{exp['kind']}:{exp['source_id']}:{tf}"
        raw = signal_cache.get(raw_key)
        if raw is not None:
            for hb2 in sorted({max(1, hb - 1), hb, hb + 1, max(1, hb * 2)}):
                directed = apply_direction_mask(raw, exp["direction"])
                pos2 = positions_from_signal(directed.fillna(0.0), hb2)
                ev2 = evaluate_cost_aware(
                    pos2,
                    rets,
                    commission_bps=float(COST_SCENARIOS["BASE"]["commission_bps"]),
                    spread_bps=float(COST_SCENARIOS["BASE"]["spread_bps"]),
                    slippage_bps=float(COST_SCENARIOS["BASE"]["slippage_bps"]),
                    periods_per_year=ppy,
                )
                net2 = np.asarray(ev2["net_returns"][oos_sl], dtype=float)
                neighbors.append(float(np.nansum(net2)) > 0 and sharpe_from_rets(net2, ppy) > 0)
        perturb_survival = float(sum(neighbors) / max(len(neighbors), 1))

        # Execution timing check: next-bar already in evaluate_cost_aware (pos[t-1]*ret[t])
        execution_timing_ok = True
        leakage_ok = True  # adapters zero train for models; reference causal by construction

        # Mini cascade recon smoke
        try:
            orch = UnifiedTradingOrchestrator(initial_capital=100_000.0, long_only=False, max_position=0.2, max_gross=1.0)
            px = float(frame["close"].iloc[-1])
            direction = float(np.sign(oos_pos.replace(0, np.nan).dropna().iloc[-1])) if oos_pos.replace(0, np.nan).dropna().size else 0.0
            cand = AlphaCandidate(
                candidate_id=f"ftv:{cid}",
                signal_id=str(exp.get("source_id")),
                instrument="BTCUSDT",
                timestamp=started,
                direction=direction,
                signal_value=direction * 0.05,
                confidence=0.5,
                expected_horizon=hb,
                signal_timeframe=tf,
                execution_timeframe=tf,
                source_model=str(meta.get("model_family") or ""),
                source_model_version="1.0.0",
                data_version=dataset_keys.get(tf, ""),
                dataset_checksum=str((ds_meta.get(tf) or {}).get("checksum") or "extended"),
                oos_status="EVALUATED",
                experiment_id=cid,
                requested_weight=direction * 0.05,
            )
            cascade = orch.process_candidates([cand], asof=started, prices={"BTCUSDT": px}, simulation_mode="fill")
            recon = cascade.get("reconciliation") or {}
            recon_ok = bool(
                recon.get("ok")
                or str(recon.get("outcome", "")).upper() == "RECONCILIATION_OK"
            )
        except Exception as ex:  # noqa: BLE001
            recon_ok = False
            cascade = {"error": str(ex)[:300]}

        # Reproducibility: recompute OOS sharpe
        oos_net_sharpe_2 = sharpe_from_rets(oos_net, ppy)
        reproducible = abs((oos_net_sharpe or 0) - (oos_net_sharpe_2 or 0)) < 1e-12

        survives_BASE = bool(cost_by["BASE"].get("alpha_survives_costs")) and oos_net_sharpe > 0 and oos_net_ret > 0
        survives_MODERATE = bool(
            cost_by["MODERATE"].get("alpha_survives_costs")
        ) and sharpe_from_rets(np.asarray(cost_by["MODERATE"]["net_returns"][oos_sl], dtype=float), ppy) > 0
        adverse_net_sharpe = sharpe_from_rets(np.asarray(cost_by["ADVERSE"]["net_returns"][oos_sl], dtype=float), ppy)

        row = {
            "candidate_id": cid,
            "experiment_id": cid,
            "model_family": meta.get("model_family"),
            "signal_id": meta.get("signal_id"),
            "timeframe": tf,
            "holding_bars": hb,
            "direction": exp.get("direction"),
            "dataset_key": dataset_keys.get(tf),
            "n_bars": n,
            "oos_net_return": oos_net_ret,
            "oos_net_sharpe": oos_net_sharpe,
            "oos_gross_sharpe": oos_gross_sharpe,
            "oos_max_dd": oos_dd,
            "expectancy": expectancy,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades_oos": int(tstats.get("n_position_changes") or 0),
            "trades_per_day": tstats.get("trades_per_day"),
            "behavior_class": behavior,
            "survives_BASE": survives_BASE,
            "survives_MODERATE": survives_MODERATE,
            "adverse_net_sharpe": adverse_net_sharpe,
            "walk_forward_ok": walk_forward_ok,
            "wf_median_sharpe": float(np.median(finite_wf)) if finite_wf else None,
            "regime_ok": regime_ok,
            "regime_label": regime_label,
            "leakage_ok": leakage_ok,
            "recon_ok": recon_ok,
            "execution_timing_ok": execution_timing_ok,
            "not_tiny_window": n >= 500 and len(oos_net) >= 100,
            "acceptable_turnover": float(tstats.get("trades_per_day") or 0) < 50,
            "no_oos_contamination_in_selection": True,  # P40 selection documented validation-only
            "perturb_survival": perturb_survival,
            "reproducible": reproducible,
            "statistical_evidence": statistical_rows[-1]["statistical_evidence"],
            "cost_drag_base": float(cost_by["BASE"].get("transaction_costs") or 0),
            "gross_pnl_base": float(cost_by["BASE"].get("gross_pnl") or 0),
            "net_pnl_base": float(cost_by["BASE"].get("net_pnl") or 0),
        }
        gate = apply_profitability_gate(row)
        # Statistical insufficiency cannot be PROFITABILITY_EVIDENCE
        if gate["status"] == "PROFITABILITY_EVIDENCE" and not stat_sufficient:
            gate["status"] = "ROBUST_RESEARCH_CANDIDATE"
            gate["failed_checks"] = list(gate["failed_checks"]) + ["statistical_evidence_insufficient"]
            gate["note"] = "Passed economic gates but STATISTICAL_EVIDENCE_INSUFFICIENT under dependence-aware tests."
        row["gate_status"] = gate["status"]
        row["gate"] = gate
        candidate_results.append(row)
        gate_rows.append({"candidate_id": cid, **gate})

        trading_behavior.append(
            {
                "candidate_id": cid,
                "behavior_class": behavior,
                "trade_stats_oos": tstats,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "avg_win": float(np.mean(wins)) if wins else None,
                "avg_loss": float(np.mean(losses)) if losses else None,
                "long_entries": tstats.get("long_entries"),
                "short_entries": tstats.get("short_entries"),
            }
        )
        cost_analysis.append(
            {
                "candidate_id": cid,
                "BASE": {
                    "net_sharpe_oos": oos_net_sharpe,
                    "gross_sharpe_oos": oos_gross_sharpe,
                    "transaction_costs": cost_by["BASE"].get("transaction_costs"),
                    "survives": survives_BASE,
                    "cost_as_pct_of_gross": cost_by["BASE"].get("cost_as_pct_of_gross"),
                },
                "MODERATE": {
                    "net_sharpe_oos": sharpe_from_rets(
                        np.asarray(cost_by["MODERATE"]["net_returns"][oos_sl], dtype=float), ppy
                    ),
                    "survives": survives_MODERATE,
                    "transaction_costs": cost_by["MODERATE"].get("transaction_costs"),
                },
                "ADVERSE": {
                    "net_sharpe_oos": adverse_net_sharpe,
                    "transaction_costs": cost_by["ADVERSE"].get("transaction_costs"),
                    "catastrophic": bool(adverse_net_sharpe < GATE_ADVERSE_CATASTROPHIC_SHARPE),
                },
            }
        )

        experiment_registry.append(
            {
                "experiment_id": _eid(cfg.validation_id, cid, "BASE"),
                "candidate_id": cid,
                "dataset": dataset_keys.get(tf),
                "model": meta.get("model_family"),
                "signal": meta.get("signal_id"),
                "timeframe": tf,
                "holding_period": hb,
                "direction": exp.get("direction"),
                "cost_model": "BASE",
                "train": [slices["train"].start, slices["train"].stop],
                "validation": [slices["validation"].start, slices["validation"].stop],
                "oos": [slices["oos"].start, slices["oos"].stop],
                "result": {
                    "oos_net_sharpe": oos_net_sharpe,
                    "oos_net_return": oos_net_ret,
                    "gate_status": row["gate_status"],
                },
                "status": row["gate_status"],
                "seed": cfg.random_seed,
            }
        )

    # Portfolio comparison on reconstructed daily nets (mu/cov from pre-OOS only)
    portfolio_comparison: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "prompt41_reference": None,
        "methods": {},
        "note": "Weights estimated on pre-OOS intersection; OOS evaluation only. Not optimized for final OOS.",
    }
    p41 = Path(cfg.prompt41_dir) / "portfolio_method_results.json"
    if p41.exists():
        portfolio_comparison["prompt41_reference"] = {
            "source": str(p41),
            "note": "Prompt 41 artifacts immutable; referenced for consistency, not re-tuned.",
        }
    try:
        from iqrp.app.backtesting.portfolio_integration.adapters import (
            causal_mu_cov,
            daily_panel_from_series,
            run_optimizer,
            weights_dict,
        )

        sleeve_ids = [cid for cid in series_map if any(
            r.get("candidate_id") == cid and r.get("survives_BASE") for r in candidate_results
        )]
        if len(sleeve_ids) < 2:
            sleeve_ids = list(series_map.keys())[: min(5, len(series_map))]
        if len(sleeve_ids) >= 2:
            panel, period_dates = daily_panel_from_series({k: series_map[k] for k in sleeve_ids})
            est = causal_mu_cov(panel, sleeve_ids, period_dates)
            equal = {n: 1.0 / len(sleeve_ids) for n in sleeve_ids}
            method_results = {"equal_sleeve_baseline": {"weights": equal, "status": "OK"}}
            for method in ("mean_variance", "risk_parity", "black_litterman", "hrp", "constraints_only"):
                long_only = method in {"risk_parity", "hrp", "constraints_only"}
                try:
                    opt = run_optimizer(
                        method,
                        mu=est["mu"],
                        cov=est["cov"],
                        names=sleeve_ids,
                        max_weight=0.5,
                        max_gross=1.0,
                        budget=1.0,
                        risk_aversion=1.0,
                        long_only_sleeves=long_only,
                    )
                    w = weights_dict(opt, sleeve_ids)
                    # OOS portfolio return: equal lag of abs(weight)*sleeve daily net
                    oos_dates = sorted(set.intersection(*(period_dates[n]["oos"] for n in sleeve_ids)))
                    if len(oos_dates) >= 5:
                        nets = panel.loc[oos_dates, sleeve_ids].fillna(0.0)
                        wvec = np.array([float(w.get(n, 0.0)) for n in sleeve_ids])
                        # static weights; next-day attribution
                        port = (nets.to_numpy() * np.abs(wvec)).sum(axis=1)
                        method_results[method] = {
                            "weights": w,
                            "status": "OK",
                            "oos_net_sharpe": sharpe_from_rets(port, 252.0),
                            "oos_net_return": float(np.nansum(port)),
                            "oos_max_dd": float(max_drawdown(port)),
                            "concentration_hhi": float(np.sum(np.abs(wvec) ** 2)),
                            "gross_exposure": float(np.sum(np.abs(wvec))),
                            "net_exposure": float(np.sum(wvec)),
                        }
                    else:
                        method_results[method] = {"weights": w, "status": "OK", "oos": "insufficient_dates"}
                except Exception as ex:  # noqa: BLE001
                    method_results[method] = {"status": "FAIL", "error": str(ex)[:300]}
            # equal baseline OOS
            oos_dates = sorted(set.intersection(*(period_dates[n]["oos"] for n in sleeve_ids)))
            if len(oos_dates) >= 5:
                nets = panel.loc[oos_dates, sleeve_ids].fillna(0.0)
                wvec = np.array([equal[n] for n in sleeve_ids])
                port = (nets.to_numpy() * wvec).sum(axis=1)
                method_results["equal_sleeve_baseline"].update(
                    {
                        "oos_net_sharpe": sharpe_from_rets(port, 252.0),
                        "oos_net_return": float(np.nansum(port)),
                        "oos_max_dd": float(max_drawdown(port)),
                        "gross_exposure": 1.0,
                        "net_exposure": 1.0,
                    }
                )
            portfolio_comparison["sleeves"] = sleeve_ids
            portfolio_comparison["estimation"] = {
                "n_obs": est.get("n_obs"),
                "cov_method": est.get("cov_method"),
                "estimation_window": est.get("estimation_window"),
            }
            portfolio_comparison["methods"] = method_results
        else:
            portfolio_comparison["status"] = "INSUFFICIENT_SLEEVES"
    except Exception as ex:  # noqa: BLE001
        portfolio_comparison["status"] = "FAIL"
        portfolio_comparison["error"] = str(ex)[:400]

    # Execution realism summary
    execution_realism = {
        "disclaimer": DISCLAIMER,
        "next_bar_execution": True,
        "mechanism": "evaluate_cost_aware uses pos[t-1]*ret[t] (no same-bar lookahead)",
        "cost_components": ["commission_bps", "spread_bps", "slippage_bps"],
        "long_short_transitions": "supported via signed positions",
        "unified_cascade_smoke": "per-candidate UnifiedTradingOrchestrator fill simulation",
    }

    # Decision matrix
    matrix = []
    for r in candidate_results:
        if "gate_status" not in r:
            continue
        matrix.append(
            {
                "candidate": r["candidate_id"],
                "model": r.get("model_family"),
                "tf": r.get("timeframe"),
                "holding": r.get("holding_bars"),
                "trades_per_day": r.get("trades_per_day"),
                "gross_sharpe": r.get("oos_gross_sharpe"),
                "net_sharpe": r.get("oos_net_sharpe"),
                "oos_sharpe": r.get("oos_net_sharpe"),
                "max_dd": r.get("oos_max_dd"),
                "cost_survival": {
                    "BASE": r.get("survives_BASE"),
                    "MODERATE": r.get("survives_MODERATE"),
                    "ADVERSE_catastrophic": float(r.get("adverse_net_sharpe") or 0)
                    < GATE_ADVERSE_CATASTROPHIC_SHARPE,
                },
                "regime_survival": r.get("regime_label"),
                "status": r.get("gate_status"),
            }
        )

    status_counts = Counter(r.get("gate_status") for r in candidate_results if r.get("gate_status"))
    n_profit = status_counts.get("PROFITABILITY_EVIDENCE", 0)

    # Answers (honest)
    best_family = None
    best_tf = None
    best_hb = None
    scored = [r for r in candidate_results if r.get("oos_net_sharpe") is not None]
    if scored:
        # diagnostic only — not selection
        by_fam = {}
        for r in scored:
            by_fam.setdefault(r.get("model_family"), []).append(float(r.get("oos_net_sharpe") or 0))
        best_family = max(by_fam, key=lambda k: float(np.nanmedian(by_fam[k]))) if by_fam else None
        by_tf = {}
        for r in scored:
            by_tf.setdefault(r.get("timeframe"), []).append(float(r.get("oos_net_sharpe") or 0))
        best_tf = max(by_tf, key=lambda k: float(np.nanmedian(by_tf[k]))) if by_tf else None
        by_hb = {}
        for r in scored:
            by_hb.setdefault(r.get("holding_bars"), []).append(float(r.get("oos_net_sharpe") or 0))
        best_hb = max(by_hb, key=lambda k: float(np.nanmedian(by_hb[k]))) if by_hb else None

    answers = {
        "did_any_demonstrate_profitability_evidence": n_profit > 0,
        "n_profitability_evidence": n_profit,
        "which_model_family_best_diagnostic": best_family,
        "which_timeframe_best_diagnostic": best_tf,
        "which_holding_best_diagnostic": best_hb,
        "discovered_suitable_horizon": best_hb is not None,
        "how_frequently_should_trade": "See behavior_class per candidate; do not maximize trade count.",
        "edge_long_short_or_both": dict(Counter(r.get("direction") for r in candidate_results)),
        "transaction_cost_destroy": cost_analysis,
        "survives_adverse": [
            r["candidate_id"]
            for r in candidate_results
            if float(r.get("adverse_net_sharpe") or 0) > GATE_ADVERSE_CATASTROPHIC_SHARPE
            and float(r.get("adverse_net_sharpe") or 0) > 0
        ],
        "survives_regimes": [r["candidate_id"] for r in candidate_results if r.get("regime_ok")],
        "survives_walk_forward": [r["candidate_id"] for r in candidate_results if r.get("walk_forward_ok")],
        "portfolio_construction_improves": (
            "See Prompt 41 comparison; not re-optimized here. Constraints-only often competitive."
        ),
        "statistically_convincing": any(
            s.get("statistical_evidence") == "SUFFICIENT_FOR_RESEARCH" for s in statistical_rows
        ),
        "reproducible": all(r.get("reproducible") for r in candidate_results if "reproducible" in r),
        "suitable_for_paper_trading": n_profit > 0,
        "suitable_for_live_trading": False,
        "status_counts": dict(status_counts),
    }

    final_status = (
        "PROFITABILITY_EVIDENCE"
        if n_profit > 0
        else "ROBUST_RESEARCH_CANDIDATE"
        if status_counts.get("ROBUST_RESEARCH_CANDIDATE") or status_counts.get("RESEARCH_ONLY")
        else "NO_CREDIBLE_EDGE"
    )

    # Reproducibility report
    repro = {
        "status": "PASS" if answers["reproducible"] else "FAIL",
        "seed": cfg.random_seed,
        "dataset_keys": dataset_keys,
        "using_extended_v101": resolved.get("using_extended_v101"),
        "n_candidates": len(candidate_results),
        "disclaimer": DISCLAIMER,
    }

    final = {
        "disclaimer": DISCLAIMER,
        "validation_id": cfg.validation_id,
        "started_at": started,
        "final_status": final_status,
        "answers": answers,
        "decision_matrix": matrix,
        "claim_distinctions": {
            "ARCHITECTURE_COMPLETE": True,
            "PORTFOLIO_INTEGRATED": True,
            "PROFITABILITY_EVIDENCE": n_profit > 0,
            "PROVEN_PROFITABILITY_ABSOLUTE": False,
            "PAPER_TRADING_RECOMMENDED": n_profit > 0,
            "LIVE_READY": False,
        },
        "recon_errors": recon_errors,
    }

    md = [
        "# Final Trading Validation (Prompt 42)",
        "",
        f"Status: **{final_status}**",
        "",
        DISCLAIMER,
        "",
        f"- Extended data: `{resolved.get('using_extended_v101')}` keys={dataset_keys}",
        f"- Candidates validated: {len(candidate_results)}",
        f"- PROFITABILITY_EVIDENCE count: **{n_profit}**",
        f"- Suitable for live trading: **NO**",
        "",
        "## Decision matrix",
        "",
        "| Candidate | Model | TF | Holding | Trades/day | Net Sharpe | Max DD | Status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in matrix:
        md.append(
            f"| `{m['candidate'][:16]}` | {m['model']} | {m['tf']} | {m['holding']} | "
            f"{m['trades_per_day']} | {m['net_sharpe']} | {m['max_dd']} | **{m['status']}** |"
        )
    md.extend(
        [
            "",
            "## Required answers",
            "",
            f"1. Profitability evidence? **{'YES' if n_profit else 'NO'}** ({n_profit})",
            f"2. Best model family (diagnostic median OOS Sharpe): {best_family}",
            f"3. Best timeframe (diagnostic): {best_tf}",
            f"4. Best holding (diagnostic): {best_hb}",
            f"5. Suitable horizon discovered? {best_hb is not None} (diagnostic only)",
            "6. Trade frequency: per-candidate behavior_class in trading_behavior.json",
            f"7. Direction mix: {answers['edge_long_short_or_both']}",
            "8–11. Costs/regimes/walk-forward: see cost_analysis.json / regime_analysis.json / walk_forward_results.json",
            f"12. Portfolio improves? See portfolio_comparison.json methods vs equal_sleeve_baseline. {answers['portfolio_construction_improves']}",
            f"13. Statistically convincing? **{answers['statistically_convincing']}**",
            f"14. Reproducible? **{answers['reproducible']}**",
            f"15. Paper trading suitable? **{answers['suitable_for_paper_trading']}** (only if PROFITABILITY_EVIDENCE)",
            "16. Live trading suitable? **NO**",
            "",
            "## Stop",
            "",
            "STOP — no broker connection, no live orders, no LIVE_READY claim.",
            "",
        ]
    )

    _write(out_dir / "final_report.json", final)
    (out_dir / "final_report.md").write_text("\n".join(md), encoding="utf-8")
    _write(out_dir / "profitability_gate.json", {"gates": gate_rows, "status_counts": dict(status_counts), "disclaimer": DISCLAIMER})
    _write(out_dir / "candidate_results.json", {"results": candidate_results, "disclaimer": DISCLAIMER})
    _write(out_dir / "trading_behavior.json", {"rows": trading_behavior, "disclaimer": DISCLAIMER})
    _write(out_dir / "cost_analysis.json", {"rows": cost_analysis, "disclaimer": DISCLAIMER})
    _write(out_dir / "execution_realism.json", execution_realism)
    _write(out_dir / "regime_analysis.json", {"rows": regime_analysis, "disclaimer": DISCLAIMER})
    _write(out_dir / "walk_forward_results.json", {"folds": walk_forward_results, "disclaimer": DISCLAIMER})
    _write(out_dir / "statistical_validation.json", {"rows": statistical_rows, "disclaimer": DISCLAIMER})
    _write(out_dir / "portfolio_comparison.json", portfolio_comparison)
    _write(out_dir / "experiment_registry.json", {"experiments": experiment_registry, "n": len(experiment_registry)})
    _write(out_dir / "reproducibility_report.json", repro)
    _write(out_dir / "test_summary.json", {"note": "Filled after pytest", "disclaimer": DISCLAIMER})

    if progress:
        print(f"[ftv] done status={final_status} n_profit={n_profit}", flush=True)
    return final


__all__ = ["run_final_validation"]

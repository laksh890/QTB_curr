"""Runner: independent OOS validation of frozen Prompt-42 candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.analytics import evaluate_cost_aware, positions_from_signal
from iqrp.app.backtesting.alpha_research.consolidation.reconstruct import (
    build_signal_cache,
    sharpe_from_rets,
)
from iqrp.app.backtesting.alpha_research.experiments import now_iso
from iqrp.app.backtesting.alpha_research.model_campaign.protocol import apply_direction_mask
from iqrp.app.backtesting.alpha_research.model_campaign.runner import _trade_stats
from iqrp.app.backtesting.alpha_research.types import COST_SCENARIOS, bars_per_day
from iqrp.app.backtesting.final_holdout.causality import audit_causality
from iqrp.app.backtesting.final_holdout.freeze import freeze_candidates
from iqrp.app.backtesting.final_validation.runner import (
    acf1,
    effective_sample_size,
    newey_west_se,
    regime_labels_from_returns,
)
from iqrp.app.backtesting.independent_validation.protocol import (
    DISCLAIMER,
    FROZEN_ALL,
    NEGATIVE_CONTROL_ID,
    PAPER_GATE_REQUIRED,
    IndependentValidationConfig,
    classify_candidate,
)
from iqrp.app.backtesting.independent_validation.provenance import build_independent_provenance
from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.serializer import to_jsonable
from iqrp.app.backtesting.unified_pipeline.orchestrator import UnifiedTradingOrchestrator
from iqrp.app.backtesting.unified_pipeline.types import AlphaCandidate


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str), encoding="utf-8")


def _sortino(rets: np.ndarray, ppy: float) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return float("nan")
    downside = r[r < 0]
    if downside.size < 2:
        return float("nan")
    dd = float(np.std(downside, ddof=1))
    if dd < 1e-15:
        return 0.0
    return float(np.mean(r) / dd * np.sqrt(ppy))


def _calmar(rets: np.ndarray, ppy: float) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return float("nan")
    ann = float(np.mean(r) * ppy)
    dd = float(max_drawdown(r))
    if dd < 1e-15:
        return float("nan")
    return ann / dd


def _load_frame(tf: str, firewall_end: pd.Timestamp, holdout_dir: Path, warmup: int) -> tuple[pd.DataFrame, np.ndarray]:
    reg = pd.read_parquet(f"data/btcusdt/btcusdt_intraday_{tf}.parquet")
    reg["timestamp"] = pd.to_datetime(reg["timestamp"], utc=True)
    reg = reg[reg["timestamp"] <= firewall_end].tail(warmup)
    hold = pd.read_parquet(holdout_dir / f"btcusdt_holdout_{tf}.parquet")
    hold["timestamp"] = pd.to_datetime(hold["timestamp"], utc=True)
    frame = (
        pd.concat([reg, hold], ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    mask = (frame["timestamp"] > firewall_end).to_numpy(dtype=bool)
    return frame, mask


def _breakeven_cost_bps(positions: pd.Series, rets: pd.Series, ppy: float) -> float | None:
    """Find total cost bps (commission+spread+slippage equally split) where net Sharpe ≈ 0."""
    # Binary search on scale of BASE total (5 bps components sum in BASE = 5)
    base_total = 5.0  # 1+2+2
    lo, hi = 0.0, 200.0
    best = None
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        # split mid across 3 equal components
        c = mid / 3.0
        ev = evaluate_cost_aware(positions, rets, commission_bps=c, spread_bps=c, slippage_bps=c, periods_per_year=ppy)
        s = float(ev["net_sharpe"])
        if s > 0:
            lo = mid
            best = mid
        else:
            hi = mid
    return best


def _block_bootstrap_sharpe(rets: np.ndarray, ppy: float, *, n_boot: int, block: int, seed: int) -> dict[str, Any]:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < max(block * 2, 10):
        return {
            "status": "INSUFFICIENT_FOR_BOOTSTRAP",
            "n": int(n),
            "note": "Holdout too short for meaningful block bootstrap.",
        }
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    samples = []
    terminals = []
    maxdds = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        pieces = [r[s : s + block] for s in starts]
        boot = np.concatenate(pieces)[:n]
        samples.append(sharpe_from_rets(boot, ppy))
        terminals.append(float(np.nansum(boot)))
        maxdds.append(float(max_drawdown(boot)))
    arr = np.asarray(samples, dtype=float)
    term = np.asarray(terminals, dtype=float)
    mdd = np.asarray(maxdds, dtype=float)
    obs = sharpe_from_rets(r, ppy)
    return {
        "status": "OK",
        "n_boot": n_boot,
        "block_len": block,
        "observed_sharpe": obs,
        "sharpe_mean": float(np.nanmean(arr)),
        "sharpe_ci_90": [float(np.nanpercentile(arr, 5)), float(np.nanpercentile(arr, 95))],
        "terminal_wealth_ci_90": [float(np.nanpercentile(term, 5)), float(np.nanpercentile(term, 95))],
        "max_dd_ci_90": [float(np.nanpercentile(mdd, 5)), float(np.nanpercentile(mdd, 95))],
        "prob_sharpe_le_0": float(np.mean(arr <= 0)),
        "prob_loss": float(np.mean(term <= 0)),
        "note": "Circular block bootstrap on holdout net returns; dependence-preserving.",
    }


def _walk_forward_holdout(net: np.ndarray, ppy: float, n_folds: int) -> list[dict[str, Any]]:
    """Rolling folds on holdout-only net returns (no retrain of frozen signal)."""
    n = len(net)
    if n < 20 or n_folds < 2:
        return [
            {
                "status": "INSUFFICIENT_FOR_WALK_FORWARD",
                "n_bars": n,
                "note": "Holdout too short for multi-segment walk-forward.",
            }
        ]
    fold = max(n // n_folds, 5)
    out = []
    for i in range(n_folds):
        a = i * fold
        b = n if i == n_folds - 1 else (i + 1) * fold
        if b - a < 3:
            continue
        seg = net[a:b]
        out.append(
            {
                "fold": i,
                "test_start_idx": a,
                "test_end_idx": b,
                "n": int(b - a),
                "oos_net_return": float(np.nansum(seg)),
                "oos_net_sharpe": sharpe_from_rets(seg, ppy),
                "oos_max_dd": float(max_drawdown(seg)),
                "note": "Frozen signal; fold is chronological holdout slice only (no parameter update).",
            }
        )
    return out


def _paper_gate(row: dict[str, Any], *, duration_ok: bool, immutable: bool, no_overlap: bool) -> dict[str, Any]:
    checks = {
        "holdout_calendar_days_ge_180": duration_ok,
        "positive_net_oos": float(row.get("net_sharpe") or -1) > 0 and float(row.get("net_return") or -1) > 0,
        "multiple_independent_periods": bool(row.get("n_wf_positive_folds", 0) >= 2),
        "acceptable_drawdown": float(row.get("max_drawdown") or 1) <= 0.5,
        "survives_BASE": bool(row.get("survives_BASE")),
        "survives_MODERATE": bool(row.get("survives_MODERATE")),
        "not_regime_concentrated": bool(row.get("not_regime_concentrated")),
        "statistically_credible": bool(row.get("statistically_credible")),
        "reproducible": bool(row.get("reproducible")),
        "no_leakage": bool(row.get("causality_pass")),
        "no_temporal_overlap": no_overlap,
        "candidate_immutable": immutable,
    }
    failed = [k for k, v in checks.items() if not v]
    return {
        "status": "PAPER_TRADING_CANDIDATE" if not failed else "NO_PAPER_TRADING_CANDIDATE",
        "checks": checks,
        "failed_checks": failed,
        "required": list(PAPER_GATE_REQUIRED),
    }


def _evaluate(
    cfg: IndependentValidationConfig,
    *,
    freeze: dict[str, Any],
    provenance: dict[str, Any],
    progress: bool,
) -> dict[str, Any]:
    hold = provenance["independent_holdout"]
    if not hold.get("holdout_available"):
        return {"status": "INVALID_HOLDOUT", "rows": [], "reason": "no independent bars"}

    firewall_end = pd.Timestamp(provenance["firewall"]["firewall_end_exclusive"])
    holdout_dir = Path("data/btcusdt/independent_holdout")
    calendar_days = int(provenance.get("calendar_days") or 0)
    duration_status = provenance["duration_status"]

    tfs = {"5m", "15m", "30m", "1h"}
    frames: dict[str, pd.DataFrame] = {}
    masks: dict[str, np.ndarray] = {}
    for tf in sorted(tfs):
        fr, m = _load_frame(tf, firewall_end, holdout_dir, int(cfg.warmup_bars.get(tf, 2000)))
        frames[tf] = fr
        masks[tf] = m
        if progress:
            print(f"[indval] {tf}: n={len(fr)} holdout={int(m.sum())}", flush=True)

    needed: set[tuple[str, str, str]] = set()
    for c in freeze["candidates"]:
        if c["candidate_id"] not in FROZEN_ALL:
            continue
        d = c["definition"]
        needed.add((d["kind"], d["source_id"], d["timeframe"]))
        sid = d["source_id"]
        if ":" in sid and "->" in sid:
            base, rest = sid.split(":", 1)
            mtf, _ = rest.split("->", 1)
            needed.add(("ref", base, mtf))

    cache, errs = build_signal_cache(frames, needed=needed, reference_lookback=20, train_frac=0.5, progress=progress)

    rows = []
    wf_all = []
    regime_all = []
    cost_all = []
    stat_all = []
    boot_all = []
    neg_control = None
    causality = {}

    for c in freeze["candidates"]:
        cid = c["candidate_id"]
        if cid not in FROZEN_ALL:
            continue
        d = c["definition"]
        tf = d["timeframe"]
        frame = frames[tf]
        mask = masks[tf]
        key = f"{d['kind']}:{d['source_id']}:{tf}"
        if key not in cache:
            rows.append({"candidate_id": cid, "status": "FAIL", "reason": f"cache miss {key}"})
            continue

        raw = cache[key]
        directed = apply_direction_mask(raw.fillna(0.0), d["direction"])
        positions = positions_from_signal(directed, int(d["holding_bars"]))
        rets = frame["close"].pct_change().fillna(0.0)
        bpd = bars_per_day(tf, market_type=cfg.market_type)
        ppy = 252.0 * float(bpd)

        causality[cid] = audit_causality(
            raw_signal=raw,
            frame=frame,
            direction=d["direction"],
            holding_bars=int(d["holding_bars"]),
            holdout_mask=mask,
        )

        cost_by = {}
        for name in ("BASE", "MODERATE", "ADVERSE"):
            cm = COST_SCENARIOS[name]
            cost_by[name] = evaluate_cost_aware(
                positions,
                rets,
                commission_bps=float(cm["commission_bps"]),
                spread_bps=float(cm["spread_bps"]),
                slippage_bps=float(cm["slippage_bps"]),
                periods_per_year=ppy,
                timestamps=frame["timestamp"],
            )

        h_net = np.asarray(cost_by["BASE"]["net_returns"], dtype=float)[mask]
        h_gross = np.asarray(cost_by["BASE"]["gross_returns"], dtype=float)[mask]
        h_pos = positions.iloc[np.where(mask)[0]]
        tstats = _trade_stats(h_pos, tf, cfg.market_type)

        net_ret = float(np.nansum(h_net))
        gross_ret = float(np.nansum(h_gross))
        net_sharpe = sharpe_from_rets(h_net, ppy)
        gross_sharpe = sharpe_from_rets(h_gross, ppy)
        max_dd = float(max_drawdown(h_net))
        cost_drag = float(cost_by["BASE"].get("transaction_costs") or 0)
        cost_pct_gross = float(abs(cost_drag) / max(abs(gross_ret), 1e-12))

        mod_net = np.asarray(cost_by["MODERATE"]["net_returns"], dtype=float)[mask]
        adv_net = np.asarray(cost_by["ADVERSE"]["net_returns"], dtype=float)[mask]
        mod_sharpe = sharpe_from_rets(mod_net, ppy)
        adv_sharpe = sharpe_from_rets(adv_net, ppy)
        survives_BASE = bool(net_ret > 0 and net_sharpe > 0)
        survives_MODERATE = bool(float(np.nansum(mod_net)) > 0 and mod_sharpe > 0)
        survives_ADVERSE = bool(float(np.nansum(adv_net)) > 0 and adv_sharpe > 0)

        # Holding duration (bars)
        pos_a = h_pos.fillna(0.0).to_numpy(dtype=float)
        hold_lens = []
        i = 0
        while i < len(pos_a):
            if pos_a[i] == 0:
                i += 1
                continue
            j = i
            while j < len(pos_a) and pos_a[j] == pos_a[i]:
                j += 1
            hold_lens.append(j - i)
            i = j

        # Regime
        regimes = regime_labels_from_returns(rets.to_numpy(), vol_win=max(24, int(bpd)))
        by_reg = {}
        pos_regs = []
        for name, rmask in regimes.items():
            m = rmask & mask
            if int(m.sum()) < 3:
                by_reg[name] = None
                continue
            s = sharpe_from_rets(np.asarray(cost_by["BASE"]["net_returns"], dtype=float)[m], ppy)
            by_reg[name] = s
            if s is not None and s > 0:
                pos_regs.append(name)
        not_regime_concentrated = not (len(pos_regs) == 1 and sum(v is not None for v in by_reg.values()) >= 2)

        # Stats
        rho = acf1(h_net)
        n_eff = effective_sample_size(max(len(h_net), 1), rho) / max(int(d["holding_bars"]), 1)
        se = newey_west_se(h_net)
        mean_r = float(np.nanmean(h_net)) if len(h_net) else 0.0
        t_hac = mean_r / se if se and np.isfinite(se) and se > 0 else float("nan")
        # Independent trading days in holdout
        ts_h = frame.loc[mask, "timestamp"]
        n_days = int(pd.to_datetime(ts_h, utc=True).dt.floor("D").nunique()) if len(ts_h) else 0
        stat_ok = bool(calendar_days >= 180 and n_eff >= 50 and np.isfinite(t_hac) and abs(t_hac) >= 1.96)
        stat_label = "SUFFICIENT_FOR_RESEARCH" if stat_ok else "STATISTICAL_EVIDENCE_INSUFFICIENT"

        wf = _walk_forward_holdout(h_net, ppy, cfg.n_walk_forward)
        n_wf_pos = sum(1 for f in wf if isinstance(f.get("oos_net_sharpe"), (int, float)) and f["oos_net_sharpe"] > 0)

        boot = _block_bootstrap_sharpe(
            h_net, ppy, n_boot=cfg.n_bootstrap, block=cfg.block_len_bars, seed=cfg.random_seed + hash(cid) % 1000
        )

        # Breakeven on holdout positions (full series costs applied; metrics on holdout slice via scale search on full then slice)
        # Use holdout-only series constructed positions/returns
        be = None
        if int(mask.sum()) >= 5:
            be = _breakeven_cost_bps(positions.iloc[np.where(mask)[0]].reset_index(drop=True), rets.iloc[np.where(mask)[0]].reset_index(drop=True), ppy)

        # Recon smoke
        try:
            orch = UnifiedTradingOrchestrator(initial_capital=100_000.0, long_only=False, max_position=0.2, max_gross=1.0)
            px = float(frame["close"].iloc[-1])
            direction = float(np.sign(h_pos.replace(0, np.nan).dropna().iloc[-1])) if h_pos.replace(0, np.nan).dropna().size else 0.0
            cand = AlphaCandidate(
                candidate_id=f"indval:{cid}",
                signal_id=str(d["source_id"]),
                instrument="BTCUSDT",
                timestamp=now_iso(),
                direction=direction,
                signal_value=direction * 0.05,
                confidence=0.5,
                expected_horizon=int(d["holding_bars"]),
                signal_timeframe=tf,
                execution_timeframe=tf,
                source_model="MTF",
                source_model_version="1.0.0",
                data_version="independent_holdout",
                dataset_checksum=str((hold.get("holdout_files") or {}).get(tf, {}).get("sha256") or "holdout"),
                oos_status="INDEPENDENT_HOLDOUT",
                experiment_id=cid,
                requested_weight=direction * 0.05,
            )
            cascade = orch.process_candidates([cand], asof=now_iso(), prices={"BTCUSDT": px}, simulation_mode="fill")
            recon = cascade.get("reconciliation") or {}
            recon_ok = bool(recon.get("ok") or str(recon.get("outcome", "")).upper() == "RECONCILIATION_OK")
        except Exception as ex:  # noqa: BLE001
            recon_ok = False
            recon = {"error": str(ex)[:300]}

        is_neg = cid == NEGATIVE_CONTROL_ID
        final_class = classify_candidate(
            duration_status=duration_status,
            net_sharpe=net_sharpe,
            survives_base=survives_BASE,
            survives_moderate=survives_MODERATE,
            regime_ok=not_regime_concentrated,
            stat_ok=stat_ok,
            is_negative_control=is_neg,
        )

        # Daily / monthly PnL
        daily = (
            pd.DataFrame({"date": pd.to_datetime(ts_h, utc=True).dt.floor("D"), "net": h_net})
            .groupby("date", sort=True)["net"]
            .sum()
        )
        monthly = daily.copy()
        if len(daily):
            monthly = daily.groupby(pd.DatetimeIndex(daily.index).to_period("M")).sum()

        row = {
            "candidate_id": cid,
            "role": "NEGATIVE_CONTROL" if is_neg else "PRIMARY_FROZEN",
            "definition_checksum": c["definition_checksum"],
            "timeframe": tf,
            "holding_bars": d["holding_bars"],
            "direction": d["direction"],
            "signal_id": d["source_id"],
            "holdout_bars": int(mask.sum()),
            "calendar_days": calendar_days,
            "duration_status": duration_status,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "gross_sharpe": gross_sharpe,
            "net_sharpe": net_sharpe,
            "sortino": _sortino(h_net, ppy),
            "max_drawdown": max_dd,
            "calmar": _calmar(h_net, ppy),
            "volatility_ann": float(np.nanstd(h_net, ddof=1) * np.sqrt(ppy)) if np.isfinite(h_net).sum() > 2 else None,
            "trades_per_day": tstats.get("trades_per_day"),
            "n_position_changes": tstats.get("n_position_changes"),
            "long_entries": tstats.get("long_entries"),
            "short_entries": tstats.get("short_entries"),
            "avg_holding_bars": float(np.mean(hold_lens)) if hold_lens else None,
            "median_holding_bars": float(np.median(hold_lens)) if hold_lens else None,
            "turnover": cost_by["BASE"].get("turnover"),
            "gross_exposure": float(np.nanmean(np.abs(pos_a))) if len(pos_a) else 0.0,
            "net_exposure": float(np.nanmean(pos_a)) if len(pos_a) else 0.0,
            "cost_drag": cost_drag,
            "cost_pct_of_gross": cost_pct_gross,
            "cost_model_label": provenance.get("ohlcv_cost_model_label"),
            "survives_BASE": survives_BASE,
            "survives_MODERATE": survives_MODERATE,
            "survives_ADVERSE": survives_ADVERSE,
            "moderate_net_sharpe": mod_sharpe,
            "adverse_net_sharpe": adv_sharpe,
            "breakeven_total_cost_bps": be,
            "causality_pass": causality[cid]["status"] == "PASS",
            "recon_ok": recon_ok,
            "not_regime_concentrated": not_regime_concentrated,
            "statistically_credible": stat_ok,
            "statistical_evidence": stat_label,
            "n_eff": n_eff,
            "t_hac": t_hac,
            "independent_trading_days": n_days,
            "n_wf_positive_folds": n_wf_pos,
            "final_class": final_class,
            "daily_pnl": {str(k): float(v) for k, v in daily.items()},
            "monthly_pnl": {str(k): float(v) for k, v in (monthly.items() if hasattr(monthly, "items") else [])},
            "selection_bias_note": (
                "Candidate selected from Prompt 39/40/42 research universe; "
                "not a random draw. Multiple-testing exposure remains."
            ),
        }
        rows.append(row)
        wf_all.append({"candidate_id": cid, "folds": wf})
        regime_all.append({"candidate_id": cid, "by_regime": by_reg})
        cost_all.append(
            {
                "candidate_id": cid,
                "label": provenance.get("ohlcv_cost_model_label"),
                "BASE": {"net_sharpe": net_sharpe, "components": COST_SCENARIOS["BASE"], "survives": survives_BASE},
                "MODERATE": {"net_sharpe": mod_sharpe, "components": COST_SCENARIOS["MODERATE"], "survives": survives_MODERATE},
                "ADVERSE": {"net_sharpe": adv_sharpe, "components": COST_SCENARIOS["ADVERSE"], "survives": survives_ADVERSE},
                "breakeven_total_cost_bps": be,
                "cost_pct_of_gross": cost_pct_gross,
            }
        )
        stat_all.append(
            {
                "candidate_id": cid,
                "acf1": rho,
                "n_eff_overlap_adj": n_eff,
                "newey_west_se": se,
                "t_hac": t_hac,
                "independent_trading_days": n_days,
                "n_trades_proxy": tstats.get("n_position_changes"),
                "statistical_evidence": stat_label,
                "multiple_testing_note": (
                    "Discovery involved large Prompt-39 grid; holdout success must overcome selection bias. "
                    "Short holdout cannot clear this burden."
                ),
            }
        )
        boot_all.append({"candidate_id": cid, **boot})
        if is_neg:
            neg_control = row

    fingerprint = hashlib.sha256(
        json.dumps(
            [{"id": r["candidate_id"], "net": r.get("net_return"), "sharpe": r.get("net_sharpe")} for r in rows],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()

    return {
        "status": duration_status,
        "rows": rows,
        "walk_forward": wf_all,
        "regime": regime_all,
        "cost": cost_all,
        "statistical": stat_all,
        "bootstrap": boot_all,
        "negative_control": neg_control,
        "causality": causality,
        "recon_errors": errs,
        "fingerprint": fingerprint,
        "capacity": {
            "capacity_status": "ESTIMATE_ONLY",
            "note": "OHLCV-only; no order-book. Do not invent market impact. No institutional capacity claim.",
        },
    }


def run_independent_validation(
    cfg: IndependentValidationConfig | None = None,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    cfg = cfg or IndependentValidationConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    _write(out_dir / "validation_config.json", {**cfg.to_dict(), "started_at": started})

    if progress:
        print("[indval] provenance + firewall", flush=True)
    provenance = build_independent_provenance(
        registry_path=cfg.registry_path,
        prompt42_dir=cfg.prompt42_dir,
        holdout_data_dir="data/btcusdt/independent_holdout",
    )
    _write(out_dir / "data_provenance.json", provenance)

    freeze = freeze_candidates(
        prompt39_dir=cfg.prompt39_dir,
        prompt42_dir=cfg.prompt42_dir,
        frozen_ids=FROZEN_ALL,
    )
    # Ensure freeze only includes our three
    freeze["candidates"] = [c for c in freeze["candidates"] if c["candidate_id"] in FROZEN_ALL]
    freeze["n_frozen"] = len(freeze["candidates"])
    _write(out_dir / "candidate_freeze.json", freeze)

    duration_status = provenance["duration_status"]
    duration_ok = duration_status == "DURATION_ADEQUATE"
    no_overlap = bool(provenance["firewall"].get("no_temporal_overlap"))
    immutable = freeze.get("status") == "PASS"

    # Always evaluate if any bars exist (diagnostic), but duration dominates verdict
    run_a = _evaluate(cfg, freeze=freeze, provenance=provenance, progress=progress)
    run_b = _evaluate(cfg, freeze=freeze, provenance=provenance, progress=False)
    repro = {
        "disclaimer": DISCLAIMER,
        "run_a": run_a.get("fingerprint"),
        "run_b": run_b.get("fingerprint"),
        "identical": run_a.get("fingerprint") == run_b.get("fingerprint"),
        "status": "PASS" if run_a.get("fingerprint") == run_b.get("fingerprint") else "FAIL",
        "aggregate_definition_checksum": freeze.get("aggregate_definition_checksum"),
        "holdout_1m_sha256": (provenance.get("independent_holdout") or {}).get("holdout_files", {}).get("1m", {}).get("sha256"),
    }

    gated_rows = []
    paper_ids = []
    for row in run_a.get("rows") or []:
        row = dict(row)
        row["reproducible"] = repro["status"] == "PASS"
        gate = _paper_gate(row, duration_ok=duration_ok, immutable=immutable, no_overlap=no_overlap)
        row["paper_gate"] = gate
        if gate["status"] == "PAPER_TRADING_CANDIDATE" and row.get("role") == "PRIMARY_FROZEN":
            paper_ids.append(row["candidate_id"])
        gated_rows.append(row)

    # Stability diagnostics (non-selective): holding ±1 only reported, not used for selection
    stability = {
        "note": "Nearby holding diagnostics are informational only; frozen candidate unchanged.",
        "ran": False,
        "reason": "Skipped while duration_status != DURATION_ADEQUATE to avoid implying optimization.",
    }
    if duration_ok:
        stability["ran"] = False
        stability["reason"] = "Duration adequate path reserved; still do not select on diagnostics."

    _write(out_dir / "holdout_results.json", {"results": gated_rows, "disclaimer": DISCLAIMER})
    _write(out_dir / "walk_forward_results.json", {"rows": run_a.get("walk_forward"), "disclaimer": DISCLAIMER})
    _write(out_dir / "regime_results.json", {"rows": run_a.get("regime"), "disclaimer": DISCLAIMER})
    _write(out_dir / "cost_results.json", {"rows": run_a.get("cost"), "disclaimer": DISCLAIMER})
    _write(out_dir / "statistical_results.json", {"rows": run_a.get("statistical"), "disclaimer": DISCLAIMER})
    _write(out_dir / "bootstrap_results.json", {"rows": run_a.get("bootstrap"), "disclaimer": DISCLAIMER})
    _write(out_dir / "capacity_results.json", run_a.get("capacity") or {"capacity_status": "ESTIMATE_ONLY"})
    _write(
        out_dir / "negative_control_results.json",
        {"control": run_a.get("negative_control"), "disclaimer": DISCLAIMER},
    )
    _write(out_dir / "reproducibility_report.json", repro)
    _write(out_dir / "stability_diagnostics.json", stability)

    primary = [r for r in gated_rows if r.get("role") == "PRIMARY_FROZEN"]
    replicated = [
        r["candidate_id"]
        for r in primary
        if r.get("survives_BASE") and float(r.get("net_sharpe") or -1) > 0 and duration_ok
    ]

    answers = {
        "1_did_either_replicate": bool(replicated),
        "1_note": (
            "Point estimates on a <30-day window are NOT accepted as replication under protocol gates."
            if not duration_ok
            else None
        ),
        "2_independent_months": max(int(provenance.get("calendar_days") or 0) // 30, 0),
        "2_calendar_days": provenance.get("calendar_days"),
        "3_independent_trades": {r["candidate_id"]: r.get("n_position_changes") for r in primary},
        "4_net_oos_sharpe": {r["candidate_id"]: r.get("net_sharpe") for r in primary},
        "5_max_drawdown": {r["candidate_id"]: r.get("max_drawdown") for r in primary},
        "6_cost_pct_of_gross": {r["candidate_id"]: r.get("cost_pct_of_gross") for r in primary},
        "7_cost_survival": {
            r["candidate_id"]: {
                "BASE": r.get("survives_BASE"),
                "MODERATE": r.get("survives_MODERATE"),
                "ADVERSE": r.get("survives_ADVERSE"),
            }
            for r in primary
        },
        "8_multiple_regimes": {r["candidate_id"]: r.get("not_regime_concentrated") for r in primary},
        "9_statistically_credible": any(r.get("statistically_credible") for r in primary),
        "10_paper_trading_candidate": paper_ids,
        "10_paper_status": "PAPER_TRADING_CANDIDATE" if paper_ids else "NO_PAPER_TRADING_CANDIDATE",
        "11_live_ready": False,
        "duration_status": duration_status,
        "per_candidate_class": {r["candidate_id"]: r.get("final_class") for r in gated_rows},
    }

    final_status = duration_status if duration_status in {"INVALID_HOLDOUT", "INSUFFICIENT_HOLDOUT"} else (
        "ROBUST_RESEARCH_EVIDENCE"
        if any(r.get("final_class") == "ROBUST_RESEARCH_EVIDENCE" for r in primary)
        else "PROMISING_OOS"
        if any(r.get("final_class") == "PROMISING_OOS" for r in primary)
        else "FAILED_REPLICATION"
    )

    md = [
        "# Independent Out-of-Sample Validation of Frozen Candidates",
        "",
        f"Status: **{final_status}**",
        "",
        DISCLAIMER,
        "",
        f"- Firewall end (exclusive): `{provenance['firewall']['firewall_end_exclusive']}`",
        f"- Independent calendar days: **{provenance.get('calendar_days')}** (need ≥180; invalid <30)",
        f"- Duration gate: **{duration_status}**",
        f"- Network extension: {'OK' if provenance['network_acquisition'].get('network_ok') else 'FAILED'}",
        f"- Candidate freeze: **{freeze.get('status')}**",
        f"- Reproducibility: **{repro['status']}**",
        f"- PAPER_TRADING_CANDIDATE: **{answers['10_paper_status']}**",
        f"- LIVE_READY: **NO**",
        "",
        "## Per-candidate (diagnostic metrics; duration gate dominates)",
        "",
        "| Candidate | Role | Net Sharpe | Max DD | Trades/day | Class | Paper |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in gated_rows:
        md.append(
            f"| `{r['candidate_id'][:16]}` | {r.get('role')} | {r.get('net_sharpe')} | {r.get('max_drawdown')} | "
            f"{r.get('trades_per_day')} | **{r.get('final_class')}** | {r['paper_gate']['status']} |"
        )
    md.extend(
        [
            "",
            "## Required answers",
            "",
            f"1. Did either replicate (under protocol)? **{answers['1_did_either_replicate']}** — {answers['1_note']}",
            f"2. Independent months≈{answers['2_independent_months']} (calendar days={answers['2_calendar_days']})",
            f"3. Independent trades: {answers['3_independent_trades']}",
            f"4. Net OOS Sharpe: {answers['4_net_oos_sharpe']}",
            f"5. Max DD: {answers['5_max_drawdown']}",
            f"6. Cost % of gross: {answers['6_cost_pct_of_gross']}",
            f"7. Cost survival: {answers['7_cost_survival']}",
            f"8. Multiple regimes: {answers['8_multiple_regimes']}",
            f"9. Statistically credible? **{answers['9_statistically_credible']}**",
            f"10. Paper trading? **{answers['10_paper_status']}** ids={paper_ids}",
            "11. LIVE_READY? **NO**",
            "",
            "## Critical conclusion",
            "",
            f"Available independent BTCUSDT history after the Prompt 35–42 firewall is "
            f"**{provenance.get('calendar_days')} day(s)**. Protocol requires ≥180 days for sufficiency "
            f"and ≥30 days even for profitability inference. Therefore the authoritative result is "
            f"**{final_status}**, not a profitability claim — regardless of any short-window Sharpe.",
            "",
            "Paid tick/L2 providers identified → **STOP_BEFORE_PURCHASE** (see data_provenance.json).",
            "",
            "## Stop",
            "",
            "STOP — no retuning, no new alphas, no broker, no live orders.",
            "",
        ]
    )
    (out_dir / "final_report.md").write_text("\n".join(md), encoding="utf-8")

    final = {
        "disclaimer": DISCLAIMER,
        "validation_id": cfg.validation_id,
        "started_at": started,
        "final_status": final_status,
        "answers": answers,
        "paper_trading_candidates": paper_ids,
        "live_ready": False,
        "claim_distinctions": {
            "INVALID_OR_INSUFFICIENT_HOLDOUT": final_status in {"INVALID_HOLDOUT", "INSUFFICIENT_HOLDOUT"},
            "PROVEN_PROFITABILITY": False,
            "PAPER_READY": False,
            "LIVE_READY": False,
        },
        "stability": stability,
    }
    _write(out_dir / "final_report.json", final)
    _write(out_dir / "test_summary.json", {"note": "Filled after pytest", "disclaimer": DISCLAIMER})
    if progress:
        print(f"[indval] done status={final_status} paper={paper_ids}", flush=True)
    return final


__all__ = ["run_independent_validation"]

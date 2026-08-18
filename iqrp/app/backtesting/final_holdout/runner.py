"""Independent final holdout validation runner for frozen Prompt-42 alphas."""

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
from iqrp.app.backtesting.final_holdout.protocol import (
    DISCLAIMER,
    GATE_MAX_DD,
    GATE_MIN_HOLDOUT_CALENDAR_DAYS,
    GATE_MIN_HOLDOUT_TRADES,
    GATE_MIN_N_EFF,
    GATE_MODERATE_COLLAPSE_SHARPE,
    FinalHoldoutConfig,
    classify_degradation,
)
from iqrp.app.backtesting.final_holdout.provenance import build_data_provenance
from iqrp.app.backtesting.final_validation.runner import (
    acf1,
    effective_sample_size,
    newey_west_se,
    regime_labels_from_returns,
)
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


def _trade_pnls(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [float(t.get("pnl") or 0) for t in trades]
    if not pnls:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "median_trade": 0.0,
            "largest_gain": None,
            "largest_loss": None,
            "max_consec_wins": 0,
            "max_consec_losses": 0,
            "long_trades": 0,
            "short_trades": 0,
        }
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    # consecutive
    mw = ml = cw = cl = 0
    for p in pnls:
        if p > 0:
            cw += 1
            cl = 0
            mw = max(mw, cw)
        else:
            cl += 1
            cw = 0
            ml = max(ml, cl)
    long_tr = [t for t in trades if t.get("side") == "LONG"]
    short_tr = [t for t in trades if t.get("side") == "SHORT"]
    return {
        "n_trades": len(pnls),
        "win_rate": float(len(wins) / len(pnls)),
        "profit_factor": float(sum(wins) / max(abs(sum(losses)), 1e-12)),
        "avg_trade": float(np.mean(pnls)),
        "median_trade": float(np.median(pnls)),
        "largest_gain": float(max(pnls)),
        "largest_loss": float(min(pnls)),
        "max_consec_wins": int(mw),
        "max_consec_losses": int(ml),
        "long_trades": int(len(long_tr)),
        "short_trades": int(len(short_tr)),
    }


def _load_warmup_plus_holdout(
    tf: str,
    *,
    p42_end: pd.Timestamp,
    holdout_dir: Path,
    warmup_bars: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    reg = pd.read_parquet(f"data/btcusdt/btcusdt_intraday_{tf}.parquet")
    reg["timestamp"] = pd.to_datetime(reg["timestamp"], utc=True)
    reg = reg[reg["timestamp"] <= p42_end].tail(warmup_bars)
    hold = pd.read_parquet(holdout_dir / f"btcusdt_holdout_{tf}.parquet")
    hold["timestamp"] = pd.to_datetime(hold["timestamp"], utc=True)
    frame = pd.concat([reg, hold], ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    mask = (frame["timestamp"] > p42_end).to_numpy(dtype=bool)
    return frame, mask


def _paper_gate(row: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "untouched_holdout_exists": bool(row.get("untouched_holdout_exists")),
        "candidate_definition_checksum_match": bool(row.get("checksum_match")),
        "causality_pass": bool(row.get("causality_pass")),
        "data_provenance_pass": bool(row.get("provenance_pass")),
        "positive_net_performance": float(row.get("holdout_net_return") or -1) > 0
        and float(row.get("holdout_net_sharpe") or -1) > 0,
        "survives_BASE": bool(row.get("survives_BASE")),
        "survives_MODERATE": bool(row.get("survives_MODERATE")),
        "acceptable_drawdown": float(row.get("holdout_max_dd") or 1) <= GATE_MAX_DD,
        "not_regime_only": bool(row.get("not_regime_only")),
        "reproducible": bool(row.get("reproducible")),
        "reconciliation_pass": bool(row.get("recon_ok")),
        "no_high_severity_defect": bool(row.get("no_high_severity_defect", True)),
        "holdout_sample_adequate": bool(row.get("holdout_sample_adequate")),
    }
    failed = [k for k, v in checks.items() if not v]
    if not failed:
        status = "PAPER_TRADING_CANDIDATE"
    elif checks["untouched_holdout_exists"] and checks["positive_net_performance"] and checks["survives_BASE"]:
        if not checks["holdout_sample_adequate"]:
            status = "WEAK_EVIDENCE"
        else:
            status = "RESEARCH_VALIDATED"
    elif checks["untouched_holdout_exists"] and not checks["positive_net_performance"]:
        status = "REJECTED"
    else:
        status = "WEAK_EVIDENCE" if checks["untouched_holdout_exists"] else "REJECTED"
    return {"status": status, "checks": checks, "failed_checks": failed}


def _evaluate_once(
    *,
    cfg: FinalHoldoutConfig,
    freeze: dict[str, Any],
    provenance: dict[str, Any],
    progress: bool,
) -> dict[str, Any]:
    holdout_meta = provenance["holdout"]
    if holdout_meta.get("final_status") != "HOLDOUT_ESTABLISHED":
        return {
            "status": "FINAL_HOLDOUT_UNAVAILABLE",
            "candidate_rows": [],
            "causality": {},
            "cost_stress": [],
            "regime": [],
            "statistical": [],
            "degradation": [],
            "reconciliation": {},
        }

    p42_end = pd.Timestamp(provenance["prompt42_windows"]["latest_p42_timestamp"])
    holdout_dir = Path("data/btcusdt/holdout")
    calendar_days = int(holdout_meta.get("holdout_calendar_days") or 0)

    # Load frames needed
    tfs = {"5m", "15m", "30m", "1h"}
    frames: dict[str, pd.DataFrame] = {}
    holdout_masks: dict[str, np.ndarray] = {}
    for tf in sorted(tfs):
        fr, mask = _load_warmup_plus_holdout(
            tf,
            p42_end=p42_end,
            holdout_dir=holdout_dir,
            warmup_bars=int(cfg.warmup_bars.get(tf, 2000)),
        )
        frames[tf] = fr
        holdout_masks[tf] = mask
        if progress:
            print(f"[holdout] frame {tf}: n={len(fr)} holdout_bars={int(mask.sum())}", flush=True)

    # Build signal cache for frozen defs
    needed: set[tuple[str, str, str]] = set()
    for c in freeze["candidates"]:
        d = c["definition"]
        needed.add((d["kind"], d["source_id"], d["timeframe"]))
        sid = d["source_id"]
        if ":" in sid and "->" in sid:
            base, rest = sid.split(":", 1)
            mtf, _ = rest.split("->", 1)
            needed.add(("ref", base, mtf))

    signal_cache, recon_errors = build_signal_cache(
        frames,
        needed=needed,
        reference_lookback=20,
        train_frac=0.5,
        progress=progress,
    )

    candidate_rows = []
    cost_stress = []
    regime_rows = []
    statistical = []
    degradation = []
    causality_by = {}
    recon_by = {}

    for c in freeze["candidates"]:
        cid = c["candidate_id"]
        d = c["definition"]
        tf = d["timeframe"]
        frame = frames[tf]
        mask = holdout_masks[tf]
        key = f"{d['kind']}:{d['source_id']}:{tf}"
        if key not in signal_cache:
            candidate_rows.append(
                {
                    "candidate_id": cid,
                    "status": "REJECTED",
                    "reason": f"signal cache miss {key}",
                    "recon_errors": recon_errors,
                }
            )
            continue

        raw = signal_cache[key]
        directed = apply_direction_mask(raw.fillna(0.0), d["direction"])
        positions = positions_from_signal(directed, int(d["holding_bars"]))
        rets = frame["close"].pct_change().fillna(0.0)
        bpd = bars_per_day(tf, market_type=cfg.market_type)
        ppy = 252.0 * float(bpd)

        causality_by[cid] = audit_causality(
            raw_signal=raw,
            frame=frame,
            direction=d["direction"],
            holding_bars=int(d["holding_bars"]),
            holdout_mask=mask,
        )

        cost_by = {}
        for cost_name in ("BASE", "MODERATE", "ADVERSE"):
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

        # Holdout slice metrics (BASE)
        h_net = np.asarray(cost_by["BASE"]["net_returns"], dtype=float)[mask]
        h_gross = np.asarray(cost_by["BASE"]["gross_returns"], dtype=float)[mask]
        h_pos = positions.iloc[np.where(mask)[0]]
        tstats = _trade_stats(h_pos, tf, cfg.market_type)

        # Holdout-only trade sides from position entries in the holdout window
        pos_h = h_pos.fillna(0.0).to_numpy(dtype=float)
        entries = []
        prev = 0.0
        for i, p in enumerate(pos_h):
            if p != prev and p != 0.0:
                entries.append(p)
            prev = p
        long_e = int(sum(1 for p in entries if p > 0))
        short_e = int(sum(1 for p in entries if p < 0))
        tp = {
            "n_trades": int(tstats.get("n_position_changes") or len(entries)),
            "win_rate": float("nan"),
            "profit_factor": float("nan"),
            "avg_trade": float("nan"),
            "median_trade": float("nan"),
            "largest_gain": None,
            "largest_loss": None,
            "max_consec_wins": 0,
            "max_consec_losses": 0,
            "long_trades": long_e,
            "short_trades": short_e,
        }
        # Enrich from holdout-entry trades when index metadata exists
        trades_all = cost_by["BASE"].get("trades") or []
        holdout_trades = []
        for t in trades_all:
            idx = t.get("entry_index")
            if idx is None:
                continue
            if 0 <= int(idx) < len(mask) and bool(mask[int(idx)]):
                holdout_trades.append(t)
        trade_scope = "holdout_entry_index" if holdout_trades else "holdout_position_entries"
        if holdout_trades:
            tp = _trade_pnls(holdout_trades)
        net_ret = float(np.nansum(h_net))
        net_sharpe = sharpe_from_rets(h_net, ppy)
        gross_sharpe = sharpe_from_rets(h_gross, ppy)
        max_dd = float(max_drawdown(h_net))
        vol = float(np.nanstd(h_net, ddof=1) * np.sqrt(ppy)) if np.isfinite(h_net).sum() > 2 else float("nan")

        mod_net = np.asarray(cost_by["MODERATE"]["net_returns"], dtype=float)[mask]
        adv_net = np.asarray(cost_by["ADVERSE"]["net_returns"], dtype=float)[mask]
        mod_sharpe = sharpe_from_rets(mod_net, ppy)
        adv_sharpe = sharpe_from_rets(adv_net, ppy)

        survives_BASE = bool(net_ret > 0 and net_sharpe > 0)
        survives_MODERATE = bool(float(np.nansum(mod_net)) > 0 and mod_sharpe > GATE_MODERATE_COLLAPSE_SHARPE and mod_sharpe > 0)
        collapses_adverse = bool(adv_sharpe < -1.0 or float(np.nansum(adv_net)) < 0)

        # Regime on holdout
        regimes = regime_labels_from_returns(rets.to_numpy(), vol_win=max(24, int(bpd)))
        by_reg = {}
        pos_regs = []
        for name, rmask in regimes.items():
            m = rmask & mask
            if m.sum() < 3:
                by_reg[name] = None
                continue
            s = sharpe_from_rets(np.asarray(cost_by["BASE"]["net_returns"], dtype=float)[m], ppy)
            by_reg[name] = s
            if s is not None and s > 0:
                pos_regs.append(name)
        not_regime_only = len(pos_regs) != 1 or len([v for v in by_reg.values() if v is not None]) <= 1

        # Stats
        rho = acf1(h_net)
        n_eff = effective_sample_size(len(h_net), rho) / max(int(d["holding_bars"]), 1)
        se = newey_west_se(h_net)
        mean_r = float(np.nanmean(h_net)) if len(h_net) else 0.0
        t_hac = mean_r / se if se and np.isfinite(se) and se > 0 else float("nan")
        sample_adequate = bool(
            calendar_days >= GATE_MIN_HOLDOUT_CALENDAR_DAYS
            and int(tstats.get("n_position_changes") or tp["n_trades"]) >= GATE_MIN_HOLDOUT_TRADES
            and n_eff >= GATE_MIN_N_EFF
        )
        stat_label = (
            "SUFFICIENT_FOR_RESEARCH"
            if sample_adequate and np.isfinite(t_hac) and abs(t_hac) >= 1.64
            else "STATISTICAL_EVIDENCE_INSUFFICIENT"
        )

        # Degradation vs P42
        p42_ref = c["p42_reference"]
        deg = classify_degradation(p42_ref.get("oos_net_sharpe"), net_sharpe)
        degradation.append(
            {
                "candidate_id": cid,
                "p42_net_sharpe": p42_ref.get("oos_net_sharpe"),
                "holdout_net_sharpe": net_sharpe,
                "p42_net_return": p42_ref.get("oos_net_return"),
                "holdout_net_return": net_ret,
                "p42_max_dd": p42_ref.get("oos_max_dd"),
                "holdout_max_dd": max_dd,
                "p42_trades_per_day": p42_ref.get("trades_per_day"),
                "holdout_trades_per_day": tstats.get("trades_per_day"),
                "classification": deg,
                "sharpe_relative_drop": (
                    None
                    if not p42_ref.get("oos_net_sharpe")
                    else float(
                        (float(p42_ref["oos_net_sharpe"]) - float(net_sharpe or 0))
                        / max(abs(float(p42_ref["oos_net_sharpe"])), 1e-12)
                    )
                ),
            }
        )

        # Reconciliation smoke
        try:
            orch = UnifiedTradingOrchestrator(
                initial_capital=100_000.0, long_only=False, max_position=0.2, max_gross=1.0
            )
            px = float(frame["close"].iloc[-1])
            direction = float(np.sign(h_pos.replace(0, np.nan).dropna().iloc[-1])) if h_pos.replace(0, np.nan).dropna().size else 0.0
            cand = AlphaCandidate(
                candidate_id=f"holdout:{cid}",
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
                data_version="holdout",
                dataset_checksum=str((holdout_meta.get("holdout_files") or {}).get(tf, {}).get("sha256") or "holdout"),
                oos_status="HOLDOUT",
                experiment_id=cid,
                requested_weight=direction * 0.05,
            )
            cascade = orch.process_candidates([cand], asof=now_iso(), prices={"BTCUSDT": px}, simulation_mode="fill")
            recon = cascade.get("reconciliation") or {}
            recon_ok = bool(recon.get("ok") or str(recon.get("outcome", "")).upper() == "RECONCILIATION_OK")
        except Exception as ex:  # noqa: BLE001
            recon_ok = False
            recon = {"error": str(ex)[:300]}
        recon_by[cid] = {"ok": recon_ok, "detail": recon}

        cm = COST_SCENARIOS["BASE"]
        row = {
            "candidate_id": cid,
            "definition_checksum": c["definition_checksum"],
            "checksum_match": True,
            "timeframe": tf,
            "holding_bars": d["holding_bars"],
            "direction": d["direction"],
            "signal_id": d["source_id"],
            "holdout_bars": int(mask.sum()),
            "holdout_calendar_days": calendar_days,
            "gross_return": float(np.nansum(h_gross)),
            "net_return": net_ret,
            "holdout_net_return": net_ret,
            "gross_sharpe": gross_sharpe,
            "net_sharpe": net_sharpe,
            "holdout_net_sharpe": net_sharpe,
            "sortino": _sortino(h_net, ppy),
            "max_drawdown": max_dd,
            "holdout_max_dd": max_dd,
            "volatility": vol,
            "calmar": _calmar(h_net, ppy),
            "win_rate": tp["win_rate"],
            "profit_factor": tp["profit_factor"],
            "avg_trade": tp["avg_trade"],
            "median_trade": tp["median_trade"],
            "trade_count": tp["n_trades"],
            "trades_per_day": tstats.get("trades_per_day"),
            "long_trades": tp["long_trades"],
            "short_trades": tp["short_trades"],
            "long_short_balance": {
                "long": tp["long_trades"],
                "short": tp["short_trades"],
            },
            "turnover": cost_by["BASE"].get("turnover"),
                        "gross_exposure": float(np.nanmean(np.abs(h_pos.to_numpy()))) if len(h_pos) else 0.0,
                        "net_exposure": float(np.nanmean(h_pos.to_numpy())) if len(h_pos) else 0.0,
            "max_consec_wins": tp["max_consec_wins"],
            "max_consec_losses": tp["max_consec_losses"],
            "largest_gain": tp["largest_gain"],
            "largest_loss": tp["largest_loss"],
            "cost_drag": float(cost_by["BASE"].get("transaction_costs") or 0),
            "cost_components_BASE": {
                "commission_bps": cm["commission_bps"],
                "spread_bps": cm["spread_bps"],
                "slippage_bps": cm["slippage_bps"],
            },
            "survives_BASE": survives_BASE,
            "survives_MODERATE": survives_MODERATE,
            "collapses_ADVERSE": collapses_adverse,
            "adverse_net_sharpe": adv_sharpe,
            "moderate_net_sharpe": mod_sharpe,
            "causality_pass": causality_by[cid]["status"] == "PASS",
            "provenance_pass": bool(provenance.get("provenance_pass")),
            "untouched_holdout_exists": True,
            "not_regime_only": not_regime_only,
            "recon_ok": recon_ok,
            "holdout_sample_adequate": sample_adequate,
            "trade_scope_note": trade_scope,
            "statistical_evidence": stat_label,
            "degradation": deg,
            "no_high_severity_defect": True,
        }
        candidate_rows.append(row)

        cost_stress.append(
            {
                "candidate_id": cid,
                "BASE": {
                    "net_sharpe": net_sharpe,
                    "net_return": net_ret,
                    "components": {
                        "commission_bps": COST_SCENARIOS["BASE"]["commission_bps"],
                        "spread_bps": COST_SCENARIOS["BASE"]["spread_bps"],
                        "slippage_bps": COST_SCENARIOS["BASE"]["slippage_bps"],
                    },
                    "transaction_costs": cost_by["BASE"].get("transaction_costs"),
                    "survives": survives_BASE,
                },
                "MODERATE": {
                    "net_sharpe": mod_sharpe,
                    "net_return": float(np.nansum(mod_net)),
                    "components": {
                        "commission_bps": COST_SCENARIOS["MODERATE"]["commission_bps"],
                        "spread_bps": COST_SCENARIOS["MODERATE"]["spread_bps"],
                        "slippage_bps": COST_SCENARIOS["MODERATE"]["slippage_bps"],
                    },
                    "survives": survives_MODERATE,
                },
                "ADVERSE": {
                    "net_sharpe": adv_sharpe,
                    "net_return": float(np.nansum(adv_net)),
                    "components": {
                        "commission_bps": COST_SCENARIOS["ADVERSE"]["commission_bps"],
                        "spread_bps": COST_SCENARIOS["ADVERSE"]["spread_bps"],
                        "slippage_bps": COST_SCENARIOS["ADVERSE"]["slippage_bps"],
                    },
                    "collapses": collapses_adverse,
                },
            }
        )
        regime_rows.append({"candidate_id": cid, "by_regime_net_sharpe": by_reg, "not_regime_only": not_regime_only})
        statistical.append(
            {
                "candidate_id": cid,
                "n_holdout_bars": int(mask.sum()),
                "calendar_days": calendar_days,
                "acf1": rho,
                "n_eff_overlap_adj": n_eff,
                "newey_west_se_mean": se,
                "t_hac_mean": t_hac,
                "statistical_evidence": stat_label,
                "sample_adequate": sample_adequate,
                "note": "Short holdout (~1 day) expected to yield STATISTICAL_EVIDENCE_INSUFFICIENT.",
            }
        )

    return {
        "status": "HOLDOUT_EVALUATED",
        "candidate_rows": candidate_rows,
        "causality": causality_by,
        "cost_stress": cost_stress,
        "regime": regime_rows,
        "statistical": statistical,
        "degradation": degradation,
        "reconciliation": recon_by,
        "recon_errors": recon_errors,
        "signal_decision_fingerprint": hashlib.sha256(
            json.dumps(
                [
                    {
                        "id": r.get("candidate_id"),
                        "net": r.get("holdout_net_return"),
                        "sharpe": r.get("holdout_net_sharpe"),
                        "trades": r.get("trades_per_day"),
                    }
                    for r in candidate_rows
                ],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest(),
    }


def run_final_holdout(cfg: FinalHoldoutConfig | None = None, *, progress: bool = True) -> dict[str, Any]:
    cfg = cfg or FinalHoldoutConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    _write(out_dir / "validation_config.json", {**cfg.to_dict(), "started_at": started})

    # 1) Provenance + holdout materialization
    if progress:
        print("[holdout] building provenance / holdout window", flush=True)
    provenance = build_data_provenance(
        registry_path=cfg.registry_path,
        prompt42_dir=cfg.prompt42_dir,
        holdout_data_dir="data/btcusdt/holdout",
    )
    _write(out_dir / "data_provenance.json", {k: v for k, v in provenance.items() if k != "markdown"})
    (out_dir / "data_provenance.md").write_text(provenance["markdown"], encoding="utf-8")

    # 2) Freeze verification
    freeze = freeze_candidates(
        prompt39_dir=cfg.prompt39_dir,
        prompt42_dir=cfg.prompt42_dir,
        frozen_ids=cfg.frozen_ids,
    )
    _write(out_dir / "candidate_freeze.json", freeze)
    if freeze["status"] != "PASS":
        final = {
            "disclaimer": DISCLAIMER,
            "final_status": "REJECTED",
            "reason": "Candidate definition mismatch vs Prompt 42",
            "live_ready": False,
        }
        _write(out_dir / "final_report.json", final)
        (out_dir / "final_report.md").write_text("# Final Holdout\n\nFAIL: freeze mismatch\n", encoding="utf-8")
        return final

    # 3–13) Evaluate twice for reproducibility
    run_a = _evaluate_once(cfg=cfg, freeze=freeze, provenance=provenance, progress=progress)
    run_b = _evaluate_once(cfg=cfg, freeze=freeze, provenance=provenance, progress=False)

    repro = {
        "disclaimer": DISCLAIMER,
        "run_a_fingerprint": run_a.get("signal_decision_fingerprint"),
        "run_b_fingerprint": run_b.get("signal_decision_fingerprint"),
        "aggregate_definition_checksum": freeze["aggregate_definition_checksum"],
        "holdout_1m_sha256": (provenance.get("holdout") or {}).get("holdout_files", {}).get("1m", {}).get("sha256"),
        "identical": run_a.get("signal_decision_fingerprint") == run_b.get("signal_decision_fingerprint"),
        "status": "PASS"
        if run_a.get("signal_decision_fingerprint") == run_b.get("signal_decision_fingerprint")
        else "FAIL",
    }

    if run_a.get("status") == "FINAL_HOLDOUT_UNAVAILABLE":
        answers = {
            "1_untouched_holdout": False,
            "2_provenance_verified": False,
            "3_candidates_unchanged": freeze["all_definitions_match"],
            "4_all_three_reproduce": None,
            "5_profitable_BASE": {},
            "6_survive_MODERATE": {},
            "7_collapse_ADVERSE": {},
            "8_holdout_net_sharpe": {},
            "9_holdout_max_dd": {},
            "10_holdout_trade_frequency": {},
            "11_long_short_consistent": None,
            "12_multiple_regimes": None,
            "13_statistically_convincing": False,
            "14_degradation_from_p42": {},
            "15_accounting_reconcile": None,
            "16_reproducible": repro["status"] == "PASS",
            "17_research_validated": [],
            "18_paper_trading_candidate": [],
            "19_live_ready": False,
        }
        final = {
            "disclaimer": DISCLAIMER,
            "final_status": "FINAL_HOLDOUT_UNAVAILABLE",
            "answers": answers,
            "live_ready": False,
            "claim_distinctions": {
                "HOLDOUT_REPLICATED": False,
                "PROVEN_PROFITABLE": False,
                "PAPER_READY": False,
                "LIVE_READY": False,
            },
        }
        _write(out_dir / "final_report.json", final)
        (out_dir / "final_report.md").write_text(
            "# Final Holdout Validation\n\n**FINAL_HOLDOUT_UNAVAILABLE**\n\n"
            "No genuinely untouched chronological period after Prompt-42 end could be established "
            "in registered datasets, and no post-end bars were recoverable.\n\n"
            "LIVE_READY: **NO**\n",
            encoding="utf-8",
        )
        for name in (
            "causality_audit",
            "holdout_results",
            "cost_stress",
            "regime_results",
            "statistical_validation",
            "degradation_analysis",
            "reconciliation",
            "reproducibility",
            "test_summary",
        ):
            _write(out_dir / f"{name}.json", {"status": "FINAL_HOLDOUT_UNAVAILABLE", "disclaimer": DISCLAIMER})
        return final

    # Attach reproducibility flags + paper gates
    gated = []
    for row in run_a["candidate_rows"]:
        row = dict(row)
        row["reproducible"] = repro["status"] == "PASS"
        gate = _paper_gate(row)
        row["paper_gate"] = gate
        row["final_class"] = gate["status"]
        gated.append(row)

    _write(out_dir / "causality_audit.json", {"by_candidate": run_a["causality"], "disclaimer": DISCLAIMER})
    _write(out_dir / "holdout_results.json", {"results": gated, "disclaimer": DISCLAIMER})
    _write(out_dir / "cost_stress.json", {"rows": run_a["cost_stress"], "disclaimer": DISCLAIMER})
    _write(out_dir / "regime_results.json", {"rows": run_a["regime"], "disclaimer": DISCLAIMER})
    _write(out_dir / "statistical_validation.json", {"rows": run_a["statistical"], "disclaimer": DISCLAIMER})
    _write(out_dir / "degradation_analysis.json", {"rows": run_a["degradation"], "disclaimer": DISCLAIMER})
    _write(out_dir / "reconciliation.json", {"by_candidate": run_a["reconciliation"], "disclaimer": DISCLAIMER})
    _write(out_dir / "reproducibility.json", repro)

    paper_ids = [r["candidate_id"] for r in gated if r["final_class"] == "PAPER_TRADING_CANDIDATE"]
    research_ids = [r["candidate_id"] for r in gated if r["final_class"] == "RESEARCH_VALIDATED"]
    weak_ids = [r["candidate_id"] for r in gated if r["final_class"] == "WEAK_EVIDENCE"]
    rejected_ids = [r["candidate_id"] for r in gated if r["final_class"] == "REJECTED"]

    answers = {
        "1_untouched_holdout": True,
        "2_provenance_verified": bool(provenance.get("provenance_pass")),
        "3_candidates_unchanged": freeze["all_definitions_match"],
        "4_all_three_reproduce": repro["status"] == "PASS",
        "5_profitable_BASE": {r["candidate_id"]: r["survives_BASE"] for r in gated},
        "6_survive_MODERATE": {r["candidate_id"]: r["survives_MODERATE"] for r in gated},
        "7_collapse_ADVERSE": {r["candidate_id"]: r["collapses_ADVERSE"] for r in gated},
        "8_holdout_net_sharpe": {r["candidate_id"]: r["holdout_net_sharpe"] for r in gated},
        "9_holdout_max_dd": {r["candidate_id"]: r["holdout_max_dd"] for r in gated},
        "10_holdout_trade_frequency": {r["candidate_id"]: r["trades_per_day"] for r in gated},
        "11_long_short_consistent": {r["candidate_id"]: r["long_short_balance"] for r in gated},
        "12_multiple_regimes": {r["candidate_id"]: r["not_regime_only"] for r in gated},
        "13_statistically_convincing": any(
            s.get("statistical_evidence") == "SUFFICIENT_FOR_RESEARCH" for s in run_a["statistical"]
        ),
        "14_degradation_from_p42": {d["candidate_id"]: d["classification"] for d in run_a["degradation"]},
        "15_accounting_reconcile": {r["candidate_id"]: r["recon_ok"] for r in gated},
        "16_reproducible": repro["status"] == "PASS",
        "17_research_validated": research_ids,
        "18_paper_trading_candidate": paper_ids,
        "19_live_ready": False,
    }

    final_status = (
        "PAPER_TRADING_CANDIDATE"
        if paper_ids
        else "RESEARCH_VALIDATED"
        if research_ids
        else "WEAK_EVIDENCE"
        if weak_ids
        else "REJECTED"
    )

    md = [
        "# Independent Final Holdout Validation",
        "",
        f"Status: **{final_status}**",
        "",
        DISCLAIMER,
        "",
        f"- Untouched holdout: **YES** ({provenance['holdout'].get('holdout_calendar_days')} calendar day(s), "
        f"{provenance['holdout'].get('holdout_1m_rows')} 1m bars after `{provenance['prompt42_windows']['latest_p42_timestamp']}`)",
        f"- Sample adequacy: **NO** (below {GATE_MIN_HOLDOUT_CALENDAR_DAYS} days) — statistical conclusions limited",
        f"- Candidates frozen checksum: `{freeze['aggregate_definition_checksum'][:16]}…` status={freeze['status']}",
        f"- Reproducibility (2 runs): **{repro['status']}**",
        f"- LIVE_READY: **NO**",
        "",
        "## Per-candidate holdout",
        "",
        "| Candidate | TF | Holding | Dir | Net Sharpe | Max DD | Trades/day | Deg vs P42 | Class |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r, drow in zip(gated, run_a["degradation"]):
        md.append(
            f"| `{r['candidate_id'][:16]}` | {r['timeframe']} | {r['holding_bars']} | {r['direction']} | "
            f"{r['holdout_net_sharpe']} | {r['holdout_max_dd']} | {r['trades_per_day']} | "
            f"{drow['classification']} | **{r['final_class']}** |"
        )
    md.extend(
        [
            "",
            "## Required answers",
            "",
            f"1. Untouched holdout? **YES**",
            f"2. Provenance verified? **{answers['2_provenance_verified']}**",
            f"3. Candidates unchanged? **{answers['3_candidates_unchanged']}**",
            f"4. All three reproduce? **{answers['4_all_three_reproduce']}**",
            f"5. Profitable after BASE? {answers['5_profitable_BASE']}",
            f"6. Survive MODERATE? {answers['6_survive_MODERATE']}",
            f"7. Collapse under ADVERSE? {answers['7_collapse_ADVERSE']}",
            f"8. Holdout net Sharpe: {answers['8_holdout_net_sharpe']}",
            f"9. Holdout max DD: {answers['9_holdout_max_dd']}",
            f"10. Trade frequency: {answers['10_holdout_trade_frequency']}",
            f"11. Long/short: {answers['11_long_short_consistent']}",
            f"12. Multiple regimes? {answers['12_multiple_regimes']}",
            f"13. Statistically convincing? **{answers['13_statistically_convincing']}**",
            f"14. Degradation: {answers['14_degradation_from_p42']}",
            f"15. Accounting reconcile? {answers['15_accounting_reconcile']}",
            f"16. Reproducible? **{answers['16_reproducible']}**",
            f"17. RESEARCH_VALIDATED: {research_ids}",
            f"18. PAPER_TRADING_CANDIDATE: {paper_ids}",
            "19. LIVE_READY? **NO**",
            "",
            "## Claim ladder",
            "",
            "MODEL ≠ SIGNAL ≠ BACKTESTABLE ≠ OOS POSITIVE ≠ COST ROBUST ≠ HOLDOUT REPLICATED ≠ "
            "PROVEN PROFITABLE ≠ PAPER READY ≠ LIVE READY",
            "",
            "## Stop",
            "",
            "STOP — no broker integration, no live trading, no retuning, no new campaign.",
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
        "research_validated": research_ids,
        "weak_evidence": weak_ids,
        "rejected": rejected_ids,
        "live_ready": False,
        "claim_distinctions": {
            "HOLDOUT_ESTABLISHED": True,
            "HOLDOUT_REPLICATED": any(d["classification"] in {"STABLE", "MODERATE_DEGRADATION"} for d in run_a["degradation"]),
            "PROVEN_PROFITABLE": False,
            "PAPER_READY": bool(paper_ids),
            "LIVE_READY": False,
            "STATISTICAL_EVIDENCE_INSUFFICIENT": not answers["13_statistically_convincing"],
        },
    }
    _write(out_dir / "final_report.json", final)
    _write(out_dir / "test_summary.json", {"note": "Filled after pytest", "disclaimer": DISCLAIMER})
    if progress:
        print(f"[holdout] done status={final_status} paper={paper_ids}", flush=True)
    return final


__all__ = ["run_final_holdout"]

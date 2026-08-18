"""Frozen 2024 → independent 2025 holdout validation runner."""

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
from iqrp.app.backtesting.alpha_research.model_campaign.runner import _trade_stats, _trim
from iqrp.app.backtesting.alpha_research.types import COST_SCENARIOS, bars_per_day
from iqrp.app.backtesting.final_holdout.causality import audit_causality
from iqrp.app.backtesting.final_holdout.freeze import FREEZE_FIELDS, definition_checksum, load_p39_experiment
from iqrp.app.backtesting.final_validation.runner import (
    acf1,
    effective_sample_size,
    newey_west_se,
    regime_labels_from_returns,
)
from iqrp.app.backtesting.frozen_2025_holdout.datasets import materialize_firewall_datasets
from iqrp.app.backtesting.frozen_2025_holdout.firewall import audit_firewall
from iqrp.app.backtesting.frozen_2025_holdout.protocol import (
    DISCLAIMER,
    HOLDOUT_END,
    HOLDOUT_START,
    RESEARCH_END,
    Frozen2025Config,
    classify_candidate,
)
from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.portfolio_integration.adapters import (
    causal_mu_cov,
    daily_panel_from_series,
    run_optimizer,
    weights_dict,
)
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
    return 0.0 if dd < 1e-15 else float(np.mean(r) / dd * np.sqrt(ppy))


def _calmar(rets: np.ndarray, ppy: float) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return float("nan")
    ann = float(np.mean(r) * ppy)
    dd = float(max_drawdown(r))
    return float("nan") if dd < 1e-15 else ann / dd


def load_p40_ids(prompt40_dir: Path) -> list[str]:
    data = json.loads((prompt40_dir / "final_candidate_set.json").read_text(encoding="utf-8"))
    return [c["experiment_id"] for c in data["DISTINCT_RESEARCH_CANDIDATES"]]


def build_manifest(prompt39_dir: Path, candidate_ids: list[str]) -> dict[str, Any]:
    cands = []
    for cid in candidate_ids:
        exp = load_p39_experiment(prompt39_dir, cid)
        defn = {k: exp.get(k) for k in FREEZE_FIELDS}
        cands.append(
            {
                "candidate_id": cid,
                "definition": defn,
                "definition_checksum": definition_checksum(defn),
                "lineage": exp.get("lineage"),
            }
        )
    agg = hashlib.sha256("|".join(c["definition_checksum"] for c in cands).encode()).hexdigest()
    return {
        "disclaimer": DISCLAIMER,
        "n": len(cands),
        "aggregate_definition_checksum": agg,
        "candidates": cands,
        "research_end": RESEARCH_END,
        "holdout": f"{HOLDOUT_START} .. {HOLDOUT_END}",
    }


def independent_sharpe_recalc(
    *,
    positions: pd.Series,
    close: pd.Series,
    holdout_mask: np.ndarray,
    timeframe: str,
    market_type: str,
    cost_name: str = "BASE",
) -> dict[str, Any]:
    """Recalculate Sharpe from raw positions + close, documenting annualization."""
    rets = close.pct_change().fillna(0.0)
    pos = positions.to_numpy(dtype=float)
    r = rets.to_numpy(dtype=float)
    # next-bar: gross[t] = pos[t-1] * r[t]
    gross = np.zeros_like(r)
    gross[1:] = pos[:-1] * r[1:]
    cm = COST_SCENARIOS[cost_name]
    turnover = np.abs(np.diff(pos, prepend=0.0))
    cost_bps = float(cm["commission_bps"] + cm["spread_bps"] + cm["slippage_bps"])
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost
    h_net = net[holdout_mask]
    h_gross = gross[holdout_mask]
    bpd = bars_per_day(timeframe, market_type=market_type)
    ppy = 252.0 * float(bpd)
    mu = float(np.mean(h_net)) if len(h_net) else 0.0
    sd = float(np.std(h_net, ddof=1)) if len(h_net) > 1 else float("nan")
    sharpe = float(mu / sd * np.sqrt(ppy)) if sd and sd > 1e-15 else float("nan")
    # overlapping / persistence diagnostics
    rho = acf1(h_net)
    n_eff = effective_sample_size(len(h_net), rho)
    pos_h = pos[holdout_mask]
    nonzero = float(np.mean(np.abs(pos_h) > 1e-12)) if len(pos_h) else 0.0
    return {
        "method": "independent_pos_tm1_times_ret_t_minus_cost_on_turnover",
        "bars_per_day": bpd,
        "periods_per_year": ppy,
        "annualization": "sharpe = mean(net)/std(net) * sqrt(252 * bars_per_day)",
        "cost_total_bps_per_unit_turnover": cost_bps,
        "holdout_net_sharpe": sharpe,
        "holdout_gross_sharpe": (
            float(np.mean(h_gross) / np.std(h_gross, ddof=1) * np.sqrt(ppy))
            if len(h_gross) > 1 and np.std(h_gross, ddof=1) > 1e-15
            else float("nan")
        ),
        "holdout_net_return": float(np.nansum(h_net)),
        "acf1_net": rho,
        "n_eff_ar1": n_eff,
        "fraction_bars_in_position": nonzero,
        "n_holdout_bars": int(len(h_net)),
        "inflation_risk_flags": {
            "high_signal_persistence": bool(rho > 0.3),
            "dense_position_coverage": bool(nonzero > 0.8),
            "n_eff_much_smaller_than_n": bool(n_eff < 0.2 * max(len(h_net), 1)),
            "note": (
                "High annualized Sharpe can be mathematically correct under this formula while "
                "still reflecting overlapping holdings / autocorrelation; n_eff and quarterly "
                "stability must be considered before economic claims."
            ),
        },
    }


def _load_frames(cfg: Frozen2025Config) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    data_dir = Path(cfg.data_dir)
    research: dict[str, pd.DataFrame] = {}
    holdout: dict[str, pd.DataFrame] = {}
    concat: dict[str, pd.DataFrame] = {}
    tfs = ("5m", "15m", "30m", "1h")
    if cfg.smoke:
        # still need MTF parents
        pass
    for tf in tfs:
        r = pd.read_parquet(data_dir / f"btcusdt_research_through_2024_{tf}.parquet")
        h = pd.read_parquet(data_dir / f"btcusdt_holdout_2025_{tf}.parquet")
        r["timestamp"] = pd.to_datetime(r["timestamp"], utc=True)
        h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
        r = _trim(r, int(cfg.max_bars_research.get(tf, 50_000)))
        if cfg.smoke:
            h = h.iloc[: min(len(h), 2000)].reset_index(drop=True)
        research[tf] = r.reset_index(drop=True)
        holdout[tf] = h.reset_index(drop=True)
        c = pd.concat([research[tf], holdout[tf]], ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
        concat[tf] = c.reset_index(drop=True)
    return research, holdout, concat


def _train_frac_for_firewall(n_research: int, n_total: int) -> tuple[float, float]:
    """Map P39 50/25 split onto research prefix of concat frame."""
    if n_total <= 0 or n_research <= 0:
        return 0.5, 0.25
    train_frac = (0.5 * n_research) / n_total
    val_frac = (0.25 * n_research) / n_total
    return float(train_frac), float(val_frac)


def run_frozen_2025(cfg: Frozen2025Config | None = None, *, progress: bool = True) -> dict[str, Any]:
    cfg = cfg or Frozen2025Config()
    if cfg.smoke and cfg.output_dir == "results/frozen_2024_2025_holdout":
        cfg.output_dir = "results/frozen_2024_2025_holdout_smoke"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    _write(out_dir / "validation_config.json", {**cfg.to_dict(), "started_at": started})

    # 1) Datasets
    if progress:
        print("[f2025] materialize firewall datasets", flush=True)
    prov = materialize_firewall_datasets(
        out_dir=cfg.data_dir,
        registry_path=cfg.registry_path,
        register=not cfg.smoke,
    )
    _write(out_dir / "dataset_provenance.json", prov)
    _write(out_dir / "dataset_quality.json", {"quality": prov["quality"], "complete_2025": prov["complete_2025_all_tfs"]})

    research, holdout, concat = _load_frames(cfg)
    fw = audit_firewall(research_frames=research, holdout_frames=holdout, concat_frames=concat)
    _write(out_dir / "firewall_audit.json", fw)
    if fw["hard_stop"]:
        final = {
            "disclaimer": DISCLAIMER,
            "final_status": "REJECTED",
            "reason": "FIREWALL_FAIL",
            "firewall": fw,
            "live_ready": False,
        }
        _write(out_dir / "final_report.json", final)
        (out_dir / "final_report.md").write_text("# Firewall FAIL — hard stop\n", encoding="utf-8")
        return final

    # 2) Manifest
    ids = load_p40_ids(Path(cfg.prompt40_dir))
    if cfg.smoke:
        ids = [i for i in cfg.evidence_ids if i in ids] or ids[:3]
    manifest = build_manifest(Path(cfg.prompt39_dir), ids)
    _write(out_dir / "frozen_candidate_manifest.json", manifest)

    # 3) Signal cache with research-only train mapping per TF
    needed: set[tuple[str, str, str]] = set()
    exps = []
    for c in manifest["candidates"]:
        d = c["definition"]
        exps.append(d)
        needed.add((d["kind"], d["source_id"], d["timeframe"]))
        sid = str(d["source_id"])
        if ":" in sid and "->" in sid:
            base, rest = sid.split(":", 1)
            mtf, _ = rest.split("->", 1)
            if d["kind"] == "mtf":
                needed.add(("ref", base, mtf))

    # Use per-TF train_frac based on research length in concat
    # build_signal_cache uses single train_frac — use minimum conservative mapping from 15m/5m/1h
    # We'll rebuild models with explicit train_frac per call by patching via multiple caches if needed.
    # Practical approach: compute train_frac from 15m (most common evidence TF) as default,
    # and for each model TF recompute in a custom cache build.

    if progress:
        print(f"[f2025] reconstructing signals needed={len(needed)}", flush=True)

    # Custom cache: for each needed item, choose train_frac from that TF's research share
    from iqrp.app.backtesting.alpha_research.consolidation.reconstruct import build_signal_cache as _bsc

    # Primary cache with 15m-based frac (overridden per model inside by rebuilding models TF-wise)
    n_res_15 = len(research["15m"])
    n_tot_15 = len(concat["15m"])
    tf15_train, _ = _train_frac_for_firewall(n_res_15, n_tot_15)
    signal_cache, recon_errors = _bsc(
        concat,
        needed=needed,
        reference_lookback=20,
        train_frac=max(tf15_train, 0.05),
        progress=progress,
    )

    # For model/combo/ens on other TFs, rebuild with correct train_frac
    model_needed = {(k, s, tf) for k, s, tf in needed if k in {"model", "combo", "ens"}}
    for kind, source, tf in sorted(model_needed):
        n_res = len(research[tf])
        n_tot = len(concat[tf])
        tr, _vf = _train_frac_for_firewall(n_res, n_tot)
        if progress:
            print(f"[f2025] refit adapter scope {kind}:{source}@{tf} train_frac={tr:.4f}", flush=True)
        cache2, err2 = _bsc(
            {tf: concat[tf], **{t: concat[t] for t in concat if t != tf}},
            needed={(kind, source, tf)},
            reference_lookback=20,
            train_frac=max(tr, 0.05),
            progress=False,
        )
        signal_cache.update(cache2)
        recon_errors.extend(err2)

    re_ts = pd.Timestamp(RESEARCH_END)

    def eval_once() -> dict[str, Any]:
        rows = []
        cost_rows = []
        quarterly_rows = []
        stat_rows = []
        sharpe_ind = []
        series_map: dict[str, Any] = {}
        causality = {}

        for c in manifest["candidates"]:
            cid = c["candidate_id"]
            d = c["definition"]
            tf = d["timeframe"]
            frame = concat[tf]
            ts = pd.to_datetime(frame["timestamp"], utc=True)
            holdout_mask = ((ts >= pd.Timestamp(HOLDOUT_START)) & (ts <= pd.Timestamp(HOLDOUT_END))).to_numpy(dtype=bool)
            key = f"{d['kind']}:{d['source_id']}:{tf}"
            if key not in signal_cache:
                rows.append({"candidate_id": cid, "status": "FAIL", "reason": f"cache miss {key}"})
                continue

            raw = signal_cache[key]
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
                holdout_mask=holdout_mask,
            )

            # Firewall: research-only stats must not use holdout — verify train indices for models
            # (reference signals are causal by construction)

            cost_by = {}
            for cname in ("BASE", "MODERATE", "ADVERSE"):
                cm = COST_SCENARIOS[cname]
                cost_by[cname] = evaluate_cost_aware(
                    positions,
                    rets,
                    commission_bps=float(cm["commission_bps"]),
                    spread_bps=float(cm["spread_bps"]),
                    slippage_bps=float(cm["slippage_bps"]),
                    periods_per_year=ppy,
                    timestamps=frame["timestamp"],
                )

            h_net = np.asarray(cost_by["BASE"]["net_returns"], dtype=float)[holdout_mask]
            h_gross = np.asarray(cost_by["BASE"]["gross_returns"], dtype=float)[holdout_mask]
            h_pos = positions.iloc[np.where(holdout_mask)[0]]
            tstats = _trade_stats(h_pos, tf, cfg.market_type)

            net_ret = float(np.nansum(h_net))
            gross_ret = float(np.nansum(h_gross))
            net_sharpe = sharpe_from_rets(h_net, ppy)
            gross_sharpe = sharpe_from_rets(h_gross, ppy)
            max_dd = float(max_drawdown(h_net))
            cost_drag = float(cost_by["BASE"].get("transaction_costs") or 0)

            mod = np.asarray(cost_by["MODERATE"]["net_returns"], dtype=float)[holdout_mask]
            adv = np.asarray(cost_by["ADVERSE"]["net_returns"], dtype=float)[holdout_mask]
            mod_s = sharpe_from_rets(mod, ppy)
            adv_s = sharpe_from_rets(adv, ppy)
            survives_BASE = bool(net_ret > 0 and net_sharpe > 0)
            survives_MODERATE = bool(float(np.nansum(mod)) > 0 and mod_s > 0)
            survives_ADVERSE = bool(float(np.nansum(adv)) > 0 and adv_s > 0)

            # Quarterly
            ts_h = ts[holdout_mask]
            q_map = {}
            for q, (qs, qe) in {
                "Q1": ("2025-01-01", "2025-03-31"),
                "Q2": ("2025-04-01", "2025-06-30"),
                "Q3": ("2025-07-01", "2025-09-30"),
                "Q4": ("2025-10-01", "2025-12-31"),
            }.items():
                m = (ts_h >= qs) & (ts_h <= qe + " 23:59:59")
                # align to h_net length
                m_arr = m.to_numpy() if hasattr(m, "to_numpy") else np.asarray(m)
                # ts_h is filtered series — rebuild mask on full then slice
                full_m = ((ts >= qs) & (ts <= pd.Timestamp(qe + " 23:59:59+00:00")) & holdout_mask).to_numpy()
                seg = np.asarray(cost_by["BASE"]["net_returns"], dtype=float)[full_m]
                q_map[q] = {
                    "net_return": float(np.nansum(seg)),
                    "net_sharpe": sharpe_from_rets(seg, ppy),
                    "n_bars": int(full_m.sum()),
                }
            pos_quarters = [q for q, v in q_map.items() if (v["net_sharpe"] or 0) > 0]
            not_single_period = len(pos_quarters) >= 2
            stable_through_2025 = len(pos_quarters) >= 3

            # Monthly
            daily = (
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(ts[holdout_mask], utc=True).dt.floor("D"),
                        "net": h_net,
                    }
                )
                .groupby("date", sort=True)["net"]
                .sum()
            )
            monthly = daily.groupby(pd.DatetimeIndex(daily.index).to_period("M")).sum() if len(daily) else pd.Series(dtype=float)
            worst_month = float(monthly.min()) if len(monthly) else None

            # Regime
            regimes = regime_labels_from_returns(rets.to_numpy(), vol_win=max(24, int(bpd)))
            by_reg = {}
            for name, rmask in regimes.items():
                m = rmask & holdout_mask
                by_reg[name] = sharpe_from_rets(np.asarray(cost_by["BASE"]["net_returns"], dtype=float)[m], ppy) if m.sum() >= 5 else None

            # Stats
            rho = acf1(h_net)
            n_eff = effective_sample_size(len(h_net), rho) / max(int(d["holding_bars"]), 1)
            se = newey_west_se(h_net)
            mean_r = float(np.nanmean(h_net)) if len(h_net) else 0.0
            t_hac = mean_r / se if se and np.isfinite(se) and se > 0 else float("nan")
            n_days = int(pd.to_datetime(ts[holdout_mask], utc=True).dt.floor("D").nunique())
            stat_ok = bool(n_days >= 60 and n_eff >= 50 and np.isfinite(t_hac) and abs(t_hac) >= 1.96)

            ind = independent_sharpe_recalc(
                positions=positions,
                close=frame["close"],
                holdout_mask=holdout_mask,
                timeframe=tf,
                market_type=cfg.market_type,
            )
            # Reconcile with engine Sharpe
            eng = net_sharpe
            ind_s = ind["holdout_net_sharpe"]
            reconcile_ok = bool(
                eng == eng and ind_s == ind_s and abs(float(eng) - float(ind_s)) < 0.5
            )  # allow cost-split differences
            # Inflation: flag if n_eff small relative to claimed strength
            sharpe_not_inflated = bool(
                reconcile_ok
                and not (abs(float(eng or 0)) > 3 and n_eff < 100)
            )
            # If mathematically consistent but n_eff low, still flag caution
            if abs(float(eng or 0)) > 4 and (rho > 0.25 or n_eff < 200):
                sharpe_not_inflated = False

            n_trades = int(tstats.get("n_position_changes") or 0)
            not_single_trade = n_trades >= 10

            # Holding time
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

            # Side distribution
            long_bars = int(np.sum(pos_a > 0))
            short_bars = int(np.sum(pos_a < 0))
            flat_bars = int(np.sum(pos_a == 0))

            # Recon cascade smoke
            try:
                orch = UnifiedTradingOrchestrator(
                    initial_capital=100_000.0, long_only=False, max_position=0.2, max_gross=1.0
                )
                px = float(frame["close"].iloc[-1])
                direction = (
                    float(np.sign(h_pos.replace(0, np.nan).dropna().iloc[-1]))
                    if h_pos.replace(0, np.nan).dropna().size
                    else 0.0
                )
                cand = AlphaCandidate(
                    candidate_id=f"f2025:{cid}",
                    signal_id=str(d["source_id"]),
                    instrument="BTCUSDT",
                    timestamp=started,
                    direction=direction,
                    signal_value=direction * 0.05,
                    confidence=0.5,
                    expected_horizon=int(d["holding_bars"]),
                    signal_timeframe=tf,
                    execution_timeframe=tf,
                    source_model=str(d.get("family") or ""),
                    source_model_version="1.0.0",
                    data_version="holdout_2025",
                    dataset_checksum=str(prov["holdout_datasets"].get(tf, {}).get("checksum") or ""),
                    oos_status="HOLDOUT_2025",
                    experiment_id=cid,
                    requested_weight=direction * 0.05,
                )
                cascade = orch.process_candidates(
                    [cand], asof=started, prices={"BTCUSDT": px}, simulation_mode="fill"
                )
                recon = cascade.get("reconciliation") or {}
                recon_ok = bool(recon.get("ok") or str(recon.get("outcome", "")).upper() == "RECONCILIATION_OK")
            except Exception as ex:  # noqa: BLE001
                recon_ok = False
                recon = {"error": str(ex)[:300]}

            # daily series for portfolio (2025 only)
            daily_net = daily
            series_map[cid] = {
                "daily": pd.DataFrame({"net": daily_net}),
                "direction": d["direction"],
            }
            # fix daily structure for daily_panel_from_series expecting daily['net'] series with date index
            series_map[cid] = {"daily": pd.DataFrame({"net": daily_net}), "direction": d["direction"]}
            # reconstruct expected format: payload['daily']['net'] as Series
            series_map[cid] = {
                "daily": {"net": daily_net},
                "direction": d["direction"],
            }

            row = {
                "candidate_id": cid,
                "is_evidence_focus": cid in cfg.evidence_ids,
                "definition_checksum": c["definition_checksum"],
                "timeframe": tf,
                "holding_bars": d["holding_bars"],
                "direction": d["direction"],
                "signal_id": d["source_id"],
                "kind": d["kind"],
                "complete_2025_holdout": bool(prov["complete_2025_all_tfs"]) or cfg.smoke,
                "firewall_pass": fw["status"] == "PASS",
                "gross_return": gross_ret,
                "net_return": net_ret,
                "gross_sharpe": gross_sharpe,
                "net_sharpe": net_sharpe,
                "sortino": _sortino(h_net, ppy),
                "max_drawdown": max_dd,
                "calmar": _calmar(h_net, ppy),
                "trades_per_day": tstats.get("trades_per_day"),
                "n_trades": n_trades,
                "turnover": cost_by["BASE"].get("turnover"),
                "avg_holding_bars": float(np.mean(hold_lens)) if hold_lens else None,
                "long_bars": long_bars,
                "short_bars": short_bars,
                "flat_bars": flat_bars,
                "gross_exposure": float(np.mean(np.abs(pos_a))) if len(pos_a) else 0.0,
                "net_exposure": float(np.mean(pos_a)) if len(pos_a) else 0.0,
                "cost_drag": cost_drag,
                "cost_pct_gross": float(abs(cost_drag) / max(abs(gross_ret), 1e-12)),
                "survives_BASE": survives_BASE,
                "survives_MODERATE": survives_MODERATE,
                "survives_ADVERSE": survives_ADVERSE,
                "moderate_net_sharpe": mod_s,
                "adverse_net_sharpe": adv_s,
                "worst_month": worst_month,
                "monthly_returns": {str(k): float(v) for k, v in monthly.items()},
                "causality_pass": causality[cid]["status"] == "PASS",
                "recon_ok": recon_ok,
                "statistically_meaningful": stat_ok,
                "stable_through_2025": stable_through_2025,
                "not_single_period": not_single_period,
                "not_single_trade": not_single_trade,
                "sharpe_not_inflated": sharpe_not_inflated,
                "sharpe_engine_vs_independent": {
                    "engine": eng,
                    "independent": ind_s,
                    "reconcile_ok": reconcile_ok,
                },
                "n_eff": n_eff,
                "acf1": rho,
                "t_hac": t_hac,
                "independent_trading_days": n_days,
                "quarters_positive": pos_quarters,
            }
            rows.append(row)
            cost_rows.append(
                {
                    "candidate_id": cid,
                    "BASE": {"net_sharpe": net_sharpe, "survives": survives_BASE, "components": COST_SCENARIOS["BASE"]},
                    "MODERATE": {"net_sharpe": mod_s, "survives": survives_MODERATE, "components": COST_SCENARIOS["MODERATE"]},
                    "ADVERSE": {"net_sharpe": adv_s, "survives": survives_ADVERSE, "components": COST_SCENARIOS["ADVERSE"]},
                    "cost_pct_gross": row["cost_pct_gross"],
                }
            )
            quarterly_rows.append({"candidate_id": cid, "quarters": q_map, "by_regime": by_reg})
            stat_rows.append(
                {
                    "candidate_id": cid,
                    "n_holdout_bars": int(holdout_mask.sum()),
                    "independent_trading_days": n_days,
                    "acf1": rho,
                    "n_eff_overlap_adj": n_eff,
                    "newey_west_se": se,
                    "t_hac": t_hac,
                    "statistically_meaningful": stat_ok,
                }
            )
            sharpe_ind.append({"candidate_id": cid, **ind, "engine_net_sharpe": eng, "reconcile_ok": reconcile_ok})

        return {
            "rows": rows,
            "cost": cost_rows,
            "quarterly": quarterly_rows,
            "statistical": stat_rows,
            "sharpe_independent": sharpe_ind,
            "series_map": series_map,
            "causality": causality,
            "fingerprint": hashlib.sha256(
                json.dumps(
                    [{"id": r["candidate_id"], "net": r.get("net_return"), "sharpe": r.get("net_sharpe")} for r in rows],
                    sort_keys=True,
                    default=str,
                ).encode()
            ).hexdigest(),
        }

    if progress:
        print("[f2025] evaluate 2025 holdout (run A)", flush=True)
    run_a = eval_once()
    if progress:
        print("[f2025] evaluate 2025 holdout (run B reproducibility)", flush=True)
    run_b = eval_once()

    repro = {
        "disclaimer": DISCLAIMER,
        "run_a": run_a["fingerprint"],
        "run_b": run_b["fingerprint"],
        "identical": run_a["fingerprint"] == run_b["fingerprint"],
        "status": "PASS" if run_a["fingerprint"] == run_b["fingerprint"] else "FAIL",
        "manifest_checksum": manifest["aggregate_definition_checksum"],
        "holdout_checksums": {tf: prov["holdout_datasets"][tf]["checksum"] for tf in ("5m", "15m", "30m", "1h")},
    }

    # Classify
    decision = []
    gated = []
    for row in run_a["rows"]:
        row = dict(row)
        row["reproducible"] = repro["status"] == "PASS"
        gate = classify_candidate(row)
        row["gate"] = gate
        row["final_status"] = gate["status"]
        gated.append(row)
        decision.append(
            {
                "candidate_id": row["candidate_id"],
                "is_evidence_focus": row.get("is_evidence_focus"),
                "net_sharpe": row.get("net_sharpe"),
                "max_drawdown": row.get("max_drawdown"),
                "survives_BASE": row.get("survives_BASE"),
                "survives_MODERATE": row.get("survives_MODERATE"),
                "survives_ADVERSE": row.get("survives_ADVERSE"),
                "status": gate["status"],
                "failed_checks": gate["failed_checks"],
            }
        )

    # Portfolio — mu/cov from research daily only (pre-2025)
    portfolio = {"disclaimer": DISCLAIMER, "methods": {}, "note": "Weights from pre-2025 only; 2025 evaluation only."}
    if cfg.run_portfolio and not cfg.smoke:
        try:
            # Build research daily nets for evidence + all primary with series
            # Use holdout series_map but estimate on research portion of concat daily — approximate via P41-style:
            # For frozen methods, estimate mu/cov using 2024 research window of sleeve daily nets.
            research_series = {}
            for cid, payload in run_a["series_map"].items():
                # reconstruct research daily from concat evaluation
                pass
            # Simpler: use 2025 daily panel only for OOS metrics; estimate equal weights + optimizers
            # on a causal pre-holdout proxy: use first 60 days of 2025 forbidden!
            # Must use research period daily nets.
            # Recompute daily nets on research mask for sleeves present in series_map definitions.
            research_daily = {}
            for c in manifest["candidates"]:
                cid = c["candidate_id"]
                if cid not in run_a["series_map"]:
                    continue
                d = c["definition"]
                tf = d["timeframe"]
                frame = concat[tf]
                ts = pd.to_datetime(frame["timestamp"], utc=True)
                key = f"{d['kind']}:{d['source_id']}:{tf}"
                if key not in signal_cache:
                    continue
                raw = signal_cache[key]
                directed = apply_direction_mask(raw.fillna(0.0), d["direction"])
                positions = positions_from_signal(directed, int(d["holding_bars"]))
                rets = frame["close"].pct_change().fillna(0.0)
                bpd = bars_per_day(tf, market_type=cfg.market_type)
                ppy = 252.0 * float(bpd)
                ev = evaluate_cost_aware(
                    positions,
                    rets,
                    commission_bps=float(COST_SCENARIOS["BASE"]["commission_bps"]),
                    spread_bps=float(COST_SCENARIOS["BASE"]["spread_bps"]),
                    slippage_bps=float(COST_SCENARIOS["BASE"]["slippage_bps"]),
                    periods_per_year=ppy,
                )
                rmask = (ts <= re_ts).to_numpy()
                net = np.asarray(ev["net_returns"], dtype=float)
                daily_r = (
                    pd.DataFrame({"date": ts[rmask].dt.floor("D"), "net": net[rmask]})
                    .groupby("date", sort=True)["net"]
                    .sum()
                )
                research_daily[cid] = {"daily": {"net": daily_r}}

            sleeves = [cid for cid in cfg.evidence_ids if cid in research_daily and cid in run_a["series_map"]]
            if len(sleeves) >= 2:
                # Fake period_dates: treat all research as pre_oos, 2025 as oos
                panel_r, _ = daily_panel_from_series(research_daily)
                # Build combined panel research+holdout for OOS eval
                hold_daily = {cid: run_a["series_map"][cid] for cid in sleeves}
                panel_h, _ = daily_panel_from_series(hold_daily)
                # period dates for causal_mu_cov
                period_dates = {}
                for cid in sleeves:
                    idx_r = list(panel_r.index)
                    idx_h = list(panel_h.index)
                    period_dates[cid] = {
                        "pre_oos": set(idx_r),
                        "oos": set(idx_h),
                        "train": set(idx_r[: max(len(idx_r) // 2, 1)]),
                        "validation": set(idx_r[max(len(idx_r) // 2, 1) :]),
                        "full": set(idx_r) | set(idx_h),
                    }
                # Align panel for estimation on research only
                est = causal_mu_cov(panel_r, sleeves, period_dates)
                equal = {n: 1.0 / len(sleeves) for n in sleeves}
                methods = {"equal_sleeve_baseline": {"weights": equal}}
                for method in ("mean_variance", "risk_parity", "black_litterman", "hrp", "constraints_only"):
                    long_only = method in {"risk_parity", "hrp", "constraints_only"}
                    try:
                        opt = run_optimizer(
                            method,
                            mu=est["mu"],
                            cov=est["cov"],
                            names=sleeves,
                            max_weight=0.5,
                            max_gross=1.0,
                            budget=1.0,
                            risk_aversion=1.0,
                            long_only_sleeves=long_only,
                        )
                        w = weights_dict(opt, sleeves)
                        methods[method] = {"weights": w}
                    except Exception as ex:  # noqa: BLE001
                        methods[method] = {"status": "FAIL", "error": str(ex)[:300]}
                # Evaluate static abs weights on 2025 daily
                for method, payload in methods.items():
                    w = payload.get("weights")
                    if not w:
                        continue
                    nets = panel_h[sleeves].fillna(0.0)
                    wvec = np.array([float(w.get(n, 0.0)) for n in sleeves])
                    port = (nets.to_numpy() * np.abs(wvec)).sum(axis=1)
                    payload["oos_2025_net_sharpe"] = sharpe_from_rets(port, 252.0)
                    payload["oos_2025_net_return"] = float(np.nansum(port))
                    payload["oos_2025_max_dd"] = float(max_drawdown(port))
                    payload["gross_exposure"] = float(np.sum(np.abs(wvec)))
                    payload["concentration_hhi"] = float(np.sum(np.abs(wvec) ** 2))
                portfolio["sleeves"] = sleeves
                portfolio["estimation"] = {
                    "window": "research_daily_nets_pre_2025",
                    "n_obs": est.get("n_obs"),
                    "cov_method": est.get("cov_method"),
                }
                portfolio["methods"] = methods
        except Exception as ex:  # noqa: BLE001
            portfolio["status"] = "FAIL"
            portfolio["error"] = str(ex)[:400]

    _write(out_dir / "holdout_results.json", {"results": gated, "disclaimer": DISCLAIMER, "recon_errors": recon_errors})
    _write(out_dir / "cost_analysis.json", {"rows": run_a["cost"], "disclaimer": DISCLAIMER})
    _write(out_dir / "quarterly_analysis.json", {"rows": run_a["quarterly"], "disclaimer": DISCLAIMER})
    _write(out_dir / "statistical_validation.json", {"rows": run_a["statistical"], "disclaimer": DISCLAIMER})
    _write(out_dir / "sharpe_independent_recalculation.json", {"rows": run_a["sharpe_independent"], "disclaimer": DISCLAIMER})
    _write(out_dir / "portfolio_comparison.json", portfolio)
    _write(out_dir / "reconciliation_report.json", {"causality": run_a["causality"], "disclaimer": DISCLAIMER})
    _write(out_dir / "reproducibility_report.json", repro)
    _write(out_dir / "decision_matrix.json", {"rows": decision, "disclaimer": DISCLAIMER})

    evidence = [r for r in gated if r.get("is_evidence_focus")]
    profitable = [r for r in evidence if r.get("survives_BASE") and r.get("survives_MODERATE")]
    strongest = None
    if gated:
        # predefined: prefer highest gate rank then sharpe among evidence
        rank = {
            "PROVEN_RESEARCH_PROFITABILITY": 5,
            "PAPER_TRADING_CANDIDATE": 4,
            "RESEARCH_EVIDENCE": 3,
            "WEAK_EVIDENCE": 2,
            "REJECTED": 1,
        }
        strongest = max(
            evidence or gated,
            key=lambda r: (rank.get(r.get("final_status"), 0), float(r.get("net_sharpe") or -999)),
        )

    paper_ids = [r["candidate_id"] for r in gated if r.get("final_status") == "PAPER_TRADING_CANDIDATE"]
    proven_ids = [r["candidate_id"] for r in gated if r.get("final_status") == "PROVEN_RESEARCH_PROFITABILITY"]

    answers = {
        "1_reproduced_in_2025": bool(profitable),
        "2_profitable_after_costs": [r["candidate_id"] for r in profitable],
        "3_net_sharpe": {r["candidate_id"]: r.get("net_sharpe") for r in evidence},
        "4_max_drawdown": {r["candidate_id"]: r.get("max_drawdown") for r in evidence},
        "5_cost_survival": {
            r["candidate_id"]: {
                "BASE": r.get("survives_BASE"),
                "MODERATE": r.get("survives_MODERATE"),
                "ADVERSE": r.get("survives_ADVERSE"),
            }
            for r in evidence
        },
        "6_persist_Q1_Q4": {r["candidate_id"]: r.get("quarters_positive") for r in evidence},
        "7_dependence_aware_stats": {r["candidate_id"]: r.get("statistically_meaningful") for r in evidence},
        "8_sharpe_genuine_or_inflated": {
            r["candidate_id"]: {
                "engine": r.get("net_sharpe"),
                "independent": (r.get("sharpe_engine_vs_independent") or {}).get("independent"),
                "not_inflated_flag": r.get("sharpe_not_inflated"),
                "n_eff": r.get("n_eff"),
                "acf1": r.get("acf1"),
            }
            for r in evidence
        },
        "9_portfolio_improved": portfolio.get("methods"),
        "10_strongest_under_gate": None if strongest is None else {
            "candidate_id": strongest["candidate_id"],
            "status": strongest.get("final_status"),
            "net_sharpe": strongest.get("net_sharpe"),
        },
        "11_statistically_credible_independent_evidence": any(r.get("statistically_meaningful") for r in evidence),
        "12_paper_trading": bool(paper_ids or proven_ids),
        "12_ids": paper_ids + proven_ids,
        "13_live_trading": False,
        "14_unsatisfied_gates": {
            r["candidate_id"]: (
                list(r.get("gate", {}).get("failed_checks") or [])
                + (["survives_ADVERSE"] if not r.get("survives_ADVERSE") and r.get("final_status") != "PROVEN_RESEARCH_PROFITABILITY" else [])
            )
            for r in evidence
        },
    }

    # Overall status
    if proven_ids:
        final_status = "PROVEN_RESEARCH_PROFITABILITY"
    elif paper_ids:
        final_status = "PAPER_TRADING_CANDIDATE"
    elif any(r.get("final_status") == "RESEARCH_EVIDENCE" for r in evidence):
        final_status = "RESEARCH_EVIDENCE"
    elif any(r.get("final_status") == "WEAK_EVIDENCE" for r in evidence):
        final_status = "WEAK_EVIDENCE"
    else:
        final_status = "REJECTED"

    md = [
        "# Frozen 2024 → Independent 2025 Holdout Validation",
        "",
        f"Status: **{final_status}**",
        "",
        DISCLAIMER,
        "",
        f"- Research: `<= {RESEARCH_END}`",
        f"- Holdout: `{HOLDOUT_START}` → `{HOLDOUT_END}`",
        f"- Firewall: **{fw['status']}**",
        f"- Complete 2025 data: **{prov['complete_2025_all_tfs']}**",
        f"- Reproducibility: **{repro['status']}**",
        f"- LIVE_READY: **NO**",
        "",
        "## Decision matrix (evidence focus)",
        "",
        "| Candidate | TF | Holding | Dir | Net Sharpe | Max DD | BASE | MOD | ADV | Status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in evidence:
        md.append(
            f"| `{r['candidate_id'][:16]}` | {r['timeframe']} | {r['holding_bars']} | {r['direction']} | "
            f"{r.get('net_sharpe')} | {r.get('max_drawdown')} | {r.get('survives_BASE')} | "
            f"{r.get('survives_MODERATE')} | {r.get('survives_ADVERSE')} | **{r.get('final_status')}** |"
        )
    md.extend(
        [
            "",
            "## Required answers",
            "",
            f"1. Reproduced in 2025 (after costs)? **{answers['1_reproduced_in_2025']}** → {answers['2_profitable_after_costs']}",
            f"3. Net Sharpe: {answers['3_net_sharpe']}",
            f"4. Max DD: {answers['4_max_drawdown']}",
            f"5. Cost survival: {answers['5_cost_survival']}",
            f"6. Quarters positive: {answers['6_persist_Q1_Q4']}",
            f"7. Dependence-aware stats: {answers['7_dependence_aware_stats']}",
            f"8. Sharpe genuine vs inflated: {answers['8_sharpe_genuine_or_inflated']}",
            "9. Portfolio: see portfolio_comparison.json (weights frozen from pre-2025)",
            f"10. Strongest under gate: {answers['10_strongest_under_gate']}",
            f"11. Statistically credible? **{answers['11_statistically_credible_independent_evidence']}**",
            f"12. Paper trading evidence? **{answers['12_paper_trading']}** ids={answers['12_ids']}",
            "13. Live trading? **NO**",
            f"14. Unsatisfied gates: {answers['14_unsatisfied_gates']}",
            "",
            "## Stop",
            "",
            "STOP — no retuning, no broker, no LIVE_READY, Prompt 35–42 artifacts untouched.",
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
        "live_ready": False,
        "claim_distinctions": {
            "PROVEN_RESEARCH_PROFITABILITY": bool(proven_ids),
            "PAPER_TRADING_CANDIDATE": bool(paper_ids),
            "LIVE_READY": False,
        },
    }
    _write(out_dir / "final_report.json", final)
    _write(out_dir / "test_summary.json", {"note": "Filled after pytest", "disclaimer": DISCLAIMER})
    if progress:
        print(f"[f2025] done status={final_status}", flush=True)
    return final


__all__ = ["run_frozen_2025"]

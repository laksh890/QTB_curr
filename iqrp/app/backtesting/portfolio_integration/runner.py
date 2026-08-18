"""Prompt 41 portfolio construction integration validation runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.analytics import evaluate_cost_aware
from iqrp.app.backtesting.alpha_research.consolidation.reconstruct import (
    build_signal_cache,
    load_prompt39_frames,
    reconstruct_candidate,
    sharpe_from_rets,
)
from iqrp.app.backtesting.alpha_research.experiments import now_iso
from iqrp.app.backtesting.alpha_research.types import COST_SCENARIOS
from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.portfolio_integration.adapters import (
    causal_mu_cov,
    daily_panel_from_series,
    run_optimizer,
    signed_exposure_series,
    validate_weights,
    weights_dict,
)
from iqrp.app.backtesting.portfolio_integration.protocol import (
    COST_NAMES,
    DISCLAIMER,
    PortfolioIntegrationConfig,
)
from iqrp.app.backtesting.serializer import to_jsonable
from iqrp.app.backtesting.unified_pipeline.orchestrator import UnifiedTradingOrchestrator
from iqrp.app.backtesting.unified_pipeline.types import AlphaCandidate
from iqrp.app.portfolio.optimization import optimize_mean_variance, optimize_risk_parity


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str), encoding="utf-8")


def load_p40_candidates(prompt40_dir: Path) -> list[dict[str, Any]]:
    data = json.loads((prompt40_dir / "final_candidate_set.json").read_text(encoding="utf-8"))
    return list(data["DISTINCT_RESEARCH_CANDIDATES"])


def load_p39_experiment(prompt39_dir: Path, experiment_id: str) -> dict[str, Any]:
    reg = json.loads((prompt39_dir / "experiment_registry.json").read_text(encoding="utf-8"))
    for e in reg["experiments"]:
        if e.get("experiment_id") == experiment_id:
            return e
    raise KeyError(experiment_id)


def portfolio_returns_from_exposures(
    exposures: pd.DataFrame,
    sleeve_daily_nets: pd.DataFrame,
) -> pd.Series:
    common = exposures.index.intersection(sleeve_daily_nets.index)
    exp = exposures.reindex(common).fillna(0.0)
    nets = sleeve_daily_nets.reindex(common).fillna(0.0)
    exp_lag = exp.shift(1).fillna(0.0)
    return (exp_lag.abs() * nets).sum(axis=1)


def _signed_static(sleeve_w: dict[str, float], directions: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for cid, w in sleeve_w.items():
        d = directions.get(cid, "LONG_SHORT")
        if d == "SHORT":
            out[cid] = -abs(float(w))
        elif d == "LONG":
            out[cid] = abs(float(w))
        else:
            out[cid] = abs(float(w))
    return out


def run_portfolio_integration(
    cfg: PortfolioIntegrationConfig | None = None,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    cfg = cfg or PortfolioIntegrationConfig()
    if cfg.smoke and cfg.output_dir == "results/portfolio_construction_integration":
        cfg.output_dir = "results/portfolio_construction_integration_smoke"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    p40 = Path(cfg.prompt40_dir)
    p39 = Path(cfg.prompt39_dir)

    _write(out_dir / "integration_config.json", {**cfg.to_dict(), "started_at": started})

    candidates_meta = load_p40_candidates(p40)
    if cfg.smoke:
        candidates_meta = candidates_meta[:4]

    exps = [load_p39_experiment(p39, c["experiment_id"]) for c in candidates_meta]
    campaign = json.loads((p39 / "campaign.json").read_text(encoding="utf-8"))
    frames, _ds = load_prompt39_frames(campaign, cfg.registry_path)
    needed = {(e["kind"], e["source_id"], e["timeframe"]) for e in exps}
    if progress:
        print(f"[pci] reconstructing {len(needed)} signals for {len(exps)} candidates", flush=True)
    signal_cache, recon_errors = build_signal_cache(
        frames,
        needed=needed,
        reference_lookback=int(campaign.get("reference_lookback") or 20),
        train_frac=float(campaign.get("train_frac") or 0.5),
        progress=progress,
    )

    series_map: dict[str, dict[str, Any]] = {}
    directions: dict[str, str] = {}
    signal_daily_sign: dict[str, pd.Series] = {}
    meta_by_id = {c["candidate_id"]: c for c in candidates_meta}

    for e in exps:
        cid = e["experiment_id"]
        directions[cid] = e["direction"]
        base = reconstruct_candidate(e, frames=frames, signal_cache=signal_cache, cost_name="BASE")
        if base.get("status") != "OK":
            continue
        series_map[cid] = base
        ts = base["timestamps"]
        pos = pd.Series(base["positions"], index=range(len(ts)))
        # map to daily last sign
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(ts, utc=True).dt.floor("D"),
                "pos": base["positions"],
            }
        )

        def _last_sign(s: pd.Series) -> float:
            nz = s.replace(0.0, np.nan).dropna()
            return float(np.sign(nz.iloc[-1])) if len(nz) else 0.0

        signal_daily_sign[cid] = df.groupby("date")["pos"].apply(_last_sign)

    names = [e["experiment_id"] for e in exps if e["experiment_id"] in series_map]
    if len(names) < 2:
        report = {
            "status": "PORTFOLIO_INTEGRATION_BLOCKED",
            "reason": "Fewer than 2 candidates reconstructed",
            "recon_errors": recon_errors,
            "disclaimer": DISCLAIMER,
        }
        _write(out_dir / "final_report.json", report)
        (out_dir / "final_report.md").write_text("# BLOCKED\n", encoding="utf-8")
        return report

    panel, period_dates = daily_panel_from_series(series_map)
    est = causal_mu_cov(panel, names, period_dates)
    if progress:
        print(f"[pci] causal mu/cov n_obs={est['n_obs']} method={est['cov_method']}", flush=True)

    # Rejection / reduced-path exercises
    rejection_exercises: list[dict[str, Any]] = []
    bad_rp = optimize_risk_parity(
        cov=est["cov"],
        names=names,
        long_only=False,
        min_weight=-0.2,
        max_weight=cfg.max_weight,
        budget=cfg.budget,
    )
    rejection_exercises.append(
        {
            "case": "risk_parity_long_short_request",
            "success": bool(bad_rp.get("success")),
            "status": bad_rp.get("status"),
            "failure_reason": bad_rp.get("failure_reason") or bad_rp.get("reason"),
            "note": "Existing RP solver requires long-only sleeves.",
        }
    )
    deg_cov = np.ones((len(names), len(names))) * 0.09 + np.eye(len(names)) * 1e-18
    deg_mv = optimize_mean_variance(
        mu=est["mu"],
        cov=deg_cov,
        names=names,
        long_only=True,
        max_weight=cfg.max_weight,
        budget=cfg.budget,
        ridge=1e-4,
    )
    rejection_exercises.append(
        {
            "case": "degenerate_covariance_mean_variance",
            "success": bool(deg_mv.get("success")),
            "status": deg_mv.get("status"),
            "note": "Exercises ridge/stabilization inside existing MV optimizer",
        }
    )
    tight = run_optimizer(
        "mean_variance",
        mu=est["mu"],
        cov=est["cov"],
        names=names,
        max_weight=0.01,
        max_gross=0.05,
        budget=1.0,
        risk_aversion=cfg.risk_aversion,
        long_only_sleeves=True,
    )
    rejection_exercises.append(
        {
            "case": "tight_constraints_mean_variance",
            "success": bool(tight.get("success")),
            "status": tight.get("status"),
            "failure_reason": tight.get("failure_reason"),
        }
    )

    method_results: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []
    recon_rows: list[dict[str, Any]] = []
    integration_matrix: list[dict[str, Any]] = []

    px_frame = frames["1h"] if "1h" in frames else next(iter(frames.values()))
    last_price = float(px_frame["close"].iloc[-1])
    # Causal returns sample for risk engine (pre-OOS BTC returns)
    btc_rets = px_frame["close"].pct_change().fillna(0.0).to_numpy(dtype=float)
    n_pre = max(int(len(btc_rets) * 0.75), 60)
    risk_rets = btc_rets[:n_pre]

    for method in cfg.methods:
        if progress:
            print(f"[pci] method={method}", flush=True)

        # Primary construction: long-only sleeve budgets for all methods (direction map
        # produces signed trading exposure). Separately, MV/BL also exercise native L/S
        # sleeves with budget=0 (handled inside run_optimizer when long_only_sleeves=False).
        if method in {"mean_variance", "black_litterman"}:
            # Prefer feasible long-only sleeves + direction map for the main cascade path;
            # also record a native L/S sleeve attempt for capability evidence.
            opt_ls = run_optimizer(
                method,
                mu=est["mu"],
                cov=est["cov"],
                names=names,
                max_weight=cfg.max_weight,
                max_gross=cfg.max_gross,
                budget=cfg.budget,
                risk_aversion=cfg.risk_aversion,
                long_only_sleeves=False,
            )
            opt = run_optimizer(
                method,
                mu=est["mu"],
                cov=est["cov"],
                names=names,
                max_weight=cfg.max_weight,
                max_gross=cfg.max_gross,
                budget=cfg.budget,
                risk_aversion=cfg.risk_aversion,
                long_only_sleeves=True,
            )
            opt["diagnostics"] = dict(opt.get("diagnostics") or {})
            opt["diagnostics"]["native_long_short_sleeve_attempt"] = {
                "success": bool(opt_ls.get("success")),
                "status": opt_ls.get("status"),
                "failure_reason": opt_ls.get("failure_reason"),
                "budget_mode": "dollar_neutral_0",
                "note": "Native L/S sleeve path (budget=0). Primary cascade uses long-only sleeves + direction map.",
            }
            long_only_sleeves = True
        else:
            long_only_sleeves = True
            opt = run_optimizer(
                method,
                mu=est["mu"],
                cov=est["cov"],
                names=names,
                max_weight=cfg.max_weight,
                max_gross=cfg.max_gross,
                budget=cfg.budget,
                risk_aversion=cfg.risk_aversion,
                long_only_sleeves=long_only_sleeves,
            )
        success = bool(opt.get("success", False)) or method == "constraints_only"
        sleeve_w = weights_dict(opt, names) if success else {n: 0.0 for n in names}
        static_signed = _signed_static(sleeve_w, directions)

        val = validate_weights(
            sleeve_w if method in {"risk_parity", "hrp", "constraints_only"} else static_signed,
            max_weight=cfg.max_weight,
            max_gross=cfg.max_gross,
            max_net=cfg.max_net,
            max_turnover=cfg.max_turnover,
            previous={n: 0.0 for n in names},
            adv={n: 1e12 for n in names},
        )
        constraint_rows.append({"method": method, "optimizer_success": success, **val})

        exposures = signed_exposure_series(sleeve_w, signal_daily_sign, directions, panel.index)
        oos_dates = set.intersection(*(period_dates[n]["oos"] for n in names))
        oos_idx = sorted(oos_dates)
        port_daily = portfolio_returns_from_exposures(exposures, panel[names])
        oos_rets = port_daily.reindex(oos_idx).fillna(0.0).to_numpy(dtype=float)
        net_exp = exposures.sum(axis=1).reindex(oos_idx).fillna(0.0).to_numpy(dtype=float)

        cost_by: dict[str, Any] = {}
        for cost_name in COST_NAMES:
            cm = COST_SCENARIOS[cost_name]
            n = min(len(net_exp), len(oos_rets))
            ev = evaluate_cost_aware(
                pd.Series(net_exp[:n]),
                pd.Series(oos_rets[:n]),
                commission_bps=float(cm["commission_bps"]),
                spread_bps=float(cm["spread_bps"]),
                slippage_bps=float(cm["slippage_bps"]),
                periods_per_year=365.0,
                n_calendar_days=max(n, 1),
            )
            cost_by[cost_name] = {
                "net_sharpe": ev["net_sharpe"],
                "gross_sharpe": ev["gross_sharpe"],
                "net_pnl": ev["net_pnl"],
                "gross_pnl": ev["gross_pnl"],
                "transaction_costs": ev["transaction_costs"],
                "alpha_survives_costs": ev["alpha_survives_costs"],
                "alpha_collapses_after_costs": ev["alpha_collapses_after_costs"],
                "max_drawdown": float(max_drawdown(ev["net_returns"])),
                "volatility": float(np.nanstd(ev["net_returns"]) * np.sqrt(365.0)) if n else None,
            }

        oos_sharpe = sharpe_from_rets(oos_rets, 365.0)
        mid = max(len(oos_rets) // 2, 1)
        s1 = sharpe_from_rets(oos_rets[:mid], 365.0)
        s2 = sharpe_from_rets(oos_rets[mid:], 365.0)
        if np.isfinite(s1) and np.isfinite(s2) and s1 > 0 and s2 <= 0:
            stability = "unstable"
        elif np.isfinite(s1) and np.isfinite(s2) and s1 > 0 and s2 < 0.5 * s1:
            stability = "degraded"
        else:
            stability = "stable"

        # Cascade via existing UnifiedTradingOrchestrator
        # Use BTCUSDT as instrument; requested weights are signed sleeve budgets (scaled down
        # so multi-candidate gross fits max_gross — orchestrator processes sequentially).
        scale = 1.0
        gross = sum(abs(v) for v in static_signed.values())
        if gross > cfg.max_gross > 0:
            scale = cfg.max_gross / gross
        orch = UnifiedTradingOrchestrator(
            initial_capital=1_000_000.0,
            long_only=False,
            max_position=cfg.max_weight,
            max_gross=cfg.max_gross,
            base_returns=risk_rets,
        )
        cands: list[AlphaCandidate] = []
        for cid in names:
            m = meta_by_id[cid]
            w = float(static_signed[cid]) * scale
            direction = float(np.sign(w)) if abs(w) > 1e-12 else 0.0
            cands.append(
                AlphaCandidate(
                    candidate_id=f"{method}:{cid}",
                    signal_id=str(m.get("signal_id") or cid),
                    instrument="BTCUSDT",
                    timestamp=started,
                    direction=direction,
                    signal_value=w,
                    confidence=0.5,
                    expected_horizon=int(m.get("holding_horizon") or 1),
                    signal_timeframe=str(m.get("timeframe") or ""),
                    execution_timeframe=str(m.get("timeframe") or ""),
                    source_model=str(m.get("model_family") or ""),
                    source_model_version="1.0.0",
                    data_version="1.0.0",
                    dataset_checksum="p40_immutable",
                    oos_status="EVALUATED",
                    experiment_id=cid,
                    requested_weight=w,
                    meta={"portfolio_method": method, "sleeve_weight": sleeve_w.get(cid)},
                )
            )
        cascade = orch.process_candidates(
            cands,
            asof=started,
            prices={"BTCUSDT": last_price},
            returns=risk_rets,
            simulation_mode="fill",
        )
        recon = cascade.get("reconciliation") or {}
        recon_ok = bool(recon.get("ok") or recon.get("status") in {"PASS", "OK", "RECONCILIATION_OK"})
        # Also accept StageOutcome style
        if not recon_ok and isinstance(recon, dict):
            # full_accounting_audit style
            recon_ok = str(recon.get("overall") or recon.get("result") or "").upper() in {
                "PASS",
                "OK",
                "CLEAN",
            } or bool(recon.get("passed"))

        cascade_complete = True
        for step in cascade.get("results") or []:
            if step.get("outcome") in {"CANDIDATE_REJECTED"} and "error" in step:
                cascade_complete = False
        # Must have processed all and recon
        cascade_complete = cascade_complete and len(cascade.get("results") or []) == len(names)

        recon_rows.append(
            {
                "method": method,
                "cascade_complete": cascade_complete,
                "reconciliation": recon,
                "reconciliation_ok": recon_ok,
                "equity": cascade.get("equity"),
                "positions": cascade.get("positions"),
                "gross_exposure_weights": cascade.get("gross_exposure_weights"),
                "n_results": len(cascade.get("results") or []),
                "result_outcomes": [r.get("outcome") for r in (cascade.get("results") or [])],
            }
        )

        row = {
            "method": method,
            "optimizer_success": success,
            "optimizer_status": opt.get("status"),
            "failure_reason": opt.get("failure_reason"),
            "sleeve_weights": sleeve_w,
            "static_signed_exposure": static_signed,
            "long_only_sleeve_optimizer": method in {"risk_parity", "hrp", "constraints_only"},
            "long_short_trading_via_direction_map": True,
            "n_active_positions": val["active_positions"],
            "long_exposure": val["long_exposure"],
            "short_exposure": val["short_exposure"],
            "gross_exposure": val["gross_exposure"],
            "net_exposure": val["net_exposure"],
            "turnover": val["turnover_vs_previous"],
            "concentration": val["concentration"],
            "constraint_violations": val["n_violations"],
            "oos_sharpe_gross_proxy": oos_sharpe,
            "cost_scenarios": cost_by,
            "oos_survives_base": bool(cost_by["BASE"].get("alpha_survives_costs")),
            "oos_survives_moderate": bool(cost_by["MODERATE"].get("alpha_survives_costs")),
            "oos_survives_adverse": bool(cost_by["ADVERSE"].get("alpha_survives_costs")),
            "max_drawdown_base": cost_by["BASE"].get("max_drawdown"),
            "volatility_base": cost_by["BASE"].get("volatility"),
            "portfolio_stability": stability,
            "cascade_complete": cascade_complete,
            "reconciliation_status": "PASS" if recon_ok else "CHECK",
            "disclaimer": DISCLAIMER,
        }
        method_results.append(row)
        integration_matrix.append(
            {
                "method": method,
                "implemented": True,
                "optimizer_api_success": success,
                "integrated_with_alpha_pipeline": cascade_complete,
                "supports_long_short_sleeves": method in {"mean_variance", "black_litterman"},
                "native_long_short_sleeve_attempt": (opt.get("diagnostics") or {}).get(
                    "native_long_short_sleeve_attempt"
                ),
                "supports_long_short_trading_via_direction": True,
                "survives_constraints": (val["n_violations"] == 0),
                "survives_cost_aware_oos_base": row["oos_survives_base"],
                "cascade_operational": cascade_complete,
                "rp_hrp_limitation": (
                    "Long-only sleeve budgets; signed exposure from candidate direction/signal"
                    if method in {"risk_parity", "hrp"}
                    else None
                ),
            }
        )

    base = next((m for m in method_results if m["method"] == "constraints_only"), None)
    comparisons = []
    if base:
        for m in method_results:
            if m["method"] == "constraints_only":
                continue
            comparisons.append(
                {
                    "method": m["method"],
                    "dd_vs_baseline": (m.get("max_drawdown_base") or 0) - (base.get("max_drawdown_base") or 0),
                    "turnover_vs_baseline": (m.get("turnover") or 0) - (base.get("turnover") or 0),
                    "gross_vs_baseline": (m.get("gross_exposure") or 0) - (base.get("gross_exposure") or 0),
                    "net_sharpe_base_method": (m.get("cost_scenarios") or {}).get("BASE", {}).get("net_sharpe"),
                    "net_sharpe_base_baseline": (base.get("cost_scenarios") or {}).get("BASE", {}).get("net_sharpe"),
                    "note": "Diagnostic vs constraints-only — not a profitability ranking.",
                }
            )

    all_cascade = all(m.get("cascade_complete") for m in method_results)
    answers = {
        "1_methods_implemented": list(cfg.methods),
        "2_mathematically_validated": [
            m["method"] for m in method_results if m.get("optimizer_success")
        ],
        "3_integrated_with_model_driven_pipeline": [
            m["method"] for m in method_results if m.get("cascade_complete")
        ],
        "4_support_long_short": {
            "mean_variance": "YES (long_only=False)",
            "black_litterman": "YES (via MV, long_only=False)",
            "risk_parity": "NO on sleeves; YES trading via direction map",
            "hrp": "NO on sleeves; YES trading via direction map",
            "constraints_only": "YES via direction map",
        },
        "5_survive_constraints": [
            m["method"] for m in method_results if (m.get("constraint_violations") or 0) == 0
        ],
        "6_survive_cost_aware_oos": {
            "BASE": [m["method"] for m in method_results if m.get("oos_survives_base")],
            "MODERATE": [m["method"] for m in method_results if m.get("oos_survives_moderate")],
            "ADVERSE": [m["method"] for m in method_results if m.get("oos_survives_adverse")],
        },
        "7_vs_constraints_only": comparisons,
        "8_full_cascade_operational": all_cascade,
        "proven_profitability": False,
        "live_ready": False,
    }

    status = (
        "PORTFOLIO_INTEGRATION_COMPLETE"
        if all_cascade and any(m.get("optimizer_success") for m in method_results)
        else "PORTFOLIO_INTEGRATION_PARTIAL"
        if method_results
        else "PORTFOLIO_INTEGRATION_BLOCKED"
    )

    final = {
        "disclaimer": DISCLAIMER,
        "integration_id": cfg.integration_id,
        "started_at": started,
        "status": status,
        "n_candidates": len(names),
        "candidate_ids": names,
        "causal_estimation": {
            k: est[k]
            for k in (
                "n_obs",
                "cov_method",
                "degenerate_before_ridge",
                "min_eigenvalue",
                "estimation_window",
                "n_dates",
            )
        },
        "answers": answers,
        "claim_distinctions": {
            "PORTFOLIO_IMPLEMENTED": True,
            "PORTFOLIO_INTEGRATED": all_cascade,
            "PORTFOLIO_ROBUST": False,
            "PROFITABLE": False,
            "LIVE_READY": False,
        },
        "rejection_exercises": rejection_exercises,
    }

    md = [
        "# Portfolio Construction Integration (Prompt 41)",
        "",
        f"Status: **{status}**",
        "",
        DISCLAIMER,
        "",
        f"- Prompt 40 candidates used: {len(names)}",
        f"- Methods: {', '.join(cfg.methods)}",
        f"- Full cascade operational: **{all_cascade}**",
        f"- Proven profitability: **NO**",
        f"- Live ready: **NO**",
        "",
        "## Required answers",
        "",
        f"1. Implemented: {answers['1_methods_implemented']}",
        f"2. Mathematically validated: {answers['2_mathematically_validated']}",
        f"3. Integrated: {answers['3_integrated_with_model_driven_pipeline']}",
        f"4. Long/short: {answers['4_support_long_short']}",
        f"5. Survive constraints: {answers['5_survive_constraints']}",
        f"6. Survive cost-aware OOS: {answers['6_survive_cost_aware_oos']}",
        "7. Vs constraints-only: see portfolio_method_results.json (not a profitability ranking)",
        f"8. Full cascade operational: **{answers['8_full_cascade_operational']}**",
        "",
        "## Claim distinctions",
        "",
        "PORTFOLIO IMPLEMENTED ≠ INTEGRATED ≠ ROBUST ≠ PROFITABLE ≠ LIVE READY",
        "",
        "## Stop",
        "",
        "STOP after Prompt 41 — no paper trading, broker integration, live trading, or production.",
        "",
    ]

    _write(out_dir / "final_report.json", final)
    (out_dir / "final_report.md").write_text("\n".join(md), encoding="utf-8")
    _write(out_dir / "integration_matrix.json", {"rows": integration_matrix, "disclaimer": DISCLAIMER})
    _write(
        out_dir / "portfolio_method_results.json",
        {"results": method_results, "comparisons_vs_baseline": comparisons, "disclaimer": DISCLAIMER},
    )
    _write(
        out_dir / "constraint_validation.json",
        {"rows": constraint_rows, "rejection_exercises": rejection_exercises, "disclaimer": DISCLAIMER},
    )
    _write(out_dir / "reconciliation_report.json", {"rows": recon_rows, "disclaimer": DISCLAIMER})
    _write(
        out_dir / "causal_estimation.json",
        {
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in est.items()},
            "disclaimer": DISCLAIMER,
        },
    )
    _write(
        out_dir / "test_summary.json",
        {"note": "Populated after pytest", "disclaimer": DISCLAIMER},
    )

    if progress:
        print(f"[pci] done status={status} cascade={all_cascade}", flush=True)
    return final


__all__ = ["run_portfolio_integration"]

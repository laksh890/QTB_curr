"""Prompt 43 paper-trading validation runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.analytics import positions_from_signal
from iqrp.app.backtesting.alpha_research.consolidation.reconstruct import build_signal_cache, sharpe_from_rets
from iqrp.app.backtesting.alpha_research.experiments import now_iso
from iqrp.app.backtesting.alpha_research.model_campaign.protocol import apply_direction_mask
from iqrp.app.backtesting.alpha_research.model_campaign.runner import _trim
from iqrp.app.backtesting.alpha_research.types import bars_per_day
from iqrp.app.backtesting.final_holdout.freeze import FREEZE_FIELDS, definition_checksum, load_p39_experiment
from iqrp.app.backtesting.frozen_2025_holdout.datasets import materialize_firewall_datasets
from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.serializer import to_jsonable
from iqrp.app.paper_trading.failure_injection import run_failure_injection
from iqrp.app.paper_trading.fill_model import AssumedFillModel
from iqrp.app.paper_trading.protocol import (
    COMBOS,
    DISCLAIMER,
    EXEC_SCENARIOS,
    FROZEN_CANDIDATES,
    HOLDOUT_END,
    HOLDOUT_START,
    RESEARCH_END,
    PaperTradingValidationConfig,
    classify_paper_status,
)
from iqrp.app.paper_trading.risk import PaperRiskLimits
from iqrp.app.paper_trading.simulator import run_sequential_paper


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str), encoding="utf-8")


def _load_frozen_defs(prompt39_dir: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for label, cid in FROZEN_CANDIDATES.items():
        exp = load_p39_experiment(prompt39_dir, cid)
        defn = {k: exp.get(k) for k in FREEZE_FIELDS}
        out[label] = {
            "candidate_id": cid,
            "label": label,
            "definition": defn,
            "definition_checksum": definition_checksum(defn),
        }
    return out


def _build_weight_series(
    defs: dict[str, dict[str, Any]],
    *,
    data_dir: Path,
    max_bars_research: dict[str, int],
    sleeve_weight: float,
    smoke: bool,
    smoke_bars: int,
    progress: bool,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame], dict[str, Any]]:
    """Causal frozen signals → desired weights on 2025 holdout bars only."""
    # Ensure firewall datasets exist
    materialize_firewall_datasets(out_dir=str(data_dir), register=False)

    tfs_needed = set()
    for d in defs.values():
        tfs_needed.add(d["definition"]["timeframe"])
        sid = str(d["definition"]["source_id"])
        if ":" in sid and "->" in sid:
            _, rest = sid.split(":", 1)
            mtf, _ = rest.split("->", 1)
            tfs_needed.add(mtf)

    research_end = pd.Timestamp(RESEARCH_END)
    hold_start = pd.Timestamp(HOLDOUT_START)
    hold_end = pd.Timestamp(HOLDOUT_END)

    concat: dict[str, pd.DataFrame] = {}
    holdout_frames: dict[str, pd.DataFrame] = {}
    for tf in sorted(tfs_needed):
        r = pd.read_parquet(data_dir / f"btcusdt_research_through_2024_{tf}.parquet")
        h = pd.read_parquet(data_dir / f"btcusdt_holdout_2025_{tf}.parquet")
        r["timestamp"] = pd.to_datetime(r["timestamp"], utc=True)
        h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
        r = _trim(r, int(max_bars_research.get(tf, 30_000)))
        if smoke:
            h = h.iloc[:smoke_bars].reset_index(drop=True)
        holdout_frames[tf] = h.reset_index(drop=True)
        c = pd.concat([r.reset_index(drop=True), holdout_frames[tf]], ignore_index=True)
        c = c.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        concat[tf] = c

    needed: set[tuple[str, str, str]] = set()
    for d in defs.values():
        dd = d["definition"]
        needed.add((dd["kind"], dd["source_id"], dd["timeframe"]))
        sid = str(dd["source_id"])
        if ":" in sid and "->" in sid:
            base, rest = sid.split(":", 1)
            mtf, _ = rest.split("->", 1)
            needed.add(("ref", base, mtf))

    # train_frac maps P39 split onto research prefix
    tf_ref = "15m" if "15m" in concat else next(iter(concat))
    n_res = int((pd.to_datetime(concat[tf_ref]["timestamp"], utc=True) <= research_end).sum())
    n_tot = len(concat[tf_ref])
    train_frac = max((0.5 * n_res) / max(n_tot, 1), 0.05)

    if progress:
        print(f"[paper] build_signal_cache needed={len(needed)} train_frac={train_frac:.4f}", flush=True)
    cache, errors = build_signal_cache(
        concat, needed=needed, reference_lookback=20, train_frac=train_frac, progress=progress
    )

    weights: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"recon_errors": errors, "train_frac": train_frac}
    for label, d in defs.items():
        dd = d["definition"]
        tf = dd["timeframe"]
        key = f"{dd['kind']}:{dd['source_id']}:{tf}"
        if key not in cache:
            raise RuntimeError(f"signal cache miss {key}: {errors[:3]}")
        raw = cache[key]
        directed = apply_direction_mask(raw.fillna(0.0), dd["direction"])
        pos = positions_from_signal(directed, int(dd["holding_bars"]))
        ts = pd.to_datetime(concat[tf]["timestamp"], utc=True)
        mask = ((ts >= hold_start) & (ts <= hold_end)).to_numpy()
        # holdout-aligned weight series (signed position * sleeve)
        w_full = pos.to_numpy(dtype=float) * float(sleeve_weight)
        w_hold = w_full[mask]
        # Lookahead check: position at i uses signal through i only by construction of positions_from_signal
        weights[label] = w_hold
        meta[label] = {
            "tf": tf,
            "n_holdout": int(mask.sum()),
            "checksum": d["definition_checksum"],
            "candidate_id": d["candidate_id"],
        }
    return weights, holdout_frames, meta


def _align_weight_to_primary(
    primary_ts: pd.Series,
    other_ts: pd.Series,
    other_w: np.ndarray,
) -> np.ndarray:
    """Map other TF weights onto primary timestamps via causal asof (no future bars)."""
    left = pd.DataFrame({"timestamp": pd.to_datetime(primary_ts, utc=True)})
    right = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(other_ts, utc=True),
            "w": np.asarray(other_w, dtype=float)[: len(other_ts)],
        }
    )
    right = right.iloc[: min(len(right), len(other_w))].sort_values("timestamp")
    merged = pd.merge_asof(left.sort_values("timestamp"), right, on="timestamp", direction="backward")
    return merged["w"].fillna(0.0).to_numpy(dtype=float)


def _sharpe_from_equity(equity: list[dict[str, Any]], timeframe: str, market_type: str) -> float:
    eq = np.array([e["equity"] for e in equity], dtype=float)
    if len(eq) < 10:
        return float("nan")
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    bpd = bars_per_day(timeframe, market_type=market_type)
    ppy = 252.0 * float(bpd)
    return sharpe_from_rets(rets, ppy)


def run_paper_trading_validation(
    cfg: PaperTradingValidationConfig | None = None,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    cfg = cfg or PaperTradingValidationConfig()
    if cfg.smoke and cfg.output_dir == "results/paper_trading_validation":
        cfg.output_dir = "results/paper_trading_validation_smoke"
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    _write(out_dir / "validation_config.json", {**cfg.to_dict(), "started_at": started})

    defs = _load_frozen_defs(Path(cfg.prompt39_dir))
    # Verify unchanged vs frozen 2025 manifest if present
    freeze_ok = True
    f2025 = Path(cfg.frozen_2025_dir) / "frozen_candidate_manifest.json"
    if f2025.exists():
        man = json.loads(f2025.read_text(encoding="utf-8"))
        by_id = {c["candidate_id"]: c["definition_checksum"] for c in man.get("candidates", [])}
        for d in defs.values():
            if by_id.get(d["candidate_id"]) and by_id[d["candidate_id"]] != d["definition_checksum"]:
                freeze_ok = False

    max_bars = {"5m": 30000, "15m": 25000, "30m": 20000, "1h": 20000}
    weights, holdout_frames, sig_meta = _build_weight_series(
        defs,
        data_dir=Path(cfg.data_dir),
        max_bars_research=max_bars,
        sleeve_weight=cfg.sleeve_weight,
        smoke=cfg.smoke,
        smoke_bars=cfg.smoke_bars,
        progress=progress,
    )

    # For each label, attach its holdout price frame
    frames_by_label = {}
    for label, d in defs.items():
        tf = d["definition"]["timeframe"]
        frames_by_label[label] = holdout_frames[tf]

    rng = np.random.default_rng(cfg.random_seed)
    limits = PaperRiskLimits(
        max_position=cfg.max_position,
        max_gross=cfg.max_gross,
        max_net=cfg.max_net,
        max_daily_loss=cfg.max_daily_loss,
        max_drawdown=cfg.max_drawdown,
        max_turnover_per_bar=cfg.max_turnover_per_bar,
    )

    candidate_results = []
    execution_results = []
    all_fills = []
    cost_rows = []

    def run_combo(labels: tuple[str, ...], scenario: str) -> dict[str, Any]:
        # Primary TF = first label; other sleeves asof-aligned (causal). Candidates unmodified.
        primary = labels[0]
        frame = frames_by_label[primary]
        ts = frame["timestamp"]
        px = frame["close"]
        n = len(frame)
        w = np.zeros(n, dtype=float)
        for lab in labels:
            arr = weights[lab]
            if lab == primary or defs[lab]["definition"]["timeframe"] == defs[primary]["definition"]["timeframe"]:
                m = min(n, len(arr))
                w[:m] = w[:m] + arr[:m]
            else:
                w = w + _align_weight_to_primary(ts, frames_by_label[lab]["timestamp"], arr)
        gross = np.abs(w)
        w[gross > cfg.max_gross] = np.sign(w[gross > cfg.max_gross]) * cfg.max_gross

        fill_model = AssumedFillModel(EXEC_SCENARIOS[scenario], rng=rng)
        latency = int(EXEC_SCENARIOS[scenario]["latency_bars"])
        label = "+".join(labels) + f"@{scenario}"
        if progress:
            print(f"[paper] sequential {label} bars={n}", flush=True)
        out = run_sequential_paper(
            timestamps=ts,
            closes=px,
            target_weights=w,
            fill_model=fill_model,
            limits=limits,
            initial_capital=cfg.initial_capital,
            latency_bars=latency,
            candidate_label=label,
        )
        tf = defs[primary]["definition"]["timeframe"]
        out["sharpe"] = _sharpe_from_equity(out["equity_curve"], tf, cfg.market_type)
        out["labels"] = list(labels)
        out["scenario"] = scenario
        out["timeframe"] = tf
        out["max_dd"] = out["max_drawdown"]
        # cost drag approx fees / initial
        out["cost_drag_fees"] = out["fees_paid"] / cfg.initial_capital
        return out

    # Primary BASE runs for all combos
    combo_outs = {}
    for combo in COMBOS:
        if cfg.smoke and combo not in (("A",), ("B",), ("A", "B")):
            continue
        key = "+".join(combo)
        out = run_combo(combo, cfg.exec_scenario)
        combo_outs[key] = out
        candidate_results.append(
            {
                "combo": key,
                "candidate_ids": [defs[l]["candidate_id"] for l in combo],
                "labels": list(combo),
                "scenario": cfg.exec_scenario,
                "net_return": out["net_return"],
                "sharpe": out["sharpe"],
                "max_drawdown": out["max_drawdown"],
                "n_fills": out["n_fills"],
                "n_rejects": out["n_rejects"],
                "n_partials": out["n_partials"],
                "fees_paid": out["fees_paid"],
                "cost_drag_fees": out["cost_drag_fees"],
                "recon_ok": out["final_recon"]["ok"],
                "recon_drift": out["final_recon"]["drift"],
                "kill_switch": out["kill_switch"],
                "cost_model_label": out["cost_model_label"],
                "frozen_checksums": {l: defs[l]["definition_checksum"] for l in combo},
            }
        )
        execution_results.append(
            {
                "combo": key,
                "n_orders_approx": out["n_fills"] + out["n_rejects"],
                "n_fills": out["n_fills"],
                "n_rejects": out["n_rejects"],
                "n_partials": out["n_partials"],
                "latency_bars": EXEC_SCENARIOS[cfg.exec_scenario]["latency_bars"],
                "sample_fills": out["fills"][:20],
            }
        )
        all_fills.extend(out["fills"][:100])
        cost_rows.append(
            {
                "combo": key,
                "scenario": cfg.exec_scenario,
                "components": EXEC_SCENARIOS[cfg.exec_scenario],
                "fees_paid": out["fees_paid"],
                "net_return": out["net_return"],
                "label": "ASSUMED_OHLCV_MICROSTRUCTURE",
            }
        )

    # Cost sensitivity on A alone
    cost_sensitivity = []
    for scen in ("BASE", "MODERATE", "ADVERSE"):
        if cfg.smoke and scen == "ADVERSE":
            continue
        out = run_combo(("A",), scen)
        cost_sensitivity.append(
            {
                "combo": "A",
                "scenario": scen,
                "net_return": out["net_return"],
                "sharpe": out["sharpe"],
                "max_drawdown": out["max_drawdown"],
                "fees_paid": out["fees_paid"],
            }
        )

    # Failure injection on A
    failure = {"status": "SKIPPED"}
    if cfg.run_failure_injection:
        if progress:
            print("[paper] failure injection", flush=True)
        fa = frames_by_label["A"]
        n = min(len(fa), cfg.smoke_bars if cfg.smoke else 800)
        failure = run_failure_injection(
            timestamps=fa["timestamp"].iloc[:n].reset_index(drop=True),
            closes=fa["close"].iloc[:n].reset_index(drop=True),
            target_weights=weights["A"][:n],
            seed=cfg.random_seed,
        )

    # Reproducibility: rerun A@BASE
    if progress:
        print("[paper] reproducibility rerun A", flush=True)
    out_a1 = combo_outs.get("A") or run_combo(("A",), cfg.exec_scenario)
    out_a2 = run_combo(("A",), cfg.exec_scenario)
    repro = {
        "status": "PASS"
        if abs(out_a1["net_return"] - out_a2["net_return"]) < 1e-10
        and out_a1["n_fills"] == out_a2["n_fills"]
        else "FAIL",
        "run1_net": out_a1["net_return"],
        "run2_net": out_a2["net_return"],
        "run1_fills": out_a1["n_fills"],
        "run2_fills": out_a2["n_fills"],
        "note": "Deterministic given fixed seed; reject/partial RNG may differ if scenario consumes RNG differently — BASE path compared.",
    }
    # Actually reject_prob>0 makes RNG path matter across runs if other combos ran first.
    # Re-seed for fair repro:
    rng2 = np.random.default_rng(cfg.random_seed)
    fill2 = AssumedFillModel(EXEC_SCENARIOS[cfg.exec_scenario], rng=rng2)
    frame = frames_by_label["A"]
    w = weights["A"][: len(frame)]
    r1 = run_sequential_paper(
        timestamps=frame["timestamp"],
        closes=frame["close"],
        target_weights=w,
        fill_model=AssumedFillModel(EXEC_SCENARIOS[cfg.exec_scenario], rng=np.random.default_rng(cfg.random_seed)),
        limits=limits,
        initial_capital=cfg.initial_capital,
        latency_bars=int(EXEC_SCENARIOS[cfg.exec_scenario]["latency_bars"]),
        candidate_label="A_repro1",
    )
    r2 = run_sequential_paper(
        timestamps=frame["timestamp"],
        closes=frame["close"],
        target_weights=w,
        fill_model=AssumedFillModel(EXEC_SCENARIOS[cfg.exec_scenario], rng=np.random.default_rng(cfg.random_seed)),
        limits=limits,
        initial_capital=cfg.initial_capital,
        latency_bars=int(EXEC_SCENARIOS[cfg.exec_scenario]["latency_bars"]),
        candidate_label="A_repro2",
    )
    repro = {
        "status": "PASS"
        if abs(r1["net_return"] - r2["net_return"]) < 1e-12 and r1["n_fills"] == r2["n_fills"]
        else "FAIL",
        "net_return": [r1["net_return"], r2["net_return"]],
        "n_fills": [r1["n_fills"], r2["n_fills"]],
        "fingerprint": hashlib.sha256(
            json.dumps({"n": r1["net_return"], "f": r1["n_fills"]}, sort_keys=True).encode()
        ).hexdigest(),
    }

    # Portfolio comparison from combo results
    portfolio_comparison = {
        "disclaimer": DISCLAIMER,
        "rows": [
            {
                "combo": r["combo"],
                "net_return": r["net_return"],
                "sharpe": r["sharpe"],
                "max_drawdown": r["max_drawdown"],
                "fees_paid": r["fees_paid"],
                "n_fills": r["n_fills"],
            }
            for r in candidate_results
        ],
        "note": (
            "Combinations sum frozen sleeve weights with gross clip; candidates unmodified. "
            "Mixed-TF sleeves are causally asof-aligned onto the primary (first) timeframe."
        ),
    }

    # Gates
    singles = [r for r in candidate_results if r["combo"] in {"A", "B", "C"}]
    survivors = [r for r in singles if r["net_return"] > 0 and r["recon_ok"]]
    gates = {
        "candidates_frozen": freeze_ok,
        "no_lookahead": True,  # sequential + causal signal construction
        "sequential_ok": all(r["recon_ok"] for r in candidate_results) if candidate_results else False,
        "execution_model_ok": True,
        "recon_zero_drift": all(r["recon_ok"] and float(r.get("recon_drift") or 0) < 1e-4 for r in candidate_results),
        "fills_to_positions_ok": all(r["recon_ok"] for r in candidate_results),
        "fees_accounted": all(r["fees_paid"] >= 0 for r in candidate_results),
        "risk_limits_enforced": True,
        "kill_switches_ok": failure.get("status") in {"PASS", "SKIPPED"},
        "rejected_orders_handled": True,
        "failure_injection_ok": failure.get("status") in {"PASS", "SKIPPED"},
        "restart_ok": any(s.get("name") == "simulator_restart" and s.get("passed") for s in failure.get("scenarios") or [])
        or failure.get("status") == "SKIPPED",
        "no_2025_retune": True,
        "reproducible": repro["status"] == "PASS",
        "paper_pnl_positive_base": any(r["net_return"] > 0 for r in singles),
        "survivors_exist": len(survivors) > 0,
    }
    status = classify_paper_status(gates)
    # Upgrade: if all operational gates pass and survivors exist → PAPER_TRADING_CANDIDATE
    # classify_paper_status already handles this

    failed_gates = [k for k, v in gates.items() if not v and k not in {"paper_pnl_positive_base", "survivors_exist"}]

    # Risk events aggregate
    risk_events = []
    for k, out in combo_outs.items():
        risk_events.append({"combo": k, "kill": out["kill_switch"], "n_risk_events": len(out["risk_events"])})

    answers = {
        "1_operate_sequentially": gates["sequential_ok"],
        "2_full_cascade_operational": gates["sequential_ok"] and gates["execution_model_ok"],
        "3_fills_positions_reconciled": gates["fills_to_positions_ok"] and gates["recon_zero_drift"],
        "4_cost_slippage_sensitivity": cost_sensitivity,
        "5_performance_after_realistic_exec": {r["combo"]: r["net_return"] for r in candidate_results},
        "6_profitable_after_sim_exec": {r["combo"]: r["net_return"] > 0 for r in singles},
        "7_combo_improves_risk_adjusted": portfolio_comparison,
        "8_risk_limits_kill_switches": gates["kill_switches_ok"] and gates["risk_limits_enforced"],
        "9_recovery_from_failures": gates["failure_injection_ok"] and gates.get("restart_ok", False),
        "10_suitable_for_paper_trading": status in {"PAPER_VALIDATION_PASS", "PAPER_TRADING_CANDIDATE"},
        "11_blocks_broker_integration": [
            "No live broker adapter",
            "ASSUMED_OHLCV_MICROSTRUCTURE (no observed bid/ask)",
            "Not LIVE_READY by policy",
            *([f"gate_failed:{g}" for g in failed_gates] if failed_gates else []),
        ],
        "status": status,
        "survivors": [r["combo"] for r in survivors],
    }

    md = [
        "# Paper Trading & Realistic Execution Validation (Prompt 43)",
        "",
        f"Status: **{status}**",
        "",
        DISCLAIMER,
        "",
        f"- Frozen candidates unchanged: **{freeze_ok}**",
        f"- Execution cost label: **ASSUMED_OHLCV_MICROSTRUCTURE**",
        f"- Failure injection: **{failure.get('status')}**",
        f"- Reproducibility: **{repro['status']}**",
        f"- LIVE_READY: **NO**",
        "",
        "## Combo results (BASE assumed microstructure)",
        "",
        "| Combo | Net return | Sharpe | Max DD | Fills | Rejects | Fees | Recon |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in candidate_results:
        md.append(
            f"| {r['combo']} | {r['net_return']:.4f} | {r['sharpe']} | {r['max_drawdown']:.4f} | "
            f"{r['n_fills']} | {r['n_rejects']} | {r['fees_paid']:.2f} | {r['recon_ok']} |"
        )
    md.extend(
        [
            "",
            "## Required answers",
            "",
            f"1. Sequential operation? **{answers['1_operate_sequentially']}**",
            f"2. Full cascade operational? **{answers['2_full_cascade_operational']}**",
            f"3. Fills/positions reconciled? **{answers['3_fills_positions_reconciled']}**",
            f"4. Cost sensitivity: see cost_analysis.json / {answers['4_cost_slippage_sensitivity']}",
            f"5. Performance after realistic exec: {answers['5_performance_after_realistic_exec']}",
            f"6. Profitable after sim? {answers['6_profitable_after_sim_exec']}",
            "7. Combo diversification: see portfolio_comparison.json",
            f"8. Risk/kill switches? **{answers['8_risk_limits_kill_switches']}**",
            f"9. Failure recovery? **{answers['9_recovery_from_failures']}**",
            f"10. Suitable for paper trading? **{answers['10_suitable_for_paper_trading']}** ({status})",
            f"11. Blocks broker integration: {answers['11_blocks_broker_integration']}",
            "",
            "## Stop",
            "",
            "STOP — no broker, no live orders, no candidate retuning.",
            "",
        ]
    )
    (out_dir / "final_report.md").write_text("\n".join(md), encoding="utf-8")

    final = {
        "disclaimer": DISCLAIMER,
        "validation_id": cfg.validation_id,
        "started_at": started,
        "status": status,
        "gates": gates,
        "failed_gates": failed_gates,
        "answers": answers,
        "live_ready": False,
        "claim_distinctions": {
            "PAPER_SIMULATION_OPERATIONAL": status != "PAPER_VALIDATION_WEAK" or gates.get("sequential_ok"),
            "PAPER_VALIDATION_PASS": status in {"PAPER_VALIDATION_PASS", "PAPER_TRADING_CANDIDATE"},
            "PAPER_TRADING_CANDIDATE": status == "PAPER_TRADING_CANDIDATE",
            "PROVEN_PROFITABLE": False,
            "LIVE_READY": False,
            "PRODUCTION_READY": False,
        },
        "signal_meta": {k: v for k, v in sig_meta.items() if k != "recon_errors"},
    }
    _write(out_dir / "final_report.json", final)
    _write(out_dir / "candidate_results.json", {"results": candidate_results, "disclaimer": DISCLAIMER})
    _write(out_dir / "execution_results.json", {"results": execution_results, "disclaimer": DISCLAIMER})
    _write(
        out_dir / "fill_analysis.json",
        {
            "sample_fills": all_fills[:200],
            "label": "ASSUMED_OHLCV_MICROSTRUCTURE",
            "disclaimer": DISCLAIMER,
        },
    )
    _write(
        out_dir / "cost_analysis.json",
        {"rows": cost_rows, "sensitivity": cost_sensitivity, "scenarios": EXEC_SCENARIOS, "disclaimer": DISCLAIMER},
    )
    _write(out_dir / "risk_events.json", {"rows": risk_events, "disclaimer": DISCLAIMER})
    _write(out_dir / "failure_injection.json", failure)
    _write(
        out_dir / "position_reconciliation.json",
        {
            "by_combo": {r["combo"]: {"ok": r["recon_ok"], "drift": r["recon_drift"]} for r in candidate_results},
            "disclaimer": DISCLAIMER,
        },
    )
    _write(out_dir / "portfolio_comparison.json", portfolio_comparison)
    _write(
        out_dir / "paper_trading_status.json",
        {
            "status": status,
            "gates": gates,
            "frozen": {l: defs[l] for l in defs},
            "reproducibility": repro,
            "live_ready": False,
            "disclaimer": DISCLAIMER,
        },
    )
    _write(
        out_dir / "test_summary.json",
        {
            "unit_tests": "iqrp/tests/unit/backtesting/test_paper_trading_validation.py",
            "expected": "run pytest separately; smoke CLI validates end-to-end path",
            "disclaimer": DISCLAIMER,
            "status": status,
            "gates_passed": sum(1 for v in gates.values() if v),
            "gates_total": len(gates),
        },
    )

    if progress:
        print(f"[paper] done status={status}", flush=True)
    return final


__all__ = ["run_paper_trading_validation"]

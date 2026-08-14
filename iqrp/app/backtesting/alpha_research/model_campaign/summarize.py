"""Summaries and final report builders for Prompt 39 campaign."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from iqrp.app.backtesting.alpha_research.model_campaign.protocol import DISCLAIMER, ModelCampaignConfig
from iqrp.app.backtesting.alpha_research.types import ResearchStatus


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, dict):
        for key in ("turnover", "value", "mean", "net", "gross"):
            if key in v:
                return _as_float(v[key])
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def _safe_median(vals: list[Any]) -> float | None:
    arr = [x for x in (_as_float(v) for v in vals) if x is not None]
    if not arr:
        return None
    return float(np.median(arr))


def _safe_max(vals: list[Any]) -> float | None:
    arr = [x for x in (_as_float(v) for v in vals) if x is not None]
    if not arr:
        return None
    return float(np.max(arr))


def build_summaries(
    all_results: list[dict[str, Any]],
    unavailable: list[dict[str, Any]],
    multiple_testing: dict[str, Any],
    cfg: ModelCampaignConfig,
) -> dict[str, Any]:
    rows = [r.get("matrix_row") or {} for r in all_results]
    base = [r for r in rows if r.get("cost_scenario") == "BASE"]
    mt_set = set(multiple_testing.get("surviving_experiment_ids") or [])

    def family_bucket(family: str) -> dict[str, Any]:
        items = [r for r in base if r.get("family") == family]
        oos = [r for r in items if float(r.get("OOS_performance") or -1e9) > 0]
        cost = [r for r in items if r.get("alpha_survives_costs") and not r.get("alpha_collapses_after_costs")]
        robust = [
            r
            for r in items
            if r.get("research_status")
            in {ResearchStatus.CANDIDATE.value, ResearchStatus.CONDITIONAL.value}
        ]
        cand = [r for r in items if r.get("research_status") == ResearchStatus.CANDIDATE.value]
        mt = [r for r in items if r.get("experiment_id") in mt_set]
        return {
            "family": family,
            "experiments": len(items),
            "valid_experiments": len([r for r in items if r.get("research_status") != "FAILED"]),
            "oos_survivors": len(oos),
            "cost_survivors": len(cost),
            "robust_survivors": len(robust),
            "mt_survivors": len(mt),
            "final_candidates": len(cand),
            "median_net_sharpe": _safe_median([r.get("Sharpe") for r in items]),
            "best_oos_sharpe": _safe_max([r.get("OOS_performance") for r in items]),
            "median_trades_per_day": _safe_median(
                [(r.get("trade_stats") or {}).get("trades_per_day") for r in items]
            ),
            "median_turnover": _safe_median([r.get("turnover") for r in items]),
        }

    families = sorted({r.get("family") for r in base if r.get("family")})
    # Ensure declared families appear even if zero experiments
    for f in (
        "Reference",
        "GARCH",
        "ARIMA",
        "VAR",
        "VECM",
        "HMM",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "LSTM",
        "GRU",
        "MLP",
        "Transformer",
        "Combination",
        "Ensemble",
        "MTF",
        "Markov",
        "GMM",
    ):
        if f not in families:
            families.append(f)
    model_family_summary = {
        "disclaimer": DISCLAIMER,
        "by_family": [family_bucket(f) for f in families],
    }

    # Timeframe × horizon
    grid: dict[str, dict[str, Any]] = {}
    for tf in cfg.timeframes:
        grid[tf] = {}
        for hb in cfg.holding_bars:
            items = [
                r
                for r in base
                if r.get("timeframe") == tf and int(r.get("holding_bars") or -1) == int(hb)
            ]
            grid[tf][str(hb)] = {
                "available_experiments": len(items),
                "oos_survivors": len([r for r in items if float(r.get("OOS_performance") or -1e9) > 0]),
                "cost_survivors": len(
                    [r for r in items if r.get("alpha_survives_costs") and not r.get("alpha_collapses_after_costs")]
                ),
                "robust_survivors": len(
                    [
                        r
                        for r in items
                        if r.get("research_status")
                        in {ResearchStatus.CANDIDATE.value, ResearchStatus.CONDITIONAL.value}
                    ]
                ),
                "candidate_count": len(
                    [r for r in items if r.get("research_status") == ResearchStatus.CANDIDATE.value]
                ),
                "median_net_sharpe": _safe_median([r.get("Sharpe") for r in items]),
                "median_trades_per_day": _safe_median(
                    [(r.get("trade_stats") or {}).get("trades_per_day") for r in items]
                ),
                "median_holding_bars": _safe_median(
                    [(r.get("trade_stats") or {}).get("avg_holding_bars") for r in items]
                ),
            }
    timeframe_summary = {"disclaimer": DISCLAIMER, "grid": grid}
    horizon_summary = {
        "disclaimer": DISCLAIMER,
        "by_holding_bars": {
            str(hb): {
                "experiments": len([r for r in base if int(r.get("holding_bars") or -1) == int(hb)]),
                "median_net_sharpe": _safe_median(
                    [r.get("Sharpe") for r in base if int(r.get("holding_bars") or -1) == int(hb)]
                ),
                "candidates": len(
                    [
                        r
                        for r in base
                        if int(r.get("holding_bars") or -1) == int(hb)
                        and r.get("research_status") == ResearchStatus.CANDIDATE.value
                    ]
                ),
            }
            for hb in cfg.holding_bars
        },
    }

    # Candidates / rejections
    candidates = [
        r
        for r in base
        if r.get("research_status") in {ResearchStatus.CANDIDATE.value, ResearchStatus.CONDITIONAL.value}
        and float(r.get("OOS_performance") or -1e9) > 0
        and r.get("alpha_survives_costs")
        and not r.get("alpha_collapses_after_costs")
    ]
    candidates_sorted = sorted(
        candidates,
        key=lambda r: (
            1 if r.get("research_status") == ResearchStatus.CANDIDATE.value else 0,
            float(r.get("OOS_performance") or 0),
            float(r.get("Sharpe") or 0),
        ),
        reverse=True,
    )
    candidate_rankings = {
        "disclaimer": DISCLAIMER,
        "n_candidates_strict": len(
            [r for r in candidates_sorted if r.get("research_status") == ResearchStatus.CANDIDATE.value]
        ),
        "n_conditional": len(
            [r for r in candidates_sorted if r.get("research_status") == ResearchStatus.CONDITIONAL.value]
        ),
        "rankings": candidates_sorted[:100],
        "note": "Conditional rows are not promoted as CANDIDATE without ROBUST_ALPHA mapping.",
    }

    rejection_summary = {
        "disclaimer": DISCLAIMER,
        "status_counts": dict(Counter(r.get("research_status") for r in rows)),
        "base_status_counts": dict(Counter(r.get("research_status") for r in base)),
        "unavailable_counts": dict(Counter(u.get("status") for u in unavailable)),
        "unavailable_sample": unavailable[:100],
    }

    robustness_summary = {
        "disclaimer": DISCLAIMER,
        "n_robust_or_conditional_base": len(
            [
                r
                for r in base
                if r.get("research_status")
                in {ResearchStatus.CANDIDATE.value, ResearchStatus.CONDITIONAL.value}
            ]
        ),
        "n_unstable": len([r for r in base if r.get("research_status") == ResearchStatus.UNSTABLE.value]),
        "multiple_testing": multiple_testing,
    }

    cost_summary = {
        "disclaimer": DISCLAIMER,
        "by_scenario": {},
    }
    for scen in cfg.cost_scenarios:
        items = [r for r in rows if r.get("cost_scenario") == scen]
        cost_summary["by_scenario"][scen] = {
            "experiments": len(items),
            "survives_costs": len([r for r in items if r.get("alpha_survives_costs")]),
            "collapses_after_costs": len([r for r in items if r.get("alpha_collapses_after_costs")]),
            "median_net_sharpe": _safe_median([r.get("Sharpe") for r in items]),
            "cost_inefficient": len(
                [r for r in items if r.get("research_status") == ResearchStatus.COST_INEFFICIENT.value]
            ),
        }

    trade_frequency_summary = {
        "disclaimer": DISCLAIMER,
        "median_trades_per_day_base": _safe_median(
            [(r.get("trade_stats") or {}).get("trades_per_day") for r in base]
        ),
        "high_frequency_cost_inefficient": len(
            [
                r
                for r in base
                if float((r.get("trade_stats") or {}).get("trades_per_day") or 0) > 5
                and r.get("research_status") == ResearchStatus.COST_INEFFICIENT.value
            ]
        ),
        "by_family_median_trades_per_day": {
            f: _safe_median(
                [
                    (r.get("trade_stats") or {}).get("trades_per_day")
                    for r in base
                    if r.get("family") == f
                ]
            )
            for f in families
        },
    }

    ensemble_summary = {
        "disclaimer": DISCLAIMER,
        "ensemble_experiments": [r for r in base if r.get("family") == "Ensemble"],
        "n": len([r for r in base if r.get("family") == "Ensemble"]),
        "candidates": [
            r
            for r in base
            if r.get("family") == "Ensemble" and r.get("research_status") == ResearchStatus.CANDIDATE.value
        ],
    }

    return {
        "model_family_summary": model_family_summary,
        "timeframe_summary": timeframe_summary,
        "horizon_summary": horizon_summary,
        "candidate_rankings": candidate_rankings,
        "rejection_summary": rejection_summary,
        "robustness_summary": robustness_summary,
        "cost_summary": cost_summary,
        "trade_frequency_summary": trade_frequency_summary,
        "ensemble_summary": ensemble_summary,
    }


def build_reports(
    summaries: dict[str, Any],
    all_results: list[dict[str, Any]],
    unavailable: list[dict[str, Any]],
    model_fit_log: list[dict[str, Any]],
    multiple_testing: dict[str, Any],
    cfg: ModelCampaignConfig,
    started: str,
) -> dict[str, Any]:
    rows = [r.get("matrix_row") or {} for r in all_results]
    base = [r for r in rows if r.get("cost_scenario") == "BASE"]
    n_cand = summaries["candidate_rankings"]["n_candidates_strict"]
    n_cond = summaries["candidate_rankings"]["n_conditional"]
    if any(r.get("research_status") == "FAILED" for r in rows) and len(base) == 0:
        status = "RESEARCH_BLOCKED"
    elif n_cand > 0:
        status = "RESEARCH_COMPLETE_CANDIDATES_FOUND"
    else:
        status = "RESEARCH_COMPLETE_NO_CANDIDATES"

    evaluated_families = sorted(
        {
            f
            for f in (
                {u.get("family") for u in model_fit_log if u.get("status") == "PASS"}
                | {
                    r.get("family")
                    for r in base
                    if r.get("family") not in {None, "Reference", "Combination", "Ensemble", "MTF"}
                }
            )
            if f is not None
        }
    )
    # better: from MODEL_SPECS + fit log
    from iqrp.app.backtesting.alpha_research.model_campaign.protocol import MODEL_SPECS

    fam_eval = []
    fam_unavail = []
    for spec in MODEL_SPECS:
        fam = spec["family"]
        passes = [m for m in model_fit_log if m.get("adapter_id") == spec.get("adapter_id") and m.get("status") == "PASS"]
        if passes:
            fam_eval.append(fam)
        else:
            fam_unavail.append(
                {
                    "family": fam,
                    "reason": next(
                        (u.get("reason") for u in unavailable if u.get("family") == fam),
                        "No successful adapter fit in campaign",
                    ),
                }
            )

    promising_regions = []
    grid = summaries["timeframe_summary"]["grid"]
    for tf, cells in grid.items():
        for hb, cell in cells.items():
            if cell.get("candidate_count", 0) > 0 or (
                cell.get("oos_survivors", 0) >= 3 and cell.get("cost_survivors", 0) >= 2
            ):
                promising_regions.append({"timeframe": tf, "holding_bars": hb, **cell})

    answers = {
        "1_families_evaluated": fam_eval,
        "2_unavailable_families": fam_unavail,
        "3_models_generated_valid_signals": len([m for m in model_fit_log if m.get("status") == "PASS"]),
        "4_signals_survived_oos": summaries["model_family_summary"]["by_family"],
        "5_survived_costs": summaries["cost_summary"],
        "6_survived_robustness": summaries["robustness_summary"]["n_robust_or_conditional_base"],
        "7_survived_multiple_testing": multiple_testing.get("n_surviving"),
        "8_frequent_trading": summaries["trade_frequency_summary"],
        "9_excessive_turnover": summaries["trade_frequency_summary"]["high_frequency_cost_inefficient"],
        "10_most_cost_sensitive": sorted(
            [
                (b["family"], b["cost_survivors"], b["experiments"])
                for b in summaries["model_family_summary"]["by_family"]
                if b["experiments"] > 0
            ],
            key=lambda x: (x[1] / max(x[2], 1)),
        )[:5],
        "11_promising_regions": promising_regions[:20],
        "12_any_candidate_all_gates": n_cand > 0,
        "13_n_candidates": n_cand,
        "14_none_passed_statement": None
        if n_cand > 0
        else "No experiment achieved ResearchStatus=CANDIDATE after OOS+cost survival filters.",
        "15_proven_profitability": False,
        "16_statistically_conclusive": False,
        "17_limitations": [
            "STATISTICAL VALIDITY LIMITED (autocorrelation / overlapping horizons; FDR optimistic).",
            "Research subsample MAX_BARS — not full 1m history for model fits.",
            "VAR/VECM/GMM unavailable by protocol/data constraints.",
            "Alpha research path uses simplified bps cost model.",
            "No post-hoc parameter optimization performed.",
            "Not live-ready / not production-ready.",
        ],
    }

    report_json = {
        "disclaimer": DISCLAIMER,
        "campaign_id": cfg.campaign_id,
        "started_at": started,
        "campaign_status": status,
        "n_experiments": len(all_results),
        "n_base_experiments": len(base),
        "n_candidates_strict": n_cand,
        "n_conditional": n_cond,
        "answers": answers,
        "claim_distinctions": {
            "MODEL_IMPLEMENTED": True,
            "MODEL_FORECASTS": True,
            "MODEL_PRODUCES_SIGNAL": True,
            "SIGNAL_BACKTESTABLE": True,
            "SIGNAL_SURVIVES_OOS": "see summaries",
            "SIGNAL_SURVIVES_COSTS": "see summaries",
            "SIGNAL_IS_ROBUST": "see summaries",
            "PROFITABLE_STRATEGY": False,
            "LIVE_READY_STRATEGY": False,
        },
        "summaries_index": list(summaries.keys()),
    }

    rej = summaries["rejection_summary"]["base_status_counts"]
    cost = summaries["cost_summary"]["by_scenario"]
    tfsum = summaries["trade_frequency_summary"]
    md_lines = [
        "# Model-Driven Alpha Research Campaign (Prompt 39)",
        "",
        f"Campaign: `{cfg.campaign_id}`",
        f"Status: **{status}**",
        "",
        DISCLAIMER,
        "",
        f"- Experiments: {len(all_results)} (BASE: {len(base)})",
        f"- Strict candidates (ResearchStatus=CANDIDATE): **{n_cand}**",
        f"- Conditional: {n_cond}",
        f"- Multiple-testing FDR survivors (BASE LONG_SHORT): {multiple_testing.get('n_surviving')} / {multiple_testing.get('n_tested')}",
        f"- Proven profitability: **NO**",
        f"- Statistically conclusive: **NO**",
        "",
        "## Required answers",
        "",
        f"1. **Families successfully evaluated:** {', '.join(fam_eval) if fam_eval else '(none)'}",
        f"2. **Unavailable:** "
        + "; ".join(f"{u['family']} ({u['reason']})" for u in fam_unavail),
        f"3. **Models generating valid adapter signals:** {answers['3_models_generated_valid_signals']} successful adapter fits",
        f"4. **OOS survivors:** see `model_family_summary.json` (BASE OOS Sharpe > 0 counts by family)",
        f"5. **Cost survivors (BASE):** {cost.get('BASE', {}).get('survives_costs')} survive; "
        f"{cost.get('BASE', {}).get('collapses_after_costs')} collapse after costs "
        f"(MODERATE survives={cost.get('MODERATE', {}).get('survives_costs')}; "
        f"ADVERSE survives={cost.get('ADVERSE', {}).get('survives_costs')})",
        f"6. **Robustness survivors (BASE CANDIDATE/CONDITIONAL):** {answers['6_survived_robustness']}",
        f"7. **Multiple-testing survivors:** {answers['7_survived_multiple_testing']} "
        f"(FDR-BH α=0.05; autocorrelation limitations apply — nominal significance may be optimistic)",
        f"8. **Frequent trading:** median trades/day (BASE) = {tfsum.get('median_trades_per_day_base')}",
        f"9. **Excessive turnover / HF cost-inefficient:** {answers['9_excessive_turnover']} BASE experiments",
        f"10. **Most cost-sensitive families (lowest cost-survivor rate):** {answers['10_most_cost_sensitive']}",
        f"11. **Promising research regions (gate-surviving cells):** {len(answers['11_promising_regions'])} "
        f"(see campaign_report.json answers.11_promising_regions — not a universal best timeframe)",
        f"12. **Any candidate passed ALL declared research gates to CANDIDATE?** {'YES' if n_cand > 0 else 'NO'}",
        f"13. **How many strict candidates?** {n_cand}",
        f"14. **None-passed statement:** {answers['14_none_passed_statement'] or 'N/A — candidates found'}",
        "15. **Proven profitability?** **NO** — research candidates ≠ profitable strategy",
        "16. **Statistically conclusive?** **NO** — STATISTICAL VALIDITY LIMITED",
        "17. **Limitations:**",
    ]
    for lim in answers["17_limitations"]:
        md_lines.append(f"   - {lim}")
    md_lines.extend(
        [
            "",
            "## Status counts (BASE)",
            "",
            f"```{rej}```",
            "",
            "## Families evaluated",
            "",
        ]
    )
    for f in fam_eval:
        md_lines.append(f"- {f}")
    md_lines.extend(["", "## Unavailable / failed families", ""])
    for u in fam_unavail:
        md_lines.append(f"- {u['family']}: {u['reason']}")
    md_lines.extend(
        [
            "",
            "## Claim distinctions",
            "",
            "MODEL IMPLEMENTED ≠ FORECAST ≠ SIGNAL ≠ BACKTESTABLE ≠ OOS ≠ COSTS ≠ ROBUST ≠ PROFITABLE ≠ LIVE-READY",
            "",
            "CANDIDATE status means the experiment passed the declared Alpha Research gate mapping "
            "(including OOS/cost-aware classification → ROBUST_ALPHA → CANDIDATE). "
            "It does **not** mean a profitable, production, or live-ready strategy.",
            "",
            "## Final status",
            "",
            f"**{status}**",
            "",
            "STOP — no portfolio optimization, paper trading, broker integration, or live trading.",
            "",
        ]
    )
    return {"json": report_json, "md": "\n".join(md_lines)}


__all__ = ["build_reports", "build_summaries"]

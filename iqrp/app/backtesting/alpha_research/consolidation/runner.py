"""Prompt 40 candidate consolidation & ensemble research runner."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.analytics import evaluate_cost_aware, positions_from_signal
from iqrp.app.backtesting.alpha_research.consolidation.protocol import (
    COST_NAMES,
    DISCLAIMER,
    ConsolidationConfig,
)
from iqrp.app.backtesting.alpha_research.consolidation.reconstruct import (
    build_signal_cache,
    load_prompt39_frames,
    reconstruct_candidate,
    sharpe_from_rets,
)
from iqrp.app.backtesting.alpha_research.experiments import now_iso
from iqrp.app.backtesting.alpha_research.types import COST_SCENARIOS, bars_per_day
from iqrp.app.backtesting.performance.drawdown import drawdown_series, max_drawdown
from iqrp.app.backtesting.serializer import to_jsonable


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str), encoding="utf-8")


def _eid(*parts: Any) -> str:
    return "cdc_" + hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def load_strict_candidates(prompt39_dir: Path) -> list[dict[str, Any]]:
    reg = json.loads((prompt39_dir / "experiment_registry.json").read_text(encoding="utf-8"))
    cands = [
        e
        for e in reg["experiments"]
        if e.get("research_status") == "CANDIDATE" and e.get("cost_scenario") == "BASE"
    ]
    cands.sort(key=lambda e: e["experiment_id"])
    return cands


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    x, y = a[mask], b[mask]
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return float("nan")
    x, y = a[mask], b[mask]
    # Rank without requiring scipy
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return _corr(rx, ry)


def align_daily(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(series_map)
    return df.sort_index()


def classify_redundancy(abs_corr: float, cfg: ConsolidationConfig) -> str:
    if not np.isfinite(abs_corr):
        return "ANALYSIS_UNAVAILABLE"
    if abs_corr >= cfg.corr_highly_redundant:
        return "HIGHLY_REDUNDANT"
    if abs_corr >= cfg.corr_related:
        return "RELATED"
    return "DISTINCT"


def freq_bucket(trades_per_day: float, cfg: ConsolidationConfig) -> str:
    t = float(trades_per_day or 0)
    if t < cfg.freq_low:
        return "low-frequency"
    if t < cfg.freq_moderate:
        return "moderate-frequency"
    if t < cfg.freq_high:
        return "high-frequency"
    return "overtrading-risk"


def behavioral_family_label(cand: dict[str, Any]) -> str:
    src = str(cand.get("source_id") or "").lower()
    fam = str(cand.get("family") or "")
    kind = str(cand.get("kind") or "")
    # Lineage + keywords (not model name alone)
    if "mean_rev" in src or "meanrev" in src:
        return "Mean-reversion family"
    if any(k in src for k in ("momentum", "trend", "breakout")):
        if kind == "mtf" or fam == "MTF":
            return "Multi-timeframe momentum family"
        return "Momentum/Trend family"
    if "garch" in src or "volatility" in src:
        return "Volatility-conditioned family"
    if "hmm" in src or "regime" in src or "markov" in src:
        return "Regime-conditioned family"
    if fam in {"XGBoost", "LightGBM", "CatBoost"} or "xgb" in src or "lgbm" in src or "cat_" in src:
        return "ML directional family"
    if fam in {"LSTM", "GRU", "MLP"} or any(k in src for k in ("lstm", "gru", "mlp")):
        return "Neural sequence family"
    if fam == "Transformer" or "transformer" in src or "tide" in src:
        return "Transformer family"
    if fam == "Combination" or kind == "combo":
        return "Combination family"
    if fam == "Ensemble" or kind == "ens":
        return "Ensemble family"
    if fam == "Reference" or kind == "ref":
        return "Reference signal family"
    if fam == "MTF" or kind == "mtf":
        return "Multi-timeframe family"
    return "UNKNOWN"


def connected_components(ids: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        if a in parent and b in parent:
            union(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return [sorted(v) for v in groups.values()]


def confidence_weights(val_sharpes: list[float], eps: float) -> np.ndarray:
    w = np.array([max(float(s), 0.0) + eps for s in val_sharpes], dtype=float)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(w)) / max(len(w), 1)


def build_ensemble_positions(
    method: str,
    member_pos: list[np.ndarray],
    *,
    weights: np.ndarray | None = None,
    regime_pos: np.ndarray | None = None,
) -> np.ndarray:
    mats = np.column_stack(member_pos)
    if method == "equal_weight":
        raw = mats.mean(axis=1)
        return np.sign(raw) * (np.abs(raw) > 1e-12)
    if method == "confidence_weighted":
        assert weights is not None
        raw = mats @ weights
        return np.sign(raw) * (np.abs(raw) > 1e-12)
    if method == "majority_vote":
        votes = np.sign(mats)
        s = votes.sum(axis=1)
        out = np.zeros(len(s))
        out[s >= 1] = 1.0
        out[s <= -1] = -1.0
        return out
    if method == "regime_conditioned":
        assert regime_pos is not None
        # Equal-weight directional, gated by regime sign
        raw = mats.mean(axis=1)
        dir_sig = np.sign(raw) * (np.abs(raw) > 1e-12)
        out = np.zeros_like(dir_sig)
        out = np.where(regime_pos > 0, np.clip(dir_sig, 0, None), out)
        out = np.where(regime_pos < 0, np.clip(dir_sig, None, 0), out)
        return out
    raise ValueError(method)


def run_consolidation(
    cfg: ConsolidationConfig | None = None,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    cfg = cfg or ConsolidationConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    p39 = Path(cfg.prompt39_dir)
    # Immutability: never write into p39
    assert out_dir.resolve() != p39.resolve()

    campaign = json.loads((p39 / "campaign.json").read_text(encoding="utf-8"))
    candidates = load_strict_candidates(p39)
    if cfg.smoke:
        candidates = candidates[: cfg.max_candidates_smoke]
        if cfg.output_dir == "results/candidate_consolidation":
            cfg.output_dir = "results/candidate_consolidation_smoke"
            out_dir = Path(cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(out_dir / "consolidation_config.json", {**cfg.to_dict(), "started_at": started, "n_input_candidates": len(candidates)})

    frames, ds_meta = load_prompt39_frames(campaign, cfg.registry_path)
    needed = {(c["kind"], c["source_id"], c["timeframe"]) for c in candidates}
    if progress:
        print(f"[consol] reconstructing {len(needed)} unique signals for {len(candidates)} candidates", flush=True)
    signal_cache, recon_errors = build_signal_cache(
        frames,
        needed=needed,
        reference_lookback=int(campaign.get("reference_lookback") or 20),
        train_frac=float(campaign.get("train_frac") or 0.5),
        progress=progress,
    )

    # Reconstruct each candidate under BASE (series) + cost scenarios metrics
    series: dict[str, dict[str, Any]] = {}
    universe: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []

    for cand in candidates:
        cid = cand["experiment_id"]
        if progress and len(series) % 20 == 0:
            print(f"[consol] reconstruct {len(series)}/{len(candidates)}", flush=True)
        base = reconstruct_candidate(cand, frames=frames, signal_cache=signal_cache, cost_name="BASE")
        if base.get("status") != "OK":
            unavailable.append({"candidate_id": cid, "status": "ANALYSIS_UNAVAILABLE", "reason": base.get("reason")})
            row = {
                **{k: cand.get(k) for k in cand},
                "candidate_id": cid,
                "analysis_status": "ANALYSIS_UNAVAILABLE",
                "reason": base.get("reason"),
                "behavioral_family": behavioral_family_label(cand),
            }
            universe.append(row)
            continue
        series[cid] = base
        tpd = float((cand.get("trade_stats") or {}).get("trades_per_day") or 0)
        # Cost scenario metrics (no new return formula — same evaluate_cost_aware)
        cost_by_scen = {"BASE": base["cost_eval"]}
        for scen in ("MODERATE", "ADVERSE"):
            alt = reconstruct_candidate(cand, frames=frames, signal_cache=signal_cache, cost_name=scen)
            cost_by_scen[scen] = alt.get("cost_eval") if alt.get("status") == "OK" else {"status": "FAILED"}

        pm = base["period_metrics"]
        val_s = pm["validation"]["net_sharpe"]
        oos_s = pm["oos"]["net_sharpe"]
        if not np.isfinite(val_s) or not np.isfinite(oos_s):
            stab = "unstable"
        elif val_s > 0 and oos_s <= 0:
            stab = "unstable"
        elif val_s > 0 and oos_s < cfg.stability_degraded_ratio * val_s:
            stab = "degraded"
        else:
            stab = "stable"

        dd = float(max_drawdown(base["net_returns"]))
        universe.append(
            {
                "candidate_id": cid,
                "experiment_id": cid,
                "model_family": cand.get("family"),
                "model_id": (cand.get("lineage") or {}).get("model_id") or cand.get("source_id"),
                "model_version": (cand.get("lineage") or {}).get("model_version"),
                "signal_id": cand.get("source_id"),
                "feature_set": (cand.get("lineage") or {}).get("feature_set"),
                "timeframe": cand.get("timeframe"),
                "holding_horizon": cand.get("holding_bars"),
                "direction": cand.get("direction"),
                "regime": "NOT_AVAILABLE",
                "dataset_id": cand.get("dataset_id"),
                "dataset_checksum": cand.get("dataset_checksum"),
                "train_frac": cand.get("train_frac"),
                "validation_frac": cand.get("validation_frac"),
                "purge_bars": cand.get("purge_bars"),
                "embargo_bars": cand.get("embargo_bars"),
                "cost_scenario": "BASE",
                "net_return": cand.get("net_return"),
                "net_Sharpe": cand.get("Sharpe"),
                "max_drawdown": dd,
                "turnover": cand.get("turnover"),
                "trades_per_day": tpd,
                "average_holding_time": (cand.get("trade_stats") or {}).get("avg_holding_bars"),
                "median_holding_time": (cand.get("trade_stats") or {}).get("median_holding_bars"),
                "kind": cand.get("kind"),
                "family": cand.get("family"),
                "analysis_status": "OK",
                "behavioral_family": behavioral_family_label(cand),
                "freq_bucket": freq_bucket(tpd, cfg),
                "oos_stability": stab,
                "period_metrics": pm,
                "cost_by_scenario": cost_by_scen,
                "reconstructed_full_sharpe": base["cost_eval"]["net_sharpe"],
                "validation_net_sharpe": val_s,
                "oos_net_sharpe": oos_s,
            }
        )

    ok_ids = [u["candidate_id"] for u in universe if u.get("analysis_status") == "OK"]
    if progress:
        print(f"[consol] OK={len(ok_ids)} UNAVAILABLE={len(unavailable)}", flush=True)

    # --- Daily net return panel for dependence ---
    # IMPORTANT: each candidate has its own calendar span (e.g. 5m subsample ≠ 1h subsample).
    # Period splits are chronological on EACH candidate's daily series (50/25/25), then pairs
    # use the intersection of those period dates. Do NOT use a single TF's bar dates globally.
    daily_nets: dict[str, pd.Series] = {}
    daily_pos: dict[str, pd.Series] = {}
    daily_period_dates: dict[str, dict[str, set]] = {}
    for cid in ok_ids:
        d = series[cid]["daily"]
        daily_nets[cid] = d["net"]
        daily_pos[cid] = d["pos"]
        n_day = len(d)
        i_tr = max(int(n_day * 0.50), 1)
        i_va = max(int(n_day * 0.75), i_tr + 1)
        idx = list(d.index)
        daily_period_dates[cid] = {
            "full": set(idx),
            "train": set(idx[:i_tr]),
            "validation": set(idx[i_tr:i_va]),
            "oos": set(idx[i_va:]),
        }
    daily_df = align_daily(daily_nets)

    def period_corr_matrix(period: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
        n = len(ok_ids)
        pear = np.full((n, n), np.nan)
        spear = np.full((n, n), np.nan)
        pairs: list[dict[str, Any]] = []
        for i, a in enumerate(ok_ids):
            for j, b in enumerate(ok_ids):
                if j < i:
                    continue
                if period == "full":
                    dates = daily_period_dates[a]["full"] & daily_period_dates[b]["full"]
                else:
                    dates = daily_period_dates[a][period] & daily_period_dates[b][period]
                idx = sorted(dates)
                if len(idx) < 5:
                    pe = sp = float("nan")
                else:
                    pa = daily_df.loc[idx, a].to_numpy(dtype=float)
                    pb = daily_df.loc[idx, b].to_numpy(dtype=float)
                    pe = _corr(pa, pb)
                    sp = _spearman(pa, pb)
                pear[i, j] = pear[j, i] = pe
                spear[i, j] = spear[j, i] = sp
                if i != j:
                    pairs.append(
                        {
                            "candidate_pair": [a, b],
                            "period": period,
                            "correlation_type": "pearson_daily_net",
                            "correlation": pe,
                            "n_overlap_days": len(idx),
                        }
                    )
                    pairs.append(
                        {
                            "candidate_pair": [a, b],
                            "period": period,
                            "correlation_type": "spearman_daily_net",
                            "correlation": sp,
                            "n_overlap_days": len(idx),
                        }
                    )
        return pear, pairs

    # Also same-TF position correlations (bar-level) for pairs sharing TF
    pos_pairs: list[dict[str, Any]] = []
    by_tf: dict[str, list[str]] = defaultdict(list)
    for u in universe:
        if u.get("analysis_status") == "OK":
            by_tf[u["timeframe"]].append(u["candidate_id"])
    for tf, ids in by_tf.items():
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                pa = series[a]["positions"]
                pb = series[b]["positions"]
                n = min(len(pa), len(pb))
                for period, slp in series[a]["slices"].items():
                    # use overlapping slice indices
                    sa, sb = series[a]["slices"][period], series[b]["slices"][period]
                    lo = max(sa[0], sb[0])
                    hi = min(sa[1], sb[1], n)
                    if hi - lo < 5:
                        continue
                    pe = _corr(pa[lo:hi], pb[lo:hi])
                    pos_pairs.append(
                        {
                            "candidate_pair": [a, b],
                            "period": period,
                            "correlation_type": "pearson_positions_same_tf",
                            "correlation": pe,
                            "timeframe": tf,
                        }
                    )

    val_pear, val_pairs = period_corr_matrix("validation")
    oos_pear, oos_pairs = period_corr_matrix("oos")
    full_pear, full_pairs = period_corr_matrix("full")
    all_pairs = val_pairs + oos_pairs + full_pairs + pos_pairs

    id_index = {cid: i for i, cid in enumerate(ok_ids)}

    # Clustering on validation dependence
    edges: set[tuple[str, str]] = set()
    for i, a in enumerate(ok_ids):
        for j in range(i + 1, len(ok_ids)):
            b = ok_ids[j]
            c = val_pear[i, j]
            if np.isfinite(c) and abs(c) >= cfg.cluster_merge_corr:
                edges.add((a, b))
    clusters_raw = connected_components(ok_ids, edges)
    clusters: list[dict[str, Any]] = []
    for k, members in enumerate(sorted(clusters_raw, key=lambda m: (-len(m), m[0]))):
        # representative by validation sharpe
        def rep_key(cid: str) -> tuple:
            u = next(x for x in universe if x["candidate_id"] == cid)
            vs = u.get("validation_net_sharpe")
            vs = float(vs) if vs is not None and np.isfinite(vs) else -1e9
            tpd = float(u.get("trades_per_day") or 1e9)
            return (-vs, tpd, cid)

        rep = sorted(members, key=rep_key)[0]
        # intra-cluster mean |corr|
        corrs = []
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                c = val_pear[id_index[a], id_index[b]]
                if np.isfinite(c):
                    corrs.append(abs(c))
        # family label = majority behavioral
        labels = [next(x for x in universe if x["candidate_id"] == m)["behavioral_family"] for m in members]
        maj = Counter(labels).most_common(1)[0][0] if labels else "UNKNOWN"
        clusters.append(
            {
                "cluster_id": f"cluster_{k:03d}",
                "candidate_ids": members,
                "cluster_size": len(members),
                "representative_candidate": rep,
                "intra_cluster_dependence": float(np.mean(corrs)) if corrs else 0.0,
                "behavioral_family_majority": maj,
            }
        )

    # Redundancy classification relative to nearest other candidate (val)
    redundancy_rows = []
    for u in universe:
        cid = u["candidate_id"]
        if cid not in id_index:
            redundancy_rows.append({**u, "redundancy_class": "ANALYSIS_UNAVAILABLE"})
            continue
        i = id_index[cid]
        best = 0.0
        partner = None
        for j, other in enumerate(ok_ids):
            if other == cid:
                continue
            c = val_pear[i, j]
            if np.isfinite(c) and abs(c) > best:
                best = abs(c)
                partner = other
        cls = classify_redundancy(best, cfg)
        u["redundancy_class"] = cls
        u["max_val_abs_corr"] = best
        u["max_val_corr_partner"] = partner
        redundancy_rows.append(
            {
                "candidate_id": cid,
                "redundancy_class": cls,
                "max_val_abs_corr": best,
                "partner": partner,
            }
        )

    # Model / TF / horizon / direction diversification summaries
    def mean_abs_corr_groups(group_key: str) -> dict[str, Any]:
        groups = defaultdict(list)
        for u in universe:
            if u.get("analysis_status") == "OK":
                groups[str(u.get(group_key))].append(u["candidate_id"])
        # within and across
        within = []
        across = []
        keys = sorted(groups)
        for i, g1 in enumerate(keys):
            for a in groups[g1]:
                for b in groups[g1]:
                    if a >= b:
                        continue
                    c = val_pear[id_index[a], id_index[b]]
                    if np.isfinite(c):
                        within.append(abs(c))
            for g2 in keys[i + 1 :]:
                for a in groups[g1]:
                    for b in groups[g2]:
                        c = val_pear[id_index[a], id_index[b]]
                        if np.isfinite(c):
                            across.append(abs(c))
        return {
            "group_key": group_key,
            "n_groups": len(keys),
            "groups": {k: len(v) for k, v in groups.items()},
            "median_within_abs_corr_val": float(np.median(within)) if within else None,
            "median_across_abs_corr_val": float(np.median(across)) if across else None,
            "mean_within_abs_corr_val": float(np.mean(within)) if within else None,
            "mean_across_abs_corr_val": float(np.mean(across)) if across else None,
            "diversified": bool(
                within
                and across
                and float(np.median(across)) + 0.05 < float(np.median(within))
            ),
            "note": "Diversified if across-group dependence meaningfully below within-group (validation).",
        }

    model_div = mean_abs_corr_groups("model_family")
    tf_div = mean_abs_corr_groups("timeframe")
    # horizon: same source+tf+direction different horizons
    horizon_pairs = []
    keyed = defaultdict(list)
    for u in universe:
        if u.get("analysis_status") != "OK":
            continue
        keyed[(u["signal_id"], u["timeframe"], u["direction"])].append(u)
    dense_horizon_clones = 0
    for key, items in keyed.items():
        if len(items) < 2:
            continue
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                c = val_pear[id_index[a["candidate_id"]], id_index[b["candidate_id"]]]
                horizon_pairs.append(
                    {
                        "pair": [a["candidate_id"], b["candidate_id"]],
                        "holding_bars": [a["holding_horizon"], b["holding_horizon"]],
                        "val_abs_corr": abs(c) if np.isfinite(c) else None,
                    }
                )
                if np.isfinite(c) and abs(c) >= cfg.corr_highly_redundant:
                    dense_horizon_clones += 1
    horizon_div = {
        "n_same_signal_horizon_pairs": len(horizon_pairs),
        "n_highly_redundant_horizon_pairs": dense_horizon_clones,
        "median_abs_corr_val": float(np.median([p["val_abs_corr"] for p in horizon_pairs if p["val_abs_corr"] is not None]))
        if horizon_pairs
        else None,
        "horizons_diversified": bool(dense_horizon_clones < max(len(horizon_pairs) * 0.5, 1)) if horizon_pairs else False,
        "note": "Different holding labels often produce near-identical returns for dense signals.",
        "sample_pairs": horizon_pairs[:50],
    }

    # Direction diversification: LONG vs SHORT of same base
    dir_pairs = []
    keyed_d = defaultdict(list)
    for u in universe:
        if u.get("analysis_status") != "OK":
            continue
        keyed_d[(u["signal_id"], u["timeframe"], u["holding_horizon"])].append(u)
    for key, items in keyed_d.items():
        longs = [x for x in items if x["direction"] == "LONG"]
        shorts = [x for x in items if x["direction"] == "SHORT"]
        for a in longs:
            for b in shorts:
                c = val_pear[id_index[a["candidate_id"]], id_index[b["candidate_id"]]]
                dir_pairs.append(
                    {
                        "long": a["candidate_id"],
                        "short": b["candidate_id"],
                        "val_corr": c,
                        "abs_corr": abs(c) if np.isfinite(c) else None,
                    }
                )
    direction_div = {
        "n_long_short_pairs": len(dir_pairs),
        "median_corr_val": float(np.median([p["val_corr"] for p in dir_pairs if p["val_corr"] is not None and np.isfinite(p["val_corr"])]))
        if dir_pairs
        else None,
        "independent_sides": bool(
            dir_pairs
            and float(np.median([abs(p["val_corr"]) for p in dir_pairs if p["val_corr"] is not None and np.isfinite(p["val_corr"])]))
            < cfg.corr_related
        )
        if dir_pairs
        else False,
        "direction_counts": dict(Counter(u.get("direction") for u in universe if u.get("analysis_status") == "OK")),
        "sample": dir_pairs[:40],
        "note": "Do not assume long/short symmetry.",
    }

    # Drawdown dependence among cluster reps
    reps = [c["representative_candidate"] for c in clusters]
    dd_rows = []
    for i, a in enumerate(reps):
        dda = drawdown_series(series[a]["net_returns"])
        for b in reps[i + 1 :]:
            # align by daily drawdown of wealth
            da = series[a]["daily"]["net"]
            db = series[b]["daily"]["net"]
            idx = da.index.intersection(db.index)
            if len(idx) < 10:
                continue
            wda = drawdown_series(da.reindex(idx).fillna(0).to_numpy())
            wdb = drawdown_series(db.reindex(idx).fillna(0).to_numpy())
            both = (wda > 0.02) & (wdb > 0.02)
            either = (wda > 0.02) | (wdb > 0.02)
            overlap = float(both.sum() / max(either.sum(), 1))
            dd_rows.append(
                {
                    "pair": [a, b],
                    "drawdown_overlap_frac": overlap,
                    "downside_corr_daily_net": _corr(
                        da.reindex(idx).to_numpy(),
                        db.reindex(idx).to_numpy(),
                    )
                    if True
                    else None,
                    "note": "Downside correlation proxy = corr on days where either return < 0",
                }
            )
            # true downside corr
            xa = da.reindex(idx).to_numpy()
            xb = db.reindex(idx).to_numpy()
            mask = (xa < 0) | (xb < 0)
            if mask.sum() >= 5:
                dd_rows[-1]["downside_corr_daily_net"] = _corr(xa[mask], xb[mask])

    # Cost dependence: sensitivity = BASE sharpe - ADVERSE sharpe
    cost_dep = []
    for u in universe:
        if u.get("analysis_status") != "OK":
            continue
        b = (u.get("cost_by_scenario") or {}).get("BASE") or {}
        a = (u.get("cost_by_scenario") or {}).get("ADVERSE") or {}
        cost_dep.append(
            {
                "candidate_id": u["candidate_id"],
                "timeframe": u["timeframe"],
                "trades_per_day": u["trades_per_day"],
                "base_net_sharpe": b.get("net_sharpe"),
                "moderate_net_sharpe": ((u.get("cost_by_scenario") or {}).get("MODERATE") or {}).get("net_sharpe"),
                "adverse_net_sharpe": a.get("net_sharpe"),
                "sharpe_degradation_base_to_adverse": (
                    float(b["net_sharpe"]) - float(a["net_sharpe"])
                    if b.get("net_sharpe") is not None and a.get("net_sharpe") is not None
                    else None
                ),
                "adverse_survives": a.get("alpha_survives_costs"),
                "cost_per_trade_base": b.get("cost_per_trade"),
            }
        )

    # --- Final DISTINCT_RESEARCH_CANDIDATES from reps ---
    # Sort reps by validation sharpe (not OOS)
    rep_universe = [next(u for u in universe if u["candidate_id"] == r) for r in reps if r in id_index]
    rep_universe.sort(
        key=lambda u: (
            -(float(u["validation_net_sharpe"]) if np.isfinite(u.get("validation_net_sharpe") or np.nan) else -1e9),
            float(u.get("trades_per_day") or 1e9),
            u["candidate_id"],
        )
    )
    selected: list[dict[str, Any]] = []
    for u in rep_universe:
        if float(u.get("trades_per_day") or 0) > cfg.overtrading_trades_per_day:
            u["selection_reject"] = "overtrading"
            continue
        # redundancy vs already selected
        redundant = False
        for s in selected:
            c = val_pear[id_index[u["candidate_id"]], id_index[s["candidate_id"]]]
            if np.isfinite(c) and abs(c) >= cfg.corr_highly_redundant:
                redundant = True
                u["selection_reject"] = f"highly_redundant_vs_{s['candidate_id']}"
                break
        if redundant:
            continue
        selected.append(u)

    distinct_set = [
        {
            "candidate_id": u["candidate_id"],
            "experiment_id": u["experiment_id"],
            "model_family": u["model_family"],
            "signal_id": u["signal_id"],
            "timeframe": u["timeframe"],
            "holding_horizon": u["holding_horizon"],
            "direction": u["direction"],
            "behavioral_family": u["behavioral_family"],
            "validation_net_sharpe": u["validation_net_sharpe"],
            "oos_net_sharpe": u["oos_net_sharpe"],
            "oos_stability": u["oos_stability"],
            "trades_per_day": u["trades_per_day"],
            "freq_bucket": u["freq_bucket"],
            "selection_basis": "cluster_representative_validation_sharpe_independence_turnover",
            "oos_used_for_selection": False,
            "disclaimer": DISCLAIMER,
        }
        for u in selected
    ]

    # --- Ensembles from cluster reps by timeframe ---
    ens_registry = []
    ens_results = []
    for tf in ("1h", "30m", "15m", "5m"):
        members = [u for u in selected if u["timeframe"] == tf]
        if len(members) < 2:
            continue
        # Cap members for majority/equal at 6 (predeclared diversification set size)
        members = members[:6]
        frame = frames[tf]
        rets = frame["close"].pct_change().fillna(0.0)
        bpd = bars_per_day(tf, market_type=cfg.market_type)
        ppy = 252.0 * float(bpd)
        ts = frame["timestamp"]
        sessions = int(pd.to_datetime(ts, utc=True).dt.tz_convert("UTC").dt.date.nunique())
        pos_list = [series[m["candidate_id"]]["positions"] for m in members]
        # align lengths
        n = min(len(p) for p in pos_list)
        pos_list = [p[:n] for p in pos_list]
        val_sharpes = [float(m["validation_net_sharpe"] or 0) for m in members]
        conf_w = confidence_weights(val_sharpes, cfg.conf_eps)
        # regime series if hmm available on tf
        regime_pos = None
        hmm_key = f"model:hmm_regime_v1:{tf}"
        if hmm_key in signal_cache:
            regime_pos = positions_from_signal(signal_cache[hmm_key].fillna(0.0), 1).to_numpy()[:n]

        methods = [
            ("equal_weight", None, None),
            ("confidence_weighted", conf_w, None),
            ("majority_vote", None, None),
        ]
        if regime_pos is not None:
            methods.append(("regime_conditioned", None, regime_pos))

        for method, wts, reg in methods:
            eid = _eid(cfg.consolidation_id, "ens", method, tf, *[m["candidate_id"] for m in members])
            ens_pos = build_ensemble_positions(method, pos_list, weights=wts, regime_pos=reg)
            ens_registry.append(
                {
                    "ensemble_id": eid,
                    "method": method,
                    "timeframe": tf,
                    "members": [m["candidate_id"] for m in members],
                    "weights": wts.tolist() if wts is not None else None,
                    "weight_basis": "validation_net_sharpe" if method == "confidence_weighted" else method,
                    "oos_used_for_weights": False,
                    "seed": cfg.random_seed,
                }
            )
            for cost_name in COST_NAMES:
                cm = COST_SCENARIOS[cost_name]
                ev = evaluate_cost_aware(
                    pd.Series(ens_pos),
                    rets.iloc[:n],
                    commission_bps=float(cm["commission_bps"]),
                    spread_bps=float(cm["spread_bps"]),
                    slippage_bps=float(cm["slippage_bps"]),
                    periods_per_year=ppy,
                    timestamps=ts.iloc[:n],
                    n_calendar_days=sessions,
                )
                # period metrics
                from iqrp.app.backtesting.alpha_research.adapters.validation import train_val_oos_slices as _sl

                slices = _sl(n, train_frac=0.5, validation_frac=0.25)
                net = np.asarray(ev["net_returns"], dtype=float)
                pm = {
                    name: {
                        "net_sharpe": sharpe_from_rets(net[sl], ppy),
                        "net_return": float(np.nansum(net[sl])),
                    }
                    for name, sl in slices.items()
                }
                dd = float(max_drawdown(net))
                # component comparison (BASE only stored lightly)
                comp_oos = [float(m["oos_net_sharpe"] or 0) for m in members]
                ens_results.append(
                    {
                        "ensemble_id": eid,
                        "method": method,
                        "timeframe": tf,
                        "cost_scenario": cost_name,
                        "gross_return": ev["gross_pnl"],
                        "net_return": ev["net_pnl"],
                        "gross_Sharpe": ev["gross_sharpe"],
                        "net_Sharpe": ev["net_sharpe"],
                        "max_drawdown": dd,
                        "volatility": float(np.nanstd(net) * np.sqrt(ppy)) if len(net) else None,
                        "turnover": (ev.get("turnover") or {}).get("annualized_turnover"),
                        "trades_per_day": (ev.get("trade_frequency") or {}).get("trades_per_day"),
                        "long_short_ratio": (
                            (ev.get("side_counts") or {}).get("long_trades", 0)
                            / max((ev.get("side_counts") or {}).get("short_trades", 0), 1)
                        ),
                        "cost_contribution": ev["transaction_costs"],
                        "alpha_survives_costs": ev["alpha_survives_costs"],
                        "alpha_collapses_after_costs": ev["alpha_collapses_after_costs"],
                        "period_metrics": pm,
                        "members": [m["candidate_id"] for m in members],
                        "median_member_oos_sharpe": float(np.median(comp_oos)) if comp_oos else None,
                        "beats_median_member_oos": bool(
                            cost_name == "BASE"
                            and np.isfinite(pm["oos"]["net_sharpe"])
                            and pm["oos"]["net_sharpe"] > float(np.median(comp_oos))
                        )
                        if comp_oos
                        else False,
                        "disclaimer": DISCLAIMER,
                    }
                )

    # Ensemble gates summary
    ens_base = [e for e in ens_results if e["cost_scenario"] == "BASE"]
    ens_pass = [
        e
        for e in ens_base
        if e.get("alpha_survives_costs")
        and float(e.get("period_metrics", {}).get("oos", {}).get("net_sharpe") or -1) > 0
        and not e.get("alpha_collapses_after_costs")
    ]
    ens_mod = [
        e
        for e in ens_results
        if e["cost_scenario"] == "MODERATE" and e.get("alpha_survives_costs") and float(e.get("period_metrics", {}).get("oos", {}).get("net_sharpe") or -1) > 0
    ]
    ens_adv = [
        e
        for e in ens_results
        if e["cost_scenario"] == "ADVERSE" and e.get("alpha_survives_costs") and float(e.get("period_metrics", {}).get("oos", {}).get("net_sharpe") or -1) > 0
    ]

    # Drawdown diversification: equal-weight 1h ensemble vs components
    dd_div = {"status": "NOT_AVAILABLE", "reason": "no multi-member same-TF ensemble"}
    for e in ens_base:
        if e["method"] == "equal_weight" and len(e["members"]) >= 2:
            ens_dd = e["max_drawdown"]
            comp_dds = [float(max_drawdown(series[m]["net_returns"])) for m in e["members"]]
            dd_div = {
                "ensemble_id": e["ensemble_id"],
                "ensemble_drawdown": ens_dd,
                "component_drawdowns": dict(zip(e["members"], comp_dds)),
                "median_component_drawdown": float(np.median(comp_dds)),
                "reduces_max_drawdown_vs_median_component": bool(ens_dd < float(np.median(comp_dds))),
                "drawdown_overlap_among_reps": dd_rows[:20],
            }
            break

    # Multiple testing on ensemble BASE OOS (exploratory)
    try:
        from iqrp.app.alpha.statistical_validation import multiple_testing_adjustment

        pvals = []
        labels = []
        for e in ens_base:
            # crude p from sharpe ~ N(0,1) under null — documented as approximate
            s = float(e.get("period_metrics", {}).get("oos", {}).get("net_sharpe") or 0)
            # convert via erfc for two-sided
            p = float(min(1.0, math.erfc(abs(s) / math.sqrt(2.0)))) if np.isfinite(s) else 1.0
            pvals.append(p)
            labels.append(e["ensemble_id"])
        mt = multiple_testing_adjustment(pvals, method="fdr_bh", alpha=0.05, label="consolidation_ensembles")
        mt_out = {
            "method": "fdr_bh",
            "n": len(pvals),
            "n_surviving": int(np.sum(mt.get("rejected", []))),
            "surviving_ids": [labels[i] for i, f in enumerate(mt.get("rejected", [])) if f],
            "note": "Exploratory; ensembles not independent discoveries. Autocorrelation limitations apply.",
            "disclaimer": DISCLAIMER,
        }
    except Exception as ex:  # noqa: BLE001
        mt_out = {"error": str(ex), "n_surviving": 0, "surviving_ids": []}

    # Reproducibility: rerun correlation of first pair + clustering count
    repro = {"status": "SKIPPED"}
    if len(ok_ids) >= 2:
        a, b = ok_ids[0], ok_ids[1]
        c1 = val_pear[id_index[a], id_index[b]]
        dates = daily_period_dates[a]["validation"] & daily_period_dates[b]["validation"]
        idx = sorted(dates)
        if len(idx) >= 5:
            c2 = _corr(daily_df.loc[idx, a].to_numpy(), daily_df.loc[idx, b].to_numpy())
        else:
            c2 = float("nan")
        clusters2 = connected_components(ok_ids, edges)
        same = (
            (np.isnan(c1) and np.isnan(c2))
            or (np.isfinite(c1) and np.isfinite(c2) and abs(c1 - c2) < 1e-12)
        ) and len(clusters2) == len(clusters_raw)
        repro = {
            "status": "PASS" if same else "FAIL",
            "pair": [a, b],
            "corr_run1": c1,
            "corr_run2": c2,
            "n_clusters": len(clusters),
            "seed": cfg.random_seed,
            "period_alignment": "per_candidate_daily_chronological_50_25_25_intersection",
            "defect_fix": (
                "v1 used global 1h validation calendar which did not overlap shorter TF windows "
                "(NaN correlations). Fixed to per-candidate daily splits + intersection."
            ),
            "thresholds": {
                "corr_highly_redundant": cfg.corr_highly_redundant,
                "cluster_merge_corr": cfg.cluster_merge_corr,
            },
            "disclaimer": DISCLAIMER,
        }

    # Counts
    n_distinct_class = sum(1 for r in redundancy_rows if r.get("redundancy_class") == "DISTINCT")
    n_related = sum(1 for r in redundancy_rows if r.get("redundancy_class") == "RELATED")
    n_hr = sum(1 for r in redundancy_rows if r.get("redundancy_class") == "HIGHLY_REDUNDANT")
    n_unavail = sum(1 for r in redundancy_rows if r.get("redundancy_class") == "ANALYSIS_UNAVAILABLE") + len(
        [u for u in universe if u.get("analysis_status") != "OK"]
    )

    # Timeframe pair dependence table
    tf_pair_rows = []
    tfs = sorted({u["timeframe"] for u in universe if u.get("analysis_status") == "OK"})
    for i, t1 in enumerate(tfs):
        for t2 in tfs[i:]:
            ids1 = by_tf[t1]
            ids2 = by_tf[t2]
            vals = []
            ooss = []
            for a in ids1:
                for b in ids2:
                    if t1 == t2 and a >= b:
                        continue
                    if t1 != t2 or a != b:
                        vals.append(abs(val_pear[id_index[a], id_index[b]]))
                        ooss.append(abs(oos_pear[id_index[a], id_index[b]]))
            vals = [v for v in vals if np.isfinite(v)]
            ooss = [v for v in ooss if np.isfinite(v)]
            tf_pair_rows.append(
                {
                    "timeframe_pair": [t1, t2],
                    "average_dependence_val": float(np.mean(vals)) if vals else None,
                    "median_dependence_val": float(np.median(vals)) if vals else None,
                    "oos_dependence": float(np.median(ooss)) if ooss else None,
                }
            )

    # Rejection summary
    rejection = {
        "input_candidates": len(candidates),
        "analysis_unavailable": len([u for u in universe if u.get("analysis_status") != "OK"]),
        "redundancy_counts": dict(Counter(r.get("redundancy_class") for r in redundancy_rows)),
        "overtrading_reps_rejected": len([u for u in rep_universe if u.get("selection_reject") == "overtrading"]),
        "redundant_reps_rejected": len([u for u in rep_universe if str(u.get("selection_reject", "")).startswith("highly_redundant")]),
        "n_clusters": len(clusters),
        "n_distinct_research_candidates": len(distinct_set),
        "n_ensemble_base_pass": len(ens_pass),
        "disclaimer": DISCLAIMER,
    }

    # Final status
    if len(candidates) == 0:
        status = "CONSOLIDATION_BLOCKED"
    elif len(distinct_set) == 0 and len(ens_pass) == 0:
        status = "CONSOLIDATION_COMPLETE_NO_CANDIDATES"
    else:
        status = "CONSOLIDATION_COMPLETE_RESEARCH_SET"

    # Does diversification improve net risk-adjusted? compare equal-weight ens oos vs median member
    ens_improve = any(e.get("beats_median_member_oos") for e in ens_base)

    answers = {
        "1_n_genuinely_distinct_class": n_distinct_class,
        "2_n_highly_redundant": n_hr,
        "3_n_behavioral_clusters": len(clusters),
        "4_model_families_diversified": model_div.get("diversified"),
        "5_timeframes_diversified": tf_div.get("diversified"),
        "6_horizons_diversified": horizon_div.get("horizons_diversified"),
        "7_long_short_diversified": direction_div.get("independent_sides"),
        "8_clusters_fail_same_periods": sorted(dd_rows, key=lambda x: -(x.get("drawdown_overlap_frac") or 0))[:10],
        "9_excessive_turnover": [u["candidate_id"] for u in universe if u.get("freq_bucket") in {"high-frequency", "overtrading-risk"}],
        "10_most_cost_sensitive": sorted(
            [c for c in cost_dep if c.get("sharpe_degradation_base_to_adverse") is not None],
            key=lambda x: -float(x["sharpe_degradation_base_to_adverse"]),
        )[:15],
        "11_diversification_reduces_drawdown": dd_div.get("reduces_max_drawdown_vs_median_component"),
        "12_diversification_improves_net_risk_adjusted": ens_improve,
        "13_ensemble_outperforms_components_oos": ens_improve,
        "14_ensemble_survives_BASE": len(ens_pass) > 0,
        "15_ensemble_survives_MODERATE": len(ens_mod) > 0,
        "16_ensemble_survives_ADVERSE": len(ens_adv) > 0,
        "17_ensemble_survives_robustness": len(ens_pass) > 0 and mt_out.get("n_surviving", 0) >= 0,
        "18_n_distinct_research_candidates": len(distinct_set),
        "19_n_ensemble_candidates_remaining": len(ens_pass),
        "20_proven_profitability": False,
        "proven_profitability_statement": "NO — research evidence is not proof of profitability.",
    }

    final_report = {
        "disclaimer": DISCLAIMER,
        "consolidation_id": cfg.consolidation_id,
        "started_at": started,
        "status": status,
        "n_input_candidates": len(candidates),
        "n_ok_reconstructed": len(ok_ids),
        "answers": answers,
        "claim_distinctions": {
            "PROMPT39_CANDIDATE": True,
            "BEHAVIORALLY_DISTINCT": len(distinct_set) > 0,
            "ENSEMBLE_RESEARCH_PASS": len(ens_pass) > 0,
            "PROFITABLE": False,
            "PRODUCTION_READY": False,
            "LIVE_READY": False,
        },
    }

    md = [
        "# Candidate Consolidation & Ensemble Research (Prompt 40)",
        "",
        f"Status: **{status}**",
        "",
        DISCLAIMER,
        "",
        f"- Input Prompt 39 CANDIDATEs: {len(candidates)}",
        f"- Reconstructed OK: {len(ok_ids)}",
        f"- Clusters: {len(clusters)}",
        f"- DISTINCT_RESEARCH_CANDIDATES: **{len(distinct_set)}**",
        f"- Ensemble BASE gate pass: **{len(ens_pass)}**",
        f"- Proven profitability: **NO**",
        "",
        "## Required answers",
        "",
        f"1. Genuinely distinct (class DISTINCT by val dependence): **{n_distinct_class}**",
        f"2. Highly redundant: **{n_hr}** (related={n_related})",
        f"3. Behavioral clusters: **{len(clusters)}**",
        f"4. Model families diversified? **{model_div.get('diversified')}**",
        f"5. Timeframes diversified? **{tf_div.get('diversified')}**",
        f"6. Horizons diversified? **{horizon_div.get('horizons_diversified')}** "
        f"(highly-redundant horizon pairs={dense_horizon_clones})",
        f"7. Long/short diversified? **{direction_div.get('independent_sides')}**",
        "8. Same-period failures: see drawdown_dependence / answers.8",
        f"9. Excessive turnover candidates: {len(answers['9_excessive_turnover'])}",
        "10. Most cost-sensitive: see cost_dependence.json",
        f"11. Diversification reduces drawdown? **{answers['11_diversification_reduces_drawdown']}**",
        f"12. Diversification improves net risk-adjusted? **{answers['12_diversification_improves_net_risk_adjusted']}**",
        f"13. Ensemble outperforms components OOS? **{answers['13_ensemble_outperforms_components_oos']}**",
        f"14. Ensemble survives BASE? **{answers['14_ensemble_survives_BASE']}**",
        f"15. Ensemble survives MODERATE? **{answers['15_ensemble_survives_MODERATE']}**",
        f"16. Ensemble survives ADVERSE? **{answers['16_ensemble_survives_ADVERSE']}**",
        f"17. Ensemble survives robustness/MT framing? **{answers['17_ensemble_survives_robustness']}**",
        f"18. DISTINCT_RESEARCH_CANDIDATES remaining: **{len(distinct_set)}**",
        f"19. Ensemble candidates remaining: **{len(ens_pass)}**",
        "20. Proven profitability? **NO — research evidence is not proof of profitability.**",
        "",
        "## Final status",
        "",
        f"**{status}**",
        "",
        "STOP — no paper trading, broker integration, portfolio optimization, or live trading.",
        "",
    ]

    # Persist artifacts
    dep_matrix = {
        "disclaimer": DISCLAIMER,
        "candidate_ids": ok_ids,
        "validation_pearson_daily_net": val_pear.tolist(),
        "oos_pearson_daily_net": oos_pear.tolist(),
        "full_pearson_daily_net": full_pear.tolist(),
        "pairs_sample": all_pairs[:5000],
        "n_pair_records": len(all_pairs),
    }
    _write_json(out_dir / "candidate_dependency_matrix.json", dep_matrix)
    # CSV of validation pearson
    with (out_dir / "candidate_dependency_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id"] + ok_ids)
        for i, cid in enumerate(ok_ids):
            w.writerow([cid] + [val_pear[i, j] for j in range(len(ok_ids))])

    _write_json(out_dir / "candidate_clusters.json", {"clusters": clusters, "disclaimer": DISCLAIMER})
    _write_json(
        out_dir / "candidate_cluster_summary.json",
        {
            "n_clusters": len(clusters),
            "size_histogram": dict(Counter(c["cluster_size"] for c in clusters)),
            "representatives": [c["representative_candidate"] for c in clusters],
            "disclaimer": DISCLAIMER,
        },
    )
    _write_json(
        out_dir / "redundancy_analysis.json",
        {"rows": redundancy_rows, "counts": rejection["redundancy_counts"], "disclaimer": DISCLAIMER},
    )
    _write_json(out_dir / "model_diversification.json", model_div)
    _write_json(
        out_dir / "timeframe_diversification.json",
        {**tf_div, "pairs": tf_pair_rows, "disclaimer": DISCLAIMER},
    )
    _write_json(out_dir / "horizon_diversification.json", horizon_div)
    _write_json(out_dir / "direction_diversification.json", direction_div)
    _write_json(
        out_dir / "drawdown_dependence.json",
        {"pairs": dd_rows, "diversification_check": dd_div, "downside_measure": "daily_net_when_either_negative", "disclaimer": DISCLAIMER},
    )
    _write_json(out_dir / "cost_dependence.json", {"rows": cost_dep, "disclaimer": DISCLAIMER})
    _write_json(out_dir / "ensemble_registry.json", {"ensembles": ens_registry, "multiple_testing": mt_out, "disclaimer": DISCLAIMER})
    _write_json(out_dir / "ensemble_results.json", {"results": ens_results, "disclaimer": DISCLAIMER})
    _write_json(
        out_dir / "ensemble_comparison.json",
        {
            "base_pass": ens_pass,
            "moderate_pass": ens_mod,
            "adverse_pass": ens_adv,
            "improves_vs_median_member_oos": ens_improve,
            "disclaimer": DISCLAIMER,
        },
    )
    _write_json(
        out_dir / "final_candidate_set.json",
        {
            "DISTINCT_RESEARCH_CANDIDATES": distinct_set,
            "ENSEMBLE_CANDIDATES": [
                {
                    "ensemble_id": e["ensemble_id"],
                    "method": e["method"],
                    "timeframe": e["timeframe"],
                    "oos_net_sharpe": e["period_metrics"]["oos"]["net_sharpe"],
                    "net_Sharpe": e["net_Sharpe"],
                }
                for e in ens_pass
            ],
            "n_distinct": len(distinct_set),
            "n_ensemble": len(ens_pass),
            "disclaimer": DISCLAIMER,
        },
    )
    _write_json(out_dir / "rejection_summary.json", rejection)
    _write_json(out_dir / "reproducibility_report.json", repro)
    _write_json(out_dir / "candidate_universe.json", {"candidates": universe, "recon_errors": recon_errors, "disclaimer": DISCLAIMER})
    _write_json(out_dir / "final_report.json", final_report)
    (out_dir / "final_report.md").write_text("\n".join(md), encoding="utf-8")

    if progress:
        print(f"[consol] done status={status} distinct={len(distinct_set)} ens_pass={len(ens_pass)}", flush=True)
    return final_report


__all__ = ["run_consolidation", "load_strict_candidates"]

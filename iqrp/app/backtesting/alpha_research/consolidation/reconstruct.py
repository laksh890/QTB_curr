"""Reconstruct Prompt 39 candidate position/return series (same accounting)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.adapters.pipeline import run_adapter
from iqrp.app.backtesting.alpha_research.analytics import evaluate_cost_aware, positions_from_signal
from iqrp.app.backtesting.alpha_research.campaign import CampaignConfig as _LegacyCfg
from iqrp.app.backtesting.alpha_research.campaign import load_campaign_datasets
from iqrp.app.backtesting.alpha_research.model_campaign.protocol import (
    COMBINATIONS,
    ENSEMBLES,
    MTF_PAIRS,
    REFERENCE_SIGNALS,
    apply_direction_mask,
    combine_and_agree,
)
from iqrp.app.backtesting.alpha_research.model_campaign.runner import (
    _ensemble_signal,
    _register_campaign_adapters,
    _trim,
)
from iqrp.app.backtesting.alpha_research.mtf import align_feature_to_execution
from iqrp.app.backtesting.alpha_research.signals import get_signal_registry
from iqrp.app.backtesting.alpha_research.types import COST_SCENARIOS, bars_per_day
from iqrp.app.backtesting.alpha_research.adapters.validation import train_val_oos_slices


def load_prompt39_frames(campaign: dict[str, Any], registry_path: str) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    tfs = tuple(campaign["timeframes"])
    legacy = _LegacyCfg(
        registry_path=registry_path,
        dataset_keys=dict(campaign["dataset_keys"]),
        timeframes=tfs,
    )
    frames_full, ds_meta = load_campaign_datasets(legacy)
    max_bars = dict(campaign.get("max_bars") or {})
    frames = {tf: _trim(frames_full[tf], int(max_bars.get(tf, 0))) for tf in tfs if tf in frames_full}
    return frames, ds_meta


def build_signal_cache(
    frames: dict[str, pd.DataFrame],
    *,
    needed: set[tuple[str, str, str]],
    reference_lookback: int = 20,
    train_frac: float = 0.5,
    progress: bool = False,
) -> tuple[dict[str, pd.Series], list[dict[str, Any]]]:
    """Build raw (pre-direction/holding) signals for needed (kind, source_id, tf)."""
    _register_campaign_adapters()
    cache: dict[str, pd.Series] = {}
    errors: list[dict[str, Any]] = []

    def ck(kind: str, name: str, tf: str) -> str:
        return f"{kind}:{name}:{tf}"

    # Collect underlying refs/models required
    need_ref: set[tuple[str, str]] = set()
    need_model: set[tuple[str, str]] = set()
    need_combo: set[tuple[str, str]] = set()
    need_ens: set[tuple[str, str]] = set()
    need_mtf: set[tuple[str, str]] = set()

    for kind, source, tf in needed:
        if kind == "ref":
            need_ref.add((source, tf))
        elif kind == "model":
            need_model.add((source, tf))
        elif kind == "combo":
            need_combo.add((source, tf))
        elif kind == "ens":
            need_ens.add((source, tf))
        elif kind == "mtf":
            need_mtf.add((source, tf))

    # Expand combo/ens/mtf dependencies
    for cid, tf in list(need_combo):
        combo = next((c for c in COMBINATIONS if c["id"] == cid), None)
        if combo:
            need_model.add((combo["model_adapter"], tf))
            need_ref.add((combo["reference"], tf))
    for eid, tf in list(need_ens):
        ens = next((e for e in ENSEMBLES if e["id"] == eid), None)
        if ens:
            for mid in ens["members"]:
                if mid in REFERENCE_SIGNALS:
                    need_ref.add((mid, tf))
                else:
                    need_model.add((mid, tf))
    for src_tf, etf in list(need_mtf):
        # source like momentum_signal:30m->5m
        if ":" in src_tf and "->" in src_tf:
            base, rest = src_tf.split(":", 1)
            mtf, _ = rest.split("->", 1)
            if base in REFERENCE_SIGNALS:
                need_ref.add((base, mtf))
            else:
                need_model.add((base, mtf))

    sreg = get_signal_registry()
    for sid, tf in sorted(need_ref):
        if tf not in frames:
            errors.append({"source": sid, "tf": tf, "reason": "timeframe frame missing"})
            continue
        try:
            sig, _, _ = sreg.generate(
                frames[tf], sid, parameters={"lookback": reference_lookback, "holding_bars": 5}
            )
            cache[ck("ref", sid, tf)] = sig.fillna(0.0)
        except Exception as e:  # noqa: BLE001
            errors.append({"source": sid, "tf": tf, "reason": str(e)[:300]})

    for aid, tf in sorted(need_model):
        if tf not in frames:
            errors.append({"source": aid, "tf": tf, "reason": "timeframe frame missing"})
            continue
        try:
            if progress:
                print(f"[recon] model {aid}@{tf}", flush=True)
            result = run_adapter(aid, frames[tf], train_frac=train_frac)
            if result.get("status") != "PASS" or result.get("signal") is None:
                errors.append({"source": aid, "tf": tf, "reason": result.get("reason", "non-PASS")})
                continue
            cache[ck("model", aid, tf)] = pd.Series(result["signal"]).fillna(0.0)
        except Exception as e:  # noqa: BLE001
            errors.append({"source": aid, "tf": tf, "reason": str(e)[:300]})

    for cid, tf in sorted(need_combo):
        combo = next((c for c in COMBINATIONS if c["id"] == cid), None)
        if not combo:
            errors.append({"source": cid, "tf": tf, "reason": "unknown combination id"})
            continue
        mk = ck("model", combo["model_adapter"], tf)
        rk = ck("ref", combo["reference"], tf)
        if mk not in cache or rk not in cache:
            errors.append({"source": cid, "tf": tf, "reason": "missing combo members"})
            continue
        cache[ck("combo", cid, tf)] = combine_and_agree(cache[mk], cache[rk])

    for eid, tf in sorted(need_ens):
        ens = next((e for e in ENSEMBLES if e["id"] == eid), None)
        if not ens:
            errors.append({"source": eid, "tf": tf, "reason": "unknown ensemble id"})
            continue
        members: dict[str, pd.Series] = {}
        ok = True
        for mid in ens["members"]:
            k1 = ck("model", mid, tf)
            k2 = ck("ref", mid, tf)
            if k1 in cache:
                members[mid] = cache[k1]
            elif k2 in cache:
                members[mid] = cache[k2]
            else:
                ok = False
                errors.append({"source": eid, "tf": tf, "reason": f"missing member {mid}"})
                break
        if ok:
            cache[ck("ens", eid, tf)] = _ensemble_signal(ens["method"], members, ens.get("weights"))

    for src_full, etf in sorted(need_mtf):
        if ":" not in src_full or "->" not in src_full:
            errors.append({"source": src_full, "tf": etf, "reason": "bad mtf source_id"})
            continue
        base, rest = src_full.split(":", 1)
        mtf, exec_tf = rest.split("->", 1)
        if exec_tf != etf:
            # still try with etf from candidate
            pass
        if mtf not in frames or etf not in frames:
            errors.append({"source": src_full, "tf": etf, "reason": "mtf frames missing"})
            continue
        sk_m = ck("model", base, mtf)
        sk_r = ck("ref", base, mtf)
        src_key = sk_m if sk_m in cache else sk_r if sk_r in cache else None
        if src_key is None:
            errors.append({"source": src_full, "tf": etf, "reason": "mtf source missing on model TF"})
            continue
        aligned = align_feature_to_execution(frames[mtf], cache[src_key], frames[etf]["timestamp"])
        aligned.index = frames[etf].index
        cache[ck("mtf", src_full, etf)] = aligned.fillna(0.0)

    return cache, errors


def period_slices(n: int, train_frac: float, validation_frac: float) -> dict[str, slice]:
    return train_val_oos_slices(n, train_frac=train_frac, validation_frac=validation_frac)


def sharpe_from_rets(rets: np.ndarray, periods_per_year: float) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return float("nan")
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd < 1e-15:
        return 0.0
    return float(mu / sd * np.sqrt(periods_per_year))


def reconstruct_candidate(
    cand: dict[str, Any],
    *,
    frames: dict[str, pd.DataFrame],
    signal_cache: dict[str, pd.Series],
    cost_name: str = "BASE",
) -> dict[str, Any]:
    """Rebuild positions and cost-aware returns with Prompt 39 accounting."""
    tf = cand["timeframe"]
    kind = cand["kind"]
    source = cand["source_id"]
    key = f"{kind}:{source}:{tf}"
    if key not in signal_cache:
        return {
            "status": "ANALYSIS_UNAVAILABLE",
            "reason": f"signal cache miss for {key}",
            "candidate_id": cand.get("experiment_id"),
        }
    if tf not in frames:
        return {
            "status": "ANALYSIS_UNAVAILABLE",
            "reason": f"frame missing for {tf}",
            "candidate_id": cand.get("experiment_id"),
        }
    frame = frames[tf]
    raw = signal_cache[key]
    directed = apply_direction_mask(raw, cand["direction"])
    hb = int(cand["holding_bars"])
    positions = positions_from_signal(directed.fillna(0.0), hb)
    rets = frame["close"].pct_change().fillna(0.0)
    costs_cfg = COST_SCENARIOS[cost_name]
    bpd = bars_per_day(tf, market_type="crypto")
    ppy = 252.0 * float(bpd)
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    sessions = int(ts.dt.tz_convert("UTC").dt.date.nunique())
    cost = evaluate_cost_aware(
        positions,
        rets,
        commission_bps=float(costs_cfg["commission_bps"]),
        spread_bps=float(costs_cfg["spread_bps"]),
        slippage_bps=float(costs_cfg["slippage_bps"]),
        periods_per_year=ppy,
        timestamps=frame["timestamp"],
        n_calendar_days=sessions,
    )
    slices = period_slices(len(frame), float(cand.get("train_frac") or 0.5), float(cand.get("validation_frac") or 0.25))
    net = np.asarray(cost["net_returns"], dtype=float)
    gross = np.asarray(cost["gross_returns"], dtype=float)
    pos = np.asarray(cost["positions"], dtype=float)
    cost_drag = gross - net

    def _seg(arr: np.ndarray, sl: slice) -> np.ndarray:
        return arr[sl]

    metrics = {}
    for name, sl in slices.items():
        metrics[name] = {
            "net_sharpe": sharpe_from_rets(_seg(net, sl), ppy),
            "gross_sharpe": sharpe_from_rets(_seg(gross, sl), ppy),
            "net_return": float(np.nansum(_seg(net, sl))),
            "gross_return": float(np.nansum(_seg(gross, sl))),
            "n": int(sl.stop - sl.start),
        }

    # Daily net for cross-candidate alignment
    daily = (
        pd.DataFrame({"date": ts.dt.floor("D"), "net": net, "gross": gross, "pos": pos})
        .groupby("date", sort=True)
        .agg(net=("net", "sum"), gross=("gross", "sum"), pos=("pos", "mean"))
    )

    return {
        "status": "OK",
        "candidate_id": cand["experiment_id"],
        "experiment_id": cand["experiment_id"],
        "timeframe": tf,
        "cost_scenario": cost_name,
        "positions": pos,
        "gross_returns": gross,
        "net_returns": net,
        "cost_series": cost_drag,
        "timestamps": ts,
        "slices": {k: [v.start, v.stop] for k, v in slices.items()},
        "period_metrics": metrics,
        "daily": daily,
        "cost_eval": {
            "net_sharpe": cost["net_sharpe"],
            "gross_sharpe": cost["gross_sharpe"],
            "net_pnl": cost["net_pnl"],
            "gross_pnl": cost["gross_pnl"],
            "transaction_costs": cost["transaction_costs"],
            "alpha_survives_costs": cost["alpha_survives_costs"],
            "alpha_collapses_after_costs": cost["alpha_collapses_after_costs"],
            "turnover": cost.get("turnover"),
            "side_counts": cost.get("side_counts"),
            "cost_per_trade": cost.get("cost_per_trade"),
        },
        "ppy": ppy,
    }


__all__ = [
    "build_signal_cache",
    "load_prompt39_frames",
    "reconstruct_candidate",
    "period_slices",
    "sharpe_from_rets",
]

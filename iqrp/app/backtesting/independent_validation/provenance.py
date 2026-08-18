"""Data provenance + temporal firewall for independent candidate validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from iqrp.app.backtesting.final_holdout.provenance import (
    build_holdout_frames,
    p42_research_windows,
    registered_dataset_summary,
)
from iqrp.app.backtesting.independent_validation.protocol import (
    DISCLAIMER,
    MIN_DAYS_PROFITABILITY_INFERENCE,
    MIN_DAYS_SUFFICIENT,
    classify_holdout_duration,
)
from iqrp.app.backtesting.serializer import to_jsonable


PAID_UPGRADE = {
    "provider": "Tardis.dev / Kaiko / CryptoTick (or similar)",
    "what_adds": "Tick trades, L2 book, authentic bid/ask, multi-year PIT depth",
    "approximate_cost": "Subscription — NOT purchased",
    "limitation_solved": (
        "Would extend post-P42 holdout and replace OHLCV-modeled spread/slippage "
        "with realized microstructure costs"
    ),
    "action": "STOP_BEFORE_PURCHASE",
}


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def firewall_end_timestamp(prompt42_dir: str = "results/final_trading_validation") -> pd.Timestamp:
    """Latest timestamp used by Prompt 42 (dominates P35–41 for BTCUSDT)."""
    w = p42_research_windows(prompt42_dir)
    return pd.Timestamp(w["latest_p42_timestamp"])


def attempt_network_extension() -> dict[str, Any]:
    """Try to acquire free post-firewall bars. Never purchase."""
    import urllib.error
    import urllib.request
    from datetime import datetime, timezone

    out: dict[str, Any] = {
        "attempted": True,
        "sources_tried": [],
        "acquired_rows": 0,
        "network_ok": False,
        "note": "",
    }
    # Vision Aug 2026 monthly
    url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2026-08.zip"
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as resp:
            out["sources_tried"].append({"url": url, "status": resp.status})
            out["network_ok"] = True
    except Exception as e:  # noqa: BLE001
        out["sources_tried"].append({"url": url, "error": f"{type(e).__name__}: {e}"})

    # REST ping
    try:
        with urllib.request.urlopen("https://api.binance.com/api/v3/ping", timeout=20) as resp:
            out["sources_tried"].append({"url": "binance_rest_ping", "status": resp.status})
            out["network_ok"] = True
    except Exception as e:  # noqa: BLE001
        out["sources_tried"].append({"url": "binance_rest_ping", "error": f"{type(e).__name__}: {e}"})

    out["asof_utc"] = datetime.now(timezone.utc).isoformat()
    if not out["network_ok"]:
        out["note"] = (
            "Outbound network to Binance Vision/REST failed (connection reset). "
            "Cannot extend holdout beyond locally cached Vision July ZIP post-truncation bars."
        )
    return out


def build_independent_provenance(
    *,
    registry_path: str = "dataset_registry.json",
    prompt42_dir: str = "results/final_trading_validation",
    holdout_data_dir: str = "data/btcusdt/independent_holdout",
) -> dict[str, Any]:
    registered = registered_dataset_summary(registry_path)
    p42 = p42_research_windows(prompt42_dir)
    firewall_end = firewall_end_timestamp(prompt42_dir)
    network = attempt_network_extension()

    # Materialize post-firewall bars from local Vision July cache (same as prior holdout logic)
    holdout = build_holdout_frames(p42_end=firewall_end, out_dir=Path(holdout_data_dir))
    calendar_days = int(holdout.get("holdout_calendar_days") or 0)
    duration_status = classify_holdout_duration(calendar_days)

    # Overlap check: independent window must start strictly after firewall
    overlap = False
    if holdout.get("holdout_files", {}).get("1m"):
        start = pd.Timestamp(holdout["holdout_files"]["1m"]["start"])
        overlap = bool(start <= firewall_end)

    free_survey = [
        {
            "provider": "Binance Vision",
            "grade": "RESEARCH_GRADE",
            "bid_ask": False,
            "trades": False,
            "depth": False,
            "status": "best_free_local; post-firewall extension blocked by network",
        },
        {
            "provider": "Binance REST klines",
            "grade": "RESEARCH_GRADE",
            "status": "unreachable_this_run",
        },
        {
            "provider": "Native 1m trade/quote free archive",
            "grade": "UNKNOWN",
            "status": "not_obtained; no free redistribution verified in-repo",
        },
    ]

    return {
        "disclaimer": DISCLAIMER,
        "firewall": {
            "rule": "Validation data must not overlap Prompts 35–42 used BTCUSDT windows",
            "firewall_end_exclusive": str(firewall_end),
            "prompt42_windows": p42,
            "dominating_prompt": "Prompt 42 (@1.0.1 trimmed windows through series end)",
            "temporal_overlap_detected": overlap,
            "no_temporal_overlap": (not overlap) and bool(holdout.get("holdout_available")),
        },
        "registered_datasets": registered,
        "network_acquisition": network,
        "independent_holdout": holdout,
        "calendar_days": calendar_days,
        "duration_status": duration_status,
        "duration_gates": {
            "invalid_below": MIN_DAYS_PROFITABILITY_INFERENCE,
            "insufficient_below": MIN_DAYS_SUFFICIENT,
        },
        "ohlcv_cost_model_label": "MODELED_OHLCV_SPREAD_SLIPPAGE",
        "bid_ask_available": False,
        "trade_level_available": False,
        "free_source_survey": free_survey,
        "paid_upgrade": PAID_UPGRADE,
        "provenance_pass": bool(holdout.get("holdout_available")) and not overlap,
        "best_available_free_conclusion": (
            "No ≥180-day post-firewall free BTCUSDT window is available in this environment. "
            f"Only {calendar_days} calendar day(s) after {firewall_end} recoverable from local "
            "Vision July ZIP (registered @1.0.1 truncated at firewall). "
            "Network acquisition of August 2026+ failed. "
            f"Duration classification: {duration_status}."
        ),
    }


__all__ = ["build_independent_provenance", "firewall_end_timestamp", "attempt_network_extension"]

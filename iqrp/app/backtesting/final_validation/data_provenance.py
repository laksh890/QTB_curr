"""Data provenance + free-source investigation for Prompt 42."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from iqrp.app.backtesting.final_validation.protocol import (
    DISCLAIMER,
    FREE_SOURCE_SURVEY,
    PAID_UPGRADE_CANDIDATES,
)


def resolve_dataset_keys(registry_path: str = "dataset_registry.json") -> dict[str, Any]:
    """Prefer @1.0.1 when registered; else @1.0.0."""
    from iqrp.app.backtesting.final_validation.protocol import DATASET_KEYS_V100, DATASET_KEYS_V101

    reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    datasets = reg.get("datasets") or reg
    # registry may be list or dict
    ids: set[str] = set()
    if isinstance(datasets, dict):
        for k, v in datasets.items():
            if isinstance(v, dict):
                did = v.get("dataset_id") or k
                ver = v.get("version") or ""
                ids.add(f"{did}@{ver}" if ver and "@" not in str(did) else str(did if "@" in str(did) else f"{did}@{ver}"))
                ids.add(str(k))
            else:
                ids.add(str(k))
    elif isinstance(datasets, list):
        for v in datasets:
            did = v.get("dataset_id", "")
            ver = v.get("version", "")
            ids.add(f"{did}@{ver}")

    use_v101 = all(k in ids or k.split("@")[0] in {x.split("@")[0] for x in ids} for k in DATASET_KEYS_V101.values())
    # simpler: check exact keys
    use_v101 = all(any(k == key or key in k for k in ids) for key in DATASET_KEYS_V101.values())
    # even simpler walk
    flat = json.dumps(reg)
    use_v101 = "btcusdt_intraday_1m@1.0.1" in flat or '"version": "1.0.1"' in flat and "btcusdt_intraday_1m" in flat
    keys = dict(DATASET_KEYS_V101 if use_v101 else DATASET_KEYS_V100)
    return {
        "dataset_keys": keys,
        "using_extended_v101": use_v101,
        "registry_path": registry_path,
    }


def summarize_parquet(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    df = pd.read_parquet(path, columns=["timestamp"])
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return {
        "exists": True,
        "path": str(path),
        "n_rows": int(len(df)),
        "start": str(ts.min()),
        "end": str(ts.max()),
        "bytes": int(path.stat().st_size),
    }


def build_data_provenance(
    *,
    registry_path: str = "dataset_registry.json",
    acquisition_log: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_dataset_keys(registry_path)
    keys = resolved["dataset_keys"]
    # map to paths
    reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    path_by_key: dict[str, str] = {}
    entries = reg.get("datasets") or reg
    if isinstance(entries, dict):
        iterable = entries.values() if entries and isinstance(next(iter(entries.values()), None), dict) else []
        if not iterable:
            # maybe keyed by dataset_id@version
            for k, v in entries.items():
                if isinstance(v, dict):
                    path_by_key[k] = v.get("path") or v.get("uri") or ""
                    did = f"{v.get('dataset_id')}@{v.get('version')}"
                    path_by_key[did] = v.get("path") or v.get("uri") or ""
        else:
            for v in iterable:
                did = f"{v.get('dataset_id')}@{v.get('version')}"
                path_by_key[did] = v.get("path") or ""
    elif isinstance(entries, list):
        for v in entries:
            did = f"{v.get('dataset_id')}@{v.get('version')}"
            path_by_key[did] = v.get("path") or v.get("uri") or ""

    files = {}
    for tf, key in keys.items():
        p = path_by_key.get(key) or f"data/btcusdt/btcusdt_intraday_{tf}.parquet"
        files[tf] = summarize_parquet(Path(p))

    # Grade decision
    primary_grade = "RESEARCH_GRADE"
    if resolved["using_extended_v101"]:
        note = (
            "Extended Binance Vision history registered (@1.0.1) after fixing µs timestamp parse. "
            "Still RESEARCH_GRADE: OHLCV-only, no bid/ask/depth, license UNKNOWN."
        )
    else:
        note = (
            "Using @1.0.0 ending 2024-12-31 if @1.0.1 not yet registered. "
            "Binance Vision remains best free in-repo BTC 1m source."
        )

    return {
        "disclaimer": DISCLAIMER,
        "primary_source": {
            "provider": "Binance Vision",
            "grade": primary_grade,
            "timezone": "UTC",
            "adjustment_status": "unadjusted",
            "source_native_timeframe": "1m",
            "derived_timeframes": ["5m", "15m", "30m", "1h"],
            "license_status": "UNKNOWN",
            "gaps_policy": "Do not silently fill missing bars",
            "note": note,
        },
        "resolved_datasets": resolved,
        "files": files,
        "free_source_survey": list(FREE_SOURCE_SURVEY),
        "paid_upgrade_candidates": list(PAID_UPGRADE_CANDIDATES),
        "best_available_free_conclusion": (
            "Binance Vision spot monthly klines remain the best legitimate free BTCUSDT "
            "intraday OHLCV source available to this repository. No free source surveyed "
            "provides superior multi-year 1m BTC with bid/ask/depth under clear research license. "
            "Paid tick/L2 providers identified but NOT purchased (STOP_BEFORE_PURCHASE)."
        ),
        "acquisition_log": acquisition_log,
    }


__all__ = ["build_data_provenance", "resolve_dataset_keys", "summarize_parquet"]

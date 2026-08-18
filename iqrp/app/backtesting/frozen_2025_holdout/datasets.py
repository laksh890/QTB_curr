"""Materialize + register immutable research (≤2024) and 2025 holdout datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from iqrp.app.backtesting.data.dataset_registry import DatasetRegistry, compute_checksum
from iqrp.app.backtesting.frozen_2025_holdout.protocol import (
    DISCLAIMER,
    HOLDOUT_END,
    HOLDOUT_START,
    RESEARCH_END,
)
from iqrp.app.data.historical.provenance import DatasetProvenance, now_utc_iso
from iqrp.app.data.historical.registry_ops import DatasetImmutabilityError, register_immutable


TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h")


def _gap_report(ts: pd.Series, tf: str) -> dict[str, Any]:
    expected = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
    }[tf]
    delta = ts.diff()
    gaps = delta > expected * 1.5
    n_gaps = int(gaps.sum())
    sample = []
    for i in list(ts.index[gaps])[:20]:
        sample.append(
            {
                "from": str(ts.loc[i - 1]) if i > 0 else None,
                "to": str(ts.loc[i]),
                "delta_s": float(delta.loc[i].total_seconds()) if pd.notna(delta.loc[i]) else None,
            }
        )
    return {
        "n_gaps": n_gaps,
        "expected_bar": str(expected),
        "gaps_sample": sample,
        "policy": "Do not silently fill missing bars",
    }


def materialize_firewall_datasets(
    *,
    source_dir: str = "data/btcusdt",
    out_dir: str = "data/btcusdt/firewall_2024_2025",
    registry_path: str = "dataset_registry.json",
    register: bool = True,
) -> dict[str, Any]:
    """Slice existing BTCUSDT parquets into immutable research / 2025 holdout files."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    research_end = pd.Timestamp(RESEARCH_END)
    hold_start = pd.Timestamp(HOLDOUT_START)
    hold_end = pd.Timestamp(HOLDOUT_END)

    registry = DatasetRegistry(registry_path) if register else None
    research_meta: dict[str, Any] = {}
    holdout_meta: dict[str, Any] = {}
    quality: dict[str, Any] = {}

    for tf in TIMEFRAMES:
        src = Path(source_dir) / f"btcusdt_intraday_{tf}.parquet"
        df = pd.read_parquet(src)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        research = df[df["timestamp"] <= research_end].reset_index(drop=True)
        holdout = df[(df["timestamp"] >= hold_start) & (df["timestamp"] <= hold_end)].reset_index(drop=True)

        r_path = out / f"btcusdt_research_through_2024_{tf}.parquet"
        h_path = out / f"btcusdt_holdout_2025_{tf}.parquet"
        research.to_parquet(r_path, index=False)
        holdout.to_parquet(h_path, index=False)

        r_sum = {
            "dataset_id": f"btcusdt_research_through_2024_{tf}",
            "version": "1.0.0",
            "path": str(r_path),
            "checksum": compute_checksum(r_path),
            "rows": int(len(research)),
            "start": str(research["timestamp"].iloc[0]) if len(research) else None,
            "end": str(research["timestamp"].iloc[-1]) if len(research) else None,
            "timezone": "UTC",
            "frequency": tf,
            "frequency_kind": "SOURCE" if tf == "1m" else "DERIVED",
            "source": f"slice_of_{src.name}_le_2024-12-31",
            "gaps": _gap_report(research["timestamp"], tf),
        }
        h_sum = {
            "dataset_id": f"btcusdt_holdout_2025_{tf}",
            "version": "1.0.0",
            "path": str(h_path),
            "checksum": compute_checksum(h_path),
            "rows": int(len(holdout)),
            "start": str(holdout["timestamp"].iloc[0]) if len(holdout) else None,
            "end": str(holdout["timestamp"].iloc[-1]) if len(holdout) else None,
            "timezone": "UTC",
            "frequency": tf,
            "frequency_kind": "SOURCE" if tf == "1m" else "DERIVED",
            "source": f"slice_of_{src.name}_calendar_2025",
            "gaps": _gap_report(holdout["timestamp"], tf),
            "complete_calendar_2025": bool(
                len(holdout)
                and holdout["timestamp"].iloc[0] <= hold_start + pd.Timedelta(hours=1)
                and holdout["timestamp"].iloc[-1] >= hold_end - pd.Timedelta(hours=1)
            ),
        }
        # Expected bar counts for continuous crypto 2025 (non-leap)
        expected = {"1m": 525600, "5m": 105120, "15m": 35040, "30m": 17520, "1h": 8760}[tf]
        h_sum["expected_rows_24x7"] = expected
        h_sum["row_count_matches_expected"] = int(len(holdout)) == expected

        research_meta[tf] = r_sum
        holdout_meta[tf] = h_sum
        quality[tf] = {
            "research_gaps": r_sum["gaps"],
            "holdout_gaps": h_sum["gaps"],
            "holdout_complete": h_sum["complete_calendar_2025"] and h_sum["row_count_matches_expected"],
        }

        if registry is not None:
            for summary, role in ((r_sum, "research"), (h_sum, "holdout_2025")):
                prov = DatasetProvenance(
                    provider="binance_vision_slice",
                    source=summary["source"],
                    acquisition_timestamp=now_utc_iso(),
                    original_symbol="BTCUSDT",
                    normalized_symbol="BTCUSDT",
                    frequency=tf,
                    timezone="UTC",
                    checksum=summary["checksum"],
                    frequency_kind=summary["frequency_kind"],
                    data_class="DEVELOPMENT/RESEARCH",
                    license_status="UNKNOWN",
                    known_limitations=[
                        "OHLCV slice from existing Binance Vision-derived parquet; not a new vendor pull.",
                        "Firewall role: " + role,
                        "Do not silently fill gaps.",
                    ],
                )
                try:
                    register_immutable(
                        registry,
                        path=summary["path"],
                        dataset_id=summary["dataset_id"],
                        version=summary["version"],
                        source=summary["source"],
                        frame=research if role == "research" else holdout,
                        provenance=prov,
                        quality_status="PASS" if quality[tf]["holdout_complete"] or role == "research" else "WARN",
                        known_limitations=list(prov.known_limitations),
                        persist=True,
                    )
                except DatasetImmutabilityError:
                    # already registered — verify checksum matches
                    existing = registry.require(summary["dataset_id"], summary["version"])
                    if existing.checksum != summary["checksum"]:
                        raise RuntimeError(
                            f"Registered {summary['dataset_id']}@{summary['version']} checksum mismatch"
                        )

    complete_2025 = all(quality[tf]["holdout_complete"] for tf in TIMEFRAMES if tf != "1m") and quality["1m"][
        "holdout_complete"
    ]

    return {
        "disclaimer": DISCLAIMER,
        "research_end": RESEARCH_END,
        "holdout_start": HOLDOUT_START,
        "holdout_end": HOLDOUT_END,
        "research_datasets": research_meta,
        "holdout_datasets": holdout_meta,
        "quality": quality,
        "complete_2025_all_tfs": complete_2025,
        "acquisition": {
            "method": "slice_existing_local_parquet",
            "purchased": False,
            "network_download": False,
            "note": (
                "Full calendar-2025 BTCUSDT already present in local Binance Vision-derived "
                "parquets (@1.0.1 lineage). Extracted to immutable firewall identities without re-download."
            ),
        },
    }


__all__ = ["materialize_firewall_datasets", "TIMEFRAMES"]

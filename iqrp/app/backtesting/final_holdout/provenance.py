"""Data provenance + holdout window establishment for Prompt final holdout."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from iqrp.app.backtesting.final_holdout.protocol import DISCLAIMER
from iqrp.app.backtesting.serializer import to_jsonable
from iqrp.app.data.historical.resampling import resample_session_aware
from iqrp.app.data.historical.calendar import crypto_24x7_calendar


_KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_vision_open_time(series: pd.Series) -> pd.Series:
    sample = float(pd.to_numeric(series.iloc[0], errors="coerce"))
    if sample > 1e18:
        unit = "ns"
    elif sample > 1e14:
        unit = "us"
    elif sample > 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit=unit, utc=True)


def load_july_2026_1m_from_cache(
    zip_path: Path = Path("data/cache/binance_vision/BTCUSDT/1m/BTCUSDT-1m-2026-07.zip"),
) -> pd.DataFrame:
    import io

    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name)
    # Vision kline CSVs are typically headerless
    df = pd.read_csv(io.BytesIO(raw), header=None, names=_KLINE_COLS)
    ts = parse_vision_open_time(df["open_time"])
    out = pd.DataFrame(
        {
            "timestamp": ts,
            "instrument": "BTCUSDT",
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
            "trade_count": pd.to_numeric(df.get("count", 0), errors="coerce").fillna(0.0),
        }
    ).dropna(subset=["timestamp", "close"])
    return out.sort_values("timestamp").reset_index(drop=True)


def registered_dataset_summary(registry_path: str = "dataset_registry.json") -> list[dict[str, Any]]:
    reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    rows = []
    for ds in reg.get("datasets") or []:
        if "btcusdt_intraday" not in str(ds.get("dataset_id", "")):
            continue
        path = Path(ds.get("path") or "")
        file_checksum = _sha256_file(path) if path.exists() else None
        rows.append(
            {
                "dataset_id": ds.get("dataset_id"),
                "version": ds.get("version"),
                "key": f"{ds.get('dataset_id')}@{ds.get('version')}",
                "path": str(ds.get("path")),
                "registry_checksum": ds.get("checksum"),
                "file_sha256": file_checksum,
                "checksum_match": bool(file_checksum and file_checksum == ds.get("checksum")),
                "start": ds.get("start"),
                "end": ds.get("end"),
                "row_count": ds.get("row_count"),
                "timezone": ds.get("timezone"),
                "source": ds.get("source"),
                "frequency": ds.get("frequency"),
                "known_limitations": ds.get("known_limitations"),
                "extra_grade": (ds.get("extra") or {}).get("data_class"),
                "provenance": (ds.get("extra") or {}).get("provenance"),
            }
        )
    return rows


def p42_research_windows(prompt42_dir: str = "results/final_trading_validation") -> dict[str, Any]:
    """Reconstruct Prompt-42 trimmed windows (last MAX_BARS of @1.0.1)."""
    from iqrp.app.backtesting.alpha_research.adapters.validation import train_val_oos_slices
    from iqrp.app.backtesting.alpha_research.model_campaign.runner import _trim

    cfg = json.loads(Path(prompt42_dir, "validation_config.json").read_text(encoding="utf-8"))
    max_bars = cfg["max_bars"]
    windows = {}
    latest_end = None
    for tf, mb in max_bars.items():
        path = Path(f"data/btcusdt/btcusdt_intraday_{tf}.parquet")
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        trimmed = _trim(df, int(mb))
        slices = train_val_oos_slices(len(trimmed), train_frac=0.5, validation_frac=0.25)
        end_ts = trimmed["timestamp"].iloc[-1]
        windows[tf] = {
            "max_bars": int(mb),
            "full_rows": int(len(df)),
            "trimmed_rows": int(len(trimmed)),
            "trimmed_start": str(trimmed["timestamp"].iloc[0]),
            "trimmed_end": str(end_ts),
            "train": [str(trimmed["timestamp"].iloc[slices["train"].start]), str(trimmed["timestamp"].iloc[slices["train"].stop - 1])],
            "validation": [
                str(trimmed["timestamp"].iloc[slices["validation"].start]),
                str(trimmed["timestamp"].iloc[slices["validation"].stop - 1]),
            ],
            "oos": [str(trimmed["timestamp"].iloc[slices["oos"].start]), str(trimmed["timestamp"].iloc[slices["oos"].stop - 1])],
            "bars_after_trimmed_end_in_registered": int((df["timestamp"] > end_ts).sum()),
        }
        if latest_end is None or end_ts > latest_end:
            latest_end = end_ts
    return {"windows": windows, "latest_p42_timestamp": str(latest_end) if latest_end is not None else None}


def build_holdout_frames(
    *,
    p42_end: pd.Timestamp,
    out_dir: Path,
) -> dict[str, Any]:
    """Materialize holdout bars AFTER p42_end from Vision July cache (never seen by P42 parquet)."""
    zip_path = Path("data/cache/binance_vision/BTCUSDT/1m/BTCUSDT-1m-2026-07.zip")
    july = load_july_2026_1m_from_cache(zip_path)
    holdout_1m = july[july["timestamp"] > p42_end].copy().reset_index(drop=True)
    status = {
        "july_zip_path": str(zip_path),
        "july_zip_sha256": _sha256_file(zip_path) if zip_path.exists() else None,
        "july_zip_rows": int(len(july)),
        "july_zip_start": str(july["timestamp"].iloc[0]) if len(july) else None,
        "july_zip_end": str(july["timestamp"].iloc[-1]) if len(july) else None,
        "p42_end": str(p42_end),
        "holdout_1m_rows": int(len(holdout_1m)),
        "holdout_available": bool(len(holdout_1m) > 0),
    }
    if len(holdout_1m) == 0:
        status["final_status"] = "FINAL_HOLDOUT_UNAVAILABLE"
        status["reason"] = "No bars after latest Prompt-42 timestamp in Vision July cache or registered data."
        return status

    out_dir.mkdir(parents=True, exist_ok=True)
    cal = crypto_24x7_calendar()
    paths = {}
    # Write 1m holdout
    p1 = out_dir / "btcusdt_holdout_1m.parquet"
    holdout_1m.to_parquet(p1, index=False)
    paths["1m"] = {
        "path": str(p1),
        "sha256": _sha256_file(p1),
        "rows": int(len(holdout_1m)),
        "start": str(holdout_1m["timestamp"].iloc[0]),
        "end": str(holdout_1m["timestamp"].iloc[-1]),
        "frequency_kind": "SOURCE",
        "source": "binance_vision_july_2026_zip_post_p42_truncation",
    }

    # For derived TFs: combine registered 1m tail + holdout for causal resample continuity,
    # then slice derived bars with timestamp > p42_end.
    reg_1m = pd.read_parquet("data/btcusdt/btcusdt_intraday_1m.parquet")
    reg_1m["timestamp"] = pd.to_datetime(reg_1m["timestamp"], utc=True)
    # Ensure no overlap
    reg_1m = reg_1m[reg_1m["timestamp"] <= p42_end]
    combined = pd.concat([reg_1m, holdout_1m], ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")
    for tf in ("5m", "15m", "30m", "1h"):
        derived_all, _prov = resample_session_aware(
            combined,
            source_frequency="1m",
            derived_frequency=tf,
            calendar=cal,
            source_dataset_id="btcusdt_holdout_combined_1m",
        )
        derived_h = derived_all[derived_all["timestamp"] > p42_end].reset_index(drop=True)
        pp = out_dir / f"btcusdt_holdout_{tf}.parquet"
        derived_h.to_parquet(pp, index=False)
        paths[tf] = {
            "path": str(pp),
            "sha256": _sha256_file(pp),
            "rows": int(len(derived_h)),
            "start": str(derived_h["timestamp"].iloc[0]) if len(derived_h) else None,
            "end": str(derived_h["timestamp"].iloc[-1]) if len(derived_h) else None,
            "frequency_kind": "DERIVED",
            "source": f"derived_from_combined_1m_holdout_slice>{p42_end}",
        }

    n_days = int(holdout_1m["timestamp"].dt.floor("D").nunique())
    status.update(
        {
            "final_status": "HOLDOUT_ESTABLISHED",
            "holdout_files": paths,
            "holdout_calendar_days": n_days,
            "holdout_hours_approx": float(len(holdout_1m) / 60.0),
            "note": (
                "Registered @1.0.1 parquet truncated at 2026-07-31 00:00:00 while Vision July ZIP "
                "contains through 2026-07-31 23:59. Those post-truncation bars were never in the "
                "P42 evaluation frame — genuine chronological holdout after P42 end. "
                "Sample is short (~1 calendar day); statistical power limited."
            ),
            "sample_adequacy_warning": n_days < 7,
        }
    )
    return status


def build_data_provenance(
    *,
    registry_path: str = "dataset_registry.json",
    prompt42_dir: str = "results/final_trading_validation",
    holdout_data_dir: str = "data/btcusdt/holdout",
) -> dict[str, Any]:
    registered = registered_dataset_summary(registry_path)
    p42 = p42_research_windows(prompt42_dir)
    latest = pd.Timestamp(p42["latest_p42_timestamp"])
    holdout = build_holdout_frames(p42_end=latest, out_dir=Path(holdout_data_dir))

    # Overlap analysis: @1.0.1 vs P42 windows
    v101 = [r for r in registered if r.get("version") == "1.0.1"]
    untouched = holdout.get("final_status") == "HOLDOUT_ESTABLISHED"

    md_lines = [
        "# Final Holdout Data Provenance",
        "",
        DISCLAIMER,
        "",
        f"- Latest Prompt-42 timestamp: `{p42['latest_p42_timestamp']}`",
        f"- Holdout status: **{holdout.get('final_status')}**",
        f"- Untouched holdout established: **{untouched}**",
        "",
        "## Registered BTCUSDT datasets",
        "",
    ]
    for r in registered:
        md_lines.append(
            f"- `{r['key']}` rows={r['row_count']} {r['start']} → {r['end']} "
            f"checksum_match={r['checksum_match']}"
        )
    md_lines.extend(
        [
            "",
            "## Independence analysis",
            "",
            "- `@1.0.1` is **not** independent of Prompt 42: P42 evaluated the last MAX_BARS of `@1.0.1`, "
            "ending at the registered series end.",
            "- Bars **after** that end exist in the local Binance Vision July 2026 ZIP but were omitted "
            "from the registered parquet (truncation defect).",
            "- Those post-end bars are the only chronological holdout available without network acquisition.",
            "",
            f"### Holdout detail",
            "",
            f"```json\n{json.dumps(to_jsonable(holdout), indent=2)}\n```",
            "",
        ]
    )

    return {
        "disclaimer": DISCLAIMER,
        "registered_datasets": registered,
        "prompt42_windows": p42,
        "holdout": holdout,
        "untouched_holdout_exists": untouched,
        "provenance_pass": bool(untouched and holdout.get("holdout_1m_rows", 0) > 0),
        "v101_contains_data_unavailable_during_p42_evaluation": bool(
            untouched
        ),  # post-truncation bars
        "overlap_note": (
            "P42 research periods fully overlap the registered @1.0.1 trimmed windows through series end. "
            "Independence comes only from Vision ZIP bars after registered truncation."
        ),
        "markdown": "\n".join(md_lines),
    }


__all__ = ["build_data_provenance", "build_holdout_frames", "p42_research_windows"]

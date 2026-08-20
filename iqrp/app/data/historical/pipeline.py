"""Acquisition pipeline: download → normalize → validate → atomic write → register."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from iqrp.app.backtesting.data.dataset_registry import DatasetRegistry, compute_checksum
from iqrp.app.backtesting.data.schema import normalize_frame
from iqrp.app.data.historical.calendar import crypto_24x7_calendar, nse_equity_calendar
from iqrp.app.data.historical.cache import CacheKey, HistoricalCache
from iqrp.app.data.historical.intraday_validation import (
    GapThresholds,
    build_intraday_quality_report,
    quality_report_markdown,
)
from iqrp.app.data.historical.provenance import DatasetProvenance, now_utc_iso
from iqrp.app.data.historical.provider import (
    HistoricalDataProvider,
    ProviderRequest,
    ProviderResponse,
)
from iqrp.app.data.historical.provider_registry import get_provider
from iqrp.app.data.historical.registry_ops import (
    DatasetImmutabilityError,
    next_version,
    register_immutable,
)
from iqrp.app.data.historical.resampling import resample_session_aware


def _select_calendar(provider_id: str, instrument: str, response_meta: dict | None = None):
    meta = dict(response_meta or {})
    market = str(meta.get("market_type") or "").upper()
    if (
        market == "CRYPTO"
        or meta.get("continuous_market")
        or str(provider_id).lower() in {"binance", "binance_vision"}
        or str(instrument).upper().endswith("USDT")
    ):
        return crypto_24x7_calendar()
    return nse_equity_calendar()


@dataclass
class AcquisitionResult:
    dataset_id: str
    version: str
    path: Path
    checksum: str
    quality_report: dict[str, Any]
    provenance: dict[str, Any]
    provider_response: dict[str, Any]
    derived: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False
    incremental: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "path": str(self.path),
            "checksum": self.checksum,
            "quality_report": self.quality_report,
            "provenance": self.provenance,
            "provider_response": self.provider_response,
            "derived": self.derived,
            "cache_hit": self.cache_hit,
            "incremental": self.incremental,
        }


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    # validate tmp non-empty before replace
    if len(pd.read_parquet(tmp)) == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("refusing to replace dataset with empty download")
    tmp.replace(path)


def _merge_incremental(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([existing, new], ignore_index=True)
    both = normalize_frame(both)
    both = both.drop_duplicates(subset=["timestamp", "instrument"], keep="last")
    return both.sort_values(["instrument", "timestamp"]).reset_index(drop=True)


class AcquisitionPipeline:
    """End-to-end historical acquisition with atomic writes and immutable registry."""

    def __init__(
        self,
        *,
        output_dir: str | Path = "data/nifty50",
        registry_path: str | Path = "dataset_registry.json",
        cache: HistoricalCache | None = None,
        gap_thresholds: GapThresholds | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.registry = DatasetRegistry(registry_path)
        self.cache = cache or HistoricalCache()
        self.gap_thresholds = gap_thresholds or GapThresholds()
        self.calendar = nse_equity_calendar()  # default; overridden per acquire

    def acquire(
        self,
        *,
        provider: str | HistoricalDataProvider = "yahoo_finance",
        instrument: str = "NIFTY50",
        start: str | datetime,
        end: str | datetime,
        frequency: str = "1m",
        adjustment_policy: str = "unadjusted",
        dataset_id: str | None = None,
        version: str = "1.0.0",
        derive: list[str] | None = None,
        use_cache: bool = True,
        incremental: bool = True,
        register: bool = True,
    ) -> AcquisitionResult:
        prov = provider if isinstance(provider, HistoricalDataProvider) else get_provider(provider)
        ds_id = dataset_id or f"{instrument.lower()}_intraday_{frequency}"
        start_s = str(pd.Timestamp(start))
        end_s = str(pd.Timestamp(end))

        # Large multi-year crypto frames should not round-trip through the small
        # HistoricalCache parquet blob unless explicitly enabled.
        provider_id = getattr(prov, "provider_id", str(provider))
        large_provider = str(provider_id).lower() in {"binance", "binance_vision"}
        if large_provider:
            use_cache = False

        cache_key = CacheKey(
            provider=prov.provider_id,
            instrument=str(instrument),
            start=start_s,
            end=end_s,
            frequency=str(frequency),
            adjustment_policy=str(adjustment_policy),
        )
        cache_hit = False
        response: ProviderResponse | None = None
        frame: pd.DataFrame

        if use_cache:
            hit = self.cache.get(cache_key)
            if hit is not None:
                frame, meta = hit
                cache_hit = True
                # rebuild minimal response meta from cache
                from iqrp.app.data.historical.provider import ProviderResponse as PR

                response = PR(
                    frame=frame,
                    provider=prov.provider_id,
                    source=str(meta.get("source", prov.provider_id)),
                    retrieval_timestamp=str(meta.get("cached_at")),
                    requested_range=(start_s, end_s),
                    actual_range=(
                        str(frame["timestamp"].min()) if len(frame) else None,
                        str(frame["timestamp"].max()) if len(frame) else None,
                    ),
                    frequency=str(frequency),
                    timezone="UTC",
                    original_timezone=str(meta.get("original_timezone", "UNKNOWN")),
                    exchange_timezone=str(meta.get("exchange_timezone", "Asia/Kolkata")),
                    adjustment_policy=adjustment_policy,
                    original_symbol=str(meta.get("original_symbol", instrument)),
                    normalized_symbol=str(instrument),
                    warnings=["served_from_cache"],
                    metadata=dict(meta),
                )

        if response is None:
            req = ProviderRequest(
                instrument=instrument,
                start=start,
                end=end,
                frequency=frequency,
                adjustment_policy=adjustment_policy,
            )
            response = prov.download(req)
            frame = response.frame
            if use_cache:
                self.cache.put(
                    cache_key,
                    frame,
                    extra_meta={
                        "source": response.source,
                        "original_timezone": response.original_timezone,
                        "exchange_timezone": response.exchange_timezone,
                        "original_symbol": response.original_symbol,
                    },
                )

        out_path = self.output_dir / f"{ds_id}.parquet"
        did_incremental = False

        # Incremental: only fetch missing period conceptually — if existing file present,
        # merge new bars and avoid replacing with partial-only file.
        if incremental and out_path.exists():
            existing = pd.read_parquet(out_path)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
            ex_end = existing["timestamp"].max()
            new = frame[frame["timestamp"] > ex_end]
            if len(new) == 0 and len(frame) <= len(existing):
                # nothing newer; keep existing if covers request partially
                frame = existing
            else:
                frame = _merge_incremental(existing, frame)
                did_incremental = True

        frame = normalize_frame(frame)
        self.calendar = _select_calendar(provider_id, instrument, response.metadata)

        # Atomic write to staging then replace
        staging = out_path.with_suffix(".parquet.staging")
        _atomic_write_parquet(frame, staging)
        # Only replace valid existing after staging verified
        if out_path.exists():
            backup = out_path.with_suffix(".parquet.bak")
            shutil.copy2(out_path, backup)
            try:
                staging.replace(out_path)
                backup.unlink(missing_ok=True)
            except Exception:
                # restore
                if backup.exists():
                    shutil.copy2(backup, out_path)
                raise
        else:
            staging.replace(out_path)

        checksum = compute_checksum(out_path)
        quality = build_intraday_quality_report(
            frame,
            frequency=str(frequency),
            dataset_id=ds_id,
            calendar=self.calendar,
            thresholds=self.gap_thresholds,
        )
        # Write quality artifacts beside dataset
        qjson = out_path.with_name(f"{ds_id}_data_quality.json")
        qmd = out_path.with_name(f"{ds_id}_data_quality.md")
        qjson.write_text(json.dumps(quality, indent=2, default=str), encoding="utf-8")
        qmd.write_text(quality_report_markdown(quality), encoding="utf-8")

        provenance = DatasetProvenance(
            provider=response.provider,
            source=response.source,
            acquisition_timestamp=response.retrieval_timestamp,
            original_symbol=response.original_symbol,
            normalized_symbol=response.normalized_symbol,
            frequency=str(frequency),
            timezone="UTC",
            original_timezone=response.original_timezone,
            exchange_timezone=response.exchange_timezone,
            currency=response.currency,
            adjustment_status=response.adjustment_policy,
            corporate_action_treatment=str(
                response.metadata.get("corporate_action_treatment", "UNKNOWN")
            ),
            checksum=checksum,
            license_status=response.license_status,
            data_class=response.data_class,
            availability_timestamp_available=response.availability_timestamp_available,
            frequency_kind="SOURCE",
            requested_range=response.requested_range,
            actual_range=response.actual_range,
            known_limitations=list(response.warnings)
            + [
                f"data_tier={response.metadata.get('data_tier', response.data_class)}",
                "DEVELOPMENT/RESEARCH data — not institutional-grade market data.",
                "License status UNKNOWN unless independently established.",
                "Availability timestamps not provided by this archive; PIT of vendor revisions unknown.",
            ],
        )
        # enrich provenance extras for crypto
        if response.metadata.get("market_type") == "CRYPTO":
            provenance.extra.update(
                {
                    "market_type": "CRYPTO",
                    "continuous_market": True,
                    "session_model": "24x7",
                    "timezone": "UTC",
                }
            )
            provenance.currency = response.currency or "USDT"
        prov_path = out_path.with_name(f"{ds_id}_provenance.json")
        prov_path.write_text(
            json.dumps(provenance.to_dict(), indent=2, default=str), encoding="utf-8"
        )

        ver = version
        if register:
            try:
                register_immutable(
                    self.registry,
                    path=out_path,
                    dataset_id=ds_id,
                    version=ver,
                    source=response.source,
                    frame=frame,
                    provenance=provenance,
                    quality_status="PASS" if quality.get("ok") else "FAIL",
                    known_limitations=provenance.known_limitations,
                )
            except DatasetImmutabilityError:
                ver = next_version(self.registry, ds_id, base=version)
                register_immutable(
                    self.registry,
                    path=out_path,
                    dataset_id=ds_id,
                    version=ver,
                    source=response.source,
                    frame=frame,
                    provenance=provenance,
                    quality_status="PASS" if quality.get("ok") else "FAIL",
                    known_limitations=provenance.known_limitations,
                )

        derived_meta: list[dict[str, Any]] = []
        for dfreq in derive or []:
            derived_meta.append(
                self._derive_and_register(
                    frame,
                    source_dataset_id=f"{ds_id}@{ver}",
                    source_frequency=str(frequency),
                    derived_frequency=dfreq,
                    instrument=instrument,
                    source_checksum=checksum,
                    register=register,
                )
            )

        return AcquisitionResult(
            dataset_id=ds_id,
            version=ver,
            path=out_path,
            checksum=checksum,
            quality_report=quality,
            provenance=provenance.to_dict(),
            provider_response=response.to_dict(),
            derived=derived_meta,
            cache_hit=cache_hit,
            incremental=did_incremental,
        )

    def _derive_and_register(
        self,
        source_frame: pd.DataFrame,
        *,
        source_dataset_id: str,
        source_frequency: str,
        derived_frequency: str,
        instrument: str,
        source_checksum: str,
        register: bool,
    ) -> dict[str, Any]:
        derived, prov = resample_session_aware(
            source_frame,
            source_frequency=source_frequency,
            derived_frequency=derived_frequency,
            calendar=self.calendar,
            source_dataset_id=source_dataset_id,
            source_checksum=source_checksum,
        )
        ds_id = f"{instrument.lower()}_intraday_{derived_frequency}"
        path = self.output_dir / f"{ds_id}.parquet"
        _atomic_write_parquet(derived, path)
        checksum = compute_checksum(path)
        prov.checksum = checksum
        quality = build_intraday_quality_report(
            derived,
            frequency=derived_frequency,
            dataset_id=ds_id,
            calendar=self.calendar,
            thresholds=self.gap_thresholds,
        )
        path.with_name(f"{ds_id}_data_quality.json").write_text(
            json.dumps(quality, indent=2, default=str), encoding="utf-8"
        )
        path.with_name(f"{ds_id}_data_quality.md").write_text(
            quality_report_markdown(quality), encoding="utf-8"
        )
        path.with_name(f"{ds_id}_provenance.json").write_text(
            json.dumps(prov.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        version = "1.0.0"
        if register:
            try:
                register_immutable(
                    self.registry,
                    path=path,
                    dataset_id=ds_id,
                    version=version,
                    source=f"derived:{source_dataset_id}",
                    frame=derived,
                    provenance=prov,
                    quality_status="PASS" if quality.get("ok") else "FAIL",
                )
            except DatasetImmutabilityError:
                version = next_version(self.registry, ds_id)
                register_immutable(
                    self.registry,
                    path=path,
                    dataset_id=ds_id,
                    version=version,
                    source=f"derived:{source_dataset_id}",
                    frame=derived,
                    provenance=prov,
                    quality_status="PASS" if quality.get("ok") else "FAIL",
                )
        return {
            "dataset_id": ds_id,
            "version": version,
            "path": str(path),
            "checksum": checksum,
            "source_frequency": source_frequency,
            "derived_frequency": derived_frequency,
            "frequency_kind": "DERIVED",
            "row_count": int(len(derived)),
            "quality_ok": quality.get("ok"),
            "provenance": prov.to_dict(),
        }


__all__ = ["AcquisitionPipeline", "AcquisitionResult"]

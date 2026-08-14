"""AlphaCandidate construction and validation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.unified_pipeline.types import (
    AlphaCandidate,
    CandidateRejectionCode,
)


_ACCEPTABLE_OOS = frozenset(
    {
        "PASS",
        "OK",
        "ACCEPTABLE",
        "UNKNOWN",
        "OOS_AVAILABLE",
        "EVALUATED",
        "",
    }
)


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            ts = pd.Timestamp(value).to_pydatetime()
        except Exception:  # noqa: BLE001
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def direction_from_signal_value(value: float, *, flat_eps: float = 1e-12) -> float:
    if not np.isfinite(value):
        return float("nan")
    if abs(value) <= flat_eps:
        return 0.0
    return float(np.sign(value))


def candidate_from_alpha_result(
    alpha_result: dict[str, Any],
    *,
    instrument: str,
    timestamp: Any | None = None,
    base_weight: float = 0.05,
    signal_timeframe: str = "",
    execution_timeframe: str = "",
    source_model: str = "reference",
    source_model_version: str = "1.0.0",
    data_version: str = "",
    dataset_checksum: str = "",
    candidate_id: str | None = None,
) -> AlphaCandidate:
    """Build a candidate from AlphaSignalResearchEngine.evaluate_candidate output."""
    sig = alpha_result.get("signal_series")
    if sig is None:
        raise ValueError("alpha_result missing signal_series")
    series = pd.Series(sig)
    if timestamp is None:
        idx = series.index[-1]
        # prefer frame timestamp if embedded
        ts_val = alpha_result.get("asof") or str(idx)
        value = float(series.iloc[-1])
    else:
        ts_val = timestamp
        # take last finite at or before — caller should pass aligned value
        value = float(series.iloc[-1])

    direction = direction_from_signal_value(value)
    oos = alpha_result.get("oos") or {}
    oos_status = "UNKNOWN"
    if isinstance(oos, dict):
        oos_status = str(oos.get("status") or oos.get("oos_status") or "EVALUATED")

    experiment = alpha_result.get("experiment") or {}
    cost = alpha_result.get("cost") or alpha_result.get("costs") or {}
    cost_id = "default_bps"
    if isinstance(cost, dict) and cost:
        cost_id = "commission_spread_slippage_bps"

    holding = int(
        alpha_result.get("holding_bars")
        or (experiment.get("holding_period") if isinstance(experiment, dict) else None)
        or 1
    )
    signal_id = str(
        alpha_result.get("signal_id")
        or (alpha_result.get("signal_meta") or {}).get("signal_id")
        or experiment.get("signal_id")
        or "unknown_signal"
    )
    exp_id = str(alpha_result.get("experiment_id") or experiment.get("experiment_id") or "")
    ds = str(
        data_version
        or alpha_result.get("dataset_id")
        or experiment.get("dataset_id")
        or ""
    )
    checksum = str(
        dataset_checksum
        or alpha_result.get("dataset_checksum")
        or experiment.get("dataset_checksum")
        or ""
    )
    cid = candidate_id or f"{signal_id}:{instrument}:{ts_val}:{direction}"
    req_w = float(direction) * abs(float(base_weight)) if direction != 0 else 0.0

    return AlphaCandidate(
        candidate_id=str(cid),
        signal_id=signal_id,
        instrument=str(instrument),
        timestamp=str(ts_val),
        direction=float(direction),
        signal_value=float(value),
        confidence=None,
        expected_horizon=holding,
        signal_timeframe=signal_timeframe or str(experiment.get("timeframe") or ""),
        execution_timeframe=execution_timeframe or signal_timeframe,
        source_model=source_model,
        source_model_version=source_model_version,
        research_configuration={
            "classification": alpha_result.get("classification"),
            "research_status": alpha_result.get("research_status"),
            "score": alpha_result.get("score"),
        },
        data_version=ds,
        dataset_checksum=checksum,
        oos_status=oos_status,
        cost_model_id=cost_id,
        experiment_id=exp_id,
        requested_weight=float(req_w) if direction != 0 else 0.0,
        meta={"from": "evaluate_candidate"},
    )


def validate_candidate(
    candidate: AlphaCandidate,
    *,
    asof: Any | None = None,
    max_staleness: Any | None = None,
    seen_ids: set[str] | None = None,
    require_model_version: bool = True,
    require_dataset: bool = True,
    acceptable_oos: set[str] | frozenset[str] | None = None,
) -> tuple[bool, list[str]]:
    """Return (ok, reason_codes). Invalid candidates must not reach Risk."""
    codes: list[str] = []
    if not candidate.candidate_id:
        codes.append(CandidateRejectionCode.MISSING_CANDIDATE_ID.value)
    if not np.isfinite(candidate.signal_value) or not np.isfinite(candidate.direction):
        codes.append(CandidateRejectionCode.NON_FINITE_SIGNAL.value)
    if candidate.direction not in (-1.0, 0.0, 1.0) and np.isfinite(candidate.direction):
        if abs(abs(candidate.direction) - 1.0) > 1e-9 and abs(candidate.direction) > 1e-12:
            codes.append(CandidateRejectionCode.INVALID_DIRECTION.value)
    ts = _parse_ts(candidate.timestamp)
    if ts is None:
        codes.append(CandidateRejectionCode.INVALID_TIMESTAMP.value)
    if not candidate.instrument:
        codes.append(CandidateRejectionCode.MISSING_INSTRUMENT.value)
    if require_model_version and not str(candidate.source_model_version).strip():
        codes.append(CandidateRejectionCode.UNKNOWN_MODEL_VERSION.value)
    if require_dataset and not (candidate.data_version or candidate.dataset_checksum):
        codes.append(CandidateRejectionCode.UNKNOWN_DATASET.value)
    oos_ok = acceptable_oos if acceptable_oos is not None else _ACCEPTABLE_OOS
    if str(candidate.oos_status).upper() not in {x.upper() for x in oos_ok}:
        # allow research classifications that are not hard OOS failure labels
        bad = {"OOS_FAILED", "OOS_FAILURE", "FAIL", "REJECTED", "REJECT"}
        if str(candidate.oos_status).upper() in bad:
            codes.append(CandidateRejectionCode.OOS_UNACCEPTABLE.value)
    if "future_" in str(candidate.meta).lower() or candidate.meta.get("contains_future"):
        codes.append(CandidateRejectionCode.FUTURE_INFORMATION.value)
    asof_ts = _parse_ts(asof) if asof is not None else None
    if asof_ts is not None and ts is not None and ts > asof_ts:
        codes.append(CandidateRejectionCode.FUTURE_INFORMATION.value)
    if max_staleness is not None and asof_ts is not None and ts is not None:
        stale = pd.Timedelta(max_staleness)
        if asof_ts - ts > stale:
            codes.append(CandidateRejectionCode.STALE_CANDIDATE.value)
    if seen_ids is not None and candidate.candidate_id in seen_ids:
        codes.append(CandidateRejectionCode.DUPLICATE_CANDIDATE.value)
    return (len(codes) == 0), codes


__all__ = [
    "candidate_from_alpha_result",
    "direction_from_signal_value",
    "validate_candidate",
]

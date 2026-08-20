"""Continuous futures contract construction and roll rules.

Distinguishes:
- **Raw** contract series (individual expiries)
- **Continuous Research** series (stitched, optionally back-adjusted)
- **Tradable** series (front contract actually tradeable at each date)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from iqrp.app.backtesting.data.schema import normalize_frame
from iqrp.app.backtesting.serializer import to_jsonable

__all__ = [
    "RollRule",
    "AdjustmentMethod",
    "ContractSeriesKind",
    "ContractSpec",
    "RollEvent",
    "ContinuousContractConfig",
    "ContinuousContractBuilder",
    "build_continuous_series",
]


class RollRule(str, Enum):
    VOLUME = "volume"
    OPEN_INTEREST = "open_interest"
    CALENDAR = "calendar"


class AdjustmentMethod(str, Enum):
    BACK_ADJUST = "back_adjust"
    RATIO = "ratio"
    UNADJUSTED = "unadjusted"


class ContractSeriesKind(str, Enum):
    RAW = "raw"
    CONTINUOUS_RESEARCH = "continuous_research"
    TRADABLE = "tradable"


@dataclass(slots=True)
class ContractSpec:
    """Static contract metadata (multiplier, tick, margin, etc.)."""

    contract: str
    root: str
    expiry: datetime
    multiplier: float = 1.0
    tick_size: float = 0.01
    tick_value: float | None = None
    currency: str | None = None
    margin: float | None = None
    exchange: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.expiry.tzinfo is None:
            raise ValueError(f"ContractSpec.expiry must be timezone-aware ({self.contract})")
        if self.tick_value is None:
            object.__setattr__(
                self, "tick_value", float(self.tick_size) * float(self.multiplier)
            )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


@dataclass(slots=True)
class RollEvent:
    """A single roll from ``from_contract`` to ``to_contract`` on ``roll_date``."""

    roll_date: datetime
    from_contract: str
    to_contract: str
    rule: RollRule
    gap: float = 0.0
    ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rule"] = self.rule.value
        return to_jsonable(d)


@dataclass(slots=True)
class ContinuousContractConfig:
    """Configuration for building a continuous futures series."""

    root: str
    continuous_symbol: str
    roll_rule: RollRule = RollRule.VOLUME
    adjustment: AdjustmentMethod = AdjustmentMethod.BACK_ADJUST
    series_kind: ContractSeriesKind = ContractSeriesKind.CONTINUOUS_RESEARCH
    calendar_days_before_expiry: int = 5
    volume_col: str = "volume"
    oi_col: str = "open_interest"
    multiplier: float = 1.0
    tick_size: float = 0.01
    tick_value: float | None = None
    margin: float | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.roll_rule, RollRule):
            object.__setattr__(self, "roll_rule", RollRule(str(self.roll_rule)))
        if not isinstance(self.adjustment, AdjustmentMethod):
            object.__setattr__(self, "adjustment", AdjustmentMethod(str(self.adjustment)))
        if not isinstance(self.series_kind, ContractSeriesKind):
            object.__setattr__(self, "series_kind", ContractSeriesKind(str(self.series_kind)))
        if self.tick_value is None:
            object.__setattr__(
                self, "tick_value", float(self.tick_size) * float(self.multiplier)
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["roll_rule"] = self.roll_rule.value
        d["adjustment"] = self.adjustment.value
        d["series_kind"] = self.series_kind.value
        return to_jsonable(d)


class ContinuousContractBuilder:
    """Build continuous research / tradable series from raw contract bars."""

    def __init__(self, config: ContinuousContractConfig) -> None:
        self.config = config

    def build(
        self,
        raw: pd.DataFrame,
        *,
        contracts: Sequence[ContractSpec] | None = None,
    ) -> tuple[pd.DataFrame, list[RollEvent]]:
        """Return ``(continuous_frame, roll_events)``.

        ``raw`` must contain timestamp, instrument (or contract), OHLCV.
        Optional ``open_interest`` required for OI-based rolls.
        """
        df = normalize_frame(raw.copy())
        if "contract" in df.columns:
            # Prefer explicit contract id when present
            df["instrument"] = df["contract"].astype(str).where(
                df["contract"].notna(), df["instrument"]
            )

        contract_meta = {
            c.contract: c for c in (contracts or [])
        }
        # Determine expiry order
        if contract_meta:
            ordered = sorted(contract_meta.values(), key=lambda c: c.expiry)
            contract_ids = [c.contract for c in ordered]
        else:
            contract_ids = sorted(df["instrument"].astype(str).unique().tolist())

        if len(contract_ids) == 0:
            empty = df.iloc[0:0].copy()
            return empty, []

        # Wide panels for close / volume / OI
        close_wide = df.pivot_table(
            index="timestamp", columns="instrument", values="close", aggfunc="last"
        ).sort_index()
        vol_wide = None
        oi_wide = None
        if self.config.volume_col in df.columns:
            vol_wide = df.pivot_table(
                index="timestamp",
                columns="instrument",
                values=self.config.volume_col,
                aggfunc="last",
            ).sort_index()
        if self.config.oi_col in df.columns:
            oi_wide = df.pivot_table(
                index="timestamp",
                columns="instrument",
                values=self.config.oi_col,
                aggfunc="last",
            ).sort_index()

        active = self._select_active_contracts(
            close_wide=close_wide,
            vol_wide=vol_wide,
            oi_wide=oi_wide,
            contract_ids=contract_ids,
            contract_meta=contract_meta,
        )
        rolls = self._detect_rolls(active)
        continuous = self._stitch(df, active, rolls)
        continuous["contract"] = active.reindex(continuous["timestamp"]).to_numpy()
        continuous["instrument"] = self.config.continuous_symbol
        continuous["multiplier"] = float(self.config.multiplier)
        continuous["tick_size"] = float(self.config.tick_size)
        continuous["tick_value"] = float(self.config.tick_value or 0.0)
        if self.config.margin is not None:
            continuous["margin"] = float(self.config.margin)
        if self.config.currency is not None:
            continuous["currency"] = self.config.currency
        continuous["series_kind"] = self.config.series_kind.value
        return normalize_frame(continuous), rolls

    def _select_active_contracts(
        self,
        *,
        close_wide: pd.DataFrame,
        vol_wide: pd.DataFrame | None,
        oi_wide: pd.DataFrame | None,
        contract_ids: list[str],
        contract_meta: Mapping[str, ContractSpec],
    ) -> pd.Series:
        idx = close_wide.index
        active = pd.Series(index=idx, dtype=object)
        rule = self.config.roll_rule

        if rule is RollRule.CALENDAR:
            if not contract_meta:
                # Fall back to lexicographic schedule: use first available contract each day
                for ts in idx:
                    row = close_wide.loc[ts]
                    available = [c for c in contract_ids if c in row.index and pd.notna(row[c])]
                    active.loc[ts] = available[0] if available else None
                return active

            expiries = sorted(contract_meta.values(), key=lambda c: c.expiry)
            lead = pd.Timedelta(days=int(self.config.calendar_days_before_expiry))
            for ts in idx:
                ts_py = pd.Timestamp(ts).to_pydatetime()
                chosen = None
                for spec in expiries:
                    if ts_py <= (spec.expiry - lead.to_pytimedelta()):
                        if spec.contract in close_wide.columns and pd.notna(
                            close_wide.at[ts, spec.contract]
                            if spec.contract in close_wide.columns
                            else np.nan
                        ):
                            chosen = spec.contract
                            break
                if chosen is None:
                    # last available
                    for spec in reversed(expiries):
                        if spec.contract in close_wide.columns and pd.notna(
                            close_wide.at[ts, spec.contract]
                        ):
                            chosen = spec.contract
                            break
                active.loc[ts] = chosen
            return active

        # Volume / OI based: pick max among contracts with a print that day
        metric = vol_wide if rule is RollRule.VOLUME else oi_wide
        if metric is None:
            raise ValueError(
                f"roll_rule={rule.value} requires column "
                f"{self.config.volume_col if rule is RollRule.VOLUME else self.config.oi_col}"
            )
        for ts in idx:
            row = metric.loc[ts] if ts in metric.index else None
            if row is None:
                active.loc[ts] = None
                continue
            candidates = [
                c for c in contract_ids if c in row.index and pd.notna(row[c])
            ]
            if not candidates:  # pragma: no cover - metric-all-NaN with closes present
                # fall back to any contract with a close
                crow = close_wide.loc[ts]
                candidates = [
                    c for c in contract_ids if c in crow.index and pd.notna(crow[c])
                ]
                active.loc[ts] = candidates[0] if candidates else None
                continue
            best = max(candidates, key=lambda c: float(row[c]))
            active.loc[ts] = best
        return active

    def _detect_rolls(self, active: pd.Series) -> list[RollEvent]:
        rolls: list[RollEvent] = []
        prev = None
        for ts, cur in active.items():
            if cur is None:
                continue
            if prev is not None and cur != prev:
                rolls.append(
                    RollEvent(
                        roll_date=pd.Timestamp(ts).to_pydatetime(),
                        from_contract=str(prev),
                        to_contract=str(cur),
                        rule=self.config.roll_rule,
                    )
                )
            prev = cur
        return rolls

    def _stitch(
        self,
        df: pd.DataFrame,
        active: pd.Series,
        rolls: list[RollEvent],
    ) -> pd.DataFrame:
        # Map timestamp -> active contract bars
        keyed = df.set_index(["timestamp", "instrument"], drop=False)
        rows: list[dict[str, Any]] = []
        for ts, contract in active.items():
            if contract is None:
                continue
            key = (ts, str(contract))
            if key not in keyed.index:  # pragma: no cover
                continue
            row = keyed.loc[key]
            if isinstance(row, pd.DataFrame):  # pragma: no cover
                row = row.iloc[-1]
            rows.append(row.to_dict())
        if not rows:  # pragma: no cover
            return df.iloc[0:0].copy()
        out = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

        method = self.config.adjustment
        if method is AdjustmentMethod.UNADJUSTED or not rolls:
            return out

        # Compute roll gaps / ratios at roll dates using closes
        close_by = df.pivot_table(
            index="timestamp", columns="instrument", values="close", aggfunc="last"
        )
        adjustments = []  # (roll_date, gap_or_log_ratio)
        for roll in rolls:
            ts = pd.Timestamp(roll.roll_date)
            if ts not in close_by.index:  # pragma: no cover - off-calendar roll date
                # nearest previous
                prior = close_by.index[close_by.index <= ts]
                if len(prior) == 0:
                    continue
                ts = prior[-1]
            if roll.from_contract not in close_by.columns or roll.to_contract not in close_by.columns:  # pragma: no cover
                continue
            old_px = close_by.at[ts, roll.from_contract]
            new_px = close_by.at[ts, roll.to_contract]
            if pd.isna(old_px) or pd.isna(new_px) or float(new_px) == 0.0:  # pragma: no cover
                continue
            gap = float(new_px) - float(old_px)
            ratio = float(old_px) / float(new_px)
            roll.gap = gap
            roll.ratio = ratio
            adjustments.append((pd.Timestamp(roll.roll_date), gap, ratio))

        if not adjustments:  # pragma: no cover
            return out

        price_cols = [c for c in ("open", "high", "low", "close", "adj_close", "settlement", "vwap") if c in out.columns]
        # Back-adjust: add cumulative gaps to *past* prices (Panama)
        # Ratio: multiply past by cumulative ratio factors
        out = out.copy()
        for roll_ts, gap, ratio in sorted(adjustments, key=lambda x: x[0], reverse=True):
            mask = out["timestamp"] < roll_ts
            if not mask.any():  # pragma: no cover
                continue
            if method is AdjustmentMethod.BACK_ADJUST:
                for col in price_cols:
                    out[col] = out[col].astype(float)
                    out.loc[mask, col] = out.loc[mask, col] + gap
            elif method is AdjustmentMethod.RATIO:
                for col in price_cols:
                    out[col] = out[col].astype(float)
                    out.loc[mask, col] = out.loc[mask, col] * ratio
        return out


def build_continuous_series(
    raw: pd.DataFrame,
    config: ContinuousContractConfig | Mapping[str, Any],
    *,
    contracts: Sequence[ContractSpec] | None = None,
) -> tuple[pd.DataFrame, list[RollEvent]]:
    """Convenience wrapper around :class:`ContinuousContractBuilder`."""
    if not isinstance(config, ContinuousContractConfig):
        payload = dict(config)
        cfg = ContinuousContractConfig(
            root=str(payload["root"]),
            continuous_symbol=str(payload.get("continuous_symbol", payload["root"] + "_c")),
            roll_rule=RollRule(str(payload.get("roll_rule", "volume"))),
            adjustment=AdjustmentMethod(str(payload.get("adjustment", "back_adjust"))),
            series_kind=ContractSeriesKind(
                str(payload.get("series_kind", "continuous_research"))
            ),
            calendar_days_before_expiry=int(payload.get("calendar_days_before_expiry", 5)),
            multiplier=float(payload.get("multiplier", 1.0)),
            tick_size=float(payload.get("tick_size", 0.01)),
            tick_value=payload.get("tick_value"),
            margin=payload.get("margin"),
            currency=payload.get("currency"),
        )
    else:
        cfg = config
    return ContinuousContractBuilder(cfg).build(raw, contracts=contracts)

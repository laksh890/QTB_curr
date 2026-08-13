"""Portfolio constraint specifications and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np


class ConstraintKind(str, Enum):
    MAX_POSITION = "max_position"
    MIN_POSITION = "min_position"
    MAX_GROSS = "max_gross"
    MAX_NET = "max_net"
    MIN_NET = "min_net"
    MAX_LONG = "max_long"
    MAX_SHORT = "max_short"
    MAX_LEVERAGE = "max_leverage"
    MAX_CONCENTRATION = "max_concentration"
    MAX_TURNOVER = "max_turnover"
    LONG_ONLY = "long_only"
    DOLLAR_NEUTRAL = "dollar_neutral"
    CUSTOM = "custom"


@dataclass(slots=True)
class ConstraintSpec:
    """A single named constraint with optional per-asset bounds."""

    kind: ConstraintKind | str
    value: float | None = None
    lower: float | None = None
    upper: float | None = None
    asset: str | None = None
    assets: list[str] = field(default_factory=list)
    hard: bool = True
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = ConstraintKind(self.kind)
        if self.name is None:
            k = self.kind.value if isinstance(self.kind, ConstraintKind) else str(self.kind)
            suffix = f":{self.asset}" if self.asset else ""
            self.name = f"{k}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind.value if isinstance(self.kind, ConstraintKind) else str(self.kind)
        return {
            "kind": kind,
            "value": float(self.value) if self.value is not None else None,
            "lower": float(self.lower) if self.lower is not None else None,
            "upper": float(self.upper) if self.upper is not None else None,
            "asset": self.asset,
            "assets": list(self.assets),
            "hard": bool(self.hard),
            "name": self.name,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstraintSpec:
        return cls(
            kind=data.get("kind", ConstraintKind.CUSTOM),
            value=float(data["value"]) if data.get("value") is not None else None,
            lower=float(data["lower"]) if data.get("lower") is not None else None,
            upper=float(data["upper"]) if data.get("upper") is not None else None,
            asset=data.get("asset"),
            assets=list(data.get("assets") or []),
            hard=bool(data.get("hard", True)),
            name=data.get("name"),
            params=dict(data.get("params") or {}),
        )


@dataclass(slots=True)
class ConstraintViolation:
    """Record of a constraint breach."""

    name: str
    kind: str
    actual: float
    limit: float | None
    message: str
    hard: bool = True
    asset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "actual": float(self.actual),
            "limit": float(self.limit) if self.limit is not None else None,
            "message": self.message,
            "hard": bool(self.hard),
            "asset": self.asset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstraintViolation:
        return cls(
            name=str(data.get("name", "")),
            kind=str(data.get("kind", "")),
            actual=float(data.get("actual", 0.0)),
            limit=float(data["limit"]) if data.get("limit") is not None else None,
            message=str(data.get("message", "")),
            hard=bool(data.get("hard", True)),
            asset=data.get("asset"),
        )


@dataclass(slots=True)
class ConstraintSet:
    """Collection of portfolio constraints with evaluation helpers."""

    constraints: list[ConstraintSpec] = field(default_factory=list)
    long_only: bool = False
    max_weight: float | None = 0.4
    min_weight: float | None = None
    max_gross: float | None = 1.5
    max_net: float | None = None
    min_net: float | None = None
    max_long: float | None = None
    max_short: float | None = None
    max_leverage: float | None = 2.0
    max_concentration: float | None = None
    max_turnover: float | None = None
    dollar_neutral: bool = False
    dollar_neutral_tol: float = 1e-6

    def __post_init__(self) -> None:
        # Materialize common scalar fields as explicit ConstraintSpec entries when absent
        existing = {
            (c.kind.value if isinstance(c.kind, ConstraintKind) else str(c.kind), c.asset)
            for c in self.constraints
        }
        extras: list[ConstraintSpec] = []
        if self.long_only and (ConstraintKind.LONG_ONLY.value, None) not in existing:
            extras.append(ConstraintSpec(kind=ConstraintKind.LONG_ONLY, value=0.0, hard=True))
        if self.max_weight is not None and (ConstraintKind.MAX_POSITION.value, None) not in existing:
            extras.append(ConstraintSpec(kind=ConstraintKind.MAX_POSITION, value=float(self.max_weight)))
        if self.min_weight is not None and (ConstraintKind.MIN_POSITION.value, None) not in existing:
            extras.append(ConstraintSpec(kind=ConstraintKind.MIN_POSITION, value=float(self.min_weight)))
        if self.max_gross is not None and (ConstraintKind.MAX_GROSS.value, None) not in existing:
            extras.append(ConstraintSpec(kind=ConstraintKind.MAX_GROSS, value=float(self.max_gross)))
        if self.max_net is not None and (ConstraintKind.MAX_NET.value, None) not in existing:
            extras.append(ConstraintSpec(kind=ConstraintKind.MAX_NET, value=float(self.max_net)))
        if self.min_net is not None and (ConstraintKind.MIN_NET.value, None) not in existing:
            extras.append(ConstraintSpec(kind=ConstraintKind.MIN_NET, value=float(self.min_net)))
        if self.max_long is not None and (ConstraintKind.MAX_LONG.value, None) not in existing:
            extras.append(ConstraintSpec(kind=ConstraintKind.MAX_LONG, value=float(self.max_long)))
        if self.max_short is not None and (ConstraintKind.MAX_SHORT.value, None) not in existing:
            extras.append(ConstraintSpec(kind=ConstraintKind.MAX_SHORT, value=float(self.max_short)))
        if self.max_leverage is not None and (ConstraintKind.MAX_LEVERAGE.value, None) not in existing:
            extras.append(ConstraintSpec(kind=ConstraintKind.MAX_LEVERAGE, value=float(self.max_leverage)))
        if self.max_concentration is not None and (ConstraintKind.MAX_CONCENTRATION.value, None) not in existing:
            extras.append(
                ConstraintSpec(kind=ConstraintKind.MAX_CONCENTRATION, value=float(self.max_concentration))
            )
        if self.max_turnover is not None and (ConstraintKind.MAX_TURNOVER.value, None) not in existing:
            extras.append(ConstraintSpec(kind=ConstraintKind.MAX_TURNOVER, value=float(self.max_turnover)))
        if self.dollar_neutral and (ConstraintKind.DOLLAR_NEUTRAL.value, None) not in existing:
            extras.append(
                ConstraintSpec(
                    kind=ConstraintKind.DOLLAR_NEUTRAL,
                    value=0.0,
                    params={"tol": float(self.dollar_neutral_tol)},
                )
            )
        if extras:
            self.constraints = list(self.constraints) + extras

    def add(self, constraint: ConstraintSpec) -> None:
        self.constraints.append(constraint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraints": [c.to_dict() for c in self.constraints],
            "long_only": bool(self.long_only),
            "max_weight": float(self.max_weight) if self.max_weight is not None else None,
            "min_weight": float(self.min_weight) if self.min_weight is not None else None,
            "max_gross": float(self.max_gross) if self.max_gross is not None else None,
            "max_net": float(self.max_net) if self.max_net is not None else None,
            "min_net": float(self.min_net) if self.min_net is not None else None,
            "max_long": float(self.max_long) if self.max_long is not None else None,
            "max_short": float(self.max_short) if self.max_short is not None else None,
            "max_leverage": float(self.max_leverage) if self.max_leverage is not None else None,
            "max_concentration": float(self.max_concentration) if self.max_concentration is not None else None,
            "max_turnover": float(self.max_turnover) if self.max_turnover is not None else None,
            "dollar_neutral": bool(self.dollar_neutral),
            "dollar_neutral_tol": float(self.dollar_neutral_tol),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstraintSet:
        return cls(
            constraints=[ConstraintSpec.from_dict(c) for c in (data.get("constraints") or [])],
            long_only=bool(data.get("long_only", False)),
            max_weight=float(data["max_weight"]) if data.get("max_weight") is not None else None,
            min_weight=float(data["min_weight"]) if data.get("min_weight") is not None else None,
            max_gross=float(data["max_gross"]) if data.get("max_gross") is not None else None,
            max_net=float(data["max_net"]) if data.get("max_net") is not None else None,
            min_net=float(data["min_net"]) if data.get("min_net") is not None else None,
            max_long=float(data["max_long"]) if data.get("max_long") is not None else None,
            max_short=float(data["max_short"]) if data.get("max_short") is not None else None,
            max_leverage=float(data["max_leverage"]) if data.get("max_leverage") is not None else None,
            max_concentration=(
                float(data["max_concentration"]) if data.get("max_concentration") is not None else None
            ),
            max_turnover=float(data["max_turnover"]) if data.get("max_turnover") is not None else None,
            dollar_neutral=bool(data.get("dollar_neutral", False)),
            dollar_neutral_tol=float(data.get("dollar_neutral_tol", 1e-6)),
        )

    def evaluate(
        self,
        weights: Sequence[float] | np.ndarray,
        *,
        names: Sequence[str] | None = None,
        current_weights: Sequence[float] | np.ndarray | None = None,
    ) -> list[ConstraintViolation]:
        return evaluate_constraints(
            weights,
            self,
            names=names,
            current_weights=current_weights,
        )


def _as_weights(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)


def gross_exposure(weights: Sequence[float] | np.ndarray) -> float:
    return float(np.sum(np.abs(_as_weights(weights))))


def net_exposure(weights: Sequence[float] | np.ndarray) -> float:
    return float(np.sum(_as_weights(weights)))


def long_exposure(weights: Sequence[float] | np.ndarray) -> float:
    w = _as_weights(weights)
    return float(np.sum(w[w > 0.0])) if w.size else 0.0


def short_exposure(weights: Sequence[float] | np.ndarray) -> float:
    w = _as_weights(weights)
    return float(np.sum(np.abs(w[w < 0.0]))) if w.size else 0.0


def leverage(weights: Sequence[float] | np.ndarray) -> float:
    return gross_exposure(weights)


def concentration_hhi(weights: Sequence[float] | np.ndarray) -> float:
    """Herfindahl–Hirschman index on absolute weights (normalized by gross)."""
    w = np.abs(_as_weights(weights))
    s = float(np.sum(w))
    if s <= 0.0:
        return 0.0
    p = w / s
    return float(np.sum(p * p))


def max_position_weight(weights: Sequence[float] | np.ndarray) -> float:
    w = _as_weights(weights)
    return float(np.max(np.abs(w))) if w.size else 0.0


def turnover(
    weights: Sequence[float] | np.ndarray,
    current_weights: Sequence[float] | np.ndarray | None,
) -> float:
    w = _as_weights(weights)
    if current_weights is None:
        return float(np.sum(np.abs(w)))
    cur = _as_weights(current_weights)
    n = max(w.size, cur.size)
    if w.size < n:
        w = np.pad(w, (0, n - w.size))
    if cur.size < n:
        cur = np.pad(cur, (0, n - cur.size))
    return float(0.5 * np.sum(np.abs(w - cur)))


def evaluate_max_position(
    weights: Sequence[float] | np.ndarray,
    limit: float,
    *,
    names: Sequence[str] | None = None,
    hard: bool = True,
) -> list[ConstraintViolation]:
    w = _as_weights(weights)
    violations: list[ConstraintViolation] = []
    for i, wi in enumerate(w):
        if abs(float(wi)) > float(limit) + 1e-12:
            asset = names[i] if names is not None and i < len(names) else str(i)
            violations.append(
                ConstraintViolation(
                    name=f"max_position:{asset}",
                    kind=ConstraintKind.MAX_POSITION.value,
                    actual=float(abs(wi)),
                    limit=float(limit),
                    message=f"Position weight |{wi:.6f}| exceeds max_position {limit}",
                    hard=hard,
                    asset=asset,
                )
            )
    return violations


def evaluate_min_position(
    weights: Sequence[float] | np.ndarray,
    limit: float,
    *,
    names: Sequence[str] | None = None,
    hard: bool = True,
    only_active: bool = True,
) -> list[ConstraintViolation]:
    """Flag active positions whose absolute weight is below ``limit``."""
    w = _as_weights(weights)
    violations: list[ConstraintViolation] = []
    for i, wi in enumerate(w):
        aw = abs(float(wi))
        if only_active and aw <= 1e-12:
            continue
        if aw + 1e-12 < float(limit):
            asset = names[i] if names is not None and i < len(names) else str(i)
            violations.append(
                ConstraintViolation(
                    name=f"min_position:{asset}",
                    kind=ConstraintKind.MIN_POSITION.value,
                    actual=aw,
                    limit=float(limit),
                    message=f"Position weight |{wi:.6f}| below min_position {limit}",
                    hard=hard,
                    asset=asset,
                )
            )
    return violations


def evaluate_gross(
    weights: Sequence[float] | np.ndarray,
    limit: float,
    *,
    hard: bool = True,
) -> list[ConstraintViolation]:
    g = gross_exposure(weights)
    if g > float(limit) + 1e-12:
        return [
            ConstraintViolation(
                name="max_gross",
                kind=ConstraintKind.MAX_GROSS.value,
                actual=g,
                limit=float(limit),
                message=f"Gross exposure {g:.6f} exceeds max_gross {limit}",
                hard=hard,
            )
        ]
    return []


def evaluate_net(
    weights: Sequence[float] | np.ndarray,
    *,
    max_net: float | None = None,
    min_net: float | None = None,
    hard: bool = True,
) -> list[ConstraintViolation]:
    n = net_exposure(weights)
    out: list[ConstraintViolation] = []
    if max_net is not None and n > float(max_net) + 1e-12:
        out.append(
            ConstraintViolation(
                name="max_net",
                kind=ConstraintKind.MAX_NET.value,
                actual=n,
                limit=float(max_net),
                message=f"Net exposure {n:.6f} exceeds max_net {max_net}",
                hard=hard,
            )
        )
    if min_net is not None and n + 1e-12 < float(min_net):
        out.append(
            ConstraintViolation(
                name="min_net",
                kind=ConstraintKind.MIN_NET.value,
                actual=n,
                limit=float(min_net),
                message=f"Net exposure {n:.6f} below min_net {min_net}",
                hard=hard,
            )
        )
    return out


def evaluate_long_short(
    weights: Sequence[float] | np.ndarray,
    *,
    max_long: float | None = None,
    max_short: float | None = None,
    hard: bool = True,
) -> list[ConstraintViolation]:
    out: list[ConstraintViolation] = []
    if max_long is not None:
        lg = long_exposure(weights)
        if lg > float(max_long) + 1e-12:
            out.append(
                ConstraintViolation(
                    name="max_long",
                    kind=ConstraintKind.MAX_LONG.value,
                    actual=lg,
                    limit=float(max_long),
                    message=f"Long exposure {lg:.6f} exceeds max_long {max_long}",
                    hard=hard,
                )
            )
    if max_short is not None:
        sh = short_exposure(weights)
        if sh > float(max_short) + 1e-12:
            out.append(
                ConstraintViolation(
                    name="max_short",
                    kind=ConstraintKind.MAX_SHORT.value,
                    actual=sh,
                    limit=float(max_short),
                    message=f"Short exposure {sh:.6f} exceeds max_short {max_short}",
                    hard=hard,
                )
            )
    return out


def evaluate_leverage(
    weights: Sequence[float] | np.ndarray,
    limit: float,
    *,
    hard: bool = True,
) -> list[ConstraintViolation]:
    lev = leverage(weights)
    if lev > float(limit) + 1e-12:
        return [
            ConstraintViolation(
                name="max_leverage",
                kind=ConstraintKind.MAX_LEVERAGE.value,
                actual=lev,
                limit=float(limit),
                message=f"Leverage {lev:.6f} exceeds max_leverage {limit}",
                hard=hard,
            )
        ]
    return []


def evaluate_concentration(
    weights: Sequence[float] | np.ndarray,
    limit: float,
    *,
    hard: bool = True,
) -> list[ConstraintViolation]:
    hhi = concentration_hhi(weights)
    if hhi > float(limit) + 1e-12:
        return [
            ConstraintViolation(
                name="max_concentration",
                kind=ConstraintKind.MAX_CONCENTRATION.value,
                actual=hhi,
                limit=float(limit),
                message=f"Concentration HHI {hhi:.6f} exceeds max_concentration {limit}",
                hard=hard,
            )
        ]
    return []


def evaluate_constraints(
    weights: Sequence[float] | np.ndarray,
    constraint_set: ConstraintSet | Iterable[ConstraintSpec],
    *,
    names: Sequence[str] | None = None,
    current_weights: Sequence[float] | np.ndarray | None = None,
) -> list[ConstraintViolation]:
    """Evaluate all constraints; never silently relax hard limits."""
    if isinstance(constraint_set, ConstraintSet):
        specs = list(constraint_set.constraints)
    else:
        specs = list(constraint_set)

    w = _as_weights(weights)
    violations: list[ConstraintViolation] = []
    seen_kinds: set[str] = set()

    for spec in specs:
        kind = spec.kind.value if isinstance(spec.kind, ConstraintKind) else str(spec.kind)
        hard = bool(spec.hard)

        if kind == ConstraintKind.MAX_POSITION.value:
            limit = float(spec.value if spec.value is not None else (spec.upper if spec.upper is not None else 1.0))
            if spec.asset is not None and names is not None:
                try:
                    idx = list(names).index(spec.asset)
                except ValueError:
                    continue
                aw = abs(float(w[idx])) if idx < w.size else 0.0
                if aw > limit + 1e-12:
                    violations.append(
                        ConstraintViolation(
                            name=spec.name or f"max_position:{spec.asset}",
                            kind=kind,
                            actual=aw,
                            limit=limit,
                            message=f"Asset {spec.asset} weight {aw:.6f} exceeds {limit}",
                            hard=hard,
                            asset=spec.asset,
                        )
                    )
            else:
                violations.extend(evaluate_max_position(w, limit, names=names, hard=hard))
            seen_kinds.add(kind)

        elif kind == ConstraintKind.MIN_POSITION.value:
            limit = float(spec.value if spec.value is not None else (spec.lower if spec.lower is not None else 0.0))
            violations.extend(evaluate_min_position(w, limit, names=names, hard=hard))
            seen_kinds.add(kind)

        elif kind == ConstraintKind.MAX_GROSS.value:
            if spec.value is not None:
                violations.extend(evaluate_gross(w, float(spec.value), hard=hard))
            seen_kinds.add(kind)

        elif kind == ConstraintKind.MAX_NET.value:
            if spec.value is not None:
                violations.extend(evaluate_net(w, max_net=float(spec.value), hard=hard))
            seen_kinds.add(kind)

        elif kind == ConstraintKind.MIN_NET.value:
            if spec.value is not None:
                violations.extend(evaluate_net(w, min_net=float(spec.value), hard=hard))
            seen_kinds.add(kind)

        elif kind == ConstraintKind.MAX_LONG.value:
            if spec.value is not None:
                violations.extend(evaluate_long_short(w, max_long=float(spec.value), hard=hard))
            seen_kinds.add(kind)

        elif kind == ConstraintKind.MAX_SHORT.value:
            if spec.value is not None:
                violations.extend(evaluate_long_short(w, max_short=float(spec.value), hard=hard))
            seen_kinds.add(kind)

        elif kind == ConstraintKind.MAX_LEVERAGE.value:
            if spec.value is not None:
                violations.extend(evaluate_leverage(w, float(spec.value), hard=hard))
            seen_kinds.add(kind)

        elif kind == ConstraintKind.MAX_CONCENTRATION.value:
            if spec.value is not None:
                violations.extend(evaluate_concentration(w, float(spec.value), hard=hard))
            seen_kinds.add(kind)

        elif kind == ConstraintKind.MAX_TURNOVER.value:
            if spec.value is not None:
                t = turnover(w, current_weights)
                if t > float(spec.value) + 1e-12:
                    violations.append(
                        ConstraintViolation(
                            name=spec.name or "max_turnover",
                            kind=kind,
                            actual=t,
                            limit=float(spec.value),
                            message=f"Turnover {t:.6f} exceeds max_turnover {spec.value}",
                            hard=hard,
                        )
                    )
            seen_kinds.add(kind)

        elif kind == ConstraintKind.LONG_ONLY.value:
            for i, wi in enumerate(w):
                if float(wi) < -1e-12:
                    asset = names[i] if names is not None and i < len(names) else str(i)
                    violations.append(
                        ConstraintViolation(
                            name=spec.name or f"long_only:{asset}",
                            kind=kind,
                            actual=float(wi),
                            limit=0.0,
                            message=f"Negative weight {wi:.6f} violates long_only",
                            hard=hard,
                            asset=asset,
                        )
                    )
            seen_kinds.add(kind)

        elif kind == ConstraintKind.DOLLAR_NEUTRAL.value:
            tol = float(spec.params.get("tol", 1e-6))
            n = net_exposure(w)
            if abs(n) > tol:
                violations.append(
                    ConstraintViolation(
                        name=spec.name or "dollar_neutral",
                        kind=kind,
                        actual=n,
                        limit=0.0,
                        message=f"Net exposure {n:.6f} violates dollar_neutral (tol={tol})",
                        hard=hard,
                    )
                )
            seen_kinds.add(kind)

    return violations


def conflicting_constraints(violations: Sequence[ConstraintViolation]) -> list[str]:
    """Return hard constraint names that conflict with a proposed portfolio."""
    return sorted({v.name for v in violations if v.hard})

"""Order validation: instrument, qty, price, side, type, tick/lot, bands, risk.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Never override hard risk limits.
- Urgency influences aggressiveness but NEVER overrides hard risk.
- Optional ``validate_risk`` callback from Risk Intelligence is authoritative
  for hard limits — rejection cannot be bypassed by urgency or confidence.
- No future information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from iqrp.app.core.exceptions import ValidationError
from iqrp.app.execution.config import ExecutionSettings
from iqrp.app.execution.order_manager.order import Order
from iqrp.app.execution.types import OrderType, Side


class RiskValidator(Protocol):
    def __call__(self, order: Order) -> tuple[bool, str]:
        """Return ``(ok, reason)``. ``ok=False`` is a hard reject."""
        ...


@dataclass
class InstrumentMeta:
    """Static instrument trading constraints known at decision time."""

    symbol: str
    tick_size: float = 0.01
    lot_size: float = 1.0
    min_qty: float = 1.0
    max_qty: float | None = None
    trading_enabled: bool = True
    reference_price: float | None = None
    currency: str = "USD"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValidationError(
                "; ".join(self.errors) or "order validation failed",
                code="ORDER_VALIDATION_FAILED",
                details={"errors": list(self.errors), "warnings": list(self.warnings)},
            )


def _is_multiple(value: float, step: float, tol: float = 1e-9) -> bool:
    if step <= 0:
        return False
    n = round(value / step)
    return abs(value - n * step) <= tol * max(1.0, abs(value))


class OrderValidator:
    """Validate orders against market microstructure and hard risk gates.

    Hard risk limits from ``validate_risk`` (or capital/band settings) are
    NEVER overridden by urgency. Invalid tick/lot/qty/price are always rejected.
    """

    def __init__(
        self,
        settings: ExecutionSettings | None = None,
        *,
        instruments: dict[str, InstrumentMeta] | None = None,
        validate_risk: Callable[[Order], tuple[bool, str]] | None = None,
        available_capital: float | None = None,
    ) -> None:
        self.settings = settings or ExecutionSettings.default()
        self.instruments = {k.upper(): v for k, v in (instruments or {}).items()}
        self.validate_risk = validate_risk
        self.available_capital = available_capital

    def register_instrument(self, meta: InstrumentMeta) -> None:
        self.instruments[meta.symbol.upper()] = meta

    def validate(self, order: Order) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        tl = self.settings.tick_lot

        if not order.instrument or not str(order.instrument).strip():
            errors.append("instrument is required")

        if order.quantity is None or float(order.quantity) <= 0:
            errors.append("quantity must be positive")

        if not isinstance(order.side, Side):
            try:
                Side(order.side)
            except Exception:  # noqa: BLE001
                errors.append(f"invalid side: {order.side!r}")

        if not isinstance(order.order_type, OrderType):
            try:
                OrderType(order.order_type)
            except Exception:  # noqa: BLE001
                errors.append(f"invalid order_type: {order.order_type!r}")

        meta = self.instruments.get(str(order.instrument).upper()) if order.instrument else None
        tick = meta.tick_size if meta else tl.default_tick_size
        lot = meta.lot_size if meta else tl.default_lot_size
        min_qty = meta.min_qty if meta else tl.min_qty
        max_qty = (meta.max_qty if meta and meta.max_qty is not None else tl.max_qty)

        if meta is not None and not meta.trading_enabled:
            errors.append(f"instrument {order.instrument} is not tradeable (trading halted)")

        qty = float(order.quantity) if order.quantity is not None else 0.0
        if qty + 1e-12 < min_qty:
            errors.append(f"quantity {qty} below min_qty {min_qty}")
        if qty > max_qty + 1e-12:
            errors.append(f"quantity {qty} exceeds max_qty {max_qty}")
        if lot > 0 and qty > 0 and not _is_multiple(qty, lot):
            errors.append(f"quantity {qty} not a multiple of lot_size {lot}")

        needs_price = order.order_type in {
            OrderType.LIMIT,
            OrderType.STOP_LIMIT,
            OrderType.LOC,
            OrderType.ICEBERG,
        }
        if needs_price and order.price is None:
            errors.append(f"{order.order_type.value} orders require price")
        if order.order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and order.stop_price is None:
            errors.append(f"{order.order_type.value} orders require stop_price")

        if order.price is not None:
            if float(order.price) <= 0:
                errors.append("price must be positive")
            elif tick > 0 and not _is_multiple(float(order.price), tick):
                errors.append(f"price {order.price} not a multiple of tick_size {tick}")

        if order.stop_price is not None:
            if float(order.stop_price) <= 0:
                errors.append("stop_price must be positive")
            elif tick > 0 and not _is_multiple(float(order.stop_price), tick):
                errors.append(f"stop_price {order.stop_price} not a multiple of tick_size {tick}")

        # Price bands — hard reject when enabled
        bands = self.settings.price_bands
        ref = meta.reference_price if meta else None
        if bands.enabled and order.price is not None:
            if ref is None and bands.require_reference:
                errors.append("reference price required for band check")
            elif ref is not None and ref > 0:
                deviation = abs(float(order.price) - float(ref)) / float(ref)
                if deviation > bands.band_pct + 1e-12:
                    errors.append(
                        f"price {order.price} outside band ({bands.band_pct:.2%}) of reference {ref}"
                    )

        # Capital check — hard limit
        cap = self.settings.capital
        if cap.check_enabled and order.price is not None:
            notional = abs(float(order.quantity) * float(order.price))
            if notional > cap.max_order_notional + 1e-9:
                errors.append(
                    f"order notional {notional} exceeds max_order_notional {cap.max_order_notional}"
                )
            if self.available_capital is not None and notional > self.available_capital + 1e-9:
                errors.append(
                    f"order notional {notional} exceeds available capital {self.available_capital}"
                )

        # Optional risk engine callback — hard reject, NEVER overridden by urgency
        if self.settings.risk.require_risk_callback and self.validate_risk is None:
            errors.append("risk validation callback required but not configured")
        if self.validate_risk is not None and self.settings.risk.enforce_hard_limits:
            try:
                ok, reason = self.validate_risk(order)
            except Exception as exc:  # noqa: BLE001
                ok, reason = False, f"risk callback error: {exc}"
            if not ok:
                errors.append(reason or "rejected by risk engine (hard limit)")

        # Urgency never bypasses — informational warning only
        if order.urgency.value == "CRITICAL" and errors:
            warnings.append(
                "CRITICAL urgency does not override hard risk/validation failures"
            )

        return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)

    def validate_or_raise(self, order: Order) -> ValidationResult:
        result = self.validate(order)
        result.raise_if_invalid()
        return result

# Slippage

Pre-trade expected slippage estimation and post-trade realized slippage analytics for the Institutional Execution Platform.

**Package:** `iqrp.app.execution.slippage`  
**Primary APIs:** `estimate_slippage`, `realized_slippage`, `compare_expected_realized`, `market_impact`

Related: [ExecutionCosts](ExecutionCosts.md) · [ExecutionPlatform](ExecutionPlatform.md) · [ImplementationShortfall](ImplementationShortfall.md)

> Portfolio Construction documents optimization-time TC under [TransactionCosts](TransactionCosts.md). This document covers **execution** slippage models only.

---

## Concepts

| Term | Meaning |
|------|---------|
| **Expected slippage** | Pre-trade model of adverse price move vs mid (bps / price / notional) |
| **Realized slippage** | Post-trade VWAP vs arrival / decision / mid benchmarks |
| **Market impact** | Temporary + permanent participation-driven impact |
| **Forecast error** | Realized − expected (model calibration signal) |

Sign convention: positive bps = adverse (buy paid more / sell received less).

---

## Expected slippage — `estimate_slippage`

```python
from iqrp.app.execution.slippage import estimate_slippage

est = estimate_slippage(
    side="buy",
    quantity=10_000,
    mid=100.0,
    spread=0.02,
    adv=5e6,
    volatility=0.02,
    liquidity=1.0,
    delay_seconds=5.0,
    horizon_seconds=300.0,
    impact_coeff=0.1,
    use_nonlinear=False,
)
print(est["expected_slippage_bps"], est["components"])
```

### Return fields

| Key | Description |
|-----|-------------|
| `expected_slippage` | Price units |
| `expected_slippage_bps` | Basis points vs mid |
| `expected_slippage_notional` | Currency |
| `components` | `spread`, `volatility`, `liquidity`, `temporary_impact`, `permanent_impact`, `delay` |
| `breakdown` | Composite `ExecutionSlippageModel` detail |
| `participation` | `qty / ADV` |

### Component helpers

```python
from iqrp.app.execution.slippage import (
    spread_slippage,
    volatility_slippage,
    liquidity_slippage,
    market_impact,
    nonlinear_impact,
    path_impact,
    effective_spread_bps,
)

spread_slippage(mid=100, spread=0.02, side="buy")
market_impact(side="buy", quantity=1e4, mid=100, adv=5e6, volatility=0.02, spread=0.02)
nonlinear_impact(quantity=1e4, mid=100, adv=5e6, volatility=0.02, exponent=0.6)
```

`ExecutionSlippageModel` / `SlippageBreakdown` compose components without double-counting in the headline total. Set `use_nonlinear=True` on `estimate_slippage` to swap square-root impact for a power-law curve.

### Historical calibration

```python
from iqrp.app.execution.slippage import HistoricalSlippageModel, historical_slippage_bps

bps = historical_slippage_bps(records=[...], participation=0.05)
model = HistoricalSlippageModel.from_records([...])
```

---

## Realized slippage — `realized_slippage`

```python
from iqrp.app.execution.slippage import realized_slippage

fills = [
    {"quantity": 4000, "price": 100.05},
    {"quantity": 6000, "price": 100.08},
]
real = realized_slippage(
    fills,
    side="buy",
    arrival_price=100.0,
    decision_price=99.98,
    mid=100.0,
)
# realized_slippage_bps vs arrival; also decision_ / mid_ variants
```

Computes fill VWAP, then side-signed slip vs arrival, decision, and mid.

---

## Expected vs realized

```python
from iqrp.app.execution.slippage import compare_expected_realized

cmp = compare_expected_realized(
    fills,
    side="buy",
    quantity=10_000,
    mid=100.0,
    arrival_price=100.0,
    spread=0.02,
    adv=5e6,
    volatility=0.02,
)
print(cmp["expected_slippage_bps"], cmp["realized_slippage_bps"], cmp["forecast_error_bps"])
```

`forecast_error_bps = realized − expected` for model monitoring — never invents fills.

---

## Engine integration

```python
from iqrp.app.execution import ExecutionEngine

engine = ExecutionEngine()
print(engine.estimate_slippage(
    side="buy", quantity=5000, market_context={"mid": 50, "spread": 0.01, "adv": 1e6}
))
# or per order / delta map
print(engine.estimate_slippage({"AAPL": 1000}, market_context={"AAPL": {"mid": 190, "adv": 4e7}}))
```

During `execute`, pre-trade slippage is stored under `report.pre_trade["by_parent"][*]["slippage"]`.

---

## Architectural notes

- Point-in-time inputs only; no look-ahead curves.  
- Impact coefficients are configuration, not alpha.  
- Urgency does not alter slippage formulas — it changes schedules that *realize* different slippage.  
- Overfill / residual policy lives in Order Manager, not here.

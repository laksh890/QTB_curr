# Execution Costs (TCA)

Execution-layer transaction cost analysis: pre-trade cost estimates and post-trade attribution (implementation shortfall, fees, opportunity).

**Package:** `iqrp.app.execution.transaction_costs`  
**Primary APIs:** `pre_trade_cost_estimate`, `post_trade_cost_analysis`

Related: [Slippage](Slippage.md) · [ExecutionPlatform](ExecutionPlatform.md) · [ImplementationShortfall](ImplementationShortfall.md)

> **Do not confuse with** Portfolio Construction’s [TransactionCosts](TransactionCosts.md) (`iqrp.app.portfolio`), which models costs *inside* optimization. This document covers the **execution TCA** package only. Filename is intentionally `ExecutionCosts.md`.

---

## Cost stack

| Component | Module | Pre-trade | Post-trade |
|-----------|--------|-----------|------------|
| Commissions | `commissions` | ✓ | ✓ |
| Exchange fees | `exchange_fees` | ✓ | ✓ |
| Spread | `spread` | informational / in slippage | attribution |
| Slippage | `slippage` | composite expected | realized vs arrival |
| Market impact | `market_impact` | optional separate | trading-cost proxy |
| Financing | `financing` | ✓ | ✓ |
| Borrow | `borrow_cost` | shorts | shorts |
| FX | inline bps | ✓ | ✓ |
| Opportunity | — | — | unfilled residual vs decision |

When `include_impact_in_slippage=True` (default), headline `total_cost` charges commissions + fees + composite slippage + financing + borrow + FX — **not** raw spread/impact again (those appear as informational fields).

---

## Pre-trade — `pre_trade_cost_estimate`

```python
from iqrp.app.execution.transaction_costs import pre_trade_cost_estimate

est = pre_trade_cost_estimate(
    side="buy",
    quantity=10_000,
    mid=100.0,
    spread=0.02,
    adv=5e6,
    volatility=0.02,
    liquidity=1.0,
    commission_bps=1.0,
    fee_bps=0.3,
    impact_coeff=0.1,
    financing_rate=0.05,
    financing_days=1.0,
    borrow_rate=0.0,
    fx_cost_bps=0.0,
)
print(est["total_cost"], est["total_cost_bps"], est["components"])
```

### Key outputs

| Field | Meaning |
|-------|---------|
| `total_cost` / `total_cost_bps` | Expected all-in cost |
| `expected_slippage` / `_bps` | From slippage composite |
| `expected_market_impact` / `_bps` | Standalone impact model |
| `components` | Currency breakdown |
| `details` | Nested per-component dicts |

Aliases: `pre_trade_estimate`.

### Engine

```python
from iqrp.app.execution import ExecutionEngine

engine = ExecutionEngine()
print(engine.estimate_costs({"AAPL": 1000}, market_context={"AAPL": {"mid": 190, "spread": 0.02, "adv": 4e7}}))
```

---

## Post-trade — `post_trade_cost_analysis`

```python
from iqrp.app.execution.transaction_costs import post_trade_cost_analysis

fills = [
    {"quantity": 4000, "price": 100.04},
    {"quantity": 5500, "price": 100.07},
]
tca = post_trade_cost_analysis(
    fills,
    side="buy",
    arrival_price=100.0,
    decision_price=99.97,
    mid=100.02,
    spread=0.02,
    parent_quantity=10_000,
    benchmark_vwap=100.03,
    benchmark_twap=100.01,
    commission_bps=1.0,
    fee_bps=0.3,
)
print(tca["implementation_shortfall"], tca["cost_attribution"], tca["fill_rate"])
```

### Attribution (`cost_attribution`)

| Bucket | Definition |
|--------|------------|
| `delay_cost` | Mid moved vs decision before/at trading |
| `trading_cost` | VWAP vs mid (execution quality) |
| `spread_cost` | Modelled half-spread cost on filled qty |
| `commissions` / `exchange_fees` | Explicit fees |
| `financing` / `borrow` / `fx` | Carry / locate / FX |
| `opportunity_cost` | Adverse mid vs decision on **unfilled** residual |

### Other outputs

- `realized_cost` / `_bps` — slippage + fees  
- `implementation_shortfall` / `_bps` — decision slip × qty + fees + opportunity  
- `benchmarks` — arrival / decision / VWAP / TWAP slip bps  
- `fill_rate` — `filled / parent`

Aliases: `post_trade_analyze`.

---

## Engine report fields

On `ExecutionEngine.execute`:

- `report.pre_trade["by_parent"]` — per-parent `costs` + `slippage`  
- `report.post_trade["by_parent"]` — `post_trade_cost_analysis` result  
- `report.analytics["by_parent"]` — `execution_quality_report` (IS, fill rate, latency, pre/post compare)

```python
report = engine.execute({"AAPL": 2000}, current={"AAPL": 0}, algo="twap",
                        market_context={"AAPL": {"mid": 190, "spread": 0.02, "adv": 4e7}})
parent_tca = report.post_trade["by_parent"][0]
```

---

## Component APIs (direct)

```python
from iqrp.app.execution.transaction_costs import (
    commission_cost,
    exchange_fees,
    spread_cost,
    slippage_cost,
    market_impact_cost,
    financing_cost,
    borrow_cost,
)

commission_cost(quantity=1000, price=100, commission_bps=1.0, side="buy")
financing_cost(notional=1e5, rate=0.05, days=2)
borrow_cost(notional=1e5, borrow_rate=0.02, days=2, is_short=True)
```

---

## Architectural rules

1. TCA never invents fills or positions — only observes provided fills + benchmarks.  
2. Pre-trade uses point-in-time mid/spread/ADV/vol only.  
3. Impact inside slippage is not double-counted in `total_cost` by default.  
4. Urgency does not alter fee schedules; it changes realized paths.  
5. Keep Portfolio [TransactionCosts](TransactionCosts.md) for optimization; use this package for execution TCA.

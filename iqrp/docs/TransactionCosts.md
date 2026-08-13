# Transaction Costs

Commission, bid–ask spread, slippage, and market-impact models aggregated by `total_transaction_cost`.

Package: `iqrp.app.portfolio.transaction_costs`  
Entry point: `total_transaction_cost`  
Facade: `PortfolioConstructionEngine.transaction_cost` / `construct(..., include_transaction_costs=True)`

Related: [TurnoverControl](TurnoverControl.md) · [PortfolioConstruction](PortfolioConstruction.md) · [LiquidityRisk](LiquidityRisk.md)

---

## Components

| Module | Function | Role |
|--------|----------|------|
| `commissions` | `commission_cost` | bps of notional + optional per-share / minimum |
| `spread` | `spread_cost` | Half-spread (default) × traded notional |
| `slippage` | `slippage_cost` | bps + optional participation / vol term |
| `market_impact` | `market_impact_cost` | Square-root / participation impact vs ADV |
| `total_cost` | `total_transaction_cost` | Sum + per-asset + turnover |

Weight delta \(\Delta w = w_{\text{new}} - w_{\text{old}}\); one-way turnover \(= \tfrac12 \sum |\Delta w_i|\).

---

## `total_transaction_cost`

```python
import numpy as np
from iqrp.app.portfolio.transaction_costs import total_transaction_cost

w0 = np.array([0.4, 0.3, 0.3])
w1 = np.array([0.5, 0.25, 0.25])

tc = total_transaction_cost(
    w0, w1,
    capital=1_000_000,
    prices=np.array([100.0, 50.0, 25.0]),
    adv=np.array([5e6, 3e6, 2e6]),
    spreads=np.array([0.0005, 0.0008, 0.0012]),
    vols=np.array([0.2, 0.3, 0.4]),
    commission_bps=1.0,
    slippage_bps=0.5,
    impact_coeff=0.1,
    half_spread=True,
    include_slippage=True,
    include_impact=True,
)
tc["total"], tc["turnover"], tc["components"].keys()
```

### Key parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `capital` | `1.0` | Portfolio NAV for notional |
| `prices` | optional | Share counts / ADV scaling |
| `adv` | optional | Average daily volume (impact / participation) |
| `spreads` | optional | Full spread; cost uses half when `half_spread=True` |
| `vols` | optional | Volatility for slippage / impact |
| `commission_bps` | `1.0` | Commission in basis points of traded notional |
| `commission_per_share` / `min_commission` | `0` | Share + floor |
| `slippage_bps` | `0.0` | Base slippage |
| `impact_coeff` | `0.1` | Market-impact scale |
| `participation_coeff` | `0.0` | Extra participation-driven slippage |
| `include_slippage` / `include_impact` | `True` | Toggle components |

Return payload includes `total`, `components` (each with own `total` / breakdowns), `turnover`, and per-asset costs when available.

---

## Engine usage

```python
from iqrp.app.portfolio import PortfolioConstructionEngine

eng = PortfolioConstructionEngine()
tc = eng.transaction_cost(w0, w1, capital=1e6, prices=prices, adv=adv, spreads=spreads)

result = eng.construct(
    forecasts=mu, returns=R, current_portfolio=w0,
    capital=1e6, prices=prices,
    include_transaction_costs=True, adv=adv, spreads=spreads, vols=vols,
)
result.transaction_cost
```

Architectural rule: when costs are configured / requested, they are included in construction audit; they do not authorize relaxing hard risk or position constraints.

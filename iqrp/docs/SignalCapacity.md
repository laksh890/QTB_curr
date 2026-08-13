# Signal Capacity

Economic capacity of an alpha candidate: ADV participation limits, turnover-implied deployable capital, market impact, scalability curves, and capacity decay with AUM.

**Package:** `iqrp.app.alpha.economics`  
**Engine entry:** `AlphaResearchEngine.analyze_capacity`  
**Parent:** [AlphaResearch](AlphaResearch.md) · Related: [Capacity](Capacity.md) · [TransactionCosts](TransactionCosts.md)

---

## Rule

Gross IC without deployable capital is not institutional alpha. Capacity and costs enter ranking, ensemble weights, and retirement — they are not optional cosmetics.

Architectural reminders:

- Historical Sharpe alone cannot approve
- Alpha approval ≠ trading approval (Risk Intelligence still gates live size)
- Point-in-time ADV / universe membership required for honest estimates

---

## ADV participation and turnover capacity

Core formula (`estimate_capacity`):

```text
max_capital ≈ ADV × max_participation / turnover
```

where `turnover` is the fraction of AUM traded **per period** (same period as ADV), unless `annualize_turnover=True`.

```python
from iqrp.app.alpha import AlphaResearchEngine
from iqrp.app.alpha.economics.capacity import estimate_capacity

eng = AlphaResearchEngine()
cap = eng.analyze_capacity(
    turnover=0.25,          # 25% of AUM traded per day
    adv=8.0e7,              # $80M average daily volume (currency)
    max_participation=0.05, # 5% of ADV
)
# {
#   "max_capital": ...,
#   "adv": ...,
#   "turnover": ...,
#   "max_participation": ...,
#   "daily_trade_budget": ADV * max_participation,
#   "capacity_formula": "adv * max_participation / turnover",
# }

direct = estimate_capacity(
    turnover=60.0,
    adv=8.0e7,
    max_participation=0.05,
    annualize_turnover=True,  # convert annual turnover → per-period
    periods_per_year=252.0,
)
```

| Input | Meaning |
|-------|---------|
| `turnover` | Per-period AUM fraction traded (or annual if `annualize_turnover`) |
| `adv` | Average daily (period) volume in currency units |
| `max_participation` | Max fraction of ADV the strategy may consume (default 0.1) |
| `daily_trade_budget` | `ADV × max_participation` |

High-frequency / short half-life signals ([SignalDecay](SignalDecay.md)) raise turnover and collapse `max_capital`.

---

## Market impact and slippage

```python
import numpy as np
from iqrp.app.alpha.economics.market_impact import market_impact_bps, market_impact_cost
from iqrp.app.alpha.economics.slippage import slippage_bps
from iqrp.app.alpha.economics.transaction_costs import estimate_transaction_cost

participation = 0.04
impact = market_impact_bps(participation, impact_coeff=0.1, vol=0.015)
# impact_bps ≈ impact_coeff * vol * sqrt(participation) * 1e4

slip = slippage_bps(participation, base_bps=1.0, vol=0.015)
dollars = market_impact_cost(notional=1e6, participation=participation, vol=0.015)

tc = estimate_transaction_cost(
    weights_old=np.array([0.0, 0.0]),
    weights_new=np.array([0.5, -0.5]),
    capital=1e6,
    cost_bps=5.0,
)
```

Net edge after impact must remain positive at intended AUM; otherwise mark capacity insufficient before APPROVED.

---

## Scalability

Map AUM → participation → cost bps → implied net Sharpe.

```python
from iqrp.app.alpha.economics.scalability import scalability_curve, scalability_report

curve = scalability_curve(
    capitals=[1e6, 5e6, 2e7, 5e7],
    gross_sharpe=1.5,
    turnover=0.2,
    adv=8e7,
    max_participation=0.05,
    vol=0.01,
    impact_coeff=0.1,
)
# participation, impact_bps, slippage_bps, total_cost_bps, net_sharpe, decay, max_capital

report = scalability_report(
    gross_sharpe=1.5,
    turnover=0.2,
    adv=8e7,
    max_participation=0.05,
    n_points=20,
)
print(report["max_viable_capital"])  # largest grid point with net_sharpe > 0
```

Use `max_viable_capital` as a research ceiling; Risk Intelligence may impose a tighter live limit.

---

## Capacity decay

As capital approaches / exceeds `max_capital`, expected edge decays:

```text
decay_factor = (1 + capital / max_capital) ** (-decay_power)
```

```python
from iqrp.app.alpha.economics.capacity import capacity_decay
import numpy as np

factors = capacity_decay(
    np.array([1e6, 1e7, 5e7]),
    max_capital=2e7,
    decay_power=1.0,
)
# values in (0, 1]
```

Ensemble weighting treats higher capacity as a positive quality component; decayed capacity feeds retirement (`capacity_collapse` / `capacity_degradation` in `evaluate_retirement`).

---

## Linking to the engine and lifecycle

```python
from iqrp.app.alpha.monitoring.retirement import evaluate_retirement

cap_now = eng.analyze_capacity(turnover=0.3, adv=5e7, max_participation=0.05)
decision = evaluate_retirement(
    ic_recent=0.01,
    ic_baseline=0.04,
    net_sharpe=-0.1,
    capacity=cap_now["max_capital"] * 0.2,
    capacity_baseline=cap_now["max_capital"],
)
# may recommend DEGRADED / RETIRED on capacity_collapse
```

Document intended AUM and ADV assumptions in experiment metadata before `approve()`. Capacity analysis is mandatory triage for short-horizon signals; it is never a substitute for economic hypothesis or statistical validation.

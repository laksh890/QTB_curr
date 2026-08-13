# Capacity Management

Liquidity- and market-impact-aware capacity scaling for institutional capital allocation.

Package: `iqrp.app.risk.capital`  
Module: `capacity.py` (`estimate_capacity`)  
Liquidity primitive: `iqrp.app.risk.market.liquidity.liquidity_risk`  
Hydra: `iqrp/configs/risk/capital/default.yaml`

Related: [Capital Allocation](CapitalAllocation.md) · [Liquidity Risk](LiquidityRisk.md) · [Risk Limits](RiskLimits.md)

---

## Purpose

Capacity management answers: **how much notional can this sleeve hold and trade without breaching participation, impact, and liquidation-time constraints?** Missing ADV or spread never implies unlimited capacity — conservative downscales apply.

`CapitalAllocator.allocate` always runs capacity adjustment after method weights. Standalone use:

```python
from iqrp.app.risk.capital import CapitalAllocator, estimate_capacity

alloc = CapitalAllocator()
cap = alloc.capacity(
    ["mom", "mr", "carry"],
    capital=10_000_000,
    weights=[0.4, 0.35, 0.25],
    adv=[5e6, 3e6, 1e6],
    spreads=[0.0004, 0.0008, 0.0015],
    vols=[0.15, 0.20, 0.25],
)
# cap["scales"], cap["scores"], cap["max_notional"],
# cap["missing_capacity"], cap["missing_liquidity"]
```

---

## Inputs and metrics

| Concept | Symbol / field | Role |
|---------|----------------|------|
| **ADV** | `adv` | Average daily volume (share basis when price=1 notional units) |
| **Participation** | `shares / ADV` | Fraction of daily volume consumed by the position |
| **Max participation** | `max_participation` | Hard daily ADV fraction (default `0.10`) |
| **Spread** | `spreads` | Bid–ask as fraction of mid |
| **Impact** | square-root model | `k · σ · √participation` via `impact_coeff` |
| **Slippage** | half-spread + impact | Temporary cost fraction from `liquidity_risk` |
| **TTL** | `capacity_ttl_days` | Days allowed to build / liquidate at participation cap |
| **Vol** | `vols` | Daily (or consistent) volatility for impact |

Underlying call per name:

```python
from iqrp.app.risk.market.liquidity import liquidity_risk

lr = liquidity_risk(
    position_size=notional,
    adv=adv_i,
    spread=spread_i,
    price=1.0,
    volatility=vol_i,
    max_participation=0.10,
    impact_coeff=0.10,
)
score = lr["score"]          # higher = more liquid
measures = lr["measures"]    # participation, slippage, time_to_liquidate, …
```

---

## Capacity math

For each name with weight \(w_i\) and capital \(C\):

\[
\text{notional}_i = C \cdot w_i
\]

\[
\text{daily\_cap}_i = \texttt{max\_participation} \cdot \mathrm{ADV}_i
\]

\[
\text{max\_notional}_i = \text{daily\_cap}_i \cdot \texttt{ttl\_days}
\]

Utilization and soft capacity scale:

- If `max_notional` ≈ 0 → use `missing_capacity_scale`.
- Else `cap_scale = clip(1 / max(util, 1), 0, 1)` where `util = notional / max_notional`.
- Fold liquidity score: `cap_scale *= clip(0.25 + 0.75 * score, 0, 1)`.

Final scale is clipped to `[0, 1]` and applied elementwise to weights before renormalization (when mass remains).

---

## TTL (time-to-liquidate / time-to-build)

TTL is the planning horizon for capacity, not a soft suggestion:

\[
\text{max\_notional} = \texttt{max\_participation} \times \mathrm{ADV} \times \texttt{ttl\_days}
\]

Default `capacity_ttl_days: 5.0`. Positions that cannot be traded within TTL at the participation cap are downscaled. The liquidity module also reports `time_to_liquidate` for the current notional at the same participation cap.

---

## Participation hard clip

After capacity soft scales and hard weight projection, `CapitalAllocator` applies `apply_participation_constraint` so allocated notionals still respect `max_participation` × ADV × TTL. Rebalance uses `rebalance_participation_cap` (default same as `max_participation`).

---

## Missing liquidity — conservative scaling

| Condition | Flag | Behavior |
|-----------|------|----------|
| ADV missing / non-finite / ≤ 0 / wrong length | `missing_capacity` | Fill with `default_adv`, then `cap_scale = min(cap_scale, missing_capacity_scale)` |
| Spread missing / invalid | contributes to `missing_liquidity` | Fill with `default_spread` |
| ADV or spread incomplete | `missing_liquidity` | Multiply scale by `missing_liquidity_scale` |

Defaults from Hydra:

```yaml
missing_capacity_scale: 0.50
missing_liquidity_scale: 0.50
default_adv: 1.0e6
default_spread: 0.002
impact_coeff: 0.10
capacity_ttl_days: 5.0
max_participation: 0.10
```

**Never assume unlimited capacity** when market data is absent. Conservative half-scales (defaults) keep allocation operable but smaller until real ADV/spreads arrive.

---

## Integration in allocation pipeline

Order inside `CapitalAllocator.allocate`:

1. Method weights (risk parity, HRP, …)
2. Optional opportunity tilt
3. **`estimate_capacity` → capacity scales**
4. Drawdown / risk-state scales
5. Correlation crowding on weights
6. Hard projection + participation clip

Method `capacity` uses equal seed × capacity scales as the primary weight driver. Method `dynamic` includes liquidity inside `dynamic_risk_scales`.

Audit on `CapitalAllocation`:

- `capacity_adjustment` — per-name scales
- `output["capacity"]` — missing flags and scores
- `reasons` — `capacity_adjustment_applied` or `missing_capacity_or_liquidity_conservative_downscale`

---

## Return payload

`estimate_capacity` returns:

| Key | Content |
|-----|---------|
| `scales` | name → `[0, 1]` capacity multiplier |
| `scores` | name → liquidity score |
| `measures` | name → liquidity measure dicts |
| `max_notional` | name → TTL capacity notional |
| `missing_capacity` / `missing_liquidity` | booleans |
| `parameters` | resolved ADV/spread defaults and coeffs |

```python
from iqrp.app.risk.capital.capacity import apply_capacity_scales
import numpy as np

w = apply_capacity_scales(np.array([0.4, 0.3, 0.3]), scales, names=names)
```

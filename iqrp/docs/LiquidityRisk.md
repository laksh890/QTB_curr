# Liquidity Risk

Liquidity metrics in `iqrp.app.risk.market.liquidity`, limits in `iqrp.app.risk.limits.liquidity_limits`, engine method `RiskIntelligenceEngine.liquidity_risk()`.

---

## Metrics

| Metric | Meaning |
|--------|---------|
| **ADV** | Average daily volume (share or notional basis) |
| **Turnover** | Position shares / ADV (= participation); also interpretable as fraction of daily volume consumed |
| **Spread** | Bid–ask as fraction of mid |
| **Impact** | Temporary square-root impact `k · σ · √participation` |
| **Participation** | `shares / ADV` |
| **Slippage** | Half-spread + temporary impact |
| **TTL** | Time-to-liquidate (days) at `max_participation` cap |
| **ADV coverage** | `ADV · price / position_notional` |
| **Liquidity-adjusted exposure** | Effective exposure after liquidity haircut — see below |
| **Score** | Composite in `[0, 1]` (higher = more liquid) |

---

## Engine / function usage

```python
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings
from iqrp.app.risk.market.liquidity import liquidity_risk

engine = RiskIntelligenceEngine(RiskSettings.default())

liq = engine.liquidity_risk(
    position_size=1_000_000,  # notional
    adv=5_000_000,            # share ADV if price given
    spread=0.0005,            # 5 bps
    price=100.0,
    volatility=0.02,          # daily σ for impact
    max_participation=0.10,   # settings.limits.max_participation
    impact_coeff=0.1,
)

m = liq["measures"]
participation = m["participation"]["value"]
ttl = m["time_to_liquidate"]["value"]
slippage = m["slippage"]["value"]
adv_coverage = m["adv_coverage"]["value"]
score = liq["score"]
```

Equivalent direct call:

```python
liq = liquidity_risk(
    position_size=1e6,
    adv=5e6,
    spread=5e-4,
    price=100.0,
    volatility=0.02,
)
```

### Model detail

- `shares = |position_size| / price`  
- `participation = shares / ADV`  
- `TTL = shares / (max_participation · ADV)`  
- `slippage = 0.5 · spread + impact_coeff · vol · √participation`  
- `slippage_cost = slippage · notional` (in measure metadata)

---

## Turnover

Within this API, **turnover vs ADV** is the participation ratio:

```python
turnover = liq["measures"]["participation"]["value"]  # position / ADV
```

For portfolio turnover (traded notional / NAV), compute upstream and feed participation into limits.

---

## Liquidity-adjusted exposure

Haircut gross exposure by the liquidity score (or inverse slippage) before limit checks / leverage:

```python
gross = float(np.sum(np.abs(weights)))
liq_adj_exposure = gross * (1.0 - slippage)           # simple cost haircut
# or
liq_adj_exposure = gross * score                      # score-scaled capacity
# or scale leverage
lev = engine.recommended_leverage(
    realized_vol=0.12,
    liquidity_score=score,   # [0, 1]
    current_drawdown=0.03,
)
```

Document the chosen haircut policy in desk procedures; the engine supplies the building blocks.

---

## Limits

```python
from iqrp.app.risk.limits import build_liquidity_limits, check_liquidity_limits

lims = build_liquidity_limits(
    max_participation=0.10,
    min_adv_coverage=0.01,
    max_time_to_liquidate=5.0,  # SOFT by default
)

breaches = check_liquidity_limits(
    participation=participation,
    adv_coverage=adv_coverage,
    time_to_liquidate=ttl,
)
```

Hydra:

```yaml
limits:
  max_participation: 0.10
  min_adv_coverage: 0.01
```

Pass `participation` and `adv_coverage` into `validate_position` / `check_limits` for the pre-trade liquidity step.

---

## Feature / execution hooks

- Pull ADV / spread features from the Feature Platform (import-only).  
- Use short-horizon EWMA vol from Volatility Forecasting for `volatility=` in impact — see [RiskIntegration.md](RiskIntegration.md).  
- Execution should respect `max_participation` and TTL when scheduling child orders; Risk validates before send.

# Position Sizing

Sizing utilities in `iqrp.app.risk.sizing`, orchestrated by `RiskIntelligenceEngine.position_size()`.

**Hard rule:** confidence, edge, and Kelly math may **reduce or scale** size within caps — they never authorize unbounded leverage.

---

## Methods

| Method | Config key | Module |
|--------|------------|--------|
| Fixed fractional | `fixed_fractional` | `volatility_target.fixed_fractional_size` |
| Volatility targeting | `volatility_target` | `volatility_target.volatility_target_size` |
| Risk parity / ERC | (weights API) | `risk_parity.risk_parity_weights` / `equal_risk_contribution` |
| Kelly (capped) | `kelly` | `kelly.kelly_fraction` |
| Fractional Kelly | `fractional_kelly` | `fractional_kelly.fractional_kelly` |
| Drawdown-adjusted | `drawdown_adjusted` | `drawdown_adjusted.drawdown_adjusted_size` |
| Confidence-adjusted | (post-process) | `confidence_adjusted_size` |
| Regime-adjusted | (post-process) | `regime_adjusted_size` |

Default method in Hydra: `sizing.method: volatility_target`.

---

## Volatility targeting

```python
from iqrp.app.risk.sizing import volatility_target_size

sz = volatility_target_size(
    realized_vol=0.20,
    target_vol=0.10,
    max_leverage=2.0,
)
# size = clip(target_vol / realized_vol, 0, max_leverage)
```

Prefer `realized_vol` / forecast σ from [Volatility Forecasting](VolatilityForecasting.md) ([RiskIntegration.md](RiskIntegration.md)).

---

## Fixed fractional

```python
from iqrp.app.risk.sizing import fixed_fractional_size

sz = fixed_fractional_size(
    equity=1_000_000,
    risk_fraction=0.01,   # settings.sizing.risk_per_trade
    stop_distance=0.02,
)
```

---

## Risk parity / equal risk contribution

```python
from iqrp.app.risk.sizing import risk_parity_weights, equal_risk_contribution

w = risk_parity_weights(cov)["weights"]
erc = equal_risk_contribution(cov)
```

Long-only ERC via iterative update; use as **proposal** weights, then pass through `validate_position`.

---

## IMPORTANT: Kelly safety

**Never use raw unlimited Kelly in production.**

`kelly_fraction` always:

1. Computes a raw Kelly estimate (binary / continuous / edge-odds forms).  
2. **Clips to `[0, max_kelly]`** (default `0.5` from config).  
3. Records `raw_kelly` in parameters for audit — the **returned value is capped**.

```python
from iqrp.app.risk.sizing import kelly_fraction, fractional_kelly

# Capped full Kelly — NEVER unbounded
k = kelly_fraction(edge=0.05, win_prob=0.55, odds=1.0, max_kelly=0.5)
assert 0.0 <= k.value <= 0.5

# Fractional Kelly = fraction * capped_kelly (still ≤ max_kelly)
fk = fractional_kelly(
    edge=0.05,
    win_prob=0.55,
    fraction=0.25,    # settings.sizing.kelly_fraction
    max_kelly=0.5,    # settings.sizing.max_kelly
)
```

### Mandatory safety stack

| Control | How |
|---------|-----|
| Fractional Kelly | `kelly_fraction` × `fraction` ≤ 1 |
| Maximum Kelly cap | `max_kelly` hard clip |
| Drawdown reduction | `drawdown_adjusted_size` / DD scalar in leverage |
| Volatility adjustment | vol-target path / `recommended_leverage` vol scalar |
| Liquidity adjustment | `liquidity_score` in `recommended_leverage` |
| Confidence adjustment | `confidence_adjusted_size` — **ceiling at `max_scale≤1`** (cannot expand past base) |

Architectural rules 5–6: forecast confidence cannot lift hard limits or unlock unlimited Kelly/leverage.

---

## Drawdown-adjusted sizing

```python
from iqrp.app.risk.sizing import drawdown_adjusted_size, volatility_target_size

base = volatility_target_size(realized_vol=0.15, target_vol=0.10, max_leverage=2.0).value
sz = drawdown_adjusted_size(
    base_size=base,
    current_drawdown=0.08,
    max_drawdown_limit=0.20,  # settings.drawdown.trading_halt
)
# Linearly → 0 as DD → limit
```

---

## Confidence & regime adjustments

```python
from iqrp.app.risk.sizing import confidence_adjusted_size, regime_adjusted_size

c = confidence_adjusted_size(base_size=1.0, confidence=0.8, min_scale=0.25, max_scale=1.0)
r = regime_adjusted_size(base_size=c.value, regime="high_vol")
# Defaults: normal=1.0, high_vol=0.5, crisis=0.25, stress=0.35, ...
```

---

## Engine orchestration

```python
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings

engine = RiskIntelligenceEngine(RiskSettings.default())

out = engine.position_size(
    realized_vol=0.12,
    edge=0.03,
    win_prob=0.54,
    current_drawdown=0.05,
    confidence=0.7,
    regime="normal",
    equity=1.0,
    method="fractional_kelly",  # or None → settings.sizing.method
)
# Final size clipped to sizing.max_leverage
print(out["size"], out["note"])
```

Pipeline inside the engine: base method → confidence scale → regime scale → `max_leverage` clip.

Hydra excerpt:

```yaml
sizing:
  method: volatility_target
  target_volatility: 0.10
  kelly_fraction: 0.25
  max_kelly: 0.5
  max_leverage: 2.0
  risk_per_trade: 0.01
```

---

## Dynamic leverage companion

```python
lev = engine.recommended_leverage(
    realized_vol=0.12,
    forecast_vol=0.14,
    current_drawdown=0.05,
    confidence=0.9,
    liquidity_score=0.8,
    regime="normal",
)
# Hard: DD ≥ max_drawdown → min_leverage; confidence_cap bounds upside
```

See [DrawdownControl.md](DrawdownControl.md) and [RiskLimits.md](RiskLimits.md).

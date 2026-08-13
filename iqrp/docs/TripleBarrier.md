# Triple Barrier Method

Implementation: `iqrp.app.labels.barrier`

## Barriers

For each entry time \(t\):

1. **Upper barrier** — take-profit / favorable move
2. **Lower barrier** — stop-loss / adverse move
3. **Time barrier** — maximum holding horizon

First touch wins. Outputs:

| Column | Meaning |
|--------|---------|
| `tb_hit_type` | `1` upper, `-1` lower, `0` time |
| `tb_hit_time` | Bars until hit |
| `tb_return` | Return realized at hit |
| `tb_upper` / `tb_lower` | Barrier prices at entry |

## Barrier modes (Hydra `triple_barrier.barrier_mode`)

- `fixed` — constant percentage bands
- `atr` — dynamic ATR-scaled bands (default)
- `volatility` — rolling realized-vol scaled bands

## Usage

```python
from iqrp.app.labels import LabelPipeline, triple_barrier_frame

out, _ = LabelPipeline().compute(ohlcv, ["triple_barrier"])
# or functional API with overrides
tb = triple_barrier_frame(ohlcv, barrier_mode="fixed", fixed_upper=0.01, horizon=30)
```

## Notes

- Path evaluation uses high/low within the horizon (not close-only).
- Trailing rows with insufficient future path are NaN — expected, not leakage.
- See `LabelValidator` for look-ahead / quality checks.

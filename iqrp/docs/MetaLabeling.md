# Meta Labeling

Implementation: `iqrp.app.labels.meta`

Meta-labeling answers: *given a primary directional signal, was the trade profitable?*

It does **not** create primary trading signals for production. Primary sides are inputs.

## Components

| Label | Role |
|-------|------|
| `meta_label` | 1 if `side * future_return > 0`, else 0 (NaN when side=0) |
| `meta_return` | Signed primary PnL proxy `side * future_return` |
| `probability_label` | Soft logistic transform of future return |
| `trade_filter_label` | 1 if `|future_return|` exceeds threshold |

## Inputs

- Preferred: `primary_signal` column in `{-1, 0, +1}`
- Optional: confirmation column (forces meta=0 when confirmation ≤ 0)
- If `primary_signal` is absent on research OHLCV frames, a synthetic side from next-bar return sign is used so pipelines remain executable

## API

```python
from iqrp.app.labels import meta_label_frame, secondary_confirmation

labeled = meta_label_frame(
    frame,
    primary_signal_column="primary_signal",
    horizon=12,
    confirmation_column="confirm",
)
confirm = secondary_confirmation(
    frame,
    primary_signal_column="primary_signal",
    confirmation_column="confirm",
)
```

## Configuration

`iqrp/configs/labels/default.yaml` → `meta_labeling.*`

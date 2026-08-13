# Feature Registry

## Automatic registration

Features subclass `Feature`, define a `FeatureMeta`, and use `@register_feature`:

```python
@register_feature
class LogReturn(Feature):
    meta = FeatureMeta(
        name="log_return",
        version="1.0.0",
        description="Log return of close",
        category="trend",
        required_columns=("close",),
        output_columns=("log_return",),
        window=1,
        parameters={"periods": 1},
    )

    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        ...
```

Importing category packages (via `ensure_features_loaded()`) executes decorators
and populates the process registry.

## Metadata fields

| Field | Purpose |
|-------|---------|
| name | Stable feature id |
| version | Semver string |
| description | Human summary |
| category | trend / momentum / … |
| dependencies | Other feature names |
| required_columns | Input columns |
| output_columns | Produced columns |
| window | Primary lookback |
| parameters | Knobs |
| source | Provenance |
| created_at | UTC timestamp |

## API

```python
from iqrp.app.features import get_registry, describe_feature, feature_dependencies

reg = get_registry()
reg.list_names(category="volatility")
describe_feature("atr")
feature_dependencies("trend_strength")
```

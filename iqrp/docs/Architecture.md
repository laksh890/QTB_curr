# Architecture

## Purpose

IQRP is structured as a long-lived research platform. The foundation separates **cross-cutting infrastructure** from **domain modules** so data, models, risk, portfolio, backtesting, and live execution can evolve independently without architectural rewrites.

Trading logic is intentionally absent in this phase.

## Layering

```text
┌─────────────────────────────────────────────────────────────┐
│  Interfaces                                                 │
│  api / dashboard / cli                                      │
├─────────────────────────────────────────────────────────────┤
│  Application domains                                        │
│  data · features · regimes · forecasting · risk ·           │
│  portfolio · execution · backtesting · live                 │
├─────────────────────────────────────────────────────────────┤
│  Shared kernel                                              │
│  core (exceptions, DI, protocols)                           │
│  config (Hydra + Pydantic)                                  │
│  logging (Loguru / Rich)                                    │
│  common (utilities)                                         │
└─────────────────────────────────────────────────────────────┘
```

### Dependency rule

Domain packages may depend on `core`, `config`, `logging`, and `common`. They must **not** depend on sibling domains through concrete implementations. Cross-domain collaboration goes through protocols registered in the DI container.

## Core concepts

### Exceptions

All platform failures inherit from `IQRPError` with a stable `code` and optional `details`:

- `DataError`
- `ConfigurationError`
- `ValidationError`
- `ModelError`
- `ExecutionError`

### Dependency injection

`Container` registers providers by type or string key. Services are resolved lazily. Prefer constructor injection at composition roots (CLI, API, workers). Use `reset_container()` in tests.

### Configuration

Hydra composes `iqrp/configs/config.yaml` with an environment overlay:

- `environment/development.yaml`
- `environment/testing.yaml`
- `environment/production.yaml`

The composed mapping is validated into a frozen `AppSettings` (Pydantic V2).

### Logging

`setup_logging` configures:

- Rich console output (development)
- JSON console/file output (production)
- Rotating, compressed file sinks

### Async

`AsyncLifecycle` and `HealthCheckable` protocols define start/stop/health contracts. Async retry and cache decorators support I/O-bound paths. CPU-bound numerical work remains synchronous (Polars / NumPy / SciPy).

## Domain module map (future)

| Package | Responsibility |
|---------|----------------|
| `data` | Ingestion, storage (DuckDB/Parquet), validation |
| `features` | Feature pipelines on Polars frames |
| `regimes` | HMM and regime detection |
| `forecasting` | Statistical and ML forecasts |
| `risk` | Risk metrics and limits |
| `portfolio` | Optimization and allocation |
| `execution` | Order abstractions |
| `backtesting` | Event-driven historical simulation |
| `live` | Live orchestration |
| `api` | Service endpoints |
| `dashboard` | Monitoring UI |

## Non-goals (this foundation)

- Strategy or signal code
- Broker integrations
- Model training pipelines
- Production market-data feeds

## Evolution guidelines

1. Keep domain packages thin at the edges; push shared policy into `core`.
2. Prefer configuration over code for environment differences.
3. Add new dependencies only when a domain module needs them — keep the foundation lean.
4. Preserve typed public surfaces; avoid leaking Hydra/OmegaConf objects past the config layer.
5. Treat tests as contracts: unit for utilities, integration for composition.

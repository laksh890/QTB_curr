# IQRP — Institutional Quantitative Research Platform

**IQRP** is the foundational repository for an institutional-grade quantitative research platform. This release establishes architecture, tooling, configuration, logging, and shared utilities only — **no trading logic**.

## What this repository is

A multi-year foundation for:

| Domain | Status |
|--------|--------|
| Data Engineering | Skeleton |
| Feature Engineering | Skeleton |
| Statistical / ML Models | Skeleton |
| Hidden Markov / Regimes | Skeleton |
| Portfolio Optimization | Skeleton |
| Risk Engine | Skeleton |
| Backtesting Engine | Skeleton |
| Live Trading Engine | Skeleton |
| Monitoring Dashboard | Skeleton |

## Design principles

- **Clean Architecture** — domain modules depend inward on core abstractions
- **SOLID** — small interfaces, explicit dependencies
- **Dependency Injection** — process container, no ambient mutable globals
- **Configuration driven** — Hydra overlays for `development` / `testing` / `production`
- **Strong typing** — Python 3.13 + MyPy strict
- **Async where appropriate** — async utilities and lifecycle protocols ready for I/O-bound work
- **Polars-first** — no Pandas unless unavoidable

## Quick start

```bash
# Prerequisites: Python 3.13+, Poetry 2.x
poetry install
poetry run pre-commit install

# Verify
poetry run iqrp version
poetry run iqrp doctor
poetry run iqrp info -e development

# Tests
poetry run pytest

# Lint / format / types
poetry run ruff check iqrp
poetry run black --check iqrp
poetry run mypy iqrp
```

## Documentation

- [Architecture](iqrp/docs/Architecture.md)
- [Getting Started](iqrp/docs/GettingStarted.md)
- [Repository Structure](iqrp/docs/RepositoryStructure.md)

## Stack

Python 3.13 · Poetry · Polars · NumPy · SciPy · Pydantic V2 · Hydra · Loguru · Rich · Typer · DuckDB · PyArrow · httpx · websockets · PyTest · Ruff · Black · MyPy

## Market data layer

See [DataArchitecture.md](iqrp/docs/DataArchitecture.md), [ExchangeAdapters.md](iqrp/docs/ExchangeAdapters.md), [Storage.md](iqrp/docs/Storage.md), and [Validation.md](iqrp/docs/Validation.md).

## Feature engineering

See [FeatureEngineering.md](iqrp/docs/FeatureEngineering.md), [FeatureRegistry.md](iqrp/docs/FeatureRegistry.md), [FeaturePipeline.md](iqrp/docs/FeaturePipeline.md), and [FeatureStore.md](iqrp/docs/FeatureStore.md).

## License

Proprietary — all rights reserved.

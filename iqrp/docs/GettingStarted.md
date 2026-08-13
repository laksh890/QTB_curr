# Getting Started

## Prerequisites

- Python **3.13+**
- [Poetry](https://python-poetry.org/) **2.x**
- Git

Optional:

- Docker (for containerized runs)
- `pre-commit` (installed via Poetry dev deps)

## Install

From the repository root:

```bash
poetry install
poetry run pre-commit install
```

Create local data/log directories:

```bash
poetry run python iqrp/scripts/bootstrap.py
```

## Verify the foundation

```bash
poetry run iqrp version
poetry run iqrp doctor
poetry run iqrp info --environment development
```

## Configuration

Configs live in `iqrp/configs/`.

| Environment | Overlay |
|-------------|---------|
| development | `environment/development.yaml` |
| testing | `environment/testing.yaml` |
| production | `environment/production.yaml` |

Load from Python:

```python
from iqrp.app.config import Environment, load_config
from iqrp.app.logging import setup_logging

settings = load_config(Environment.DEVELOPMENT)
setup_logging(settings.logging)
```

Override the config directory with `IQRP_CONFIG_DIR` if needed.

## Run tests

```bash
# All tests with coverage
poetry run pytest

# Unit only
poetry run pytest -m unit

# Integration only
poetry run pytest -m integration
```

## Code quality

```bash
poetry run ruff check iqrp
poetry run ruff format iqrp
poetry run black iqrp
poetry run mypy iqrp
```

Pre-commit runs Ruff, Black, and related hooks on commit.

## Docker

```bash
docker compose -f iqrp/docker/docker-compose.yml build
docker compose -f iqrp/docker/docker-compose.yml run --rm iqrp iqrp doctor
```

## Project layout (summary)

See [RepositoryStructure.md](RepositoryStructure.md) for the full tree. Application code lives under `iqrp/app/`; tests under `iqrp/tests/`.

## Next steps (out of scope for this foundation)

1. Implement data ingestion adapters under `app/data/ingestion`
2. Add storage repositories under `app/data/storage`
3. Introduce feature transformers under `app/features`
4. Grow regimes, forecasting, risk, and portfolio modules behind DI protocols

# Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml                 # Lint, type-check, test
├── iqrp/
│   ├── __init__.py                # Package version
│   ├── app/
│   │   ├── __init__.py
│   │   ├── cli.py                 # Typer CLI entrypoint
│   │   ├── core/                  # Exceptions, DI, protocols
│   │   ├── config/                # Hydra loader + Pydantic settings
│   │   ├── common/                # Singleton, timer, retry, cache, UUID, datetime
│   │   ├── logging/               # Centralized Loguru setup
│   │   ├── data/
│   │   │   ├── ingestion/         # (skeleton)
│   │   │   ├── storage/           # (skeleton)
│   │   │   └── validation/        # (skeleton)
│   │   ├── features/              # (skeleton)
│   │   ├── regimes/               # (skeleton)
│   │   ├── forecasting/           # (skeleton)
│   │   ├── risk/                  # (skeleton)
│   │   ├── portfolio/             # (skeleton)
│   │   ├── execution/             # (skeleton)
│   │   ├── backtesting/           # (skeleton)
│   │   ├── live/                  # (skeleton)
│   │   ├── api/                   # (skeleton)
│   │   └── dashboard/             # (skeleton)
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── unit/
│   │   └── integration/
│   ├── configs/
│   │   ├── config.yaml
│   │   └── environment/
│   │       ├── development.yaml
│   │       ├── testing.yaml
│   │       └── production.yaml
│   ├── docs/
│   │   ├── Architecture.md
│   │   ├── GettingStarted.md
│   │   └── RepositoryStructure.md
│   ├── notebooks/                 # Research notebooks (empty scaffold)
│   ├── scripts/
│   │   └── bootstrap.py
│   └── docker/
│       ├── Dockerfile
│       └── docker-compose.yml
├── pyproject.toml                 # Poetry, tool configs
├── .pre-commit-config.yaml
├── .gitignore
└── README.md
```

## Package boundaries

| Path | Role |
|------|------|
| `iqrp/app/core` | Kernel: errors, DI, shared protocols |
| `iqrp/app/config` | Configuration composition and validation |
| `iqrp/app/common` | Pure utilities with no domain knowledge |
| `iqrp/app/logging` | Observability setup |
| `iqrp/app/<domain>` | Future domain implementations |
| `iqrp/tests` | Unit and integration contracts |
| `iqrp/configs` | Environment-specific Hydra overlays |

## Naming conventions

- Modules: `snake_case.py`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Error codes: `UPPER_SNAKE_CASE` strings on exceptions
- Tests: `test_*.py` mirroring the unit under test

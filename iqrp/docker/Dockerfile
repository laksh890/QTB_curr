# syntax=docker/dockerfile:1.7

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.1.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    IQRP_CONFIG_DIR=/app/iqrp/configs

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir "poetry==${POETRY_VERSION}"

COPY pyproject.toml README.md ./
COPY iqrp ./iqrp

RUN poetry install --no-ansi --only main --no-root \
    && pip install --no-cache-dir --no-deps .

RUN useradd --create-home --shell /bin/bash iqrp \
    && mkdir -p /app/logs /app/data \
    && chown -R iqrp:iqrp /app

USER iqrp

ENTRYPOINT ["iqrp"]
CMD ["info", "--environment", "development"]

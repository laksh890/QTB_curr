"""Time-series validation splits and model benchmarking."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.intelligence.config import BenchmarkConfig, IntelligenceSettings
from iqrp.app.forecasting.intelligence.ranking import compute_metrics
from iqrp.app.forecasting.intelligence.registry import create_model, list_discovered_models


@dataclass(slots=True)
class BenchmarkResult:
    name: str
    family: str
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "metrics": dict(self.metrics),
            "fold_metrics": list(self.fold_metrics),
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }


def make_splits(
    n: int,
    config: BenchmarkConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    method = config.method
    if method in {"walk_forward", "rolling"}:
        return _walk_forward(n, config)
    if method == "time_series_split":
        return _time_series_split(n, config.n_splits)
    if method in {"nested_cv", "purged_kfold", "embargo"}:
        return _purged_kfold(n, config)
    return _walk_forward(n, config)


def _walk_forward(n: int, config: BenchmarkConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    train_size = max(int(config.train_size), 8)
    test_size = max(int(config.test_size), 4)
    gap = max(int(config.gap), 0)
    start = 0
    while True:
        tr_end = start + train_size
        te_start = tr_end + gap
        te_end = te_start + test_size
        if te_end > n:
            break
        splits.append((np.arange(start, tr_end), np.arange(te_start, te_end)))
        start += test_size if config.method == "rolling" else test_size
        if len(splits) >= max(int(config.n_splits), 1) and config.method == "walk_forward":
            # keep generating until end for rolling; for walk_forward cap splits
            if config.method == "walk_forward" and len(splits) >= config.n_splits:
                break
    if not splits and n > train_size + test_size:
        splits.append((np.arange(0, n - test_size), np.arange(n - test_size, n)))
    return splits


def _time_series_split(n: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = max(int(n_splits), 1)
    splits = []
    for k in range(1, n_splits + 1):
        tr_end = int(n * k / (n_splits + 1))
        te_end = int(n * (k + 1) / (n_splits + 1))
        if te_end <= tr_end or tr_end < 4:
            continue
        splits.append((np.arange(0, tr_end), np.arange(tr_end, te_end)))
    return splits


def _purged_kfold(n: int, config: BenchmarkConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    k = max(int(config.n_splits), 2)
    embargo = max(int(config.embargo), 0)
    purge = max(int(config.purge), 0)
    fold = max(n // k, 4)
    splits = []
    for i in range(k):
        te_start = i * fold
        te_end = n if i == k - 1 else min((i + 1) * fold, n)
        test_idx = np.arange(te_start, te_end)
        train_mask = np.ones(n, dtype=bool)
        lo = max(te_start - purge, 0)
        hi = min(te_end + purge + embargo, n)
        train_mask[lo:hi] = False
        train_idx = np.where(train_mask)[0]
        if train_idx.size < 8 or test_idx.size < 2:
            continue
        splits.append((train_idx, test_idx))
    return splits


def benchmark_model(
    name: str,
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    settings: IntelligenceSettings,
    model_kwargs: dict[str, Any] | None = None,
) -> BenchmarkResult:
    cfg = settings.benchmark
    n = frame.height
    splits = make_splits(n, cfg)
    fold_metrics: list[dict[str, float]] = []
    latencies: list[float] = []
    family = ""
    for tr_idx, te_idx in splits:
        train = frame[tr_idx.tolist()]
        test = frame[te_idx.tolist()]
        model = create_model(name, **(model_kwargs or {}))
        family = getattr(getattr(model, "meta", None), "algorithm_family", "") or family
        t0 = time.perf_counter()
        model.fit(train, feature_columns=feature_columns, target_column=target_column)
        pred = model.predict(test, feature_columns=feature_columns)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)
        y_true = test[target_column].to_numpy().astype(np.float64)
        proba = None
        if getattr(model.meta, "supports_proba", False):
            try:
                proba = model.predict_proba(test, feature_columns=feature_columns)
            except Exception:  # pragma: no cover
                proba = None
        fold_metrics.append(compute_metrics(y_true, pred, probabilities=proba, latency_ms=latency))
    if not fold_metrics:
        # single holdout fallback
        split = max(n // 5, 8)
        train, test = frame[:-split], frame[-split:]
        model = create_model(name, **(model_kwargs or {}))
        family = getattr(getattr(model, "meta", None), "algorithm_family", "") or family
        t0 = time.perf_counter()
        model.fit(train, feature_columns=feature_columns, target_column=target_column)
        pred = model.predict(test, feature_columns=feature_columns)
        latency = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency)
        fold_metrics.append(
            compute_metrics(test[target_column].to_numpy(), pred, latency_ms=latency)
        )
    agg = _aggregate_folds(fold_metrics)
    return BenchmarkResult(
        name=name,
        family=str(family),
        metrics=agg,
        fold_metrics=fold_metrics,
        latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        metadata={"n_folds": len(fold_metrics), "method": cfg.method},
    )


def benchmark_candidates(
    frame: pl.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    settings: IntelligenceSettings,
    candidates: list[str] | None = None,
) -> list[BenchmarkResult]:
    if candidates is None:
        candidates = [m.name for m in list_discovered_models(settings)]
    results: list[BenchmarkResult] = []

    def _one(name: str) -> BenchmarkResult:
        try:
            return benchmark_model(
                name,
                frame,
                feature_columns=feature_columns,
                target_column=target_column,
                settings=settings,
            )
        except Exception as exc:
            return BenchmarkResult(
                name=name,
                family="",
                metrics={"rmse": float("inf"), "mae": float("inf"), "n": 0.0},
                metadata={"error": str(exc)},
            )

    if settings.benchmark.parallel and len(candidates) > 1:
        with ThreadPoolExecutor(
            max_workers=min(settings.benchmark.max_workers, len(candidates))
        ) as ex:
            futs = {ex.submit(_one, n): n for n in candidates}
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for n in candidates:
            results.append(_one(n))
    return results


def _aggregate_folds(folds: list[dict[str, float]]) -> dict[str, float]:
    keys = set()
    for f in folds:
        keys.update(f)
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(f[k]) for f in folds if k in f and np.isfinite(f[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out

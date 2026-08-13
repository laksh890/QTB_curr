"""Feature pipeline: dependency resolution, parallel & incremental execution."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl
from loguru import logger

from iqrp.app.core.exceptions import ConfigurationError, DataError
from iqrp.app.features.base.cache import FeatureCache
from iqrp.app.features.base.registry import FeatureRegistry, ensure_features_loaded, get_registry


@dataclass
class PipelineBenchmarks:
    feature_times_ms: dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0
    memory_bytes_estimate: int = 0
    cache_hit_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_times_ms": dict(self.feature_times_ms),
            "total_time_ms": self.total_time_ms,
            "memory_bytes_estimate": self.memory_bytes_estimate,
            "cache_hit_rate": self.cache_hit_rate,
        }


class FeaturePipeline:
    """Execute registered features with topological ordering and caching."""

    def __init__(
        self,
        *,
        registry: FeatureRegistry | None = None,
        cache: FeatureCache | None = None,
        max_workers: int = 4,
        use_cache: bool = True,
        lazy: bool = False,
    ) -> None:
        ensure_features_loaded()
        self.registry = registry or get_registry()
        self.cache = cache or FeatureCache()
        self.max_workers = max_workers
        self.use_cache = use_cache
        self.lazy = lazy
        self.last_benchmarks = PipelineBenchmarks()

    def resolve_order(self, feature_names: list[str]) -> list[str]:
        """Return features in dependency-satisfying order (all transitive deps)."""
        needed: set[str] = set()
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in needed:
                return
            if name in visiting:
                raise ConfigurationError(
                    "Feature dependency cycle detected",
                    code="FEATURE_DEPENDENCY_CYCLE",
                    details={"requested": feature_names, "cycle_at": name},
                )
            visiting.add(name)
            meta = self.registry.describe(name)
            for dep in meta.dependencies:
                visit(dep)
            visiting.remove(name)
            needed.add(name)

        for name in feature_names:
            visit(name)

        indegree: dict[str, int] = dict.fromkeys(needed, 0)
        edges: dict[str, list[str]] = defaultdict(list)
        for name in needed:
            for dep in self.registry.dependencies(name):
                if dep not in needed:
                    continue
                edges[dep].append(name)
                indegree[name] += 1

        queue = deque(sorted(n for n, d in indegree.items() if d == 0))
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in sorted(edges[node]):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(needed):
            raise ConfigurationError(
                "Feature dependency cycle detected",
                code="FEATURE_DEPENDENCY_CYCLE",
                details={"requested": feature_names},
            )
        return order

    def compute(
        self,
        frame: pl.DataFrame,
        feature_names: list[str] | None = None,
        *,
        parallel: bool = True,
        since: datetime | None = None,
    ) -> tuple[pl.DataFrame, PipelineBenchmarks]:
        """Compute features and return enriched frame + benchmarks.

        If ``lazy`` is True, only resolve order and return the input frame
        unchanged (order is available via :meth:`resolve_order`).
        """
        started = time.perf_counter()
        names = feature_names or self.registry.list_names()
        order = self.resolve_order(names)
        if self.lazy:
            bench = PipelineBenchmarks(total_time_ms=(time.perf_counter() - started) * 1000)
            self.last_benchmarks = bench
            return frame, bench

        work = frame
        if since is not None and "open_time" in work.columns:
            # Incremental: keep history for windows, but only append new outputs later.
            history = work.filter(pl.col("open_time") < since)
            incremental = work.filter(pl.col("open_time") >= since)
            if incremental.is_empty():
                bench = PipelineBenchmarks(
                    total_time_ms=(time.perf_counter() - started) * 1000,
                    cache_hit_rate=self.cache.stats.hit_rate,
                )
                self.last_benchmarks = bench
                return work, bench
            compute_input = pl.concat([history, incremental], how="diagonal_relaxed")
        else:
            compute_input = work
            incremental = work

        # Levels of the dependency graph for parallel batches.
        levels = self._levels(order)
        result = compute_input
        times: dict[str, float] = {}

        for level in levels:
            if parallel and len(level) > 1 and self.max_workers > 1:
                outputs: dict[str, pl.DataFrame] = {}
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {pool.submit(self._compute_one, name, result): name for name in level}
                    for fut in as_completed(futures):
                        name = futures[fut]
                        try:
                            out, elapsed = fut.result()
                        except Exception as exc:
                            raise DataError(
                                f"Feature '{name}' failed: {exc}",
                                code="FEATURE_COMPUTE_FAILED",
                                details={"feature": name},
                            ) from exc
                        outputs[name] = out
                        times[name] = elapsed
                for name in level:
                    result = self._join_outputs(result, outputs[name])
            else:
                for name in level:
                    try:
                        out, elapsed = self._compute_one(name, result)
                    except Exception as exc:
                        raise DataError(
                            f"Feature '{name}' failed: {exc}",
                            code="FEATURE_COMPUTE_FAILED",
                            details={"feature": name},
                        ) from exc
                    result = self._join_outputs(result, out)
                    times[name] = elapsed

        if since is not None and "open_time" in result.columns:
            result = result.filter(pl.col("open_time") >= since)

        mem = int(result.estimated_size())
        bench = PipelineBenchmarks(
            feature_times_ms=times,
            total_time_ms=(time.perf_counter() - started) * 1000,
            memory_bytes_estimate=mem,
            cache_hit_rate=self.cache.stats.hit_rate,
        )
        self.last_benchmarks = bench
        logger.info(
            "feature_pipeline features={} total_ms={:.2f} cache_hit_rate={:.2f}",
            len(order),
            bench.total_time_ms,
            bench.cache_hit_rate,
        )
        return result, bench

    def _compute_one(self, name: str, frame: pl.DataFrame) -> tuple[pl.DataFrame, float]:
        feature = self.registry.get(name)
        key = FeatureCache.make_key(
            feature.meta.name,
            feature.meta.version,
            feature.meta.parameters,
            frame,
            columns=feature.meta.required_columns or tuple(frame.columns),
        )
        t0 = time.perf_counter()
        if self.use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached, (time.perf_counter() - t0) * 1000
        out = feature.run(frame)
        keep = [c for c in ("open_time", *feature.meta.output_columns) if c in out.columns]
        out = out.select(keep)
        if self.use_cache:
            self.cache.put(key, out)
        return out, (time.perf_counter() - t0) * 1000

    @staticmethod
    def _join_outputs(base: pl.DataFrame, feature_frame: pl.DataFrame) -> pl.DataFrame:
        if feature_frame.is_empty():
            return base
        if "open_time" in base.columns and "open_time" in feature_frame.columns:
            cols = [c for c in feature_frame.columns if c == "open_time" or c not in base.columns]
            return base.join(feature_frame.select(cols), on="open_time", how="left")
        # Column-bind when no time key (same row order assumed).
        new_cols = [c for c in feature_frame.columns if c not in base.columns]
        if not new_cols:
            return base
        return base.hstack(feature_frame.select(new_cols))

    def _levels(self, order: list[str]) -> list[list[str]]:
        index = {n: i for i, n in enumerate(order)}
        levels: list[list[str]] = []
        remaining = set(order)
        while remaining:
            ready = [
                n
                for n in order
                if n in remaining and all(d not in remaining for d in self.registry.dependencies(n))
            ]
            if not ready:
                # Should not happen after resolve_order; fall back to sequential.
                ready = [min(remaining, key=lambda x: index[x])]
            levels.append(ready)
            for n in ready:
                remaining.discard(n)
        return levels

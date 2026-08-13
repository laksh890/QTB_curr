"""Label pipeline with dependency resolution and parallel execution."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import polars as pl
from loguru import logger

from iqrp.app.core.exceptions import ConfigurationError, DataError
from iqrp.app.labels.base.registry import LabelRegistry, ensure_labels_loaded, get_registry


@dataclass
class LabelPipelineBenchmarks:
    label_times_ms: dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_times_ms": dict(self.label_times_ms),
            "total_time_ms": self.total_time_ms,
        }


class LabelPipeline:
    def __init__(
        self,
        *,
        registry: LabelRegistry | None = None,
        max_workers: int = 4,
    ) -> None:
        ensure_labels_loaded()
        self.registry = registry or get_registry()
        self.max_workers = max_workers
        self.last_benchmarks = LabelPipelineBenchmarks()

    def resolve_order(self, label_names: list[str]) -> list[str]:
        needed: set[str] = set()
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in needed:
                return
            if name in visiting:
                raise ConfigurationError(
                    "Label dependency cycle detected",
                    code="LABEL_DEPENDENCY_CYCLE",
                    details={"cycle_at": name, "requested": label_names},
                )
            visiting.add(name)
            for dep in self.registry.dependencies(name):
                visit(dep)
            visiting.remove(name)
            needed.add(name)

        for name in label_names:
            visit(name)

        indegree: dict[str, int] = dict.fromkeys(needed, 0)
        edges: dict[str, list[str]] = defaultdict(list)
        for name in needed:
            for dep in self.registry.dependencies(name):
                if dep in needed:
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
                "Label dependency cycle detected",
                code="LABEL_DEPENDENCY_CYCLE",
                details={"requested": label_names},
            )
        return order

    def compute(
        self,
        frame: pl.DataFrame,
        label_names: list[str] | None = None,
        *,
        parallel: bool = True,
    ) -> tuple[pl.DataFrame, LabelPipelineBenchmarks]:
        started = time.perf_counter()
        names = label_names or self.registry.list_names()
        order = self.resolve_order(names)
        levels = self._levels(order)
        result = frame
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
                                f"Label '{name}' failed: {exc}",
                                code="LABEL_COMPUTE_FAILED",
                                details={"label": name},
                            ) from exc
                        outputs[name] = out
                        times[name] = elapsed
                for name in level:
                    result = self._join(result, outputs[name])
            else:
                for name in level:
                    try:
                        out, elapsed = self._compute_one(name, result)
                    except Exception as exc:
                        raise DataError(
                            f"Label '{name}' failed: {exc}",
                            code="LABEL_COMPUTE_FAILED",
                            details={"label": name},
                        ) from exc
                    result = self._join(result, out)
                    times[name] = elapsed

        bench = LabelPipelineBenchmarks(
            label_times_ms=times,
            total_time_ms=(time.perf_counter() - started) * 1000,
        )
        self.last_benchmarks = bench
        logger.info("label_pipeline labels={} total_ms={:.2f}", len(order), bench.total_time_ms)
        return result, bench

    def _compute_one(self, name: str, frame: pl.DataFrame) -> tuple[pl.DataFrame, float]:
        t0 = time.perf_counter()
        label = self.registry.get(name)
        out = label.run(frame)
        keep = [c for c in ("open_time", *label.meta.output_columns) if c in out.columns]
        return out.select(keep), (time.perf_counter() - t0) * 1000

    @staticmethod
    def _join(base: pl.DataFrame, label_frame: pl.DataFrame) -> pl.DataFrame:
        if label_frame.is_empty():
            return base
        if "open_time" in base.columns and "open_time" in label_frame.columns:
            cols = [c for c in label_frame.columns if c == "open_time" or c not in base.columns]
            return base.join(label_frame.select(cols), on="open_time", how="left")
        new_cols = [c for c in label_frame.columns if c not in base.columns]
        if not new_cols:
            return base
        return base.hstack(label_frame.select(new_cols))

    def _levels(self, order: list[str]) -> list[list[str]]:
        levels: list[list[str]] = []
        remaining = set(order)
        while remaining:
            ready = [
                n
                for n in order
                if n in remaining and all(d not in remaining for d in self.registry.dependencies(n))
            ]
            if not ready:
                ready = [sorted(remaining)[0]]
            levels.append(ready)
            for n in ready:
                remaining.discard(n)
        return levels

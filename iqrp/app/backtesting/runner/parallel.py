"""Parallel parameter sweeps with process isolation."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def _run_single(payload: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint (must be importable / picklable)."""
    from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
    from iqrp.app.backtesting.runner.runner import BacktestRunner
    from iqrp.app.backtesting.strategy import (
        BuyAndHoldStrategy,
        CrossSectionalMomentumStrategy,
        StrategyRegistry,
    )

    # Ensure reference strategies exist in the worker process
    for cls in (BuyAndHoldStrategy, CrossSectionalMomentumStrategy):
        try:
            StrategyRegistry.register(cls, overwrite=True)
        except Exception:
            pass

    cfg = BacktestRunConfig.from_dict(payload["config"])
    runner = BacktestRunner(cfg)
    runner.validate()
    runner.prepare()
    runner.run()
    res = runner.result()
    return {
        "experiment_id": payload.get("experiment_id"),
        "seed": payload.get("seed"),
        "status": str(
            runner.status().value if hasattr(runner.status(), "value") else runner.status()
        ),
        "equity_end": float(res.equity_curve[-1]) if res.equity_curve else None,
        "n_fills": len(res.fills),
        "result": res.to_dict(),
    }


def parameter_sweep_parallel(
    base_config: Mapping[str, Any],
    grid: Sequence[Mapping[str, Any]],
    *,
    max_workers: int | None = None,
    seed0: int = 42,
) -> list[dict[str, Any]]:
    """Run isolated backtests for each parameter dict in ``grid``.

    Each experiment receives a unique ``backtest_id``, ``seed``, and deep-copied
    configuration so workers cannot share mutable state.
    """
    jobs: list[dict[str, Any]] = []
    for i, params in enumerate(grid):
        cfg = copy.deepcopy(dict(base_config))
        cfg.update(dict(params))
        seed = int(params.get("seed", seed0 + i))
        exp_id = str(params.get("experiment_id") or f"{cfg.get('backtest_id', 'sweep')}_{i}")
        cfg["backtest_id"] = exp_id
        cfg["seed"] = seed
        # Isolate outputs
        out = cfg.get("output_dir", "results")
        cfg["output_dir"] = str(out)
        jobs.append({"config": cfg, "experiment_id": exp_id, "seed": seed})

    results: list[dict[str, Any]] = []
    if max_workers == 1 or len(jobs) <= 1:
        for job in jobs:
            results.append(_run_single(job))
        return results

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_run_single, job): job for job in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda r: str(r.get("experiment_id", "")))
    return results


__all__ = ["parameter_sweep_parallel"]

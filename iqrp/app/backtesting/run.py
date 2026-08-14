"""CLI entrypoint: ``python -m iqrp.app.backtesting.run``."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iqrp.app.backtesting.run",
        description="Operational institutional backtest runner (Phase 13).",
    )
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--strategy", type=str, default=None, help="strategy_id")
    p.add_argument("--strategy-version", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None, help="dataset path")
    p.add_argument("--adapter", type=str, default=None, choices=["parquet", "csv"])
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--capital", type=float, default=None)
    p.add_argument("--universe", type=str, default=None, help="comma-separated instruments")
    p.add_argument("--output", type=str, default=None, help="output_dir")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", type=str, default=None, help="checkpoint path")
    p.add_argument("--parallel", action="store_true", help="enable parallel sweep if configured")
    p.add_argument("--backtest-id", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from iqrp.app.backtesting.runner import BacktestRunner
    from iqrp.app.backtesting.runner.configuration import BacktestRunConfig
    from iqrp.app.backtesting.strategy import (
        BuyAndHoldStrategy,
        CrossSectionalMomentumStrategy,
        StrategyRegistry,
    )

    for cls in (BuyAndHoldStrategy, CrossSectionalMomentumStrategy):
        try:
            StrategyRegistry.register(cls, overwrite=True)
        except Exception:
            pass

    if args.config:
        cfg = BacktestRunConfig.from_yaml(args.config)
    else:
        cfg = BacktestRunConfig()

    updates: dict = {}
    if args.strategy:
        updates["strategy_id"] = args.strategy
    if args.strategy_version:
        updates["strategy_version"] = args.strategy_version
    if args.dataset:
        updates["dataset_path"] = args.dataset
    if args.adapter:
        updates["adapter"] = args.adapter
    if args.start:
        updates["start"] = args.start
    if args.end:
        updates["end"] = args.end
    if args.capital is not None:
        updates["initial_capital"] = float(args.capital)
    if args.universe:
        updates["universe"] = [s.strip() for s in args.universe.split(",") if s.strip()]
    if args.output:
        updates["output_dir"] = args.output
    if args.seed is not None:
        updates["seed"] = int(args.seed)
    if args.resume:
        updates["resume_from"] = args.resume
    if args.backtest_id:
        updates["backtest_id"] = args.backtest_id
    if updates:
        cfg = cfg.with_updates(**updates)

    runner = BacktestRunner(cfg)
    runner.validate()
    runner.prepare()

    if args.parallel and cfg.parallel.get("grid"):
        results = runner.parameter_sweep(list(cfg.parallel["grid"]))
        print(f"parallel sweep complete: n={len(results)}")
        for row in results:
            print(row.get("experiment_id"), row.get("status"), row.get("equity_end"))
        return 0

    runner.run()
    report = runner.report()
    res = runner.result()
    print(
        f"status={runner.status().value} "
        f"equity_end={res.equity_curve[-1] if res.equity_curve else None} "
        f"report={report}"
    )
    return 0 if runner.status().value == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())

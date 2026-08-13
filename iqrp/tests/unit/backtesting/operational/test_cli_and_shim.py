"""CLI smoke for ``python -m iqrp.app.backtesting.run`` and shim."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from iqrp.app.backtesting.data.synthetic import write_synthetic_ohlcv
from iqrp.app.backtesting.run import build_parser, main
from iqrp.app.backtesting.strategy import (
    BuyAndHoldStrategy,
    CrossSectionalMomentumStrategy,
    StrategyRegistry,
)


def test_build_parser_flags():
    p = build_parser()
    args = p.parse_args(
        [
            "--strategy",
            "buy_and_hold",
            "--dataset",
            "x.parquet",
            "--adapter",
            "parquet",
            "--capital",
            "1000",
            "--seed",
            "3",
            "--universe",
            "AAA,BBB",
            "--parallel",
        ]
    )
    assert args.strategy == "buy_and_hold"
    assert args.parallel is True


def test_main_cli_synthetic(tmp_path: Path):
    StrategyRegistry.clear()
    data = tmp_path / "bars.parquet"
    write_synthetic_ohlcv(data, n_days=35, instruments=["AAA", "BBB"], seed=7)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "backtest_id": "cli_smoke",
                "strategy_id": "buy_and_hold",
                "strategy_version": "1.0.0",
                "dataset_path": str(data),
                "adapter": "parquet",
                "start": "2020-01-01",
                "end": "2020-02-20",
                "initial_capital": 1_000_000,
                "seed": 7,
                "output_dir": str(tmp_path / "results"),
                "spread_bps": 1.0,
                "strategy_params": {"mode": "equal_weight"},
            }
        ),
        encoding="utf-8",
    )
    rc = main(["--config", str(cfg_path)])
    assert rc == 0


def test_main_cli_overrides_and_parallel(tmp_path: Path):
    data = tmp_path / "bars.parquet"
    write_synthetic_ohlcv(data, n_days=30, instruments=["AAA"], seed=1)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "backtest_id": "cli_par",
                "strategy_id": "buy_and_hold",
                "dataset_path": str(data),
                "output_dir": str(tmp_path / "out"),
                "seed": 1,
                "parallel": {
                    "grid": [
                        {"strategy_params": {"mode": "equal_weight"}, "experiment_id": "p0"},
                        {"strategy_params": {"mode": "first_instrument"}, "experiment_id": "p1"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    rc = main(
        [
            "--config",
            str(cfg_path),
            "--parallel",
            "--capital",
            "250000",
            "--output",
            str(tmp_path / "out2"),
            "--backtest-id",
            "cli_par2",
        ]
    )
    assert rc == 0


def test_module_invocation_subprocess(tmp_path: Path):
    data = tmp_path / "bars.parquet"
    write_synthetic_ohlcv(data, n_days=25, instruments=["AAA", "BBB"], seed=5)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "backtest_id": "mod_smoke",
                "strategy_id": "buy_and_hold",
                "dataset_path": str(data),
                "output_dir": str(tmp_path / "results"),
                "seed": 5,
                "start": "2020-01-01",
                "end": "2020-02-10",
            }
        ),
        encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[4]
    py = repo / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)
    proc = subprocess.run(
        [str(py), "-m", "iqrp.app.backtesting.run", "--config", str(cfg_path)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout

    # Shim package
    proc2 = subprocess.run(
        [str(py), "-m", "iqrp.backtesting", "--config", str(cfg_path), "--seed", "5"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc2.returncode == 0, proc2.stderr + proc2.stdout

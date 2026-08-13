"""Nested YAML/config flattening and remaining configuration coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from iqrp.app.backtesting.runner.configuration import BacktestRunConfig, _to_plain


def test_nested_institutional_yaml(tmp_path: Path) -> None:
    payload = {
        "backtest": {
            "id": "nested_bt",
            "strategy_id": "buy_and_hold",
            "strategy_version": "1.0.0",
            "seed": 11,
            "output_dir": str(tmp_path / "out"),
        },
        "market": {
            "universe": ["AAA", "BBB"],
            "frequency": "1d",
            "start": "2020-01-01",
            "end": "2020-02-01",
        },
        "capital": {"initial": 250_000, "currency": "INR"},
        "data": {
            "adapter": "parquet",
            "path": "/tmp/data.parquet",
            "dataset_id": "ds1",
            "dataset_version": "2.0.0",
        },
        "strategy": {"id": "buy_and_hold", "version": "1.0.0", "params": {"mode": "equal_weight"}},
        "execution": {"simulation": "realistic"},
        "risk": {"enabled": True},
        "portfolio": {"enabled": True},
        "walk_forward": {"enabled": False},
        "scenarios": {"enabled": False},
        "output": {"dir": str(tmp_path / "out2")},
    }
    path = tmp_path / "nested.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = BacktestRunConfig.from_yaml(path)
    assert cfg.backtest_id == "nested_bt"
    assert cfg.strategy_id == "buy_and_hold"
    assert cfg.strategy_version == "1.0.0"
    assert cfg.universe == ["AAA", "BBB"]
    assert cfg.frequency == "1d"
    assert cfg.start == "2020-01-01"
    assert cfg.end == "2020-02-01"
    assert cfg.initial_capital == 250_000
    assert cfg.currency == "INR"
    assert cfg.adapter == "parquet"
    assert cfg.dataset_path == "/tmp/data.parquet"
    assert cfg.dataset_id == "ds1"
    assert cfg.dataset_version == "2.0.0"
    assert cfg.strategy_params["mode"] == "equal_weight"
    assert cfg.execution_config.get("simulation") == "realistic"
    assert cfg.risk_config.get("enabled") is True
    assert cfg.portfolio_config.get("enabled") is True
    assert cfg.output_dir == str(tmp_path / "out")  # backtest.output_dir wins over later output


def test_output_block_string_and_capital_scalar() -> None:
    cfg = BacktestRunConfig.from_dict(
        {
            "id": "x",
            "capital": 12345,
            "output": "results/x",
            "strategy": {"id": "buy_and_hold"},
        }
    )
    assert cfg.backtest_id == "x"
    assert cfg.initial_capital == 12345
    assert cfg.output_dir == "results/x"
    assert cfg.strategy_id == "buy_and_hold"


def test_universe_string_and_none() -> None:
    cfg = BacktestRunConfig.from_dict({"universe": "AAA, BBB", "strategy_id": "buy_and_hold"})
    assert cfg.universe == ["AAA", "BBB"]
    cfg2 = BacktestRunConfig.from_dict({"universe": None, "strategy_id": "buy_and_hold"})
    assert cfg2.universe == []


def test_with_updates_and_results_root(tmp_path: Path) -> None:
    cfg = BacktestRunConfig(backtest_id="a", output_dir=str(tmp_path))
    cfg2 = cfg.with_updates(seed=99, strategy_id="buy_and_hold")
    assert cfg2.seed == 99
    assert cfg2.strategy_id == "buy_and_hold"
    assert cfg2.results_root() == tmp_path / "a"


def test_from_omegaconf_roundtrip() -> None:
    from omegaconf import OmegaConf

    cfg = BacktestRunConfig.from_omegaconf(
        OmegaConf.create({"backtest_id": "oc", "strategy_id": "buy_and_hold", "seed": 5})
    )
    assert cfg.backtest_id == "oc"
    assert cfg.seed == 5


def test_to_plain_path_and_namespace() -> None:
    class NS:
        def __init__(self) -> None:
            self.a = 1
            self._hidden = 2

    assert _to_plain(Path("/tmp/x")) == "/tmp/x"
    assert _to_plain(NS())["a"] == 1
    assert "_hidden" not in _to_plain(NS())
    assert _to_plain({"k": (1, 2)}) == {"k": [1, 2]}


def test_from_yaml_invalid_root(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(TypeError):
        BacktestRunConfig.from_yaml(path)

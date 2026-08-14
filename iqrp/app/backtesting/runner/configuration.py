"""Backtest run configuration (YAML / dict / OmegaConf)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def _to_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    # OmegaConf / pydantic / namespace
    if hasattr(obj, "items") and callable(obj.items):
        try:
            return {str(k): _to_plain(v) for k, v in obj.items()}
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {str(k): _to_plain(v) for k, v in vars(obj).items() if not str(k).startswith("_")}
    return obj


def _flatten_nested_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept both flat and nested institutional YAML layouts."""
    out = dict(raw)
    bt = out.pop("backtest", None)
    if isinstance(bt, Mapping):
        if "id" in bt and "backtest_id" not in out:
            out["backtest_id"] = bt["id"]
        for k in ("strategy_id", "strategy_version", "seed", "output_dir"):
            if k in bt and k not in out:
                out[k] = bt[k]

    market = out.pop("market", None)
    if isinstance(market, Mapping):
        if "universe" in market and "universe" not in out:
            out["universe"] = market["universe"]
        if "frequency" in market and "frequency" not in out:
            out["frequency"] = market["frequency"]
        if "start" in market and "start" not in out:
            out["start"] = market["start"]
        if "end" in market and "end" not in out:
            out["end"] = market["end"]

    data = out.pop("data", None)
    if isinstance(data, Mapping):
        if "adapter" in data and "adapter" not in out:
            out["adapter"] = data["adapter"]
        if "path" in data and "dataset_path" not in out:
            out["dataset_path"] = data["path"]
        if "dataset_id" in data and "dataset_id" not in out:
            out["dataset_id"] = data["dataset_id"]
        if "dataset_version" in data and "dataset_version" not in out:
            out["dataset_version"] = data["dataset_version"]

    strategy = out.pop("strategy", None)
    if isinstance(strategy, Mapping):
        if "id" in strategy and "strategy_id" not in out:
            out["strategy_id"] = strategy["id"]
        if "version" in strategy and "strategy_version" not in out:
            out["strategy_version"] = strategy["version"]
        if "params" in strategy and "strategy_params" not in out:
            out["strategy_params"] = strategy["params"]

    for src, dest in (
        ("execution", "execution_config"),
        ("risk", "risk_config"),
        ("portfolio", "portfolio_config"),
        ("walk_forward", "walk_forward_config"),
        ("scenarios", "scenario_config"),
        ("scenario", "scenario_config"),
        ("output", "output_dir"),
    ):
        block = out.pop(src, None)
        if block is None:
            continue
        if dest == "output_dir":
            if isinstance(block, Mapping) and "dir" in block and "output_dir" not in out:
                out["output_dir"] = block["dir"]
            elif isinstance(block, str) and "output_dir" not in out:
                out["output_dir"] = block
        elif isinstance(block, Mapping) and dest not in out:
            out[dest] = dict(block)

    return out


@dataclass
class BacktestRunConfig:
    """Operational configuration for a single backtest run."""

    backtest_id: str = "backtest"
    strategy_id: str = ""
    strategy_version: str = "1.0.0"
    dataset_id: str | None = None
    dataset_version: str | None = None
    dataset_path: str | None = None
    adapter: str = "parquet"
    universe: list[str] = field(default_factory=list)
    start: str | None = None
    end: str | None = None
    frequency: str = "daily"
    initial_capital: float = 1_000_000.0
    currency: str = "USD"
    timezone: str = "UTC"
    seed: int = 42
    output_dir: str = "results"
    commission_bps: float = 0.0
    spread_bps: float = 1.0
    slippage_bps: float = 0.0
    financing_bps: float = 0.0
    risk_config: dict[str, Any] = field(default_factory=dict)
    portfolio_config: dict[str, Any] = field(default_factory=dict)
    execution_config: dict[str, Any] = field(default_factory=dict)
    tcost_config: dict[str, Any] = field(default_factory=dict)
    slippage_config: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)
    walk_forward_config: dict[str, Any] = field(default_factory=dict)
    scenario_config: dict[str, Any] = field(default_factory=dict)
    strategy_params: dict[str, Any] = field(default_factory=dict)
    checkpoint_dir: str | None = None
    resume_from: str | None = None
    parallel: dict[str, Any] = field(default_factory=dict)
    enforce_pit: bool = True
    reconciliation_tolerance: float = 1e-4
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> BacktestRunConfig:
        raw = _to_plain(dict(data or {}))
        raw = _flatten_nested_config(raw)
        if "capital" in raw and "initial_capital" not in raw:
            cap = raw.pop("capital")
            if isinstance(cap, Mapping):
                if "initial" in cap:
                    raw["initial_capital"] = cap["initial"]
                if "currency" in cap and "currency" not in raw:
                    raw["currency"] = cap["currency"]
            else:
                raw["initial_capital"] = cap
        if "id" in raw and "backtest_id" not in raw:
            raw["backtest_id"] = raw.pop("id")
        allowed = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in raw.items() if k in allowed}
        if "universe" in kwargs and kwargs["universe"] is None:
            kwargs["universe"] = []
        if "universe" in kwargs and isinstance(kwargs["universe"], str):
            kwargs["universe"] = [s.strip() for s in kwargs["universe"].split(",") if s.strip()]
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BacktestRunConfig:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(path)
            plain = OmegaConf.to_container(cfg, resolve=True)
            if not isinstance(plain, Mapping):
                raise TypeError("YAML root must be a mapping")
            return cls.from_dict(plain)
        except Exception:
            import yaml

            plain = yaml.safe_load(text) or {}
            if not isinstance(plain, Mapping):
                raise TypeError("YAML root must be a mapping") from None
            return cls.from_dict(plain)

    @classmethod
    def from_omegaconf(cls, cfg: Any) -> BacktestRunConfig:
        try:
            from omegaconf import OmegaConf

            plain = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            plain = _to_plain(cfg)
        if not isinstance(plain, Mapping):
            raise TypeError("OmegaConf root must be a mapping")
        return cls.from_dict(plain)

    def results_root(self) -> Path:
        return Path(self.output_dir) / str(self.backtest_id)

    def with_updates(self, **kwargs: Any) -> BacktestRunConfig:
        base = self.to_dict()
        base.update(kwargs)
        return BacktestRunConfig.from_dict(base)


__all__ = ["BacktestRunConfig"]

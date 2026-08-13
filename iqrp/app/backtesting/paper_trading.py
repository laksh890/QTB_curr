"""Paper-trading handoff preserving strategy versions and configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping

from iqrp.app.backtesting.serializer import to_jsonable

__all__ = ["PaperTradingConfig", "PaperTradingInterface"]


@dataclass
class PaperTradingConfig:
    """Immutable handoff package for paper trading.

    Preserves experiment id, lineage versions, strategy config, and seed so
    live paper runs remain comparable to the validated backtest.
    """

    experiment_id: str
    strategy_name: str = "strategy"
    seed: int = 42
    config: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    scorecard: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = (
        "Paper trading must preserve strategy version, feature/model versions, "
        "and execution config from the promoting backtest."
    )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PaperTradingConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class PaperTradingInterface:
    """Build and track paper-trading configs from promoted experiments."""

    def __init__(self) -> None:
        self._configs: dict[str, PaperTradingConfig] = {}

    def from_result(
        self,
        result: Any,
        *,
        gates: Mapping[str, Any] | None = None,
        strategy_name: str | None = None,
    ) -> PaperTradingConfig:
        eid = getattr(result, "experiment_id", None) or result.get("experiment_id")  # type: ignore[union-attr]
        lineage = getattr(result, "lineage", None)
        if hasattr(lineage, "to_dict"):
            lineage_d = lineage.to_dict()
        elif isinstance(lineage, Mapping):
            lineage_d = dict(lineage)
        else:
            lineage_d = {}

        config = getattr(result, "config", None)
        if hasattr(config, "model_dump"):
            cfg = config.model_dump()
        elif isinstance(config, Mapping):
            cfg = dict(config)
        else:
            cfg = {}

        scorecard = getattr(result, "scorecard", None)
        if scorecard is None and hasattr(result, "metrics"):
            scorecard = getattr(result, "metrics", {})
        if hasattr(scorecard, "to_dict"):
            sc = scorecard.to_dict()
        elif isinstance(scorecard, Mapping):
            sc = dict(scorecard)
        else:
            sc = {}

        seed = int(getattr(result, "seed", lineage_d.get("seed", 42)))
        name = strategy_name or cfg.get("name") or "strategy"
        pt = PaperTradingConfig(
            experiment_id=str(eid),
            strategy_name=str(name),
            seed=seed,
            config=cfg,
            lineage=lineage_d,
            scorecard=sc,
            gates=dict(gates or {}),
        )
        self._configs[pt.experiment_id] = pt
        return pt

    def from_experiment(
        self,
        experiment_id: str,
        registry: Any,
        *,
        gates: Mapping[str, Any] | None = None,
    ) -> PaperTradingConfig:
        rec = registry.require(experiment_id)
        pt = PaperTradingConfig(
            experiment_id=rec.experiment_id,
            strategy_name=rec.name,
            seed=int(rec.lineage.seed),
            config=dict(rec.config),
            lineage=rec.lineage.to_dict(),
            scorecard=dict(rec.metrics),
            gates=dict(gates or {}),
        )
        self._configs[pt.experiment_id] = pt
        return pt

    def get(self, experiment_id: str) -> PaperTradingConfig | None:
        return self._configs.get(experiment_id)

    def list(self) -> list[PaperTradingConfig]:
        return list(self._configs.values())

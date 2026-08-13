"""Component registry for the Institutional Backtesting Platform.

Tracks importable platform modules used by Phase 13 validation and tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ComponentSpec", "BacktestingRegistry", "default_registry"]


@dataclass
class ComponentSpec:
    name: str
    category: str
    import_path: str
    symbol: str
    docs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "import_path": self.import_path,
            "symbol": self.symbol,
            "docs": list(self.docs),
        }


class BacktestingRegistry:
    """Catalog of backtesting platform components."""

    def __init__(self, components: list[ComponentSpec] | None = None) -> None:
        self._components: dict[str, ComponentSpec] = {}
        for c in components or []:
            self.register(c)

    def register(self, spec: ComponentSpec) -> None:
        self._components[spec.name] = spec

    def get(self, name: str) -> ComponentSpec | None:
        return self._components.get(name)

    def list(self, category: str | None = None) -> list[ComponentSpec]:
        items = list(self._components.values())
        if category is not None:
            items = [c for c in items if c.category == category]
        return items

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self._components.values()]


def default_registry() -> BacktestingRegistry:
    """Canonical Phase 13 component catalog."""
    specs = [
        ComponentSpec("Event Engine", "engine", "iqrp.app.backtesting.event_engine", "EventDrivenEngine", ["EventEngine.md", "BacktestingPlatform.md"]),
        ComponentSpec("Walk-Forward", "walk_forward", "iqrp.app.backtesting.walk_forward", "WalkForwardEngine", ["WalkForward.md"]),
        ComponentSpec("Rolling Retraining", "rolling", "iqrp.app.backtesting.rolling_retraining", "RollingRetrainer", ["RollingRetraining.md"]),
        ComponentSpec("Performance Metrics", "performance", "iqrp.app.backtesting.performance", "build_scorecard", ["PerformanceMetrics.md"]),
        ComponentSpec("Scenario Testing", "scenarios", "iqrp.app.backtesting.scenarios", "ScenarioEngine", ["ScenarioTesting.md"]),
        ComponentSpec("Capacity Testing", "capacity", "iqrp.app.backtesting.capacity", "capacity_curve", ["CapacityTesting.md"]),
        ComponentSpec("Parameter Robustness", "robustness", "iqrp.app.backtesting.robustness", "parameter_sweep", ["ParameterRobustness.md"]),
        ComponentSpec("Validation Gates", "gates", "iqrp.app.backtesting.validation_gates", "evaluate_gates", ["StrategyValidation.md"]),
        ComponentSpec("Paper Trading", "paper", "iqrp.app.backtesting.paper_trading", "PaperTradingInterface", ["StrategyValidation.md"]),
        ComponentSpec("Experiment Registry", "registry", "iqrp.app.backtesting.experiment_registry", "ExperimentRegistry", ["Reproducibility.md"]),
        ComponentSpec("PIT / Leakage", "pit", "iqrp.app.backtesting.pit", "detect_leakage", ["Reproducibility.md"]),
        ComponentSpec("Corporate Actions", "corporate", "iqrp.app.backtesting.corporate_actions", "actions_asof", ["BacktestingPlatform.md"]),
        ComponentSpec("Comparison", "comparison", "iqrp.app.backtesting.comparison", "compare_strategies", ["PerformanceMetrics.md"]),
        ComponentSpec("Reports", "reports", "iqrp.app.backtesting.reports", "full_report", ["BacktestingPlatform.md"]),
        ComponentSpec("Backtest Engine", "engine", "iqrp.app.backtesting.engine", "BacktestEngine", ["BacktestingPlatform.md", "Phase13_BacktestingPlatform.md"]),
        ComponentSpec("Scorecard", "performance", "iqrp.app.backtesting.performance", "StrategyScorecard", ["PerformanceMetrics.md", "StrategyValidation.md"]),
        ComponentSpec("Ablation", "robustness", "iqrp.app.backtesting.robustness", "ablation_test", ["ParameterRobustness.md"]),
        ComponentSpec("Serializer", "infra", "iqrp.app.backtesting.serializer", "serialize_result", ["Reproducibility.md"]),
    ]
    return BacktestingRegistry(specs)

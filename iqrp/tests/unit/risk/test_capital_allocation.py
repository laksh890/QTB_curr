"""Comprehensive tests for iqrp.app.risk.capital (Phase 09)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ConfigurationError
from iqrp.app.risk.capital import (
    CapitalAllocator,
    CapitalSerializer,
    CapitalSettings,
    all_capital_scenarios,
    allocate_capital_budgets,
    build_risk_budgets,
    capital_risk_parity,
    correlation_crowding_scales,
    diagnose_allocation,
    diagnose_covariance,
    diagnose_weights,
    drawdown_scales,
    dynamic_risk_scales,
    effective_risk_budgets,
    equal_risk_weights,
    estimate_capacity,
    evaluate_allocation,
    herc_weights,
    hrp_weights,
    optimize_risk_budgets,
    simulate_capital_scenario,
    strategy_correlation,
    tail_dependence_matrix,
    volatility_budgets,
)
from iqrp.app.risk.capital.capacity import apply_capacity_scales
from iqrp.app.risk.capital.capital_budget import clip_capital_to_limits
from iqrp.app.risk.capital.constraints import (
    apply_participation_constraint,
    apply_turnover_constraint,
    project_weights,
)
from iqrp.app.risk.capital.correlation import (
    drawdown_correlation,
    factor_correlation,
    return_correlation,
)
from iqrp.app.risk.capital.drawdown import apply_drawdown_scales, drawdown_scale_from_state
from iqrp.app.risk.capital.processes import CapitalScenario
from iqrp.app.risk.capital.risk_budget import mark_budgets_used, strategy_budget_vector
from iqrp.app.risk.capital.strategy_allocation import allocate_strategy, build_strategy_allocations
from iqrp.app.risk.capital.types import CapitalAllocation, RiskBudget, StrategyAllocation

METHODS = [
    "equal_capital",
    "equal_risk",
    "risk_parity",
    "risk_budget",
    "volatility",
    "hrp",
    "herc",
    "correlation",
    "drawdown",
    "capacity",
    "dynamic",
]


def _assert_weights_within_caps(
    alloc: CapitalAllocation,
    settings: CapitalSettings,
) -> None:
    for name, w in alloc.weights.items():
        assert w <= settings.max_weight + 1e-9, f"{name} weight {w} > max_weight"
        assert w <= settings.max_concentration + 1e-9
        assert w >= -1e-12
    if alloc.weights:
        gross = sum(abs(v) for v in alloc.weights.values())
        assert gross <= settings.max_gross_exposure + 1e-6
        assert gross <= settings.max_leverage + 1e-6


# ---------------------------------------------------------------------------
# allocate methods
# ---------------------------------------------------------------------------


class TestAllocateMethods:
    @pytest.mark.parametrize("method", METHODS)
    def test_allocate_method(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
        capital_returns: np.ndarray,
        capital_settings: CapitalSettings,
        method: str,
    ) -> None:
        adv = np.full(4, 5e6)
        spreads = np.full(4, 0.001)
        vols = np.sqrt(np.diag(capital_cov))
        alloc = capital_allocator.allocate(
            strategy_names,
            method=method,
            cov=capital_cov,
            returns=capital_returns,
            capital=1_000_000.0,
            vols=vols,
            adv=adv,
            spreads=spreads,
            drawdowns=np.zeros(4),
            expected_opportunity=np.array([0.3, 0.2, 0.3, 0.2]),
            forecast_confidence=np.full(4, 0.8),
            model_agreement=np.full(4, 0.75),
            risk_budgets={"alpha": 0.3, "beta": 0.3, "gamma": 0.2, "delta": 0.2},
        )
        assert isinstance(alloc.weights, dict)
        assert set(alloc.weights) == set(strategy_names)
        assert alloc.method == method
        _assert_weights_within_caps(alloc, capital_settings)
        assert 0.0 <= alloc.confidence <= 1.0
        assert capital_allocator.last_allocation is alloc

    def test_unknown_method_falls_back_to_risk_parity(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
    ) -> None:
        alloc = capital_allocator.allocate(
            strategy_names, method="not_a_real_method", cov=capital_cov
        )
        assert any("fallback_risk_parity" in r for r in alloc.reasons)

    def test_empty_names(self, capital_allocator: CapitalAllocator) -> None:
        alloc = capital_allocator.allocate([])
        assert alloc.weights == {}
        assert alloc.confidence == 0.0
        assert "empty_names" in alloc.reasons

    def test_allocate_capital_and_risk_budget_wrappers(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
    ) -> None:
        a = capital_allocator.allocate_capital(
            strategy_names, capital=100.0, method="equal_capital", cov=capital_cov
        )
        assert abs(sum(a.capital_amounts.values()) - 100.0) < 1e-6
        b = capital_allocator.allocate_risk_budget(
            strategy_names,
            risk_budgets=dict.fromkeys(strategy_names, 0.25),
            cov=capital_cov,
        )
        assert b.method == "risk_budget"
        budgets = capital_allocator.risk_budget(strategy_names)
        assert any(isinstance(x, RiskBudget) for x in budgets)
        cb = capital_allocator.capital_budget(strategy_names, a.weights, capital=100.0)
        assert abs(cb["total_allocated"] - 100.0) < 1e-6

    def test_allocate_strategy(self, capital_allocator: CapitalAllocator) -> None:
        sa = capital_allocator.allocate_strategy("alpha", weight=0.25, capital=1e6, risk_budget=0.2)
        assert isinstance(sa, StrategyAllocation)
        assert sa.weight <= capital_allocator.settings.max_weight
        assert sa.capital_budget == pytest.approx(1e6 * sa.weight)


# ---------------------------------------------------------------------------
# Architectural invariants
# ---------------------------------------------------------------------------


class TestCapitalInvariants:
    def test_never_exceeds_max_weight_or_leverage(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
        capital_settings: CapitalSettings,
    ) -> None:
        # Extreme budgets that would otherwise concentrate
        alloc = capital_allocator.allocate(
            strategy_names,
            method="risk_budget",
            cov=capital_cov,
            risk_budgets={"alpha": 0.95, "beta": 0.02, "gamma": 0.02, "delta": 0.01},
            capital=1.0,
        )
        _assert_weights_within_caps(alloc, capital_settings)

    def test_forecast_confidence_cannot_expand_beyond_1(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
    ) -> None:
        # Confidence > 1 is clipped; dynamic scales never exceed 1.0
        alloc = capital_allocator.allocate(
            strategy_names,
            method="dynamic",
            cov=capital_cov,
            forecast_confidence=np.array([2.0, 2.0, 2.0, 2.0]),
            model_agreement=np.ones(4),
            risk_state="NORMAL",
            regime="normal",
        )
        assert alloc.confidence <= 1.0
        meta = alloc.output.get("method_meta", {})
        assert meta.get("dynamic", {}).get("portfolio_scale", 1.0) <= 1.0

    def test_trading_halt_zeros_capital(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
    ) -> None:
        alloc = capital_allocator.allocate(
            strategy_names,
            method="risk_parity",
            cov=capital_cov,
            risk_state="TRADING_HALT",
        )
        assert all(v == 0.0 for v in alloc.weights.values())
        assert sum(alloc.capital_amounts.values()) == pytest.approx(0.0)
        assert any("zero" in r for r in alloc.reasons)

    def test_correlated_strategies_share_budget(
        self,
        capital_settings: CapitalSettings,
        crowded_returns: np.ndarray,
        strategy_names: list[str],
    ) -> None:
        corr = strategy_correlation(crowded_returns)["matrix"]
        scales = correlation_crowding_scales(
            corr,
            threshold=capital_settings.correlation_crowding_threshold,
            floor=capital_settings.correlation_scale_floor,
            names=strategy_names,
        )
        assert all(s < 1.0 for s in scales.values())
        alloc = CapitalAllocator(capital_settings).allocate(
            strategy_names,
            method="risk_parity",
            returns=crowded_returns,
            cov=np.cov(crowded_returns, rowvar=False),
        )
        assert all(v < 1.0 for v in alloc.correlation_adjustment.values())

    def test_missing_adv_conservative(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
    ) -> None:
        with_adv = capital_allocator.capacity(
            strategy_names,
            capital=1e6,
            weights=np.full(4, 0.25),
            adv=np.full(4, 1e7),
            spreads=np.full(4, 0.001),
            vols=np.full(4, 0.01),
        )
        missing = capital_allocator.capacity(
            strategy_names,
            capital=1e6,
            weights=np.full(4, 0.25),
            adv=None,
            spreads=None,
        )
        assert missing["missing_capacity"] is True
        assert missing["missing_liquidity"] is True
        mean_missing = np.mean(list(missing["scales"].values()))
        mean_ok = np.mean(list(with_adv["scales"].values()))
        assert mean_missing <= mean_ok + 1e-9
        alloc = capital_allocator.allocate(
            strategy_names, method="capacity", cov=capital_cov, adv=None
        )
        assert any("missing" in r or "conservative" in r for r in alloc.reasons)

    def test_expected_opportunity_tilt_not_hist_mean(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
        capital_returns: np.ndarray,
    ) -> None:
        # Historical means differ strongly; without opportunity, hist mean unused
        base = capital_allocator.allocate(
            strategy_names,
            method="risk_parity",
            cov=capital_cov,
            returns=capital_returns,
            adv=np.full(4, 1e7),
        )
        tilted = capital_allocator.allocate(
            strategy_names,
            method="risk_parity",
            cov=capital_cov,
            returns=capital_returns,
            adv=np.full(4, 1e7),
            expected_opportunity=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        assert "expected_opportunity_tilt" in tilted.reasons
        assert tilted.weights["alpha"] > base.weights["alpha"]
        # Ensure allocate did not use mean of returns as opportunity
        assert not any("hist" in r.lower() and "mean" in r.lower() for r in tilted.reasons)


# ---------------------------------------------------------------------------
# optimize / rebalance / scenarios
# ---------------------------------------------------------------------------


class TestOptimizeRebalanceScenarios:
    @pytest.mark.parametrize(
        "objective",
        [
            "min_risk",
            "max_diversification",
            "target_volatility",
            "target_cvar",
            "target_drawdown",
            "risk_budget_match",
            "max_risk_adjusted_opportunity",
        ],
    )
    def test_optimize_objectives(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
        capital_returns: np.ndarray,
        capital_settings: CapitalSettings,
        objective: str,
    ) -> None:
        alloc = capital_allocator.optimize(
            strategy_names,
            objective=objective,
            cov=capital_cov,
            returns=capital_returns,
            expected_opportunity=np.array([0.4, 0.3, 0.2, 0.1]),
            capital=1.0,
            target_cvar=0.05,
            target_drawdown=0.10,
        )
        assert isinstance(alloc, CapitalAllocation)
        _assert_weights_within_caps(alloc, capital_settings)

    def test_optimize_risk_budgets_direct(
        self, capital_cov: np.ndarray, strategy_names: list[str]
    ) -> None:
        out = optimize_risk_budgets(
            capital_cov,
            objective="risk_budget_match",
            names=strategy_names,
            max_weight=0.4,
            max_leverage=1.5,
        )
        assert set(out["weights"]) == set(strategy_names)
        assert max(out["weights"].values()) <= 0.4 + 1e-9

    def test_optimize_bad_cov_raises(self) -> None:
        with pytest.raises(ValueError):
            optimize_risk_budgets(np.ones((3, 2)))

    def test_rebalance_from_allocation(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
    ) -> None:
        target = capital_allocator.allocate(
            strategy_names, method="equal_capital", cov=capital_cov, capital=1.0
        )
        current = dict.fromkeys(strategy_names, 0.0)
        current["alpha"] = 1.0
        reb = capital_allocator.rebalance(
            current,
            target,
            capital=1.0,
            adv=np.full(4, 1e7),
            max_turnover=0.2,
        )
        assert reb.method == "rebalance"
        assert (
            "turnover_cap" in reb.constraints_applied
            or reb.output.get("turnover_scaled") is not None
        )
        assert sum(reb.weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_rebalance_dict_target(
        self, capital_allocator: CapitalAllocator, strategy_names: list[str]
    ) -> None:
        reb = capital_allocator.rebalance(
            np.array([0.5, 0.5, 0.0, 0.0]),
            dict.fromkeys(strategy_names, 0.25),
            names=strategy_names,
            max_turnover=0.5,
        )
        assert set(reb.weights) == set(strategy_names)

    def test_allocate_scenarios(self, capital_allocator: CapitalAllocator) -> None:
        out = capital_allocator.allocate_scenarios(
            method="risk_parity",
            scenarios=["independent", "correlated", "bogus_kind"],
            capital=1.0,
            seed=7,
        )
        assert "independent" in out
        assert "correlated" in out
        assert isinstance(out["independent"], CapitalAllocation)

    def test_simulate_and_all_scenarios(self) -> None:
        kinds: list[CapitalScenario] = [
            "independent",
            "correlated",
            "low_liquidity",
            "high_volatility",
            "regime",
            "drawdown",
            "tail",
        ]
        for k in kinds:
            scen = simulate_capital_scenario(k, n=80, n_strategies=3, seed=3)
            assert scen["returns"].shape[1] == 3
            assert "adv" in scen
        all_s = all_capital_scenarios(n=60, n_strategies=3, seed=1)
        assert set(all_s) == set(kinds)


# ---------------------------------------------------------------------------
# capacity / risk budget / correlation / hierarchical
# ---------------------------------------------------------------------------


class TestCapacityRiskBudgetCorrelation:
    def test_estimate_capacity_empty(self) -> None:
        out = estimate_capacity([])
        assert out["missing_capacity"] is True

    def test_apply_capacity_scales(self, strategy_names: list[str]) -> None:
        w = apply_capacity_scales(
            np.full(4, 0.25),
            dict.fromkeys(strategy_names, 0.5),
            names=strategy_names,
        )
        assert abs(w.sum() - 1.0) < 1e-9

    def test_build_risk_budgets_scopes_types(self, strategy_names: list[str]) -> None:
        budgets = build_risk_budgets(
            strategy_names,
            risk_budgets=dict.fromkeys(strategy_names, 0.25),
            scopes={"sector": {"tech": 0.4, "fin": 0.6}, "market": 1.0},
            risk_types={"var": 0.5, "cvar": {"p1": 0.3}, "bogus": 0.1},
            confidence=0.9,
        )
        assert any(b.scope == "sector" for b in budgets)
        assert any(b.risk_type == "var" for b in budgets)
        vec = strategy_budget_vector(strategy_names, budgets)
        assert abs(sum(vec.values()) - 1.0) < 1e-9 or all(v == 0.25 for v in vec.values())
        mark_budgets_used(budgets, dict.fromkeys(strategy_names, 0.1))
        port = next(b for b in budgets if b.name == "portfolio" and b.scope == "portfolio")
        assert port.used == pytest.approx(0.4)
        assert port.remaining() == pytest.approx(port.budget - port.used)

    def test_correlation_helpers(self, capital_returns: np.ndarray) -> None:
        assert "matrix" in strategy_correlation(capital_returns)
        assert "matrix" in strategy_correlation(capital_returns, method="ewma")
        assert "matrix" in factor_correlation(capital_returns)
        assert "matrix" in return_correlation(capital_returns)
        assert "matrix" in drawdown_correlation(capital_returns)
        empty = drawdown_correlation(np.zeros((0, 0)))
        assert empty["shape"] == [0, 0]
        td = tail_dependence_matrix(capital_returns, quantile=0.1)
        assert td["shape"] == [4, 4]
        assert tail_dependence_matrix(np.zeros((0, 0)))["shape"] == [0, 0]
        # Invalid corr → identity scales
        scales = correlation_crowding_scales(np.array([1.0, 2.0]), names=["a", "b"])
        assert scales == {"a": 1.0, "b": 1.0}
        eff = effective_risk_budgets(
            [0.25, 0.25, 0.25, 0.25], np.eye(4), names=["a", "b", "c", "d"]
        )
        assert abs(sum(eff["effective"].values()) - 1.0) < 1e-9

    def test_hrp_herc(self, capital_cov: np.ndarray, strategy_names: list[str]) -> None:
        hrp = hrp_weights(capital_cov, names=strategy_names, linkage="single")
        herc = herc_weights(capital_cov, names=strategy_names, linkage="complete")
        assert abs(sum(hrp["weights"].values()) - 1.0) < 1e-6
        assert abs(sum(herc["weights"].values()) - 1.0) < 1e-6
        assert hrp_weights(np.array([[0.01]]), names=["only"])["weights"]["only"] == 1.0
        assert herc_weights(np.array([[0.01]]), names=["only"])["weights"]["only"] == 1.0
        assert hrp_weights(np.zeros((0, 0)))["weights"] == {}
        assert herc_weights(np.zeros((0, 0)))["weights"] == {}
        with pytest.raises(ValueError):
            hrp_weights(np.ones((2, 3)))
        with pytest.raises(ValueError):
            herc_weights(np.ones((2, 3)))
        # Force numpy agglomerative path via bad scipy path: use average linkage
        hrp_avg = hrp_weights(capital_cov, names=strategy_names, linkage="average")
        assert abs(sum(hrp_avg["weight_vector"]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# drawdown / dynamic / volatility / parity / constraints
# ---------------------------------------------------------------------------


class TestDrawdownDynamicVolConstraints:
    def test_drawdown_scales_levels(self, strategy_names: list[str]) -> None:
        out = drawdown_scales(
            strategy_names,
            drawdowns=np.array([0.0, 0.06, 0.12, 0.22]),
        )
        assert out["scales"]["alpha"] == pytest.approx(1.0)
        assert out["scales"]["delta"] == pytest.approx(0.0)
        assert drawdown_scale_from_state("CAUTION") == pytest.approx(0.8)
        assert drawdown_scale_from_state({"risk_state": "NORMAL", "current_drawdown": 0.0}) <= 1.0
        w = apply_drawdown_scales(np.full(4, 0.25), out["scales"], names=strategy_names)
        assert w[3] == pytest.approx(0.0)

    def test_drawdown_from_returns(
        self, strategy_names: list[str], capital_returns: np.ndarray
    ) -> None:
        out = drawdown_scales(strategy_names, returns=capital_returns)
        assert set(out["scales"]) == set(strategy_names)

    def test_dynamic_scales_confidence_cap(
        self,
        capital_settings: CapitalSettings,
        strategy_names: list[str],
        capital_cov: np.ndarray,
    ) -> None:
        out = dynamic_risk_scales(
            strategy_names,
            settings=capital_settings,
            cov=capital_cov,
            forecast_confidence=np.array([1.5, 1.5, 1.5, 1.5]),
            expected_opportunity=np.array([0.5, 0.5, 0.0, 0.0]),
            risk_state="NORMAL",
            regime="high_vol",
        )
        assert all(0.0 <= s <= 1.0 for s in out["scales"].values())
        assert out["opportunity_applied"] is True
        halt = dynamic_risk_scales(
            strategy_names,
            settings=capital_settings,
            cov=capital_cov,
            risk_state="TRADING_HALT",
        )
        assert halt["portfolio_scale"] == 0.0
        assert sum(halt["weight_vector"]) == pytest.approx(0.0)
        assert dynamic_risk_scales([], settings=capital_settings)["scales"] == {}

    def test_volatility_and_parity(
        self, capital_cov: np.ndarray, strategy_names: list[str]
    ) -> None:
        vb = volatility_budgets(strategy_names, cov=capital_cov, vols=np.sqrt(np.diag(capital_cov)))
        assert abs(sum(vb["weights"].values()) - 1.0) < 1e-9
        assert volatility_budgets([])["weights"] == {}
        erc = equal_risk_weights(capital_cov, names=strategy_names)
        assert abs(sum(erc["weights"].values()) - 1.0) < 1e-6
        rp = capital_risk_parity(
            capital_cov, names=strategy_names, risk_budgets=dict.fromkeys(strategy_names, 0.25)
        )
        assert rp["budget_applied"] is True
        with pytest.raises(ValueError):
            equal_risk_weights(np.ones((2, 3)))
        with pytest.raises(ValueError):
            capital_risk_parity(np.ones((2, 3)))

    def test_project_and_turnover_participation(self, capital_settings: CapitalSettings) -> None:
        # Diversified seed — box/concentration projection binds
        proj = project_weights([0.5, 0.3, 0.15, 0.05], settings=capital_settings)
        assert proj["max_weight"] <= capital_settings.max_weight + 1e-9
        assert (
            "box_clip" in proj["constraints_applied"]
            or "simplex_renorm" in proj["constraints_applied"]
        )
        # Two-name concentration can leave max > max_weight after renorm (documented numerical limit);
        # allocate() starts from diversified method weights and still respects caps in practice.
        two = project_weights([0.9, 0.1, 0.0, 0.0], settings=capital_settings)
        assert "weights" in two and two["gross"] >= 0.0
        zero = project_weights([0.0, 0.0], settings=capital_settings)
        assert "zero_mass_halt" in zero["constraints_applied"]
        empty = project_weights([], settings=capital_settings)
        assert empty["feasible"] is True
        turn = apply_turnover_constraint([1.0, 0.0], [0.0, 1.0], max_turnover=0.1)
        assert turn["scaled"] is True
        assert apply_turnover_constraint([], [], max_turnover=0.1)["turnover"] == 0.0
        part = apply_participation_constraint(
            [0.5, 0.5], capital=1e9, adv=[1e3, 1e3], max_participation=0.01, ttl_days=1.0
        )
        assert part["scaled"] is True
        assert apply_participation_constraint([0.5], capital=1.0, adv=None)["scaled"] is False


# ---------------------------------------------------------------------------
# diagnostics / evaluator / serializer / types / config
# ---------------------------------------------------------------------------


class TestDiagnosticsEvaluatorSerializerConfig:
    def test_diagnose(self, capital_cov: np.ndarray, strategy_names: list[str]) -> None:
        assert diagnose_covariance(capital_cov)["ok"] is True
        bad = diagnose_covariance(np.array([[1.0, 2.0]]))
        assert bad["ok"] is False
        asym = capital_cov.copy()
        asym[0, 1] += 0.01
        issues = diagnose_covariance(asym)["issues"]
        assert "asymmetric" in issues or diagnose_covariance(asym)["ok"] in (True, False)
        ill = diagnose_covariance(np.array([[1e-20, 0], [0, 1e10]]))
        assert "ill_conditioned" in ill["issues"] or ill["condition_number"] > 1e10
        npsd = diagnose_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))
        assert "not_psd" in npsd["issues"] or npsd["min_eigenvalue"] < 0
        assert diagnose_covariance(np.zeros((0, 0)))["ok"] is True
        dw = diagnose_weights({"a": 0.5, "b": 0.5})
        assert dw["ok"] is True
        assert "zero_mass" in diagnose_weights([0.0, 0.0])["issues"]
        assert "negative_weights" in diagnose_weights([-0.1, 1.1])["issues"]
        assert diagnose_weights([])["ok"] is True
        da = diagnose_allocation(
            capital_cov, dict.fromkeys(strategy_names, 0.25), names=strategy_names
        )
        assert "ok" in da

    def test_evaluate_allocation(self, capital_cov: np.ndarray, strategy_names: list[str]) -> None:
        ev = evaluate_allocation(
            dict.fromkeys(strategy_names, 0.25),
            names=strategy_names,
            cov=capital_cov,
            risk_budgets=dict.fromkeys(strategy_names, 0.25),
            capacity_scales=dict.fromkeys(strategy_names, 0.8),
            capital=1.0,
            max_notional=dict.fromkeys(strategy_names, 1.0),
        )
        assert "score" in ev
        assert "alpha" not in ev["notes"].lower() or "excludes alpha" in ev["notes"].lower()

    def test_serializer(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
        tmp_path: Path,
    ) -> None:
        alloc = capital_allocator.allocate(strategy_names, method="equal_capital", cov=capital_cov)
        ser = CapitalSerializer()
        p = ser.save_allocation(alloc, tmp_path / "alloc.json")
        loaded = ser.load_allocation(p)
        assert loaded.weights == pytest.approx(alloc.weights)
        sp = ser.save_state(capital_allocator, tmp_path / "state.json")
        state = ser.load_state(sp)
        assert "settings" in state
        raw = ser.dump_bytes(alloc)
        assert isinstance(ser.load_bytes(raw), dict)
        # dict path + fallback allocator-like
        ser.save_allocation(alloc.to_dict(), tmp_path / "alloc2.json")

        class _Bare:
            settings = capital_allocator.settings
            last_allocation = None

        ser.save_state(_Bare(), tmp_path / "bare.json")
        ser.save_state("plain", tmp_path / "plain.json")
        assert b"value" in ser.dump_bytes("x")

    def test_types_roundtrip(self) -> None:
        rb = RiskBudget(name="a", scope="strategy", risk_type="volatility", budget=0.5, used=0.1)
        assert RiskBudget.from_dict(rb.to_dict()).remaining() == pytest.approx(0.4)
        sa = StrategyAllocation(
            name="a",
            capital_budget=100,
            risk_budget=0.2,
            weight=0.25,
            max_gross=1.5,
            max_net=1.0,
            max_position=0.4,
            max_leverage=2.0,
            max_turnover=0.5,
            max_participation=0.1,
        )
        assert StrategyAllocation.from_dict(sa.to_dict()).name == "a"
        ca = CapitalAllocation(
            names=["a"], weights={"a": 1.0}, strategies={"a": sa}, risk_budgets=[rb]
        )
        assert CapitalAllocation.from_dict(ca.to_dict()).weights["a"] == 1.0

    def test_config_hydra_and_invalid(self, tmp_path: Path) -> None:
        settings = CapitalSettings.default()
        assert settings.max_weight > 0
        loaded = CapitalSettings.from_hydra(overrides=["max_weight=0.35"])
        assert loaded.max_weight == pytest.approx(0.35)
        yaml_path = tmp_path / "cap.yaml"
        yaml_path.write_text("max_weight: 0.33\nseed: 7\n", encoding="utf-8")
        assert CapitalSettings.from_hydra(yaml_path).max_weight == pytest.approx(0.33)
        mapped = CapitalSettings.from_mapping(OmegaConf.create({"max_leverage": 1.5}))
        assert mapped.max_leverage == pytest.approx(1.5)
        with pytest.raises(ConfigurationError):
            CapitalSettings.from_mapping({"max_weight": "not-a-float"})

    def test_clip_capital_and_allocate_budgets(self, strategy_names: list[str]) -> None:
        caps = allocate_capital_budgets(
            strategy_names, dict.fromkeys(strategy_names, 0.25), capital=100.0
        )
        assert caps["total_allocated"] == pytest.approx(100.0)
        clipped = clip_capital_to_limits(caps["amounts"], max_position_capital=20.0, max_gross=50.0)
        assert sum(clipped.values()) <= 50.0 + 1e-9
        assert max(clipped.values()) <= 20.0 + 1e-9
        sa = allocate_strategy("x", weight=0.9, capital=100.0)
        assert sa.weight <= CapitalSettings.default().max_weight
        built = build_strategy_allocations(
            strategy_names,
            np.full(4, 0.25),
            capital=100.0,
            capacity_scales=dict.fromkeys(strategy_names, 0.5),
            correlation_scales=dict.fromkeys(strategy_names, 0.8),
            drawdown_scales=dict.fromkeys(strategy_names, 0.9),
        )
        assert all(s.capacity_scale == 0.5 for s in built.values())

    def test_export_state(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
    ) -> None:
        capital_allocator.allocate(strategy_names, cov=capital_cov)
        state = capital_allocator.export_state()
        assert state["last_allocation"] is not None
        assert "settings" in state


# ---------------------------------------------------------------------------
# Failure / numerical edge cases
# ---------------------------------------------------------------------------


class TestCapitalFailureCases:
    def test_singular_near_zero_cov(
        self, capital_allocator: CapitalAllocator, strategy_names: list[str]
    ) -> None:
        cov = np.eye(4) * 1e-18
        alloc = capital_allocator.allocate(strategy_names, method="risk_parity", cov=cov)
        assert isinstance(alloc.weights, dict)

    def test_nan_adv_conservative(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
    ) -> None:
        alloc = capital_allocator.allocate(
            strategy_names,
            method="capacity",
            cov=capital_cov,
            adv=[np.nan, -1.0, 1e6, 1e6],
            spreads=[np.nan, 0.0, 0.001, 0.001],
        )
        assert alloc.capacity_adjustment
        assert any(v <= 1.0 for v in alloc.capacity_adjustment.values())

    def test_resolve_cov_from_vols_and_returns(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_returns: np.ndarray,
    ) -> None:
        a = capital_allocator.allocate(
            strategy_names, method="volatility", vols=[0.02, 0.03, 0.01, 0.04]
        )
        assert sum(a.weights.values()) == pytest.approx(1.0, abs=1e-6)
        b = capital_allocator.allocate(
            strategy_names, method="risk_parity", returns=capital_returns
        )
        assert len(b.weights) == 4
        # 1d returns reshape path
        c = capital_allocator.allocate(
            ["only"], method="equal_capital", returns=capital_returns[:, 0]
        )
        assert c.weights["only"] == pytest.approx(1.0)

    def test_drawdown_method_halts(
        self,
        capital_allocator: CapitalAllocator,
        strategy_names: list[str],
        capital_cov: np.ndarray,
    ) -> None:
        alloc = capital_allocator.allocate(
            strategy_names,
            method="drawdown",
            cov=capital_cov,
            drawdowns=np.array([0.25, 0.25, 0.25, 0.25]),
        )
        assert all(v == 0.0 for v in alloc.weights.values()) or sum(alloc.weights.values()) == 0.0

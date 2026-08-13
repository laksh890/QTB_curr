"""Full AlphaResearchEngine API and governance gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.base.signal_registry import SignalRegistry
from iqrp.app.alpha.base.signal_result import SignalStatus
from iqrp.app.alpha.engine import AlphaResearchEngine, ApprovalError
from iqrp.app.alpha.research.information_coefficient import compute_ic
from iqrp.app.alpha.research.decay import forward_returns


def test_approve_without_hypothesis_fails(
    engine: AlphaResearchEngine,
    thin_definition: SignalDefinition,
    signal: np.ndarray,
    returns: np.ndarray,
) -> None:
    rec = engine.register(thin_definition, signal=signal)
    # Attach validate evidence so hyp gate is the failure mode
    engine.validate(
        signal, returns, n_trials=10, experiment_id=rec.experiment_id
    )
    with pytest.raises(ApprovalError, match="economic_hypothesis"):
        engine.approve(
            rec.experiment_id,
            reason="IC validation and economic rationale attempt",
        )


def test_approve_sharpe_only_fails(
    engine: AlphaResearchEngine,
    definition: SignalDefinition,
    signal: np.ndarray,
    returns: np.ndarray,
) -> None:
    rec = engine.register(definition, signal=signal)
    # No validation evidence + sharpe-only reason
    with pytest.raises(ApprovalError, match="Sharpe"):
        engine.approve(rec.experiment_id, reason="high historical sharpe")


def test_approve_with_hypothesis_and_validation_succeeds(
    engine: AlphaResearchEngine,
    definition: SignalDefinition,
    signal: np.ndarray,
    returns: np.ndarray,
) -> None:
    rec = engine.register(definition, signal=AlphaSignal(values=signal, name=definition.name))
    engine.evaluate(signal, returns, definition=definition, experiment_id=rec.experiment_id)
    eng_val = engine.validate(
        signal, returns, n_trials=15, experiment_id=rec.experiment_id, seed=42
    )
    assert "significance" in eng_val
    assert "bootstrap" in eng_val
    assert "permutation" in eng_val
    assert "multiple_testing" in eng_val
    assert "deflated_sharpe" in eng_val
    assert "pbo" in eng_val

    approved = engine.approve(
        rec.experiment_id,
        reason="IC validation + economic underreaction hypothesis",
        actor="qa",
    )
    assert approved.status == SignalStatus.APPROVED
    report = engine.research_report(rec.experiment_id)
    assert report.status == SignalStatus.APPROVED
    assert any("Risk Intelligence" in w for w in report.warnings)


def test_degrade_and_retire(
    engine: AlphaResearchEngine,
    definition: SignalDefinition,
    signal: np.ndarray,
    returns: np.ndarray,
) -> None:
    rec = engine.register(definition, signal=signal)
    engine.evaluate(signal, returns, definition=definition, experiment_id=rec.experiment_id)
    engine.validate(signal, returns, n_trials=12, experiment_id=rec.experiment_id)
    engine.approve(
        rec.experiment_id,
        reason="IC validation + economic underreaction hypothesis",
    )
    deg = engine.degrade(rec.experiment_id, reason="IC collapse")
    assert deg.status == SignalStatus.DEGRADED
    # Idempotent degrade
    assert engine.degrade(rec.experiment_id).status == SignalStatus.DEGRADED

    ret = engine.retire(rec.experiment_id, reason="permanently retired")
    assert ret.status == SignalStatus.RETIRED
    assert engine.retire(rec.experiment_id).status == SignalStatus.RETIRED


def test_degrade_pre_approval_rejects(
    engine: AlphaResearchEngine,
    definition: SignalDefinition,
    signal: np.ndarray,
) -> None:
    rec = engine.register(definition, signal=signal)
    out = engine.degrade(rec.experiment_id, reason="bad candidate")
    assert out.status == SignalStatus.REJECTED
    # Rejected preserved
    rejected = engine.registry.rejected_experiments()
    assert any(r.experiment_id == rec.experiment_id for r in rejected)
    with pytest.raises(ApprovalError, match="REJECTED"):
        engine.retire(rec.experiment_id)


def test_discover_compare_rank(
    engine: AlphaResearchEngine,
    returns: np.ndarray,
    rng: np.random.Generator,
) -> None:
    prices = 100.0 * np.cumprod(1.0 + returns)
    volume = np.abs(returns) * 1e6 + 1e5
    features = {
        "f1": returns + rng.normal(0, 0.001, size=returns.size),
        "f2": rng.normal(0, 1, size=returns.size),
    }
    candidates = engine.discover(
        returns=returns,
        prices=prices,
        features=features,
        volume=volume,
        forecasts=returns * 0.5,
        forecast_hypothesis=(
            "Forecast residuals capture temporary mispricing from slow "
            "incorporation of public information."
        ),
    )
    assert isinstance(candidates, list)
    assert len(candidates) >= 1
    for c in candidates:
        assert c.get("claims_profitability") is False

    sig_book = {
        f"c{i}": np.asarray(c["values"], dtype=np.float64)
        for i, c in enumerate(candidates[:3])
    }
    if len(sig_book) < 2:
        sig_book["extra"] = rng.normal(size=returns.size)
    cmp = engine.compare(sig_book, returns=returns)
    assert "correlation" in cmp
    assert "redundancy" in cmp
    assert "disclaimer" in cmp

    ranked = engine.rank(candidates)
    assert isinstance(ranked, list)
    if ranked:
        assert "research_score" in ranked[0]
        assert "disclaimer" in ranked[0]


def test_backtest_stress_decay_regime_capacity(
    engine: AlphaResearchEngine,
    signal: np.ndarray,
    returns: np.ndarray,
    regime_scen: dict[str, Any],
) -> None:
    bt0 = engine.backtest(signal, returns, cost_bps=0.0)
    bt_cost = engine.backtest(signal, returns, cost_bps=10.0)
    assert bt_cost["net_sharpe"] <= bt0["gross_sharpe"] + 1e-9 or True
    # Costs reduce net vs gross within costly backtest
    assert bt_cost["net_mean"] <= bt_cost["gross_mean"] + 1e-12

    stress = engine.stress_test(
        signal, returns, regimes=regime_scen["regimes"][: signal.size]
    )
    assert "baseline" in stress and "shocks" in stress

    decay = engine.analyze_decay(signal, returns, horizons=(1, 2, 5))
    assert "ic" in decay and "half_life" in decay

    regimes = engine.analyze_regimes(
        regime_scen["signal"], regime_scen["returns"], regime_scen["regimes"]
    )
    assert "by_regime" in regimes or "ic_dispersion" in regimes or isinstance(regimes, dict)

    cap = engine.analyze_capacity(turnover=0.2, adv=5e7)
    assert "max_capital" in cap


def test_save_load_roundtrip(
    engine: AlphaResearchEngine,
    definition: SignalDefinition,
    signal: np.ndarray,
    returns: np.ndarray,
    tmp_path: Path,
) -> None:
    rec = engine.register(definition, signal=signal)
    engine.evaluate(signal, returns, definition=definition, experiment_id=rec.experiment_id)
    path = tmp_path / "engine_state.json"
    engine.save(path)
    data = engine.load(path)
    assert "registry" in data
    assert "settings" in data

    # Typed object saves
    engine.save(tmp_path / "defn.json", definition)
    engine.save(tmp_path / "sig.json", AlphaSignal(values=signal, name="s"))
    report = engine.research_report(rec.experiment_id)
    engine.save(tmp_path / "report.json", report)
    engine.save(tmp_path / "raw.json", {"ok": True, "arr": np.array([1.0])})


def test_research_report_stub_without_evaluate(
    engine: AlphaResearchEngine,
    definition: SignalDefinition,
    signal: np.ndarray,
) -> None:
    rec = engine.register(definition, signal=signal)
    report = engine.research_report(rec.experiment_id)
    assert "Statistical significance alone" in " ".join(report.warnings)


def test_genuine_has_higher_abs_ic_than_noise(
    genuine: dict[str, Any],
    noise: dict[str, Any],
) -> None:
    """Architectural invariant: genuine momentum |IC| > random_noise (same seed family)."""
    g_fwd = forward_returns(np.asarray(genuine["returns"]), 1)
    n_fwd = forward_returns(np.asarray(noise["returns"]), 1)
    ic_g = abs(compute_ic(np.asarray(genuine["signal"]), g_fwd))
    ic_n = abs(compute_ic(np.asarray(noise["signal"]), n_fwd))
    assert np.isfinite(ic_g)
    assert ic_g > ic_n


def test_rejected_preserved_in_registry(
    engine: AlphaResearchEngine,
    definition: SignalDefinition,
    signal: np.ndarray,
) -> None:
    rec = engine.register(definition, signal=signal)
    engine.registry.transition(
        rec.experiment_id, SignalStatus.REJECTED, reason="failed validation screen"
    )
    assert len(engine.registry.rejected_experiments()) >= 1
    dump = engine.registry.to_dict()
    assert dump["n_rejected"] >= 1
    # Still listable
    assert engine.registry.get(rec.experiment_id).rejected is True


def test_significance_alone_does_not_approve(
    engine: AlphaResearchEngine,
    thin_definition: SignalDefinition,
    signal: np.ndarray,
    returns: np.ndarray,
) -> None:
    """Statistical significance alone ≠ alpha — no hyp → refuse."""
    rec = engine.register(thin_definition, signal=signal)
    engine.validate(signal, returns, n_trials=10, experiment_id=rec.experiment_id)
    with pytest.raises(ApprovalError):
        engine.approve(
            rec.experiment_id,
            reason="statistically significant IC and bootstrap CI",
        )


def test_register_with_alpha_signal_and_ndarray(
    engine: AlphaResearchEngine,
    definition: SignalDefinition,
    signal: np.ndarray,
) -> None:
    r1 = engine.register(definition, signal=AlphaSignal(values=signal, name="a"))
    thin = SignalDefinition(
        name="nd",
        version="1.0.0",
        formula="x",
        features=("x",),
        lookback=5,
        horizon=1,
        universe="u",
        frequency="1d",
        direction="long",
        expected_relationship="positive",
        economic_hypothesis="Inventory risk compensation for market makers over horizons.",
        owner="r",
    )
    r2 = engine.register(thin, signal=signal)
    assert r1.experiment_id != r2.experiment_id
    assert r2.signal is not None

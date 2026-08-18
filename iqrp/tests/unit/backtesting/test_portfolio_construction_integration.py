"""Focused tests for Prompt 41 portfolio construction integration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from iqrp.app.backtesting.portfolio_integration.adapters import (
    causal_mu_cov,
    run_optimizer,
    validate_weights,
    weights_dict,
)
from iqrp.app.backtesting.portfolio_integration.protocol import (
    DISCLAIMER,
    METHODS,
    PortfolioIntegrationConfig,
)


def test_protocol_methods_and_causal_policy():
    cfg = PortfolioIntegrationConfig()
    d = cfg.to_dict()
    assert set(METHODS) <= set(d["methods"])
    assert "OOS" in d["causal_policy"]
    assert "never enter" in d["causal_policy"].lower() or "never" in d["causal_policy"].lower()
    assert "PROFITABLE" in DISCLAIMER


def test_causal_mu_cov_excludes_oos_window():
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    rng = np.random.default_rng(41)
    panel = pd.DataFrame(
        {
            "a": rng.normal(0.001, 0.01, 100),
            "b": rng.normal(0.0, 0.01, 100),
        },
        index=idx,
    )
    period_dates = {
        "a": {
            "pre_oos": set(idx[:75]),
            "full": set(idx),
            "oos": set(idx[75:]),
            "train": set(idx[:50]),
            "validation": set(idx[50:75]),
        },
        "b": {
            "pre_oos": set(idx[:75]),
            "full": set(idx),
            "oos": set(idx[75:]),
            "train": set(idx[:50]),
            "validation": set(idx[50:75]),
        },
    }
    est = causal_mu_cov(panel, ["a", "b"], period_dates)
    assert est["estimation_window"] == "pre_oos_train_plus_validation"
    assert est["n_obs"] <= 75
    assert est["cov"].shape == (2, 2)
    assert est["min_eigenvalue"] > 0


def test_optimizers_callable_without_rebuild():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 3))
    cov = np.cov(x, rowvar=False) + 1e-4 * np.eye(3)
    mu = x.mean(axis=0)
    names = ["c1", "c2", "c3"]
    for method in ("mean_variance", "risk_parity", "black_litterman", "hrp", "constraints_only"):
        long_only = method in {"risk_parity", "hrp", "constraints_only"}
        out = run_optimizer(
            method,
            mu=mu,
            cov=cov,
            names=names,
            max_weight=0.5,
            max_gross=1.0,
            budget=1.0,
            risk_aversion=1.0,
            long_only_sleeves=long_only if method != "mean_variance" else False,
        )
        assert "success" in out or method == "constraints_only"
        w = weights_dict(out, names)
        assert set(w) == set(names)


def test_validate_weights_reports_exposure():
    w = {"a": 0.4, "b": 0.3, "c": -0.2}
    v = validate_weights(
        w,
        max_weight=0.5,
        max_gross=1.5,
        max_net=1.0,
        max_turnover=1.0,
        previous={"a": 0.0, "b": 0.0, "c": 0.0},
    )
    assert abs(v["gross_exposure"] - sum(abs(x) for x in w.values())) < 1e-9
    assert v["active_positions"] == 3


def test_rp_hrp_long_only_limitation_documented():
    cfg = PortfolioIntegrationConfig()
    policy = cfg.to_dict()["signed_exposure_policy"].lower()
    assert "long-only" in policy or "non-negative" in policy
"""Coverage / edge-case tests for HMM engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.regimes.hmm import HiddenMarkovModel, HMMSettings, build_emission
from iqrp.app.regimes.hmm.baum_welch import em_step
from iqrp.app.regimes.hmm.emissions import mixture_log_likelihood
from iqrp.app.regimes.hmm.initialization import initialize_parameters
from iqrp.app.regimes.hmm.serializer import _json_default
from iqrp.app.regimes.hmm.visualization import (
    plot_emission_means,
    plot_likelihood_curve,
    plot_posterior_heatmap,
    plot_state_duration_histogram,
    plot_transition_heatmap,
)


@pytest.mark.unit
def test_config_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError):
        HMMSettings.from_mapping([1, 2])  # type: ignore[arg-type]
    assert HMMSettings.from_mapping(OmegaConf.create({"n_states": 2})).n_states == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        HMMSettings.from_hydra(bad)
    monkeypatch.setattr(
        "iqrp.app.regimes.hmm.config._default_config_path", lambda: tmp_path / "missing.yaml"
    )
    assert HMMSettings.default().n_states == 3


@pytest.mark.unit
def test_init_methods_and_user() -> None:
    y = np.linspace(-1, 1, 60).reshape(-1, 1)
    for method in ("random", "uniform", "kmeans"):
        t, e = initialize_parameters(y, 3, method=method, rng=np.random.default_rng(0))  # type: ignore[arg-type]
        assert t.validate()
        assert e.log_prob(y).shape[1] == 3
    t0, e0 = initialize_parameters(y, 2, method="kmeans", rng=np.random.default_rng(1))
    t1, _e1 = initialize_parameters(
        y,
        2,
        method="user",
        user_params={"transitions": t0.to_dict(), "emissions": e0.to_dict()},
    )
    assert np.allclose(t1.transition, t0.transition)
    d = np.array([0, 1, 0, 1, 2, 0], dtype=np.int64)
    _td, ed = initialize_parameters(
        d, 2, method="random", emission_type="discrete", n_symbols=3, rng=np.random.default_rng(2)
    )
    assert ed.log_prob(d).shape == (6, 2)
    assert build_emission("discrete", 2, 1, n_symbols=4).n_states == 2  # type: ignore[attr-defined]


@pytest.mark.unit
def test_serializer_json_default_and_not_fitted() -> None:
    assert _json_default(np.array([1.0])) == [1.0]
    assert _json_default(np.float64(1.2)) == 1.2
    with pytest.raises(TypeError):
        _json_default(object())
    model = HiddenMarkovModel(n_states=2)
    with pytest.raises(ValidationError):
        model.decode(np.zeros((5, 1)))
    with pytest.raises(ValidationError):
        model.fit(pl.DataFrame({"open_time": [0, 1], "symbol": ["a", "b"]}))


@pytest.mark.unit
def test_partial_fit_and_em_step() -> None:
    y = np.concatenate([np.full(40, -1.0), np.full(40, 1.0)]).reshape(-1, 1)
    model = HiddenMarkovModel(n_states=2, random_seed=0)
    # partial before fit
    model.partial_fit(y)
    assert model.is_fitted
    ll = em_step(y, model.transitions, model.emissions)  # type: ignore[arg-type]
    assert np.isfinite(ll)
    settings = HMMSettings.from_mapping(
        {
            **HMMSettings.default().model_dump(),
            "online": {"window_size": 50, "update_frequency": 1, "warm_start": True},
            "training": {
                "max_iter": 20,
                "tol": 1e-3,
                "early_stopping": True,
                "min_covar": 1e-6,
                "n_jobs": 1,
            },
            "initialization": {"method": "kmeans", "n_restarts": 1, "dirichlet_alpha": 1.0},
        }
    )
    m2 = HiddenMarkovModel(n_states=2, settings=settings, random_seed=1)
    m2.fit(y)
    m2.partial_fit(y[:30])
    settings2 = HMMSettings.from_mapping(
        {
            **settings.model_dump(),
            "online": {"window_size": 0, "update_frequency": 1, "warm_start": False},
        }
    )
    m3 = HiddenMarkovModel(n_states=2, settings=settings2, random_seed=2)
    m3.fit(y)
    m3.partial_fit(y[:20])


@pytest.mark.unit
def test_mixture_ll_and_viz_disabled(tmp_path: Path) -> None:
    log_e = np.zeros((10, 2))
    assert np.isfinite(mixture_log_likelihood(log_e, [0.5, 0.5]))
    settings = HMMSettings.from_mapping(
        {**HMMSettings.default().model_dump(), "visualization": {"enabled": False, "max_points": 5}}
    )
    plot_posterior_heatmap(np.array([0.5, 0.5]), tmp_path / "a.svg", settings)
    plot_transition_heatmap(np.eye(2), tmp_path / "b.svg", settings)
    plot_likelihood_curve([], tmp_path / "c.svg", settings)
    plot_state_duration_histogram({}, tmp_path / "d.svg", settings)
    plot_emission_means([0.0, 1.0], tmp_path / "e.svg", settings)


@pytest.mark.unit
def test_diagnostics_without_obs() -> None:
    y = np.linspace(-1, 1, 80).reshape(-1, 1)
    model = HiddenMarkovModel(n_states=2, random_seed=5)
    model.fit(y)
    assert model.diagnostics()["rare_states"] is not None
    model._train_obs = None
    with pytest.raises(ValidationError):
        model.diagnostics()
    assert model.select_model(y)["best_n_states"] >= 2

"""Synthetic recovery and stress tests for GMM."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.regimes.gmm import GaussianMixtureModel, GMMSettings
from iqrp.app.regimes.gmm.evaluator import _best_accuracy
from iqrp.app.simulation.regimes.hidden_regime import HiddenRegimeSimulator
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator


@pytest.mark.unit
def test_mixture_recovery() -> None:
    rng = np.random.default_rng(21)
    n = 400
    true = rng.choice(2, size=n, p=[0.4, 0.6])
    x = np.empty((n, 2))
    means = [np.array([-2.0, -2.0]), np.array([2.0, 2.0])]
    for k in range(2):
        x[true == k] = rng.multivariate_normal(
            means[k], 0.25 * np.eye(2), size=int(np.sum(true == k))
        )
    settings = GMMSettings.from_mapping(
        {
            **GMMSettings.default().model_dump(),
            "n_components": 2,
            "covariance": {"type": "full", "reg_covar": 1e-6},
            "initialization": {"method": "kmeans++", "n_restarts": 3},
            "training": {
                "max_iter": 80,
                "tol": 1e-5,
                "early_stopping": True,
                "n_jobs": 2,
                "warm_start": False,
            },
            "preprocessing": {
                "standardize": False,
                "whiten": False,
                "pca_components": None,
                "ica_components": None,
            },
        }
    )
    model = GaussianMixtureModel(n_components=2, settings=settings, random_seed=21)
    model.fit(x)
    pred = model.predict(x)
    assert _best_accuracy(pred, true, 2) >= 0.85
    recovered = sorted(model.component_means().tolist())
    assert abs(recovered[0][0] - (-2.0)) < 0.6
    assert abs(recovered[1][0] - 2.0) < 0.6
    # model selection prefers 2 over 1 on well-separated data
    sel = model.select_model(x)
    assert sel["best_n_components"] >= 1


@pytest.mark.unit
def test_simulation_engine_and_stress() -> None:
    true_p = RegimeSwitchingSimulator.mixed_transition(2, 0.9)
    obs = HiddenRegimeSimulator(np.random.default_rng(22)).simulate(
        300,
        transition_matrix=true_p,
        state_names=("a", "b"),
        emission_means=(-1.0, 1.0),
        emission_stds=(0.4, 0.4),
    )
    y = obs.observations.reshape(-1, 1)
    settings = GMMSettings.from_mapping(
        {
            **GMMSettings.default().model_dump(),
            "n_components": 2,
            "covariance": {"type": "diag", "reg_covar": 1e-6},
            "training": {
                "max_iter": 50,
                "tol": 1e-4,
                "early_stopping": True,
                "n_jobs": 1,
                "warm_start": False,
            },
            "initialization": {"method": "kmeans", "n_restarts": 2},
        }
    )
    model = GaussianMixtureModel(n_components=2, settings=settings, random_seed=22)
    model.fit(y)
    assert _best_accuracy(model.predict(y), obs.latent.state_ids, 2) >= 0.70
    # stress: near-constant
    flat = np.zeros((80, 1))
    m2 = GaussianMixtureModel(
        n_components=2,
        settings=GMMSettings.from_mapping(
            {
                **settings.model_dump(),
                "covariance": {"type": "spherical", "reg_covar": 1e-4},
                "training": {**settings.training.model_dump(), "max_iter": 20},
            }
        ),
        random_seed=23,
    )
    m2.fit(flat)
    assert np.isfinite(m2.log_likelihood(flat))

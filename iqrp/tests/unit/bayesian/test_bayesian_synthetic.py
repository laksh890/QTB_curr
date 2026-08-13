"""Synthetic recovery and calibration tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.bayesian import BayesianRegimeSwitchingModel, BayesianSettings
from iqrp.app.regimes.bayesian.evaluator import _best_accuracy
from iqrp.app.simulation.regimes.hidden_regime import HiddenRegimeSimulator
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator


def _settings(**kw: object) -> BayesianSettings:
    data = {
        **BayesianSettings.default().model_dump(),
        "n_states": 2,
        "inference": {
            "algorithm": "gibbs",
            "n_chains": 2,
            "n_samples": 80,
            "burn_in": 20,
            "thin": 1,
            "target_accept": 0.65,
            "step_size": 0.05,
            "leapfrog_steps": 5,
            "n_jobs": 2,
            "checkpoint_every": 0,
            "resume": False,
        },
    }
    data.update(kw)
    return BayesianSettings.from_mapping(data)


@pytest.mark.unit
def test_synthetic_posterior_recovery() -> None:
    true_p = RegimeSwitchingSimulator.mixed_transition(2, 0.92)
    obs = HiddenRegimeSimulator(np.random.default_rng(11)).simulate(
        500,
        transition_matrix=true_p,
        state_names=("bear", "bull"),
        emission_means=(-1.5, 1.5),
        emission_stds=(0.35, 0.35),
    )
    y = obs.observations.reshape(-1, 1)
    model = BayesianRegimeSwitchingModel(n_states=2, settings=_settings(), random_seed=11)
    model.fit(y)
    pred = model.predict(y)
    acc = _best_accuracy(pred, obs.latent.state_ids, 2)
    assert acc >= 0.70
    tm = model.transition_matrix()
    persist = sorted(np.diag(tm).tolist(), reverse=True)
    true_persist = sorted(np.diag(true_p).tolist(), reverse=True)
    assert abs(persist[0] - true_persist[0]) < 0.25
    means = model.posterior_summary().mean_means().reshape(-1)  # type: ignore[union-attr]
    recovered = sorted(means.tolist())
    assert abs(recovered[0] - (-1.5)) < 0.75
    assert abs(recovered[1] - 1.5) < 0.75
    ci = model.credible_intervals("means", level=0.95)
    lows = np.asarray(ci["low"]).reshape(-1)
    highs = np.asarray(ci["high"]).reshape(-1)
    covered = any(lo <= -1.5 <= hi or lo <= 1.5 <= hi for lo, hi in zip(lows, highs, strict=False))
    assert covered
    report = model.evaluate(y, true_states=obs.latent.state_ids)
    assert report["metrics"]["prediction_accuracy"] >= 0.70


@pytest.mark.unit
def test_full_covariance_and_stress() -> None:
    rng = np.random.default_rng(12)
    y = []
    state = 0
    p = np.array([[0.9, 0.1], [0.15, 0.85]])
    means = [np.array([-1.0, -1.0]), np.array([1.0, 1.0])]
    covs = [np.array([[0.2, 0.05], [0.05, 0.2]]), np.array([[0.2, -0.05], [-0.05, 0.2]])]
    for _ in range(250):
        y.append(rng.multivariate_normal(means[state], covs[state]))
        state = int(rng.choice(2, p=p[state]))
    y_arr = np.asarray(y)
    settings = _settings(
        emission={"type": "multivariate_gaussian", "covariance_type": "full"},
        inference={
            **_settings().inference.model_dump(),
            "n_samples": 40,
            "burn_in": 10,
            "n_chains": 1,
            "n_jobs": 1,
        },
    )
    model = BayesianRegimeSwitchingModel(n_states=2, settings=settings, random_seed=12)
    model.fit(y_arr)
    assert model.emissions is not None and model.emissions.covars.ndim == 3
    flat = np.zeros((100, 1))
    m2 = BayesianRegimeSwitchingModel(
        n_states=2,
        settings=_settings(
            inference={
                **_settings().inference.model_dump(),
                "n_samples": 20,
                "burn_in": 5,
                "n_chains": 1,
                "algorithm": "variational",
            }
        ),
        random_seed=13,
    )
    m2.fit(flat)
    assert np.isfinite(m2.log_likelihood(flat))
    frame = pl.DataFrame({"a": y_arr[:, 0], "b": y_arr[:, 1]})
    model.predict(frame, observation_columns=["a", "b"])

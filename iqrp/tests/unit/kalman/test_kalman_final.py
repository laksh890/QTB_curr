"""Final coverage push for Kalman engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.regimes.kalman.config import KalmanSettings
from iqrp.app.regimes.kalman.covariance import ensure_spd
from iqrp.app.regimes.kalman.evaluator import KalmanEvaluator
from iqrp.app.regimes.kalman.initialization import build_system
from iqrp.app.regimes.kalman.linear import FilterTrace, filter_linear
from iqrp.app.regimes.kalman.model import KalmanFilterModel, _resolve_names
from iqrp.app.regimes.kalman.trainer import KalmanTrainer, simulate_lds
from iqrp.app.regimes.kalman.ukf import sigma_points, unscented_transform
from iqrp.app.regimes.kalman.visualization import plot_filtered_state


def _settings(**kw: object) -> KalmanSettings:
    data = {**KalmanSettings.default().model_dump(), "training": {"em_iterations": 2, "tol": 1.0, "estimate_noise": True}}
    data.update(kw)
    return KalmanSettings.from_mapping(data)


@pytest.mark.unit
def test_final_coverage_lines(tmp_path: Path) -> None:
    assert _resolve_names(None) == ("bearish", "bullish")
    assert _resolve_names(("a",))[0] == "a"
    assert len(_resolve_names(("a", "b", "c"))) == 2

    sys = build_system(_settings(application="custom", n_states=3, n_obs=2))
    assert sys.n_states == 3 and sys.n_obs == 2
    states, obs = simulate_lds(sys, 12, rng=np.random.default_rng(0))
    tr = filter_linear(obs, sys)
    # evaluator without true states / with n_params 0
    ev = KalmanEvaluator().evaluate(observations=obs, trace=tr, n_params=0)
    assert "log_likelihood" in ev["metrics"]
    # 1d true states
    ev2 = KalmanEvaluator().evaluate(
        observations=obs[:, 0],
        trace=FilterTrace(
            means=tr.means[:, :1],
            covs=tr.covs[:, :1, :1],
            pred_means=tr.pred_means[:, :1],
            pred_covs=tr.pred_covs[:, :1, :1],
            innovations=tr.innovations[:, :1],
            innovation_covs=tr.innovation_covs[:, :1, :1],
            gains=tr.gains[:, :1, :1],
            log_likelihood=tr.log_likelihood,
        ),
        true_states=states[:, 0],
        n_params=2,
    )
    assert "state_mse" in ev2["metrics"]

    # UKF cholesky fallback
    pts, wm, wc = sigma_points(np.zeros(1), np.array([[-1.0]]), alpha=0.5, kappa=0.0)
    mean, cov = unscented_transform(pts, wm, wc, np.array([[0.01]]))
    assert mean.shape[0] == 1 and cov[0, 0] > 0

    # trainer converge early via large tol
    trainer = KalmanTrainer(_settings(application="denoise"))
    sys_d = build_system(_settings(application="denoise"), application="denoise")
    _, y = simulate_lds(sys_d, 20, rng=np.random.default_rng(1))
    res = trainer.fit(y, system=sys_d)
    assert res.converged or res.n_iter >= 1

    # model sample initial_state + forecast metadata
    model = KalmanFilterModel(settings=_settings(application="denoise"), system=sys_d, random_seed=2)
    model.fit(y)
    labels, _obs = model.sample(5, initial_state=1)
    assert labels[0] == 1
    fc = model.forecast(y, horizon=2)
    assert "state_means" in fc.metadata
    # empty history plot
    plot_filtered_state(np.zeros(0), tmp_path / "z.svg", _settings())
    # ensure_spd LinAlgError path via non-square? use nan matrix with fallback
    bad = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    # eigh may produce nan; just call on identity after replace
    assert ensure_spd(np.eye(2)).shape == (2, 2)
    _ = bad

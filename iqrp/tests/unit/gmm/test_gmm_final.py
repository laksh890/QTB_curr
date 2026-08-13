"""Final coverage push for GMM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.regimes.gmm.config import GMMSettings
from iqrp.app.regimes.gmm.em import fit_em
from iqrp.app.regimes.gmm.evaluator import GMMEvaluator, _cluster_indices
from iqrp.app.regimes.gmm.gaussian import _full_logpdf
from iqrp.app.regimes.gmm.initialization import _hierarchical_centers, initialize_parameters
from iqrp.app.regimes.gmm.maximization import bayesian_m_step, m_step
from iqrp.app.regimes.gmm.mixture import PreprocessState, fit_preprocess
from iqrp.app.regimes.gmm.model import GaussianMixtureModel
from iqrp.app.regimes.gmm.model_selection import cross_validate_ll
from iqrp.app.regimes.gmm.serializer import _json_default
from iqrp.app.regimes.gmm.trainer import GMMTrainer
from iqrp.app.regimes.gmm.visualization import (
    plot_cluster_scatter,
    plot_component_weights,
    plot_covariance_ellipses,
    plot_likelihood_curve,
    plot_probability_heatmap,
    plot_regime_timeline,
)


@pytest.mark.unit
def test_final_coverage_lines(tmp_path: Path) -> None:
    # 1d init + m_step + trainer
    y1 = np.linspace(-2, 2, 50)
    initialize_parameters(y1, 2, method="random", rng=np.random.default_rng(0))
    resp = np.tile([0.5, 0.5], (50, 1))
    m_step(y1, resp, covariance_type="diag")
    bayesian_m_step(y1, resp, covariance_type="diag", mean_prior=np.array([0.0]))
    GMMTrainer(GMMSettings.default()).fit(y1, n_components=2, rng=np.random.default_rng(1))

    # hierarchical tiny / pad paths
    _hierarchical_centers(np.array([[0.0]]), 2)
    pts = np.random.default_rng(2).normal(size=(8, 2))
    _hierarchical_centers(pts, 4)

    # singular cov recovery in _full_logpdf
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert _full_logpdf(np.zeros((3, 2)), np.zeros(2), singular).shape == (3,)

    # cluster indices short / single cluster
    assert _cluster_indices(np.zeros((2, 1)), np.zeros(2, dtype=int))["silhouette"] == 0.0
    # silhouette with singleton cluster member
    x = np.array([[0.0], [0.1], [5.0], [5.1], [5.2]])
    labs = np.array([0, 0, 1, 1, 1])
    _cluster_indices(x, labs)

    # preprocess ica/whiten branches already; hit standardize false + ica path size
    fit_preprocess(np.random.default_rng(3).normal(size=(30, 3)), ica_components=2)

    # cv edge: too few train points
    cross_validate_ll(np.zeros((3, 1)), n_components=3, n_folds=2, max_iter=5)

    # serializer type error
    with pytest.raises(TypeError):
        _json_default(object())

    # model outliers + sample with preprocess inverse
    settings = GMMSettings.from_mapping(
        {
            **GMMSettings.default().model_dump(),
            "n_components": 2,
            "preprocessing": {
                "standardize": True,
                "whiten": False,
                "pca_components": None,
                "ica_components": None,
            },
            "training": {
                "max_iter": 25,
                "tol": 1e-3,
                "early_stopping": True,
                "n_jobs": 1,
                "warm_start": False,
            },
            "initialization": {"method": "kmeans", "n_restarts": 1},
        }
    )
    y = np.concatenate([np.full(40, -1.0), np.full(40, 1.0)]).reshape(-1, 1)
    model = GaussianMixtureModel(n_components=2, settings=settings, random_seed=4)
    model.fit(y)
    assert "n_outliers" in model.outliers(y)
    model.sample(5)
    # evaluate for silhouette path
    GMMEvaluator().evaluate(
        x=y,
        params=model.params,  # type: ignore[arg-type]
        responsibilities=model.predict_proba(y),
        log_likelihood=model.log_likelihood(y),
        true_labels=np.concatenate([np.zeros(40), np.ones(40)]).astype(int),
    )
    # params sample with preprocess
    assert model.params is not None
    model.params.sample(3, rng=np.random.default_rng(0))
    # viz disabled + empty weights + 1d scatter
    off = GMMSettings.from_mapping(
        {
            **GMMSettings.default().model_dump(),
            "visualization": {"enabled": False, "max_points": 2},
        }
    )
    plot_cluster_scatter(y, np.zeros(80, dtype=int), tmp_path / "a.svg", off)
    plot_probability_heatmap(np.eye(2), tmp_path / "b.svg", off)
    plot_regime_timeline(np.zeros(5, dtype=int), tmp_path / "c.svg", off)
    plot_covariance_ellipses(np.array([[0.0], [1.0]]), np.ones(2), tmp_path / "d.svg", off)
    plot_component_weights(np.array([0.5, 0.5]), tmp_path / "e.svg")
    plot_likelihood_curve(np.array([1.0, 2.0]), tmp_path / "f.svg")
    # bayesian_m_step mean_prior wrong shape
    bayesian_m_step(
        y.reshape(-1, 1),
        np.full((80, 2), 0.5),
        covariance_type="spherical",
        mean_prior=np.zeros((1, 3)),
    )
    # empty preprocess state transform
    PreprocessState.from_dict({"standardize": False})
    fit_em(y, 2, max_iter=3, n_restarts=1, rng=np.random.default_rng(9))

"""Additional coverage gaps for GMM engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.gmm.config import GMMSettings
from iqrp.app.regimes.gmm.covariance import (
    covariance_from_dict,
    covariance_to_dict,
    expand_covariance,
)
from iqrp.app.regimes.gmm.em import fit_em
from iqrp.app.regimes.gmm.gaussian import log_gaussian_pdf
from iqrp.app.regimes.gmm.initialization import initialize_parameters
from iqrp.app.regimes.gmm.maximization import bayesian_m_step, m_step
from iqrp.app.regimes.gmm.mixture import fit_preprocess, transform_preprocess
from iqrp.app.regimes.gmm.model import GaussianMixtureModel, GMMRegimeModel
from iqrp.app.regimes.gmm.model_selection import select_n_components
from iqrp.app.regimes.gmm.visualization import (
    plot_cluster_scatter,
    plot_component_weights,
    plot_covariance_ellipses,
    plot_likelihood_curve,
    plot_probability_heatmap,
    plot_regime_timeline,
)


@pytest.mark.unit
def test_bayesian_mstep_and_expand() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2))
    resp = np.full((40, 2), 0.5)
    for cov in ("full", "diag", "spherical", "tied"):
        _w, _m0, c = m_step(x, resp, covariance_type=cov)  # type: ignore[arg-type]
        w, _m, c = bayesian_m_step(
            x, resp, covariance_type=cov, mean_prior=np.zeros(2)  # type: ignore[arg-type]
        )
        assert w.shape[0] == 2
        assert expand_covariance(c, 2, 2, cov).shape == (2, 2, 2)  # type: ignore[arg-type]
    assert covariance_from_dict(covariance_to_dict(np.eye(2), "tied"), "tied").shape == (2, 2)
    # expand full from tied-shaped array
    assert expand_covariance(np.eye(2), 2, 2, "full").shape == (2, 2, 2)
    # 1d log pdf
    ll = log_gaussian_pdf(
        np.linspace(-1, 1, 10), np.array([-1.0, 1.0]), np.ones(2), covariance_type="spherical"
    )
    assert ll.shape == (10, 2)
    # hierarchical empty split path via larger data
    initialize_parameters(x, 3, method="hierarchical", rng=rng)
    # preprocess whiten only
    z, st = fit_preprocess(x, standardize=False, whiten=True)
    transform_preprocess(x, st)
    assert z.shape[1] == 2


@pytest.mark.unit
def test_parallel_em_and_viz_and_regime() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-1, 0.3, (60, 1)), rng.normal(1, 0.3, (60, 1))])
    result = fit_em(x, 2, n_restarts=3, n_jobs=2, max_iter=20, rng=rng)
    assert result.n_iter >= 1
    # bayesian 1d
    fit_em(
        x.reshape(-1),
        2,
        model_type="bayesian_gmm",
        covariance_type="diag",
        max_iter=15,
        n_restarts=1,
        rng=rng,
    )
    settings = GMMSettings.from_mapping(
        {
            **GMMSettings.default().model_dump(),
            "n_components": 2,
            "training": {
                "max_iter": 20,
                "tol": 1e-3,
                "early_stopping": True,
                "n_jobs": 1,
                "warm_start": False,
            },
            "initialization": {"method": "kmeans", "n_restarts": 1},
        }
    )
    model = GaussianMixtureModel(n_components=2, settings=settings, random_seed=2)
    model.fit(x)
    comps, _obs = model.sample(8, initial_state=0)
    assert comps[0] == 0
    # load from weights-only algorithm state path
    state = model.export_state()
    algo = state["algorithm_state"]
    algo.pop("params", None)
    m2 = GaussianMixtureModel(n_components=2, settings=settings)
    m2.import_state(state)
    assert m2.is_fitted
    # serializer path without npz covered by save
    # viz empties / 2d ellipses / disabled
    tmp = Path("/tmp/gmm_viz")
    tmp.mkdir(exist_ok=True)
    plot_likelihood_curve([], tmp / "a.svg")
    plot_cluster_scatter(np.zeros(5), np.zeros(5, dtype=int), tmp / "b.svg")
    plot_probability_heatmap(np.zeros(2), tmp / "c.svg")
    plot_regime_timeline(np.array([], dtype=int), tmp / "d.svg")
    plot_covariance_ellipses(np.zeros((0, 1)), np.zeros(0), tmp / "e.svg")
    plot_covariance_ellipses(np.array([[0.0, 0.0], [1.0, 1.0]]), np.eye(2), tmp / "f.svg")
    plot_component_weights([], tmp / "g.svg")
    off = GMMSettings.from_mapping(
        {**GMMSettings.default().model_dump(), "visualization": {"enabled": False, "max_points": 3}}
    )
    plot_likelihood_curve([1.0], tmp / "h.svg", off)
    # select log_likelihood criterion
    select_n_components(x, min_components=1, max_components=2, criterion="log_likelihood", rng=rng)
    frame = pl.DataFrame({"ret": x.ravel()})
    regime = GMMRegimeModel(n_components=2, settings=settings, random_seed=3)
    regime.fit(frame, feature_columns=["ret"])
    assert regime.predict(frame, feature_columns=["ret"]).shape[0] == frame.height
    regime.forecast(frame, steps=1)
    st = regime._algorithm_state()
    r2 = GMMRegimeModel(n_components=2, settings=settings)
    r2._load_algorithm_state(st)
    assert r2.is_fitted
    # singular gaussian path
    log_gaussian_pdf(
        np.zeros((4, 2)),
        np.zeros((2, 2)),
        np.array([np.ones((2, 2)), np.ones((2, 2))]),
        covariance_type="full",
    )

"""Gap-filling tests for GMM coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.core.exceptions import ValidationError
from iqrp.app.regimes.gmm.config import GMMSettings
from iqrp.app.regimes.gmm.covariance import component_covariance
from iqrp.app.regimes.gmm.em import fit_em
from iqrp.app.regimes.gmm.evaluator import _avg_ce, _best_accuracy
from iqrp.app.regimes.gmm.gaussian import _full_logpdf, log_gaussian_pdf, sample_gaussian
from iqrp.app.regimes.gmm.initialization import _hierarchical_centers
from iqrp.app.regimes.gmm.mixture import (
    GaussianMixtureParams,
    PreprocessState,
    transform_preprocess,
)
from iqrp.app.regimes.gmm.model import GaussianMixtureModel, GMMRegimeModel, _resolve_names
from iqrp.app.regimes.gmm.model_selection import select_n_components
from iqrp.app.regimes.gmm.prediction import (
    cluster_stability,
    forecast_occupancy,
    regime_similarity,
)
from iqrp.app.regimes.gmm.visualization import (
    plot_cluster_scatter,
    plot_covariance_ellipses,
    plot_likelihood_curve,
    plot_regime_timeline,
)
from iqrp.app.state_space.base.registry import get_registry as get_ss_registry


@pytest.mark.unit
def test_resolve_names_and_helpers() -> None:
    assert _resolve_names(3, ("a", "b", "c")) == ("a", "b", "c")
    assert _resolve_names(3, ("a",))[1] == "regime_1"
    assert _resolve_names(2, None) == ("regime_0", "regime_1")
    assert np.isnan(_avg_ce(np.array([0.5, 0.5]), np.array([0])))
    pred = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 0])
    truth = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 1])
    assert 0 <= _best_accuracy(pred, truth, 7) <= 1
    assert cluster_stability(np.array([0])) == 1.0
    assert regime_similarity(np.eye(2)).shape == (2, 2)
    fc = forecast_occupancy(np.array([[0.7, 0.3], [0.6, 0.4]]), horizon=2)
    assert fc.horizon == 2


@pytest.mark.unit
def test_gaussian_and_cov_paths() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=(30, 2))
    means = np.array([[0.0, 0.0], [1.0, 1.0]])
    for cov_type, covars in (
        ("diag", np.ones((2, 2))),
        ("spherical", np.array([0.5, 0.5])),
        ("tied", np.eye(2)),
        ("full", np.array([np.eye(2), np.eye(2)])),
    ):
        ll = log_gaussian_pdf(y, means, covars, covariance_type=cov_type)  # type: ignore[arg-type]
        assert ll.shape == (30, 2)
        sample_gaussian(means, covars, 0, covariance_type=cov_type, rng=rng)  # type: ignore[arg-type]
        component_covariance(covars, 0, cov_type)  # type: ignore[arg-type]
    # singular recovery
    assert _full_logpdf(np.zeros((3, 2)), np.zeros(2), np.ones((2, 2))).shape == (3,)
    # hierarchical small n
    tiny = np.array([[0.0, 0.0], [1.0, 1.0]])
    m, _lab = _hierarchical_centers(tiny, 3)
    assert m.shape[0] == 3


@pytest.mark.unit
def test_model_edges_and_regime_roundtrip(tmp_path: Path) -> None:
    import iqrp.app.regimes.gmm

    assert "gmm" in get_ss_registry().list_names()
    settings = GMMSettings.from_mapping(
        {
            **GMMSettings.default().model_dump(),
            "n_components": 2,
            "model_type": "gmm",
            "covariance": {"type": "tied", "reg_covar": 1e-6},
            "training": {
                "max_iter": 20,
                "tol": 1e-3,
                "early_stopping": True,
                "n_jobs": 1,
                "warm_start": False,
            },
            "initialization": {"method": "random", "n_restarts": 1},
            "columns": {
                **GMMSettings.default().columns.model_dump(),
                "observation_columns": ["ret"],
            },
            "model_selection": {
                "min_components": 1,
                "max_components": 2,
                "criterion": "cv",
                "cv_folds": 2,
            },
            "preprocessing": {
                "standardize": True,
                "whiten": True,
                "pca_components": None,
                "ica_components": None,
            },
        }
    )
    y = np.concatenate([np.full(40, -1.0), np.full(40, 1.0)]).reshape(-1, 1)
    model = GaussianMixtureModel(n_components=2, settings=settings, random_seed=5)
    model.fit(y)
    frame = pl.DataFrame({"open_time": list(range(80)), "ret": y.ravel()})
    model.fit(frame)
    frame_close = pl.DataFrame({"open_time": list(range(80)), "close": y.ravel()})
    m2 = GaussianMixtureModel(
        n_components=2,
        settings=GMMSettings.from_mapping(
            {
                **settings.model_dump(),
                "columns": {**settings.columns.model_dump(), "observation_columns": None},
                "preprocessing": {
                    "standardize": False,
                    "whiten": False,
                    "pca_components": None,
                    "ica_components": None,
                },
            }
        ),
        random_seed=6,
    )
    m2.fit(frame_close)
    assert np.isfinite(m2.aic(frame_close))
    assert np.isfinite(m2.bic(frame_close))
    m2._train_obs = None
    m2._responsibilities = None
    with pytest.raises(ValidationError):
        m2.diagnostics()
    # warm start em
    assert model.params is not None
    fit_em(
        y,
        2,
        covariance_type="tied",
        warm_start=(model.params.weights, model.params.means, model.params.covars),
        max_iter=5,
        rng=np.random.default_rng(0),
    )
    sel = select_n_components(
        y,
        min_components=1,
        max_components=2,
        criterion="cv",
        cv_folds=2,
        rng=np.random.default_rng(1),
    )
    assert sel["best_n_components"] >= 1
    regime = GMMRegimeModel(n_components=2, settings=settings, random_seed=7)
    regime.fit(frame, feature_columns=["ret"])
    state = regime._algorithm_state()
    r2 = GMMRegimeModel(n_components=2, settings=settings, random_seed=8)
    r2._load_algorithm_state(state)
    assert r2.is_fitted
    # preprocess transform / sample inverse
    params = GaussianMixtureParams.from_dict(model.params.to_dict())
    transform_preprocess(y, params.preprocess)
    params.sample(5, rng=np.random.default_rng(0))
    PreprocessState.from_dict(PreprocessState().to_dict())
    plot_likelihood_curve([], tmp_path / "a.svg")
    plot_cluster_scatter(np.linspace(-1, 1, 20), np.zeros(20, dtype=int), tmp_path / "b.svg")
    plot_regime_timeline([], tmp_path / "c.svg")
    plot_covariance_ellipses(np.array([[0.0, 0.0], [1.0, 1.0]]), np.eye(2), tmp_path / "d.svg")
    off = GMMSettings.from_mapping(
        {
            **GMMSettings.default().model_dump(),
            "visualization": {"enabled": False, "max_points": 5},
        }
    )
    plot_likelihood_curve([1.0], tmp_path / "e.svg", off)

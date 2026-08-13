"""Core unit tests for GMM regime engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.gmm import (
    GaussianMixtureModel,
    GMMRegimeModel,
    GMMSettings,
    fit_em,
    select_n_components,
)
from iqrp.app.regimes.gmm.covariance import expand_covariance, n_covariance_params
from iqrp.app.regimes.gmm.initialization import initialize_parameters
from iqrp.app.regimes.gmm.visualization import (
    plot_cluster_scatter,
    plot_component_weights,
    plot_likelihood_curve,
    plot_probability_heatmap,
    plot_regime_timeline,
)


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 120
    labels = rng.choice(2, size=n, p=[0.45, 0.55])
    x = np.empty((n, 2))
    x[labels == 0] = rng.normal([-1.5, -1.5], 0.35, size=(np.sum(labels == 0), 2))
    x[labels == 1] = rng.normal([1.5, 1.5], 0.35, size=(np.sum(labels == 1), 2))
    return x, labels


def _settings(**kw: object) -> GMMSettings:
    data = {
        **GMMSettings.default().model_dump(),
        "n_components": 2,
        "preprocessing": {
            "standardize": True,
            "whiten": False,
            "pca_components": None,
            "ica_components": None,
        },
        "training": {
            "max_iter": 40,
            "tol": 1e-4,
            "early_stopping": True,
            "n_jobs": 2,
            "warm_start": False,
        },
        "initialization": {"method": "kmeans", "n_restarts": 2},
        "model_selection": {
            "min_components": 2,
            "max_components": 2,
            "criterion": "bic",
            "cv_folds": 2,
        },
    }
    data.update(kw)
    return GMMSettings.from_mapping(data)


@pytest.mark.unit
def test_init_cov_types_and_em() -> None:
    x, _ = _data(1)
    for method in ("random", "kmeans", "kmeans++", "hierarchical"):
        w, m, _c = initialize_parameters(x, 2, method=method, rng=np.random.default_rng(1))  # type: ignore[arg-type]
        assert w.shape[0] == 2 and m.shape == (2, 2)
    for cov in ("full", "diag", "tied", "spherical"):
        result = fit_em(
            x,
            2,
            covariance_type=cov,  # type: ignore[arg-type]
            max_iter=25,
            n_restarts=1,
            n_jobs=1,
            rng=np.random.default_rng(2),
        )
        assert result.converged or result.n_iter >= 1
        assert n_covariance_params(2, 2, cov) > 0  # type: ignore[arg-type]
        assert expand_covariance(result.covars, 2, 2, cov).shape == (2, 2, 2)  # type: ignore[arg-type]


@pytest.mark.unit
def test_model_api(tmp_path: Path) -> None:
    x, labels = _data(2)
    model = GaussianMixtureModel(n_components=2, settings=_settings(), random_seed=2)
    model.fit(x)
    pred = model.predict(x)
    proba = model.predict_proba(x)
    assert pred.shape[0] == x.shape[0]
    assert proba.shape == (x.shape[0], 2)
    assert np.isfinite(model.score(x))
    assert model.component_means().shape[0] == 2
    assert model.component_covariances().shape[0] == 2
    assert "weights" in model.cluster_statistics()
    fc = model.forecast(x, horizon=3)
    assert fc.horizon == 3
    comps, obs = model.sample(20)
    assert comps.shape[0] == 20 and obs.shape[0] == 20
    path = model.save(tmp_path / "gmm.json")
    loaded = GaussianMixtureModel.load(path)
    assert loaded.is_fitted
    diag = model.diagnostics(x)
    assert "occupancy" in diag
    out = model.outliers(x)
    assert "n_outliers" in out
    plot_likelihood_curve(model._history, tmp_path / "ll.svg")
    plot_cluster_scatter(x, pred, tmp_path / "sc.svg")
    plot_probability_heatmap(proba, tmp_path / "hm.svg")
    plot_regime_timeline(pred, tmp_path / "tl.svg")
    plot_component_weights(model.params.weights, tmp_path / "w.svg")  # type: ignore[union-attr]
    report = model.evaluate(x, true_states=labels)
    assert report["metrics"]["prediction_accuracy"] >= 0.75


@pytest.mark.unit
def test_bayesian_selection_and_regime() -> None:
    x, _ = _data(3)
    settings = _settings(model_type="bayesian_gmm")
    model = GaussianMixtureModel(n_components=2, settings=settings, random_seed=3)
    model.fit(x)
    assert model.params is not None
    sel = select_n_components(
        x, min_components=2, max_components=2, criterion="aic", rng=np.random.default_rng(3)
    )
    assert sel["best_n_components"] == 2
    frame = pl.DataFrame({"a": x[:, 0], "b": x[:, 1]})
    regime = GMMRegimeModel(n_components=2, settings=_settings(), random_seed=4)
    regime.fit(frame, feature_columns=["a", "b"])
    assert regime.predict_proba(frame, feature_columns=["a", "b"]).shape[1] == 2
    assert regime.forecast(frame, steps=2).steps == 2

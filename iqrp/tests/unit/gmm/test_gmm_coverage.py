"""Coverage / edge-case tests for GMM engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.regimes.gmm import GaussianMixtureModel, GMMSettings
from iqrp.app.regimes.gmm.initialization import initialize_parameters
from iqrp.app.regimes.gmm.mixture import fit_preprocess
from iqrp.app.regimes.gmm.model_selection import cross_validate_ll
from iqrp.app.regimes.gmm.serializer import _json_default
from iqrp.app.regimes.gmm.visualization import plot_covariance_ellipses, plot_probability_heatmap


@pytest.mark.unit
def test_config_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError):
        GMMSettings.from_mapping([1, 2])  # type: ignore[arg-type]
    assert GMMSettings.from_mapping(OmegaConf.create({"n_components": 2})).n_components == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        GMMSettings.from_hydra(bad)
    monkeypatch.setattr(
        "iqrp.app.regimes.gmm.config._default_config_path",
        lambda: tmp_path / "missing.yaml",
    )
    assert GMMSettings.default().n_components == 3


@pytest.mark.unit
def test_preprocess_user_init_partial_fit(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 3))
    z, st = fit_preprocess(x, standardize=True, pca_components=2, whiten=True)
    assert z.shape[1] == 2 and st.pca_components is not None
    z2, st2 = fit_preprocess(x, standardize=True, ica_components=2)
    assert z2.shape[1] == 2 and st2.ica_components is not None
    w, m, c = initialize_parameters(x[:, :2], 2, method="kmeans", rng=rng)
    w2, m2, _c2 = initialize_parameters(
        x[:, :2],
        2,
        method="user",
        user_params={"weights": w, "means": m, "covars": c},
    )
    assert np.allclose(w2, w)
    settings = GMMSettings.from_mapping(
        {
            **GMMSettings.default().model_dump(),
            "n_components": 2,
            "online": {
                "window_size": 40,
                "update_frequency": 2,
                "warm_start": True,
                "adaptive_covariance": True,
            },
            "training": {
                "max_iter": 25,
                "tol": 1e-3,
                "early_stopping": True,
                "n_jobs": 1,
                "warm_start": False,
            },
            "initialization": {"method": "kmeans", "n_restarts": 1},
            "preprocessing": {
                "standardize": True,
                "whiten": False,
                "pca_components": None,
                "ica_components": None,
            },
        }
    )
    model = GaussianMixtureModel(n_components=2, settings=settings, random_seed=1)
    y = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)]).reshape(-1, 1)
    model.partial_fit(y)  # fits first
    model.partial_fit(y[:20])  # skip
    model.partial_fit(y[20:40])  # update
    settings2 = GMMSettings.from_mapping(
        {
            **settings.model_dump(),
            "online": {
                "window_size": 0,
                "update_frequency": 1,
                "warm_start": False,
                "adaptive_covariance": False,
            },
        }
    )
    m2 = GaussianMixtureModel(n_components=2, settings=settings2, random_seed=2)
    m2.fit(y)
    m2.partial_fit(y[:30])
    assert _json_default(np.array([1.0])) == [1.0]
    with pytest.raises(TypeError):
        _json_default(object())
    bare = GaussianMixtureModel(n_components=2)
    with pytest.raises(ValidationError):
        bare.predict(y)
    with pytest.raises(ValidationError):
        bare.fit(pl.DataFrame({"open_time": [0, 1], "symbol": ["a", "b"]}))
    plot_probability_heatmap(np.array([0.5, 0.5]), tmp_path / "p.svg")
    plot_covariance_ellipses(np.array([[0.0], [1.0]]), np.ones((2, 1)), tmp_path / "e.svg")
    cv = cross_validate_ll(y, 2, n_folds=2, max_iter=15, rng=np.random.default_rng(0))
    assert np.isfinite(cv) or cv == float("-inf")

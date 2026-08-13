"""Training orchestration for GMM regime detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.regimes.gmm.config import GMMSettings
from iqrp.app.regimes.gmm.em import EMResult, fit_em
from iqrp.app.regimes.gmm.mixture import fit_preprocess
from iqrp.app.regimes.gmm.model_selection import select_n_components


class GMMTrainer:
    def __init__(self, settings: GMMSettings | None = None) -> None:
        self.settings = settings or GMMSettings.default()

    def fit(
        self,
        observations: np.ndarray,
        *,
        n_components: int | None = None,
        warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[EMResult, Any]:
        s = self.settings
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_proc, prep = fit_preprocess(
            y,
            standardize=s.preprocessing.standardize,
            whiten=s.preprocessing.whiten,
            pca_components=s.preprocessing.pca_components,
            ica_components=s.preprocessing.ica_components,
        )
        k = int(n_components if n_components is not None else s.n_components)
        result = fit_em(
            y_proc,
            k,
            model_type=s.model_type,
            covariance_type=s.covariance.type,
            init_method=s.initialization.method,
            max_iter=s.training.max_iter,
            tol=s.training.tol,
            early_stopping=s.training.early_stopping,
            reg_covar=s.covariance.reg_covar,
            n_restarts=1 if warm_start is not None else s.initialization.n_restarts,
            n_jobs=s.training.n_jobs,
            warm_start=warm_start,
            bayesian_params=s.bayesian.model_dump(),
            rng=rng,
        )
        return result, prep

    def select(
        self,
        observations: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        s = self.settings
        return select_n_components(
            observations,
            min_components=s.model_selection.min_components,
            max_components=s.model_selection.max_components,
            criterion=s.model_selection.criterion,
            cv_folds=s.model_selection.cv_folds,
            preprocess=True,
            rng=rng,
            model_type=s.model_type,
            covariance_type=s.covariance.type,
            max_iter=s.training.max_iter,
            tol=s.training.tol,
            reg_covar=s.covariance.reg_covar,
        )

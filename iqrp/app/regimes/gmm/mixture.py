"""Mixture model parameter container and preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.math.matrices.eigen import principal_components
from iqrp.app.regimes.gmm.covariance import CovarianceType, expand_covariance, n_covariance_params
from iqrp.app.regimes.gmm.expectation import e_step, pointwise_log_density
from iqrp.app.regimes.gmm.gaussian import sample_gaussian


@dataclass
class PreprocessState:
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    pca_components: np.ndarray | None = None
    ica_components: np.ndarray | None = None
    whiten: bool = False
    standardize: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
            "pca_components": None if self.pca_components is None else self.pca_components.tolist(),
            "ica_components": None if self.ica_components is None else self.ica_components.tolist(),
            "whiten": self.whiten,
            "standardize": self.standardize,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessState:
        return cls(
            mean=None if data.get("mean") is None else np.asarray(data["mean"], dtype=np.float64),
            scale=(
                None if data.get("scale") is None else np.asarray(data["scale"], dtype=np.float64)
            ),
            pca_components=(
                None
                if data.get("pca_components") is None
                else np.asarray(data["pca_components"], dtype=np.float64)
            ),
            ica_components=(
                None
                if data.get("ica_components") is None
                else np.asarray(data["ica_components"], dtype=np.float64)
            ),
            whiten=bool(data.get("whiten", False)),
            standardize=bool(data.get("standardize", True)),
        )


def fit_preprocess(
    x: np.ndarray,
    *,
    standardize: bool = True,
    whiten: bool = False,
    pca_components: int | None = None,
    ica_components: int | None = None,
) -> tuple[np.ndarray, PreprocessState]:
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    state = PreprocessState(standardize=standardize, whiten=whiten)
    if standardize:
        state.mean = y.mean(axis=0)
        state.scale = np.clip(y.std(axis=0), 1e-12, None)
        y = (y - state.mean) / state.scale
    if pca_components is not None and pca_components > 0:
        k = min(int(pca_components), y.shape[1])
        pca = principal_components(y, n_components=k)
        state.pca_components = pca["components"]
        y = pca["scores"]
        if whiten:
            ev = np.clip(pca["eigenvalues"], 1e-12, None)
            y = y / np.sqrt(ev)
    if ica_components is not None and ica_components > 0:
        # FastICA-lite: PCA whitening + random orthogonal rotation (deterministic via SVD)
        k = min(int(ica_components), y.shape[1])
        pca = principal_components(y, n_components=k)
        z = pca["scores"]
        ev = np.clip(pca["eigenvalues"], 1e-12, None)
        z = z / np.sqrt(ev)
        # symmetric decorrelation via SVD of kurtosis-proxy gradient
        w = z.T @ (z**3) / max(z.shape[0], 1)
        u, _, vt = np.linalg.svd(w + 1e-9 * np.eye(k), full_matrices=False)
        rot = u @ vt
        state.ica_components = pca["components"] @ np.diag(1.0 / np.sqrt(ev)) @ rot
        y = z @ rot
        state.whiten = True
    elif whiten and state.pca_components is None and y.shape[1] > 1:
        pca = principal_components(y, n_components=y.shape[1])
        state.pca_components = pca["components"]
        ev = np.clip(pca["eigenvalues"], 1e-12, None)
        y = pca["scores"] / np.sqrt(ev)
        state.whiten = True
    return y, state


def transform_preprocess(x: np.ndarray, state: PreprocessState) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if state.standardize and state.mean is not None and state.scale is not None:
        y = (y - state.mean) / state.scale
    if state.ica_components is not None:
        y = y @ state.ica_components
    elif state.pca_components is not None:
        y = y @ state.pca_components
        if state.whiten:
            # approximate — components already store rotation; scale by feature std of scores
            s = np.clip(np.std(y, axis=0), 1e-12, None)
            y = y / s
    return y


@dataclass
class GaussianMixtureParams:
    weights: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    covariance_type: CovarianceType = "full"
    preprocess: PreprocessState = field(default_factory=PreprocessState)
    model_type: str = "gmm"

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    def n_params(self) -> int:
        k, d = self.n_components, self.n_features
        # weights K-1 + means K*d + cov
        return (k - 1) + k * d + n_covariance_params(k, d, self.covariance_type)

    def responsibilities(self, x: np.ndarray) -> np.ndarray:
        z = transform_preprocess(x, self.preprocess)
        resp, _ = e_step(
            z, self.weights, self.means, self.covars, covariance_type=self.covariance_type
        )
        return resp

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        z = transform_preprocess(x, self.preprocess)
        return pointwise_log_density(
            z, self.weights, self.means, self.covars, covariance_type=self.covariance_type
        )

    def sample(
        self,
        n_samples: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng()
        comps = rng.choice(self.n_components, size=n_samples, p=self.weights)
        out = np.empty((n_samples, self.n_features), dtype=np.float64)
        for i, c in enumerate(comps):
            out[i] = sample_gaussian(
                self.means, self.covars, int(c), covariance_type=self.covariance_type, rng=rng
            )
        # inverse preprocess (standardize only)
        if (
            (
                self.preprocess.standardize
                and self.preprocess.mean is not None
                and self.preprocess.scale is not None
            )
            and self.preprocess.pca_components is None
            and self.preprocess.ica_components is None
        ):
            out = out * self.preprocess.scale + self.preprocess.mean
        return comps.astype(np.int64), out

    def expanded_covariances(self) -> np.ndarray:
        return expand_covariance(
            self.covars, self.n_components, self.n_features, self.covariance_type
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "covars": self.covars.tolist(),
            "covariance_type": self.covariance_type,
            "preprocess": self.preprocess.to_dict(),
            "model_type": self.model_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GaussianMixtureParams:
        return cls(
            weights=np.asarray(data["weights"], dtype=np.float64),
            means=np.asarray(data["means"], dtype=np.float64),
            covars=np.asarray(data["covars"], dtype=np.float64),
            covariance_type=data.get("covariance_type", "full"),
            preprocess=PreprocessState.from_dict(data.get("preprocess") or {}),
            model_type=str(data.get("model_type", "gmm")),
        )

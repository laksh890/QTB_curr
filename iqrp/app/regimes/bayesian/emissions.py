"""Bayesian emission models (Gaussian / multivariate Gaussian)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from iqrp.app.regimes.bayesian.priors import ModelPriors, sample_invgamma, sample_wishart

CovarianceType = Literal["diag", "full"]


@dataclass
class BayesianEmissions:
    """Gaussian emissions with Normal-Inverse-Gamma / Wishart priors."""

    n_states: int
    n_features: int
    means: np.ndarray
    covars: np.ndarray
    covariance_type: CovarianceType
    priors: ModelPriors

    @classmethod
    def from_priors(
        cls,
        priors: ModelPriors,
        n_states: int,
        n_features: int,
        *,
        covariance_type: CovarianceType = "diag",
        rng: np.random.Generator,
    ) -> BayesianEmissions:
        k, d = int(n_states), int(n_features)
        means = priors.mean_loc.copy()
        if covariance_type == "diag":
            var = sample_invgamma(priors.invgamma_shape, priors.invgamma_scale, k * d, rng)
            covars = var.reshape(k, d)
            for i in range(k):
                means[i] = rng.normal(
                    priors.mean_loc[i],
                    np.sqrt(np.clip(covars[i] / max(priors.mean_strength, 1e-6), 1e-12, None)),
                )
        else:
            covars = np.empty((k, d, d), dtype=np.float64)
            for i in range(k):
                precision = sample_wishart(
                    priors.wishart_df, np.linalg.inv(priors.wishart_scale + 1e-9 * np.eye(d)), rng
                )
                cov = np.linalg.inv(precision + 1e-9 * np.eye(d))
                covars[i] = 0.5 * (cov + cov.T)
                means[i] = rng.multivariate_normal(
                    priors.mean_loc[i], covars[i] / max(priors.mean_strength, 1e-6)
                )
        return cls(k, d, means, covars, covariance_type, priors)

    def log_prob(self, observations: np.ndarray) -> np.ndarray:
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        t, _ = y.shape
        out = np.empty((t, self.n_states), dtype=np.float64)
        for k in range(self.n_states):
            out[:, k] = _gaussian_logpdf(y, self.means[k], self.covars[k], self.covariance_type)
        return out

    def sample_posterior(
        self,
        observations: np.ndarray,
        states: np.ndarray,
        *,
        rng: np.random.Generator,
        min_covar: float = 1e-6,
    ) -> BayesianEmissions:
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        means = self.means.copy()
        covars = self.covars.copy()
        pri = self.priors
        for k in range(self.n_states):
            mask = s == k
            n_k = int(mask.sum())
            if self.covariance_type == "diag":
                if n_k == 0:
                    var = sample_invgamma(
                        pri.invgamma_shape, pri.invgamma_scale, self.n_features, rng
                    )
                    covars[k] = np.clip(var, min_covar, None)
                    means[k] = rng.normal(
                        pri.mean_loc[k],
                        np.sqrt(np.clip(covars[k] / max(pri.mean_strength, 1e-6), 1e-12, None)),
                    )
                    continue
                yk = y[mask]
                ybar = yk.mean(axis=0)
                sse = np.sum((yk - ybar) ** 2, axis=0)
                kappa_n = pri.mean_strength + n_k
                m_n = (pri.mean_strength * pri.mean_loc[k] + n_k * ybar) / kappa_n
                a_n = pri.invgamma_shape + 0.5 * n_k
                b_n = (
                    pri.invgamma_scale
                    + 0.5 * sse
                    + 0.5 * (pri.mean_strength * n_k / kappa_n) * (ybar - pri.mean_loc[k]) ** 2
                )
                var = np.array(
                    [
                        sample_invgamma(float(a_n), float(b_n[j]), 1, rng)[0]
                        for j in range(self.n_features)
                    ]
                )
                covars[k] = np.clip(var, min_covar, None)
                means[k] = rng.normal(m_n, np.sqrt(np.clip(covars[k] / kappa_n, 1e-12, None)))
            else:
                d = self.n_features
                if n_k == 0:
                    precision = sample_wishart(
                        pri.wishart_df,
                        np.linalg.inv(pri.wishart_scale + 1e-9 * np.eye(d)),
                        rng,
                    )
                    cov = np.linalg.inv(precision + 1e-9 * np.eye(d))
                    covars[k] = 0.5 * (cov + cov.T) + min_covar * np.eye(d)
                    means[k] = rng.multivariate_normal(
                        pri.mean_loc[k], covars[k] / max(pri.mean_strength, 1e-6)
                    )
                    continue
                yk = y[mask]
                ybar = yk.mean(axis=0)
                scatter = (yk - ybar).T @ (yk - ybar)
                kappa_n = pri.mean_strength + n_k
                m_n = (pri.mean_strength * pri.mean_loc[k] + n_k * ybar) / kappa_n
                diff = (ybar - pri.mean_loc[k]).reshape(-1, 1)
                scale_n = (
                    pri.wishart_scale
                    + scatter
                    + (pri.mean_strength * n_k / kappa_n) * (diff @ diff.T)
                )
                df_n = pri.wishart_df + n_k
                precision = sample_wishart(df_n, np.linalg.inv(scale_n + 1e-9 * np.eye(d)), rng)
                cov = np.linalg.inv(precision + 1e-9 * np.eye(d))
                covars[k] = 0.5 * (cov + cov.T) + min_covar * np.eye(d)
                means[k] = rng.multivariate_normal(m_n, covars[k] / kappa_n)
        return BayesianEmissions(
            self.n_states,
            self.n_features,
            means,
            covars,
            self.covariance_type,
            self.priors,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "n_features": self.n_features,
            "means": self.means.tolist(),
            "covars": self.covars.tolist(),
            "covariance_type": self.covariance_type,
            "priors": self.priors.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BayesianEmissions:
        return cls(
            n_states=int(data["n_states"]),
            n_features=int(data["n_features"]),
            means=np.asarray(data["means"], dtype=np.float64),
            covars=np.asarray(data["covars"], dtype=np.float64),
            covariance_type=data.get("covariance_type", "diag"),
            priors=ModelPriors.from_dict(data["priors"]),
        )


def _gaussian_logpdf(
    y: np.ndarray,
    mean: np.ndarray,
    covar: np.ndarray,
    covariance_type: CovarianceType,
) -> np.ndarray:
    diff = y - mean
    d = y.shape[1]
    if covariance_type == "diag":
        var = np.clip(np.asarray(covar, dtype=np.float64).reshape(-1), 1e-12, None)
        out = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(var)) + np.sum(diff**2 / var, axis=1))
        return np.asarray(out, dtype=np.float64)
    cov = np.asarray(covar, dtype=np.float64) + 1e-9 * np.eye(d)
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise np.linalg.LinAlgError
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-3 * np.eye(d)
        _, logdet = np.linalg.slogdet(cov)
        inv = np.linalg.pinv(cov)
    quad = np.einsum("ti,ij,tj->t", diff, inv, diff)
    return np.asarray(-0.5 * (d * np.log(2 * np.pi) + logdet + quad), dtype=np.float64)

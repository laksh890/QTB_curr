"""Posterior containers, summaries, and credible intervals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.math.statistics.confidence import ConfidenceInterval


@dataclass
class ParameterDraw:
    """One MCMC / VI draw of regime-switching parameters."""

    transition: np.ndarray
    initial: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    states: np.ndarray | None = None
    log_joint: float = 0.0
    chain_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition": self.transition.tolist(),
            "initial": self.initial.tolist(),
            "means": self.means.tolist(),
            "covars": self.covars.tolist(),
            "states": None if self.states is None else self.states.tolist(),
            "log_joint": float(self.log_joint),
            "chain_id": int(self.chain_id),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterDraw:
        states = data.get("states")
        return cls(
            transition=np.asarray(data["transition"], dtype=np.float64),
            initial=np.asarray(data["initial"], dtype=np.float64),
            means=np.asarray(data["means"], dtype=np.float64),
            covars=np.asarray(data["covars"], dtype=np.float64),
            states=None if states is None else np.asarray(states, dtype=np.int64),
            log_joint=float(data.get("log_joint", 0.0)),
            chain_id=int(data.get("chain_id", 0)),
        )


@dataclass
class Posterior:
    """Collection of posterior draws with summary helpers."""

    draws: list[ParameterDraw] = field(default_factory=list)
    burn_in: int = 0
    thin: int = 1
    n_chains: int = 1
    algorithm: str = "gibbs"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return len(self.draws)

    def kept(self) -> list[ParameterDraw]:
        return self.draws

    def mean_transition(self) -> np.ndarray:
        if not self.draws:
            return np.zeros((0, 0))
        return np.mean([d.transition for d in self.draws], axis=0)

    def mean_initial(self) -> np.ndarray:
        if not self.draws:
            return np.zeros(0)
        return np.mean([d.initial for d in self.draws], axis=0)

    def mean_means(self) -> np.ndarray:
        if not self.draws:
            return np.zeros((0, 0))
        return np.mean([d.means for d in self.draws], axis=0)

    def mean_covars(self) -> np.ndarray:
        if not self.draws:
            return np.zeros((0, 0))
        return np.mean([d.covars for d in self.draws], axis=0)

    def state_occupancy(self) -> np.ndarray:
        mats = []
        for d in self.draws:
            if d.states is None:
                continue
            k = d.transition.shape[0]
            counts = np.bincount(d.states.astype(np.int64), minlength=k).astype(np.float64)
            mats.append(counts / max(float(counts.sum()), 1.0))
        if not mats:
            k = self.draws[0].transition.shape[0] if self.draws else 1
            return np.full(k, 1.0 / k)
        return np.mean(mats, axis=0)

    def posterior_state_probabilities(self, n_steps: int | None = None) -> np.ndarray:
        """Empirical P(Z_t = k | data) from sampled latent paths."""
        paths = [d.states for d in self.draws if d.states is not None]
        if not paths:
            k = self.draws[0].transition.shape[0] if self.draws else 1
            t = int(n_steps or 1)
            return np.full((t, k), 1.0 / k)
        t = int(n_steps or paths[0].size)
        k = int(self.draws[0].transition.shape[0])
        proba = np.zeros((t, k), dtype=np.float64)
        for s in paths:
            ss = np.asarray(s, dtype=np.int64)[:t]
            for i, st in enumerate(ss):
                if 0 <= int(st) < k:
                    proba[i, int(st)] += 1.0
        proba /= max(len(paths), 1)
        return proba

    def credible_intervals(
        self,
        parameter: str = "means",
        *,
        level: float = 0.95,
    ) -> dict[str, Any]:
        alpha = 1.0 - float(level)
        lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
        if not self.draws:
            empty = np.zeros(0, dtype=np.float64)
            return {
                "parameter": parameter,
                "level": float(level),
                "mean": empty,
                "low": empty,
                "high": empty,
            }
        if parameter == "transition":
            stack = np.stack([d.transition for d in self.draws], axis=0)
        elif parameter == "initial":
            stack = np.stack([d.initial for d in self.draws], axis=0)
        elif parameter == "covars":
            stack = np.stack([d.covars for d in self.draws], axis=0)
        else:
            stack = np.stack([d.means for d in self.draws], axis=0)
        low = np.percentile(stack, lo_q, axis=0)
        high = np.percentile(stack, hi_q, axis=0)
        mean = np.mean(stack, axis=0)
        return {
            "parameter": parameter,
            "level": float(level),
            "mean": mean,
            "low": low,
            "high": high,
        }

    def marginal_summary(self, parameter: str = "means") -> dict[str, Any]:
        ci = self.credible_intervals(parameter, level=0.95)
        return {
            "mean": ci["mean"],
            "std": np.std(
                np.stack(
                    [
                        getattr(d, parameter if parameter != "means" else "means")
                        for d in self.draws
                    ],
                    axis=0,
                ),
                axis=0,
            ),
            "credible_95": {"low": ci["low"], "high": ci["high"]},
        }

    def scalar_ci(self, values: np.ndarray, *, level: float = 0.95) -> ConfidenceInterval:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        alpha = 1.0 - float(level)
        low, high = np.quantile(v, [alpha / 2, 1 - alpha / 2])
        return ConfidenceInterval(
            float(low), float(high), float(level), "posterior", float(np.mean(v))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "draws": [d.to_dict() for d in self.draws],
            "burn_in": self.burn_in,
            "thin": self.thin,
            "n_chains": self.n_chains,
            "algorithm": self.algorithm,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Posterior:
        return cls(
            draws=[ParameterDraw.from_dict(d) for d in data.get("draws") or []],
            burn_in=int(data.get("burn_in", 0)),
            thin=int(data.get("thin", 1)),
            n_chains=int(data.get("n_chains", 1)),
            algorithm=str(data.get("algorithm", "gibbs")),
            metadata=dict(data.get("metadata") or {}),
        )


def posterior_predictive_observations(
    posterior: Posterior,
    *,
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw predictive observations from posterior parameter draws."""
    rng = rng or np.random.default_rng()
    if not posterior.draws:
        return np.zeros((0, 1))
    outs = []
    for d in posterior.draws:
        means = np.asarray(d.means, dtype=np.float64)
        covars = np.asarray(d.covars, dtype=np.float64)
        p = np.asarray(d.transition, dtype=np.float64)
        pi = np.asarray(d.initial, dtype=np.float64)
        pi = pi / max(float(pi.sum()), 1e-300)
        k, feat = means.shape[0], means.shape[1] if means.ndim == 2 else 1
        if means.ndim == 1:
            means = means.reshape(-1, 1)
            feat = 1
        state = int(rng.choice(k, p=pi))
        y = np.empty((n_steps, feat), dtype=np.float64)
        for t in range(n_steps):
            mu = means[state]
            if covars.ndim == 2 and covars.shape[0] == k and covars.shape[1] == feat:
                std = np.sqrt(np.clip(covars[state], 1e-12, None))
                y[t] = rng.normal(mu, std)
            else:
                cov = (
                    covars[state]
                    if covars.ndim == 3
                    else np.diag(np.clip(covars[state], 1e-12, None))
                )
                y[t] = rng.multivariate_normal(mu, cov + 1e-9 * np.eye(feat))
            state = int(rng.choice(k, p=p[state]))
        outs.append(y)
    return np.stack(outs, axis=0)

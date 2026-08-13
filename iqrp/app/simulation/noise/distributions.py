"""Innovation noise models for synthetic market paths."""

from __future__ import annotations

from typing import Literal

import numpy as np

NoiseName = Literal["gaussian", "student_t", "laplace", "cauchy", "uniform", "mixture"]


class NoiseSampler:
    """Draw i.i.d. innovations from a configured heavy-tailed or mixture law."""

    def __init__(
        self,
        distribution: NoiseName = "gaussian",
        *,
        df: float = 5.0,
        mixture_weights: tuple[float, ...] = (0.9, 0.1),
        mixture_scales: tuple[float, ...] = (1.0, 3.0),
        rng: np.random.Generator | None = None,
    ) -> None:
        self.distribution = distribution
        self.df = max(df, 2.01)
        self.mixture_weights = np.asarray(mixture_weights, dtype=np.float64)
        self.mixture_weights = self.mixture_weights / self.mixture_weights.sum()
        self.mixture_scales = np.asarray(mixture_scales, dtype=np.float64)
        self.rng = rng or np.random.default_rng()

    def sample(self, size: int | tuple[int, ...]) -> np.ndarray:
        return sample_noise(
            size,
            self.distribution,
            df=self.df,
            mixture_weights=self.mixture_weights,
            mixture_scales=self.mixture_scales,
            rng=self.rng,
        )


def sample_noise(
    size: int | tuple[int, ...],
    distribution: NoiseName = "gaussian",
    *,
    df: float = 5.0,
    mixture_weights: np.ndarray | tuple[float, ...] | None = None,
    mixture_scales: np.ndarray | tuple[float, ...] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample standardized innovations (approx unit variance where defined)."""
    rng = rng or np.random.default_rng()
    if distribution == "gaussian":
        return rng.standard_normal(size)
    if distribution == "student_t":
        # Student-t with variance scaled to ~1
        raw = rng.standard_t(df, size=size)
        scale = np.sqrt(df / (df - 2.0))
        return np.asarray(raw / scale, dtype=np.float64)
    if distribution == "laplace":
        # Laplace(0, b) with Var=2b^2 => b=1/sqrt(2) for unit variance
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=size)
    if distribution == "cauchy":
        # Cauchy has no variance; clip extremes for numerical safety
        raw = rng.standard_cauchy(size)
        return np.clip(raw, -20.0, 20.0) / 5.0
    if distribution == "uniform":
        # Uniform[-a,a] with Var=a^2/3 => a=sqrt(3)
        a = np.sqrt(3.0)
        return rng.uniform(-a, a, size=size)
    if distribution == "mixture":
        weights = np.asarray(
            mixture_weights if mixture_weights is not None else (0.9, 0.1),
            dtype=np.float64,
        )
        weights = weights / weights.sum()
        scales = np.asarray(
            mixture_scales if mixture_scales is not None else (1.0, 3.0),
            dtype=np.float64,
        )
        flat = int(np.prod(size)) if isinstance(size, tuple) else int(size)
        comps = rng.choice(len(weights), size=flat, p=weights)
        z = rng.standard_normal(flat) * scales[comps]
        return z.reshape(size)
    from iqrp.app.core.exceptions import ValidationError

    raise ValidationError(
        f"Unknown noise distribution '{distribution}'",
        code="SIM_NOISE_UNKNOWN",
    )

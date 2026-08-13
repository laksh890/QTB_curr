# Black–Litterman

Equilibrium prior, absolute/relative views, view confidence (\(\Omega\)), and posterior \(\mu\) / \(\Sigma\) for mean–variance construction.

Package: `iqrp.app.portfolio.optimization` / `iqrp.app.portfolio.expected_returns`  
Entry points: `optimize_black_litterman`, `black_litterman_posterior`

Related: [MeanVariance](MeanVariance.md) · [PortfolioConstruction](PortfolioConstruction.md) · [ForecastIntelligence](ForecastIntelligence.md)

---

## Model

1. **Equilibrium** \(\pi = \delta\, \Sigma\, w_m\) (or caller-supplied `equilibrium_returns` / `mu` as prior).
2. **Views** \(P \mu = Q + \varepsilon\), \(\varepsilon \sim \mathcal{N}(0, \Omega)\).
3. **Uncertainty scale** \(\tau\) on the prior covariance \(\tau\Sigma\).
4. **Posterior mean** (classic form):

\[
\mu_{\text{post}} = \pi + \tau\Sigma\, P^\top (P\,\tau\Sigma\, P^\top + \Omega)^{-1}(Q - P\pi)
\]

5. Optional **posterior covariance** for robust / uncertainty-aware steps when `use_posterior_cov=True`.

If `P` / `Q` are omitted, posterior equals the equilibrium prior (no views).

Default \(\Omega\) (when omitted): He–Litterman diagonal proportional to \(\mathrm{diag}(P\,(\tau\Sigma)\,P^\top)\).

---

## `optimize_black_litterman`

Builds posterior \(\mu\) via `black_litterman_posterior` (preferred) or a local classical implementation, then runs `optimize_mean_variance`.

```python
import numpy as np
from iqrp.app.portfolio.optimization import optimize_black_litterman

cov = np.diag([0.04, 0.09, 0.16])
# One relative view: asset 0 outperforms asset 1 by 2%
P = np.array([[1.0, -1.0, 0.0]])
Q = np.array([0.02])

out = optimize_black_litterman(
    cov=cov,
    market_weights=np.array([0.4, 0.35, 0.25]),
    risk_aversion=2.5,
    P=P, Q=Q, tau=0.05,
    long_only=True, max_weight=0.5,
    names=["a", "b", "c"],
)
out["success"], out["weights"], out.get("diagnostics", {}).get("posterior_mu")
```

Via engine (Hydra `expected_returns.bl_tau` / `bl_risk_aversion` apply as defaults):

```python
from iqrp.app.portfolio import PortfolioConstructionEngine

eng = PortfolioConstructionEngine()
eng.optimize(
    cov=cov, method="black_litterman",
    P=P, Q=Q, market_weights=[0.4, 0.35, 0.25], tau=0.05,
    names=["a", "b", "c"],
)
```

Estimator-only posterior:

```python
from iqrp.app.portfolio.expected_returns import black_litterman_posterior

post = black_litterman_posterior(
    cov, market_weights=[1/3, 1/3, 1/3], P=P, Q=Q, tau=0.05,
)
post["posterior_mu"], post.get("posterior_cov")
```

---

## Multi-forecast views

Stack several forecast views as rows of \(P\) with matching \(Q\) and optional diagonal/full \(\Omega\):

```python
# Absolute views from two forecast sources on assets 0 and 2
P = np.eye(3)[[0, 2]]          # shape (2, 3)
Q = np.array([0.08, 0.04])     # view returns
omega = np.diag([0.001, 0.002])  # lower = higher confidence

out = optimize_black_litterman(
    cov=cov, P=P, Q=Q, omega=omega, tau=0.05,
    equilibrium_returns=np.array([0.05, 0.05, 0.05]),
    long_only=True, max_weight=0.4,
)
```

Confidence from Forecast Intelligence maps naturally to larger \(\Omega\) (less certainty) — never to relaxing portfolio hard constraints. Engine `construct` still applies `require_risk_validation` after the BL solve.

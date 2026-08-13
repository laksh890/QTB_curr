# GARCH Family Models

## Recursions

Implemented in `iqrp/app/forecasting/volatility/base/recursion.py` with optional Numba acceleration.

- **ARCH(p)** — σ²_t = ω + Σ α_i ε²_{t-i}
- **GARCH(p,q)** — adds Σ β_j σ²_{t-j}
- **GJR-GARCH** — leverage via I(ε<0) ε² terms
- **EGARCH** — log-variance; positivity unconstrained
- **APARCH** — (|ε| − γ ε)^δ power dynamics
- **FIGARCH** — fractional integration via truncated ARCH(∞) weights
- **Component GARCH** — permanent q_t and transitory h_t − q_t

## Estimation

Maximum likelihood via `base/likelihood.py`:

- Backends: L-BFGS-B, SLSQP, Nelder-Mead, robust (NM → L-BFGS-B)
- Automatic initialization from sample variance
- Multiple random restarts
- Distributions: Gaussian, Student-t, Skewed-t, GED, Laplace, custom

## Persistence & half-life

Diagnostics compute persistence ≈ α + β (+ ½γ for GJR) and half-life = ln(0.5)/ln(persistence).

## Forecasting

Multi-step variance uses the closed-form GARCH(1,1) path (mean-reversion to unconditional variance). EGARCH / FIGARCH use persistence approximations on the appropriate scale.

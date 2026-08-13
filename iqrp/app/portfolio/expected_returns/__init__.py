"""Expected-return estimators for portfolio construction."""

from iqrp.app.portfolio.expected_returns.black_litterman import (
    black_litterman_posterior,
    equilibrium_returns,
)
from iqrp.app.portfolio.expected_returns.forecast import forecast_expected_returns
from iqrp.app.portfolio.expected_returns.historical import historical_expected_returns
from iqrp.app.portfolio.expected_returns.shrinkage import (
    james_stein_shrinkage,
    shrinkage_expected_returns,
)

__all__ = [
    "black_litterman_posterior",
    "equilibrium_returns",
    "forecast_expected_returns",
    "historical_expected_returns",
    "james_stein_shrinkage",
    "shrinkage_expected_returns",
]

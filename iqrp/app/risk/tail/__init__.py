"""Tail risk subpackage."""

from iqrp.app.risk.tail.cvar import historical_cvar, monte_carlo_cvar, parametric_cvar
from iqrp.app.risk.tail.drawdown import (
    downside_deviation,
    drawdown_series,
    drawdown_state,
    expected_drawdown,
    max_drawdown,
    ulcer_index,
)
from iqrp.app.risk.tail.expected_shortfall import (
    conditional_tail_expectation,
    expected_shortfall,
)
from iqrp.app.risk.tail.tail_dependence import empirical_tail_dependence
from iqrp.app.risk.tail.var import (
    filtered_historical_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
)

__all__ = [
    "conditional_tail_expectation",
    "downside_deviation",
    "drawdown_series",
    "drawdown_state",
    "empirical_tail_dependence",
    "expected_drawdown",
    "expected_shortfall",
    "filtered_historical_var",
    "historical_cvar",
    "historical_var",
    "max_drawdown",
    "monte_carlo_cvar",
    "monte_carlo_var",
    "parametric_cvar",
    "parametric_var",
    "ulcer_index",
]

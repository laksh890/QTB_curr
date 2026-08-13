"""Time-aware alpha backtesting with purge / embargo support.

All splits and backtests in this package are chronological. Signals at time
``t`` may only earn returns that realize strictly after ``t`` (forward returns
or an explicit lag). Purge and embargo gaps separate train and test folds to
block label leakage from overlapping horizons.
"""

from __future__ import annotations

from iqrp.app.alpha.backtesting.embargo import apply_embargo, embargo_splits
from iqrp.app.alpha.backtesting.nested_cv import nested_cv_splits
from iqrp.app.alpha.backtesting.portfolio_backtest import portfolio_backtest
from iqrp.app.alpha.backtesting.purged_cv import purged_kfold_splits
from iqrp.app.alpha.backtesting.signal_backtest import signal_backtest
from iqrp.app.alpha.backtesting.walk_forward import walk_forward_backtest, walk_forward_splits

__all__ = [
    "apply_embargo",
    "embargo_splits",
    "nested_cv_splits",
    "portfolio_backtest",
    "purged_kfold_splits",
    "signal_backtest",
    "walk_forward_backtest",
    "walk_forward_splits",
]

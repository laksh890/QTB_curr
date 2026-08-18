"""Frozen 2024 → 2025 independent holdout package."""

from iqrp.app.backtesting.frozen_2025_holdout.protocol import DISCLAIMER, Frozen2025Config
from iqrp.app.backtesting.frozen_2025_holdout.runner import run_frozen_2025

__all__ = ["DISCLAIMER", "Frozen2025Config", "run_frozen_2025"]

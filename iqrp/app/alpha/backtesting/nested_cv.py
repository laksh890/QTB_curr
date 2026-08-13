"""Nested cross-validation with purged/embargo outer and inner loops.

Look-ahead prevention
---------------------
Outer test folds are chronological blocks. Inner model-selection folds are
drawn only from the outer training region after purge/embargo relative to the
outer test fold, so hyperparameters never see outer OOS data.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from iqrp.app.alpha.backtesting.embargo import apply_embargo, embargo_splits
from iqrp.app.alpha.backtesting.purged_cv import purged_kfold_splits


def nested_cv_splits(
    n: int,
    *,
    n_outer: int = 5,
    n_inner: int = 3,
    purge: int = 5,
    embargo: int = 5,
) -> Iterator[dict[str, Any]]:
    """Yield nested CV fold descriptors.

    Each item::

        {
          "outer_train": idx,
          "outer_test": idx,
          "inner_folds": [(inner_train, inner_val), ...],
        }
    """
    n = int(n)
    for outer_tr, outer_te in embargo_splits(n, n_splits=n_outer, embargo=embargo, purge=purge):
        # Map outer train indices to a contiguous local view for inner CV
        outer_tr = np.asarray(outer_tr, dtype=int)
        outer_te = np.asarray(outer_te, dtype=int)
        # Further enforce embargo relative to outer test
        outer_tr = apply_embargo(outer_tr, outer_te, embargo=embargo, purge=purge)
        if outer_tr.size < max(n_inner * 2, 10):
            continue

        # Build inner splits on the ordered outer-train timeline
        # Use positional ranks within outer_tr
        m = int(outer_tr.size)
        inner_folds: list[tuple[np.ndarray, np.ndarray]] = []
        for loc_tr, loc_val in purged_kfold_splits(m, n_splits=n_inner, purge=purge):
            # loc_* are positions into outer_tr
            inner_train = outer_tr[loc_tr]
            inner_val = outer_tr[loc_val]
            # purge/embargo val against train within outer train
            inner_train = apply_embargo(inner_train, inner_val, embargo=embargo, purge=purge)
            if inner_train.size and inner_val.size:
                inner_folds.append((inner_train, inner_val))

        yield {
            "outer_train": outer_tr,
            "outer_test": outer_te,
            "inner_folds": inner_folds,
            "purge": int(purge),
            "embargo": int(embargo),
            "look_ahead_guard": "nested_purged_embargo",
        }

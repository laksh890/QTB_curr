"""Walk-forward evaluation engine.

Orchestrates window generation, purged/embargoed splits, fold evaluation, and
OOS metric aggregation. Training is always strictly causal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from iqrp.app.backtesting.walk_forward.evaluator import (
    FoldFn,
    WalkForwardEvaluator,
    WalkForwardReport,
)
from iqrp.app.backtesting.walk_forward.windows import (
    WalkForwardWindow,
    assert_no_future_training,
    generate_windows,
)


class WalkForwardEngine:
    """Production walk-forward runner.

    Example
    -------
    >>> eng = WalkForwardEngine()
    >>> report = eng.run(
    ...     n=100,
    ...     train_size=40,
    ...     test_size=10,
    ...     step=10,
    ...     evaluate_fold=lambda tr, te: {"n_train": len(tr), "n_test": len(te)},
    ... )
    """

    def __init__(self, evaluator: WalkForwardEvaluator | None = None) -> None:
        self.evaluator = evaluator or WalkForwardEvaluator()

    def windows(
        self,
        n: int,
        train_size: int,
        test_size: int,
        *,
        mode: str = "rolling",
        step: int | None = None,
        purge: int = 0,
        embargo: int = 0,
        anchor: int = 0,
        validation_size: int = 0,
        n_splits: int = 5,
    ) -> list[WalkForwardWindow]:
        wins = generate_windows(
            n,
            train_size=train_size,
            test_size=test_size,
            mode=mode,
            step=step,
            purge=purge,
            embargo=embargo,
            anchor=anchor,
            validation_size=validation_size,
            n_splits=n_splits,
        )
        assert_no_future_training(wins)
        return wins

    def run(
        self,
        *,
        n: int,
        train_size: int,
        test_size: int,
        evaluate_fold: FoldFn,
        step: int | None = None,
        mode: str = "rolling",
        purge: int = 0,
        embargo: int = 0,
        anchor: int = 0,
        validation_size: int = 0,
        n_splits: int = 5,
        as_dict: bool = True,
    ) -> dict[str, Any] | WalkForwardReport:
        """Generate windows, evaluate each fold, aggregate OOS metrics.

        ``evaluate_fold(train_idx, test_idx)`` must return a mapping of metrics.
        """
        wins = self.windows(
            n,
            train_size=train_size,
            test_size=test_size,
            mode=mode,
            step=step,
            purge=purge,
            embargo=embargo,
            anchor=anchor,
            validation_size=validation_size,
            n_splits=n_splits,
        )
        report = self.evaluator.evaluate(
            wins, evaluate_fold, mode=mode, purge=purge, embargo=embargo
        )
        return report.to_dict() if as_dict else report

    def run_on_windows(
        self,
        windows: Sequence[WalkForwardWindow],
        evaluate_fold: FoldFn,
        *,
        as_dict: bool = True,
    ) -> dict[str, Any] | WalkForwardReport:
        assert_no_future_training(list(windows))
        report = self.evaluator.evaluate(windows, evaluate_fold)
        return report.to_dict() if as_dict else report

    def run_arrays(
        self,
        *,
        X: Any,
        y: Any | None = None,
        train_size: int,
        test_size: int,
        step: int | None = None,
        mode: str = "rolling",
        purge: int = 0,
        embargo: int = 0,
        fit_predict: Callable[..., Mapping[str, Any]] | None = None,
        evaluate_fold: FoldFn | None = None,
    ) -> dict[str, Any]:
        """Convenience runner over array-like inputs of length ``n``.

        Prefer ``evaluate_fold`` for index-only callbacks. If ``fit_predict`` is
        provided it is called as ``fit_predict(X_train, y_train, X_test, y_test)``.
        """
        X_arr = np.asarray(X)
        n = int(X_arr.shape[0])
        y_arr = None if y is None else np.asarray(y)

        if evaluate_fold is None:
            if fit_predict is None:
                raise ValueError("Provide evaluate_fold or fit_predict")

            def evaluate_fold(tr: np.ndarray, te: np.ndarray) -> Mapping[str, Any]:
                X_tr, X_te = X_arr[tr], X_arr[te]
                if y_arr is None:
                    return dict(fit_predict(X_tr, X_te))
                return dict(fit_predict(X_tr, y_arr[tr], X_te, y_arr[te]))

        result = self.run(
            n=n,
            train_size=train_size,
            test_size=test_size,
            step=step,
            mode=mode,
            purge=purge,
            embargo=embargo,
            evaluate_fold=evaluate_fold,
            as_dict=True,
        )
        assert isinstance(result, dict)
        return result

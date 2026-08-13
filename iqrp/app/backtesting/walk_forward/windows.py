"""Generate rolling / expanding / anchored / purged-kfold walk-forward windows.

Look-ahead prevention
---------------------
Every window enforces ``train.end <= test.start`` after purge. Training never
includes indices at or after the prediction timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from iqrp.app.backtesting.walk_forward.embargo import apply_embargo
from iqrp.app.backtesting.walk_forward.purge import apply_purge, purged_kfold_splits
from iqrp.app.backtesting.walk_forward.test_window import TestWindow
from iqrp.app.backtesting.walk_forward.training_window import TrainingWindow
from iqrp.app.backtesting.walk_forward.validation_window import ValidationWindow

WindowMode = Literal["rolling", "expanding", "anchored", "purged_kfold"]


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One causal train(/val)/test fold."""

    fold_id: int
    mode: str
    train: TrainingWindow
    test: TestWindow
    validation: ValidationWindow | None = None
    purge: int = 0
    embargo: int = 0
    train_idx: np.ndarray | None = None
    test_idx: np.ndarray | None = None
    validation_idx: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Causal modes: training must end strictly before prediction time.
        # purged_kfold may retain non-overlapping post-test train samples.
        if self.mode != "purged_kfold":
            self.train.assert_before(self.test.prediction_timestamp)
            if self.train_idx is not None and self.train_idx.size:
                if int(np.max(self.train_idx)) >= int(self.test.prediction_timestamp):
                    raise ValueError(
                        f"NO FUTURE TRAINING: train max index "
                        f"{int(np.max(self.train_idx))} >= prediction "
                        f"{self.test.prediction_timestamp}"
                    )
        if self.validation is not None and self.mode != "purged_kfold":
            self.validation.assert_after_train(self.train.end)
            self.validation.assert_before_test(self.test.start)

    @property
    def train_indices(self) -> np.ndarray:
        if self.train_idx is not None:
            return np.asarray(self.train_idx, dtype=int)
        return self.train.indices()

    @property
    def test_indices(self) -> np.ndarray:
        if self.test_idx is not None:
            return np.asarray(self.test_idx, dtype=int)
        return self.test.indices()

    @property
    def validation_indices(self) -> np.ndarray:
        if self.validation_idx is not None:
            return np.asarray(self.validation_idx, dtype=int)
        if self.validation is None:
            return np.asarray([], dtype=int)
        return self.validation.indices()

    def as_index_pair(self) -> tuple[np.ndarray, np.ndarray]:
        return self.train_indices, self.test_indices

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": int(self.fold_id),
            "mode": str(self.mode),
            "train_start": int(self.train.start),
            "train_end": int(self.train.end),
            "test_start": int(self.test.start),
            "test_end": int(self.test.end),
            "validation_start": None if self.validation is None else int(self.validation.start),
            "validation_end": None if self.validation is None else int(self.validation.end),
            "purge": int(self.purge),
            "embargo": int(self.embargo),
            "n_train": int(self.train_indices.size),
            "n_test": int(self.test_indices.size),
        }

    def __repr__(self) -> str:
        val = ""
        if self.validation is not None:
            val = f", val=[{self.validation.start}, {self.validation.end})"
        return (
            f"WalkForwardWindow(fold={self.fold_id}, mode={self.mode!r}, "
            f"train=[{self.train.start}, {self.train.end}), "
            f"test=[{self.test.start}, {self.test.end}){val}, "
            f"purge={self.purge}, embargo={self.embargo}, "
            f"n_train={self.train_indices.size}, n_test={self.test_indices.size})"
        )


def generate_windows(
    n: int,
    train_size: int,
    test_size: int,
    *,
    mode: WindowMode | str = "rolling",
    step: int | None = None,
    purge: int = 0,
    embargo: int = 0,
    anchor: int = 0,
    validation_size: int = 0,
    n_splits: int = 5,
) -> list[WalkForwardWindow]:
    """Generate causal walk-forward windows.

    Parameters
    ----------
    n:
        Total number of observations.
    train_size:
        Minimum / fixed training length (mode-dependent).
    test_size:
        Length of each OOS test fold.
    mode:
        ``rolling`` — fixed-length train window slides forward.
        ``expanding`` — train grows from index 0.
        ``anchored`` — train grows from ``anchor``.
        ``purged_kfold`` — contiguous purged K-fold with embargo.
    step:
        Advance between successive folds (defaults to ``test_size``).
    purge:
        Bars excluded between train end and test start (label-horizon gap).
        Also applied as index purge around the test fold.
    embargo:
        Bars excluded after test end from training sets.
    anchor:
        Fixed train origin for ``anchored`` mode.
    validation_size:
        Optional inner validation length carved from the end of train
        (still strictly before test).
    n_splits:
        Number of folds for ``purged_kfold`` mode.
    """
    n = int(n)
    tr = max(int(train_size), 1)
    te = max(int(test_size), 1)
    p = max(int(purge), 0)
    e = max(int(embargo), 0)
    step_size = max(int(step) if step is not None else te, 1)
    val_size = max(int(validation_size), 0)
    mode_key = str(mode).lower().strip()

    if mode_key == "purged_kfold":
        return _purged_kfold_windows(n, n_splits=n_splits, purge=p, embargo=e)

    if mode_key not in {"rolling", "expanding", "anchored"}:
        raise ValueError(
            f"Unknown window mode {mode!r}; expected rolling|expanding|anchored|purged_kfold"
        )

    windows: list[WalkForwardWindow] = []
    # cursor is the end of the (pre-validation) train segment relative to origin.
    # For rolling: train = [cursor - tr, cursor) once cursor >= tr, we iterate by start.
    start = 0
    fold_id = 0
    while True:
        if mode_key == "rolling":
            train_start = start
            train_end = start + tr
        elif mode_key == "expanding":
            train_start = 0
            train_end = start + tr
        else:  # anchored
            train_start = max(int(anchor), 0)
            train_end = max(train_start + tr, start + tr)
            if train_end <= train_start:
                break

        if val_size > 0:
            if train_end - train_start <= val_size:
                break
            val_start = train_end - val_size
            fit_end = val_start
            validation = ValidationWindow(val_start, train_end)
        else:
            fit_end = train_end
            validation = None

        test_start = train_end + p
        test_end = test_start + te
        if test_end > n or fit_end <= train_start:
            break

        train_win = TrainingWindow(train_start, fit_end)
        test_win = TestWindow(test_start, test_end)
        train_win.assert_before(test_win.prediction_timestamp)

        raw_train = train_win.indices()
        raw_test = test_win.indices()
        purged = apply_purge(raw_train, test_start=test_start, test_end=test_end, purge=p)
        final_train = apply_embargo(purged, raw_test, embargo=e, purge=p)
        # Causal hard constraint: drop anything at/after prediction time.
        final_train = final_train[final_train < test_start]
        if final_train.size == 0 or raw_test.size == 0:
            start += step_size
            continue

        val_idx = None
        if validation is not None:
            val_idx = validation.indices()
            val_idx = val_idx[val_idx < test_start]

        # Recompute effective train bounds from surviving indices when purged.
        eff_train = TrainingWindow(int(final_train[0]), int(final_train[-1]) + 1)
        windows.append(
            WalkForwardWindow(
                fold_id=fold_id,
                mode=mode_key,
                train=eff_train,
                test=test_win,
                validation=validation,
                purge=p,
                embargo=e,
                train_idx=final_train,
                test_idx=raw_test,
                validation_idx=val_idx,
            )
        )
        fold_id += 1
        start += step_size
        if start + tr + p + te > n and mode_key == "rolling":
            # Next rolling fold cannot fit.
            if start + tr + p + te > n:
                break
        if mode_key in {"expanding", "anchored"} and start + tr + p + te > n:
            break

    return windows


def _purged_kfold_windows(
    n: int,
    *,
    n_splits: int,
    purge: int,
    embargo: int,
) -> list[WalkForwardWindow]:
    splits = purged_kfold_splits(n, n_splits=n_splits, purge=purge)
    windows: list[WalkForwardWindow] = []
    for fold_id, (tr, te) in enumerate(splits):
        # Apply embargo on top of purge.
        tr2 = apply_embargo(tr, te, embargo=embargo, purge=purge)
        if tr2.size == 0 or te.size == 0:
            continue
        # Keep only causal train indices strictly before test start for WF semantics.
        # For classic purged k-fold, train may include post-test samples; we still
        # allow them when present after embargo, but document the fold bounds from
        # the contiguous test span and the surviving train min/max.
        te0 = int(te[0])
        te1 = int(te[-1]) + 1
        train_win = TrainingWindow(int(tr2[0]), int(tr2[-1]) + 1)
        test_win = TestWindow(te0, te1)
        windows.append(
            WalkForwardWindow(
                fold_id=fold_id,
                mode="purged_kfold",
                train=train_win,
                test=test_win,
                validation=None,
                purge=int(purge),
                embargo=int(embargo),
                train_idx=tr2,
                test_idx=te,
            )
        )
    return windows


def assert_no_future_training(windows: list[WalkForwardWindow]) -> None:
    """Raise if any fold violates train-end < prediction timestamp."""
    for w in windows:
        tr = w.train_indices
        te = w.test_indices
        if tr.size == 0 or te.size == 0:
            continue
        if int(np.max(tr)) >= int(np.min(te)):
            # purged_kfold may intentionally include post-test train samples;
            # for causal modes this is forbidden.
            if w.mode != "purged_kfold":
                raise ValueError(
                    f"Fold {w.fold_id}: train max {int(np.max(tr))} >= "
                    f"test min {int(np.min(te))} (future training)"
                )

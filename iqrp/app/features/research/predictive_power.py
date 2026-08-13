"""Time-series predictive power evaluation (no trading signals)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.features.research._numeric import (
    binary_classification_metrics,
    information_coefficient,
    mutual_information,
    r_squared,
    rank_information_coefficient,
    ridge_fit_predict,
    safe_nanmean,
)
from iqrp.app.features.research.config import ResearchSettings
from iqrp.app.features.research.targets import TARGET_NAMES, build_targets
from iqrp.app.features.research.timeseries_cv import iter_splits


@dataclass
class TargetPredictiveMetrics:
    target: str
    information_coefficient: float
    rank_information_coefficient: float
    mutual_information: float
    predictive_r2: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    n_splits: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeaturePredictiveReport:
    feature: str
    by_target: dict[str, TargetPredictiveMetrics] = field(default_factory=dict)
    mean_abs_ic: float = float("nan")
    mean_r2: float = float("nan")
    mean_auc: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "by_target": {k: v.to_dict() for k, v in self.by_target.items()},
            "mean_abs_ic": self.mean_abs_ic,
            "mean_r2": self.mean_r2,
            "mean_auc": self.mean_auc,
        }


class PredictivePowerEngine:
    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings.default()

    def evaluate(
        self, frame: pl.DataFrame, columns: list[str]
    ) -> dict[str, FeaturePredictiveReport]:
        targets = build_targets(frame, self.settings)
        target_cols = targets.select(pl.exclude(self.settings.columns.timestamp))
        joined = pl.concat([frame, target_cols], how="horizontal")
        out: dict[str, FeaturePredictiveReport] = {}

        def one(feature: str) -> FeaturePredictiveReport:
            return self._evaluate_feature(joined, feature)

        with ThreadPoolExecutor(max_workers=max(1, self.settings.n_jobs)) as pool:
            reports = list(pool.map(one, columns))
        for rep in reports:
            out[rep.feature] = rep
        return out

    def _evaluate_feature(self, frame: pl.DataFrame, feature: str) -> FeaturePredictiveReport:
        x = frame[feature].cast(pl.Float64).to_numpy()
        by_target: dict[str, TargetPredictiveMetrics] = {}
        for target in TARGET_NAMES:
            y = frame[target].cast(pl.Float64).to_numpy()
            by_target[target] = self._walk_forward_metrics(x, y, target)
        abs_ics = [
            abs(m.information_coefficient)
            for m in by_target.values()
            if np.isfinite(m.information_coefficient)
        ]
        r2s = [m.predictive_r2 for m in by_target.values() if np.isfinite(m.predictive_r2)]
        aucs = [m.auc for m in by_target.values() if np.isfinite(m.auc)]
        return FeaturePredictiveReport(
            feature=feature,
            by_target=by_target,
            mean_abs_ic=float(np.mean(abs_ics)) if abs_ics else float("nan"),
            mean_r2=float(np.mean(r2s)) if r2s else float("nan"),
            mean_auc=float(np.mean(aucs)) if aucs else float("nan"),
        )

    def _walk_forward_metrics(
        self, x: np.ndarray, y: np.ndarray, target: str
    ) -> TargetPredictiveMetrics:
        cfg = self.settings.predictive
        ics: list[float] = []
        rics: list[float] = []
        mis: list[float] = []
        r2s: list[float] = []
        accs: list[float] = []
        precs: list[float] = []
        recs: list[float] = []
        f1s: list[float] = []
        aucs: list[float] = []
        n_splits = 0
        for split in iter_splits(len(x), cfg):
            x_tr = x[split.train_start : split.train_end]
            y_tr = y[split.train_start : split.train_end]
            x_te = x[split.test_start : split.test_end]
            y_te = y[split.test_start : split.test_end]
            if np.isfinite(x_tr).sum() < 10 or np.isfinite(y_tr).sum() < 10:
                continue
            n_splits += 1
            # In-sample IC on test block between feature and target (proper OOS association)
            ics.append(information_coefficient(x_te, y_te))
            rics.append(rank_information_coefficient(x_te, y_te))
            mis.append(mutual_information(x_te, y_te, bins=cfg.mi_bins))
            pred = ridge_fit_predict(x_tr, y_tr, x_te, alpha=self.settings.importance.ridge_alpha)
            r2s.append(r_squared(y_te, pred))
            cls = binary_classification_metrics(y_te, pred, threshold=cfg.classification_threshold)
            # Direction/regime are inherently classification-friendly
            if target in {"future_direction", "future_regime"}:
                cls = binary_classification_metrics(
                    y_te, x_te, threshold=cfg.classification_threshold
                )
            accs.append(cls["accuracy"])
            precs.append(cls["precision"])
            recs.append(cls["recall"])
            f1s.append(cls["f1"])
            aucs.append(cls["auc"])

        return TargetPredictiveMetrics(
            target=target,
            information_coefficient=safe_nanmean(np.asarray(ics, dtype=np.float64)),
            rank_information_coefficient=safe_nanmean(np.asarray(rics, dtype=np.float64)),
            mutual_information=safe_nanmean(np.asarray(mis, dtype=np.float64)),
            predictive_r2=safe_nanmean(np.asarray(r2s, dtype=np.float64)),
            accuracy=safe_nanmean(np.asarray(accs, dtype=np.float64)),
            precision=safe_nanmean(np.asarray(precs, dtype=np.float64)),
            recall=safe_nanmean(np.asarray(recs, dtype=np.float64)),
            f1=safe_nanmean(np.asarray(f1s, dtype=np.float64)),
            auc=safe_nanmean(np.asarray(aucs, dtype=np.float64)),
            n_splits=n_splits,
        )

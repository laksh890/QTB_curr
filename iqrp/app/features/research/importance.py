"""Feature importance: permutation, SHAP-lite, LOO, drop-one, RFE, SFS."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from iqrp.app.features.research._numeric import r_squared, ridge_fit_predict
from iqrp.app.features.research.config import ResearchSettings
from iqrp.app.features.research.targets import build_targets
from iqrp.app.features.research.timeseries_cv import iter_splits


@dataclass
class ImportanceReport:
    permutation: dict[str, float]
    shap_values: dict[str, float]
    leave_one_out: dict[str, float]
    drop_one: dict[str, float]
    rfe_ranking: list[str]
    sfs_selected: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImportanceAnalyzer:
    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings.default()
        self._rng = np.random.default_rng(self.settings.random_seed)

    def analyze(self, frame: pl.DataFrame, columns: list[str]) -> ImportanceReport:
        if not columns:
            return ImportanceReport({}, {}, {}, {}, [], [])
        targets = build_targets(frame, self.settings)
        y = targets["future_return"].cast(pl.Float64).to_numpy()
        x_mat = np.column_stack([frame[c].cast(pl.Float64).to_numpy() for c in columns])
        notes: list[str] = []

        baseline = self._oos_r2(x_mat, y)
        perm = self._permutation(x_mat, y, columns, baseline)
        shap_vals, shap_note = self._shap(x_mat, y, columns)
        if shap_note:
            notes.append(shap_note)
        loo = self._leave_one_out(x_mat, y, columns, baseline)
        drop = {k: baseline - v for k, v in loo.items()}  # drop-one performance delta
        rfe = self._rfe(x_mat, y, columns)
        sfs = self._sfs(x_mat, y, columns)
        return ImportanceReport(
            permutation=perm,
            shap_values=shap_vals,
            leave_one_out=loo,
            drop_one=drop,
            rfe_ranking=rfe,
            sfs_selected=sfs,
            notes=notes,
        )

    def _oos_r2(self, x: np.ndarray, y: np.ndarray, cols: list[int] | None = None) -> float:
        cfg = self.settings.predictive
        alpha = self.settings.importance.ridge_alpha
        scores: list[float] = []
        use = x if cols is None else x[:, cols]
        if use.ndim == 1:
            use = use.reshape(-1, 1)
        if use.shape[1] == 0:
            return float("nan")
        for split in iter_splits(len(y), cfg):
            pred = ridge_fit_predict(
                use[split.train_start : split.train_end],
                y[split.train_start : split.train_end],
                use[split.test_start : split.test_end],
                alpha=alpha,
            )
            scores.append(r_squared(y[split.test_start : split.test_end], pred))
        finite = [s for s in scores if np.isfinite(s)]
        return float(np.mean(finite)) if finite else float("nan")

    def _permutation(
        self, x: np.ndarray, y: np.ndarray, columns: list[str], baseline: float
    ) -> dict[str, float]:
        n_perm = self.settings.importance.n_permutations
        out: dict[str, float] = {}
        for j, name in enumerate(columns):
            drops: list[float] = []
            for _ in range(n_perm):
                xp = x.copy()
                self._rng.shuffle(xp[:, j])
                # Preserve temporal structure of other cols; only shuffle feature j
                # within full sample; importance still measured via OOS R2 drop.
                score = self._oos_r2(xp, y)
                if np.isfinite(baseline) and np.isfinite(score):
                    drops.append(baseline - score)
            out[name] = float(np.mean(drops)) if drops else float("nan")
        return out

    def _shap(
        self, x: np.ndarray, y: np.ndarray, columns: list[str]
    ) -> tuple[dict[str, float], str | None]:
        if not self.settings.importance.shap_enabled:
            return {c: float("nan") for c in columns}, "SHAP disabled by config"
        # Prefer shap + tree model when available (optional deps).
        try:
            import importlib

            shap: Any = importlib.import_module("shap")
            gbr_mod: Any = importlib.import_module("sklearn.ensemble")
            model = gbr_mod.GradientBoostingRegressor(random_state=self.settings.random_seed)
            m = np.isfinite(y)
            for j in range(x.shape[1]):
                m &= np.isfinite(x[:, j])
            if m.sum() < 40:
                raise RuntimeError("insufficient rows for shap")
            model.fit(x[m], y[m])
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(x[m][: min(200, int(m.sum()))])
            mean_abs = np.mean(np.abs(values), axis=0)
            return (
                {c: float(mean_abs[i]) for i, c in enumerate(columns)},
                "SHAP via shap.TreeExplainer + GradientBoostingRegressor",
            )
        except Exception as exc:  # optional dependency path
            logger.debug("shap_unavailable fallback=linear_shap_lite err={}", exc)
        # Linear SHAP approximation: |beta_j * (x_j - E[x_j])| mean
        alpha = self.settings.importance.ridge_alpha
        m = np.isfinite(y)
        for j in range(x.shape[1]):
            m &= np.isfinite(x[:, j])
        xt, yt = x[m], y[m]
        if len(yt) < x.shape[1] + 5:
            return {c: float("nan") for c in columns}, "SHAP-lite insufficient data"
        mu = xt.mean(axis=0)
        xc = xt - mu
        gram = xc.T @ xc + alpha * np.eye(xc.shape[1])
        try:
            beta = np.linalg.solve(gram, xc.T @ (yt - yt.mean()))
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ (xc.T @ (yt - yt.mean()))
        contrib = np.mean(np.abs(xc * beta), axis=0)
        return (
            {c: float(contrib[i]) for i, c in enumerate(columns)},
            "Linear SHAP-lite (|beta_j (x_j - E x_j)|)",
        )

    def _leave_one_out(
        self, x: np.ndarray, y: np.ndarray, columns: list[str], baseline: float
    ) -> dict[str, float]:
        _ = baseline
        out: dict[str, float] = {}
        idx_all = list(range(len(columns)))
        for j, name in enumerate(columns):
            cols = [i for i in idx_all if i != j]
            out[name] = self._oos_r2(x, y, cols)
        return out

    def _rfe(self, x: np.ndarray, y: np.ndarray, columns: list[str]) -> list[str]:
        remaining = list(range(len(columns)))
        ranking: list[str] = []
        target_n = min(self.settings.importance.rfe_n_features_to_select, len(columns))
        while len(remaining) > target_n:
            # Drop feature with smallest |coef|
            alpha = self.settings.importance.ridge_alpha
            m = np.isfinite(y)
            for j in remaining:
                m &= np.isfinite(x[:, j])
            xt = x[m][:, remaining]
            yt = y[m]
            if len(yt) < len(remaining) + 2:
                break
            xc = xt - xt.mean(axis=0)
            gram = xc.T @ xc + alpha * np.eye(len(remaining))
            try:
                beta = np.linalg.solve(gram, xc.T @ (yt - yt.mean()))
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(gram) @ (xc.T @ (yt - yt.mean()))
            drop_local = int(np.argmin(np.abs(beta)))
            drop = remaining.pop(drop_local)
            ranking.append(columns[drop])
        ranking.extend([columns[i] for i in remaining])  # best last
        ranking.reverse()  # most important first
        return ranking

    def _sfs(self, x: np.ndarray, y: np.ndarray, columns: list[str]) -> list[str]:
        selected: list[int] = []
        remaining = list(range(len(columns)))
        target_n = min(self.settings.importance.sfs_n_features_to_select, len(columns))
        while len(selected) < target_n and remaining:
            best_i = None
            best_score = -np.inf
            for i in remaining:
                trial = [*selected, i]
                score = self._oos_r2(x, y, trial)
                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_i = i
            if best_i is None:
                break
            selected.append(best_i)
            remaining.remove(best_i)
        return [columns[i] for i in selected]

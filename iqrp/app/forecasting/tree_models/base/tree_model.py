"""Base class for institutional tree-based forecasting models."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport
from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.forecast_model import ForecastModel
from iqrp.app.forecasting.base.metadata import TrainingMetadata
from iqrp.app.forecasting.base.prediction import PredictionInterval
from iqrp.app.forecasting.explainability.importance import ExplanationResult
from iqrp.app.forecasting.postprocessing.intervals import residual_intervals
from iqrp.app.forecasting.tree_models.base.backends import (
    BackendName,
    create_estimator,
    estimator_feature_importances,
    estimator_predict,
    estimator_predict_proba,
)
from iqrp.app.forecasting.tree_models.calibration.calibrators import apply_calibration, fit_calibrator
from iqrp.app.forecasting.tree_models.config import TreeSettings
from iqrp.app.forecasting.tree_models.evaluation.metrics import evaluate_tree_predictions
from iqrp.app.forecasting.tree_models.explainability.importance import (
    compute_feature_importance,
    shap_values as compute_shap,
)
from iqrp.app.forecasting.tree_models.optimization.cv import make_time_splits
from iqrp.app.forecasting.tree_models.optimization.hpo import optimize_hyperparameters
from iqrp.app.forecasting.tree_models.preprocessing.pipeline import (
    TreePreprocessor,
    select_features,
)


class TreeForecastModel(ForecastModel):
    """Shared API for XGBoost / LightGBM / CatBoost / sklearn tree ensembles."""

    backend: BackendName = "hist_gradient_boosting"

    def __init__(self, settings: TreeSettings | Any | None = None, **params: Any) -> None:
        if settings is None:
            settings = TreeSettings.default()
        elif isinstance(settings, dict):
            settings = TreeSettings.from_mapping(settings)
        super().__init__(settings=settings)
        self._tree_settings: TreeSettings = settings  # type: ignore[assignment]
        self.backend = self._backend_name()
        self._params_kw: dict[str, Any] = dict(params)

        self._estimator: Any = None
        self._quantile_estimators: dict[float, Any] = {}
        self._regime_estimators: dict[Any, Any] = {}
        self._preprocessor: TreePreprocessor | None = None
        self._selected_features: list[str] = []
        self._calibrator: Any = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self._train_pred: np.ndarray | None = None
        self._best_params: dict[str, Any] = {}
        self._cv_scores: list[float] = []
        self._update_count: int = 0

    @property
    def best_params(self) -> dict[str, Any]:
        return dict(self._best_params)

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> TreeForecastModel:
        tgt = self._resolve_target(frame, target_column)
        cols = self._resolve_feature_columns(frame, feature_columns)
        cols = [c for c in cols if c != tgt]
        if not cols:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No feature columns for tree model", code="TREE_NO_FEATURES")
        y = frame[tgt].to_numpy().astype(np.float64)
        regimes = self._maybe_regime(frame, regime_column)
        # regime-as-feature
        work = frame
        if (
            regimes is not None
            and self._tree_settings.regime.enabled
            and self._tree_settings.regime.mode == "feature"
            and self._regime_column
            and self._regime_column not in cols
        ):
            cols = cols + [self._regime_column]
        X_raw = work.select(cols).to_numpy().astype(np.float64)
        # feature selection
        if self._tree_settings.feature_selection.enabled:
            cols = select_features(
                X_raw,
                y,
                cols,
                method=self._tree_settings.feature_selection.method,
                max_features=self._tree_settings.feature_selection.max_features,
                correlation_threshold=self._tree_settings.feature_selection.correlation_threshold,
            )
            X_raw = work.select(cols).to_numpy().astype(np.float64)
        self._preprocessor = TreePreprocessor().fit(X_raw)
        X = self._preprocessor.transform(X_raw)
        task = self._tree_settings.task.type
        hp = self._hyperparams()
        # HPO
        if self._tree_settings.optimization.method != "none":
            best, scores = optimize_hyperparameters(
                self.backend,
                X,
                y,
                task=task,
                base_params=hp,
                method=self._tree_settings.optimization.method,
                n_trials=self._tree_settings.optimization.n_trials,
                validation=self._tree_settings.validation,
                pruning=self._tree_settings.optimization.pruning,
                parallel=self._tree_settings.optimization.parallel,
            )
            hp = {**hp, **best}
            self._cv_scores = scores
        self._best_params = dict(hp)
        self._selected_features = list(cols)
        # fit primary / quantile / regime models
        if task == "quantile":
            self._quantile_estimators = {}
            for alpha in self._tree_settings.task.quantile_alphas:
                est = create_estimator(self.backend, task="quantile", params=hp, quantile_alpha=alpha)
                self._fit_estimator(est, X, y)
                self._quantile_estimators[float(alpha)] = est
            self._estimator = self._quantile_estimators.get(0.5) or next(
                iter(self._quantile_estimators.values())
            )
        elif (
            regimes is not None
            and self._tree_settings.regime.enabled
            and self._tree_settings.regime.mode in {"separate", "routing", "weighted"}
        ):
            self._regime_estimators = {}
            for reg in np.unique(regimes):
                mask = regimes == reg
                if int(mask.sum()) < 20:
                    continue
                est = create_estimator(self.backend, task=task, params=hp)
                self._fit_estimator(est, X[mask], y[mask])
                self._regime_estimators[reg] = est
            self._estimator = create_estimator(self.backend, task=task, params=hp)
            self._fit_estimator(self._estimator, X, y)
        else:
            self._estimator = create_estimator(self.backend, task=task, params=hp)
            self._fit_estimator(self._estimator, X, y)
        # calibration
        train_pred = estimator_predict(self._estimator, X)
        if self._tree_settings.calibration.enabled and task in {"binary", "probability", "multiclass"}:
            proba = estimator_predict_proba(self._estimator, X)
            self._calibrator = fit_calibrator(
                y, proba, method=self._tree_settings.calibration.method
            )
        self._X, self._y = X, y
        self._train_pred = train_pred
        self._residuals = y - train_pred
        self._target_column = tgt
        self._feature_columns = list(cols)
        self._training_meta = TrainingMetadata(
            n_samples=int(y.size),
            n_features=int(X.shape[1]),
            feature_columns=tuple(cols),
            target_column=tgt,
            regime_column=self._regime_column,
            horizon=self._tree_settings.forecast.default_horizon,
            extra={"backend": self.backend, "best_params": self._best_params, "cv_scores": self._cv_scores},
        )
        self._fitted = True
        self._update_count = 0
        return self

    def partial_fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> TreeForecastModel:
        mode = self._tree_settings.online.mode
        if not self._fitted or mode == "refit":
            return self.fit(
                frame, feature_columns, target_column=target_column, regime_column=regime_column
            )
        self._update_count += 1
        if self._update_count % max(int(self._tree_settings.online.refresh_every), 1) == 0:
            return self.fit(
                frame, feature_columns or self._feature_columns,
                target_column=target_column or self._target_column,
                regime_column=regime_column or self._regime_column,
            )
        # warm start / incremental: refit on concatenated window
        if self._X is None or self._y is None:
            return self.fit(
                frame, feature_columns, target_column=target_column, regime_column=regime_column
            )
        tgt = target_column or self._target_column or self._tree_settings.columns.target
        cols = list(self._feature_columns or feature_columns or [])
        X_new = self._transform(frame, cols)
        y_new = frame[tgt].to_numpy().astype(np.float64)

        w = int(self._tree_settings.online.window)
        X_all = np.vstack([self._X, X_new])[-w:]
        y_all = np.concatenate([self._y, y_new])[-w:]
        if mode == "warm_start" and hasattr(self._estimator, "set_params"):
            try:
                self._estimator.set_params(warm_start=True)
            except Exception:  # noqa: BLE001
                pass
        self._fit_estimator(self._estimator, X_all, y_all)
        self._X, self._y = X_all, y_all
        self._train_pred = estimator_predict(self._estimator, X_all)
        self._residuals = y_all - self._train_pred
        return self

    def predict(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        cols = list(feature_columns or self._feature_columns)
        # always prefer fitted schema when caller passes a superset / different set
        if self._feature_columns and (
            not cols
            or len(cols) != len(self._feature_columns)
            or any(c not in frame.columns for c in cols)
        ):
            cols = list(self._feature_columns)
        X = self._transform(frame, cols)
        if self._regime_estimators and self._regime_column and self._regime_column in frame.columns:
            regimes = frame[self._regime_column].to_numpy()
            out = np.empty(X.shape[0])
            mode = self._tree_settings.regime.mode
            if mode == "weighted" and self._regime_estimators:
                preds = {k: estimator_predict(est, X) for k, est in self._regime_estimators.items()}
                # frequency weights from training regimes if available
                weights = {k: 1.0 for k in preds}
                total = sum(weights.values()) or 1.0
                out = sum(weights[k] / total * preds[k] for k in preds)
                return np.asarray(out, dtype=np.float64)
            for i in range(X.shape[0]):
                reg = regimes[i]
                est = self._regime_estimators.get(reg, self._estimator)
                out[i] = float(estimator_predict(est, X[i : i + 1])[0])
            return out
        return estimator_predict(self._estimator, X)

    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        if not self.meta.supports_proba:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Model '{self.meta.name}' does not support predict_proba",
                code="TREE_NO_PROBA",
            )
        cols = feature_columns or self._feature_columns
        X = self._transform(frame, cols)
        proba = estimator_predict_proba(self._estimator, X)
        if self._calibrator is not None:
            proba = apply_calibration(self._calibrator, proba)
        return proba

    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        self._require_fitted()
        h = max(int(horizon if horizon is not None else self._tree_settings.forecast.default_horizon), 1)
        pred = self.predict(frame, feature_columns)
        # multi-horizon: repeat last prediction with residual growth
        last = float(pred[-1]) if pred.size else 0.0
        path = np.full(h, last)
        if self._tree_settings.forecast.multi_horizon and self._residuals is not None and self._residuals.size:
            # direct-style: use residual std drift
            sig = float(np.std(self._residuals))
            path = path + sig * np.linspace(0, 0.1 * h, h)
        intervals = residual_intervals(
            path,
            residual_std=np.maximum(
                float(np.std(self._residuals)) if self._residuals is not None else 1e-3,
                1e-6,
            ),
            level=self._tree_settings.forecast.interval_level,
        )
        quantiles = None
        meta: dict[str, Any] = {"backend": self.backend, "best_params": self._best_params}
        if self._quantile_estimators:
            qdict = {
                str(a): float(estimator_predict(est, self._transform(frame, self._feature_columns))[-1])
                for a, est in self._quantile_estimators.items()
            }
            meta["quantiles"] = qdict
        return Forecast.from_values(
            path,
            horizon=h,
            model_name=self.meta.name,
            model_version=self.meta.version,
            features_used=tuple(self._feature_columns),
            regime_used=frame[self._regime_column].to_numpy()[-1]
            if self._regime_column and self._regime_column in frame.columns
            else None,
            strategy="direct",
            intervals=intervals,
            metadata=meta,
        )

    def forecast_interval(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        level: float | None = None,
        feature_columns: list[str] | None = None,
    ) -> list[PredictionInterval]:
        fc = self.forecast(frame, horizon=horizon, feature_columns=feature_columns)
        if fc.intervals is not None:
            return fc.intervals
        lvl = float(level if level is not None else self._tree_settings.forecast.interval_level)
        return residual_intervals(fc.path(), level=lvl)

    def evaluate(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        probabilities: np.ndarray | None = None,
    ) -> EvaluationReport:
        self._require_fitted()
        tgt = target_column or self._target_column or self._tree_settings.columns.target
        y_true = frame[tgt].to_numpy().astype(np.float64)
        y_pred = self.predict(frame, feature_columns)
        proba = probabilities
        if proba is None and self.meta.supports_proba:
            try:
                proba = self.predict_proba(frame, feature_columns)
            except Exception:  # noqa: BLE001
                proba = None
        metrics = evaluate_tree_predictions(
            y_true, y_pred, proba=proba, task=self._tree_settings.task.type
        )
        return EvaluationReport(
            metrics=metrics,
            method=f"tree_{self.backend}",
            n_samples=int(y_true.size),
        )

    def explain(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        method: str = "permutation",
    ) -> ExplanationResult:
        self._require_fitted()
        cols = feature_columns or self._feature_columns
        if method == "shap":
            sv = self.shap_values(frame, cols)
            means = np.mean(np.abs(sv), axis=0)
            return ExplanationResult(
                method="shap",
                importances={c: float(means[i]) for i, c in enumerate(cols)},
                attributions=sv,
            )
        if method == "builtin":
            imp = self.feature_importance(kind="gain")
            return ExplanationResult(method="builtin", importances=imp)
        from iqrp.app.forecasting.explainability.importance import permutation_importance

        return permutation_importance(
            self, frame, cols, target_column=self._target_column, n_repeats=3
        )

    def feature_importance(self, *, kind: str = "gain") -> dict[str, float]:
        self._require_fitted()
        return compute_feature_importance(
            self._estimator,
            self._feature_columns,
            kind=kind,
            X=self._X,
            y=self._y,
            model=self,
        )

    def shap_values(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        cols = feature_columns or self._feature_columns
        X = self._transform(frame, cols)
        return compute_shap(self._estimator, X, feature_names=cols)

    def cross_validate(self, frame: pl.DataFrame | None = None) -> dict[str, Any]:
        self._require_fitted()
        assert self._X is not None and self._y is not None
        splits = make_time_splits(self._X.shape[0], self._tree_settings.validation)
        scores = []
        for tr, te in splits:
            est = create_estimator(
                self.backend, task=self._tree_settings.task.type, params=self._best_params or self._hyperparams()
            )
            self._fit_estimator(est, self._X[tr], self._y[tr])
            pred = estimator_predict(est, self._X[te])
            scores.append(float(np.sqrt(np.mean((self._y[te] - pred) ** 2))))
        return {"rmse_folds": scores, "rmse_mean": float(np.mean(scores)) if scores else float("nan")}

    def diagnostics(self) -> Any:
        self._require_fitted()
        from iqrp.app.forecasting.tree_models.diagnostics.report import run_tree_diagnostics

        assert self._X is not None and self._y is not None
        return run_tree_diagnostics(
            self._estimator,
            self._X,
            self._y,
            backend=self.backend,
            task=self._tree_settings.task.type,
            params=self._best_params or self._hyperparams(),
            feature_names=self._feature_columns,
        )

    def _fit_estimator(self, est: Any, X: np.ndarray, y: np.ndarray) -> None:
        # early stopping where supported
        kwargs: dict[str, Any] = {}
        rounds = int(self._tree_settings.hyperparameters.early_stopping_rounds)
        if (
            self._tree_settings.optimization.early_stopping
            and rounds > 0
            and X.shape[0] > 40
            and hasattr(est, "fit")
        ):
            n = X.shape[0]
            split = int(n * 0.8)
            try:
                # xgboost / lgbm style
                est.fit(
                    X[:split],
                    y[:split],
                    eval_set=[(X[split:], y[split:])],
                    verbose=False,
                )
                return
            except TypeError:
                pass
            except Exception:  # noqa: BLE001
                pass
        est.fit(X, y, **kwargs)

    def _transform(self, frame: pl.DataFrame, cols: list[str]) -> np.ndarray:
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(f"Missing columns: {missing}", code="TREE_MISSING_COLS")
        X = frame.select(cols).to_numpy().astype(np.float64)
        if self._preprocessor is not None:
            return self._preprocessor.transform(X)
        return X

    def _hyperparams(self) -> dict[str, Any]:
        hp = self._tree_settings.hyperparameters
        base = {
            "n_estimators": hp.n_estimators,
            "max_depth": hp.max_depth,
            "learning_rate": hp.learning_rate,
            "subsample": hp.subsample,
            "colsample_bytree": hp.colsample_bytree,
            "min_child_weight": hp.min_child_weight,
            "reg_lambda": hp.reg_lambda,
            "reg_alpha": hp.reg_alpha,
            "random_state": hp.random_state,
            "n_jobs": hp.n_jobs,
            "device": hp.device,
        }
        base.update(self._params_kw)
        return base

    def _resolve_target(self, frame: pl.DataFrame, target_column: str | None) -> str:
        if target_column:
            return target_column
        if self._target_column:
            return self._target_column
        cfg = self._tree_settings.columns.target
        if cfg in frame.columns:
            return cfg
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError("Target column required", code="TREE_NO_TARGET")

    def _maybe_regime(self, frame: pl.DataFrame, regime_column: str | None) -> np.ndarray | None:
        col = regime_column or (
            self._tree_settings.regime.column if self._tree_settings.regime.enabled else None
        )
        if col and col in frame.columns:
            self._regime_column = col
            return frame[col].to_numpy()
        return None

    def _algorithm_state(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "best_params": dict(self._best_params),
            "selected_features": list(self._selected_features),
            "feature_columns": list(self._feature_columns),
            "X": None if self._X is None else self._X.tolist(),
            "y": None if self._y is None else self._y.tolist(),
            "residuals": None if self._residuals is None else self._residuals.tolist(),
            "train_pred": None if self._train_pred is None else self._train_pred.tolist(),
            "cv_scores": list(self._cv_scores),
            "update_count": self._update_count,
            "preprocessor": None if self._preprocessor is None else self._preprocessor.to_dict(),
            "params_kw": dict(self._params_kw),
            # store predictions for cold restore without pickling estimators
            "estimator_importances": None
            if self._estimator is None
            else estimator_feature_importances(
                self._estimator, len(self._feature_columns) or 1
            ).tolist(),
        }

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        self._best_params = dict(state.get("best_params") or {})
        self._selected_features = list(state.get("selected_features") or [])
        self._feature_columns = list(state.get("feature_columns") or self._selected_features)
        self._X = None if state.get("X") is None else np.asarray(state["X"], dtype=np.float64)
        self._y = None if state.get("y") is None else np.asarray(state["y"], dtype=np.float64)
        self._residuals = (
            None if state.get("residuals") is None else np.asarray(state["residuals"], dtype=np.float64)
        )
        self._train_pred = (
            None if state.get("train_pred") is None else np.asarray(state["train_pred"], dtype=np.float64)
        )
        self._cv_scores = list(state.get("cv_scores") or [])
        self._update_count = int(state.get("update_count", 0))
        self._params_kw = dict(state.get("params_kw") or {})
        prep = state.get("preprocessor")
        if prep:
            self._preprocessor = TreePreprocessor.from_dict(prep)
        # rebuild estimator from stored data
        if self._X is not None and self._y is not None:
            task = self._tree_settings.task.type
            hp = self._best_params or self._hyperparams()
            self._estimator = create_estimator(self.backend, task=task, params=hp)
            self._fit_estimator(self._estimator, self._X, self._y)

    @abstractmethod
    def _backend_name(self) -> BackendName:
        ...

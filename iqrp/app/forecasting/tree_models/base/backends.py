"""Estimator factory with optional library backends and numpy fallbacks."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

BackendName = Literal[
    "xgboost",
    "lightgbm",
    "catboost",
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
]


def _task_is_class(task: str) -> bool:
    return task in {"binary", "multiclass", "probability"}


def create_estimator(
    backend: BackendName,
    *,
    task: str = "regression",
    params: dict[str, Any] | None = None,
    quantile_alpha: float | None = None,
) -> Any:
    p = dict(params or {})
    n_estimators = int(p.get("n_estimators", 100))
    max_depth = int(p.get("max_depth", 4))
    learning_rate = float(p.get("learning_rate", 0.1))
    subsample = float(p.get("subsample", 0.9))
    colsample = float(p.get("colsample_bytree", 0.9))
    reg_lambda = float(p.get("reg_lambda", 1.0))
    reg_alpha = float(p.get("reg_alpha", 0.0))
    random_state = int(p.get("random_state", 42))
    n_jobs = int(p.get("n_jobs", -1))
    device = str(p.get("device", "cpu"))
    is_cls = _task_is_class(task)
    is_quantile = task == "quantile"

    if backend == "xgboost":
        return _make_xgboost(
            is_cls,
            is_quantile,
            n_estimators,
            max_depth,
            learning_rate,
            subsample,
            colsample,
            reg_lambda,
            reg_alpha,
            random_state,
            n_jobs,
            device,
            quantile_alpha,
        )
    if backend == "lightgbm":
        return _make_lightgbm(
            is_cls,
            is_quantile,
            n_estimators,
            max_depth,
            learning_rate,
            subsample,
            colsample,
            reg_lambda,
            reg_alpha,
            random_state,
            n_jobs,
            device,
            quantile_alpha,
        )
    if backend == "catboost":
        return _make_catboost(
            is_cls,
            is_quantile,
            n_estimators,
            max_depth,
            learning_rate,
            subsample,
            reg_lambda,
            random_state,
            n_jobs,
            device,
            quantile_alpha,
        )
    if backend == "hist_gradient_boosting":
        return _make_hist_gb(
            is_cls,
            is_quantile,
            n_estimators,
            max_depth,
            learning_rate,
            random_state,
            quantile_alpha,
        )
    if backend == "random_forest":
        return _make_rf(is_cls, n_estimators, max_depth, random_state, n_jobs, extra=False)
    if backend == "extra_trees":
        return _make_rf(is_cls, n_estimators, max_depth, random_state, n_jobs, extra=True)
    from iqrp.app.core.exceptions import ValidationError

    raise ValidationError(f"Unknown backend {backend}", code="TREE_BAD_BACKEND")


def _make_xgboost(
    is_cls: bool,
    is_quantile: bool,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample: float,
    reg_lambda: float,
    reg_alpha: float,
    random_state: int,
    n_jobs: int,
    device: str,
    quantile_alpha: float | None,
) -> Any:
    try:
        from xgboost import XGBClassifier, XGBRegressor

        common = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }
        if device in {"gpu", "cuda"}:
            common["device"] = "cuda"
            common["tree_method"] = "hist"
        if is_cls:
            return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
        if is_quantile and quantile_alpha is not None:
            return XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=quantile_alpha, **common
            )
        return XGBRegressor(objective="reg:squarederror", **common)
    except Exception:
        from iqrp.app.forecasting.tree_models.base.native import NativeGBM

        return NativeGBM(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            task="classification" if is_cls else "regression",
            random_state=random_state,
            quantile_alpha=quantile_alpha if is_quantile else None,
        )


def _make_lightgbm(
    is_cls: bool,
    is_quantile: bool,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample: float,
    reg_lambda: float,
    reg_alpha: float,
    random_state: int,
    n_jobs: int,
    device: str,
    quantile_alpha: float | None,
) -> Any:
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        common = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbosity": -1,
            "device": "gpu" if device in {"gpu", "cuda"} else "cpu",
        }
        if is_cls:
            return LGBMClassifier(**common)
        if is_quantile and quantile_alpha is not None:
            return LGBMRegressor(objective="quantile", alpha=quantile_alpha, **common)
        return LGBMRegressor(**common)
    except Exception:
        from iqrp.app.forecasting.tree_models.base.native import NativeGBM

        return NativeGBM(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            task="classification" if is_cls else "regression",
            random_state=random_state,
            quantile_alpha=quantile_alpha if is_quantile else None,
        )


def _make_catboost(
    is_cls: bool,
    is_quantile: bool,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    reg_lambda: float,
    random_state: int,
    n_jobs: int,
    device: str,
    quantile_alpha: float | None,
) -> Any:
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        common = {
            "iterations": n_estimators,
            "depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "l2_leaf_reg": reg_lambda,
            "random_seed": random_state,
            "thread_count": n_jobs if n_jobs > 0 else -1,
            "verbose": False,
            "allow_writing_files": False,
            "task_type": "GPU" if device in {"gpu", "cuda"} else "CPU",
        }
        if is_cls:
            return CatBoostClassifier(**common)
        if is_quantile and quantile_alpha is not None:
            return CatBoostRegressor(loss_function=f"Quantile:alpha={quantile_alpha}", **common)
        return CatBoostRegressor(loss_function="RMSE", **common)
    except Exception:
        from iqrp.app.forecasting.tree_models.base.native import NativeGBM

        return NativeGBM(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            task="classification" if is_cls else "regression",
            random_state=random_state,
            quantile_alpha=quantile_alpha if is_quantile else None,
        )


def _make_hist_gb(
    is_cls: bool,
    is_quantile: bool,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
    quantile_alpha: float | None,
) -> Any:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        if is_cls:
            return HistGradientBoostingClassifier(
                max_iter=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        if is_quantile and quantile_alpha is not None:
            return HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile_alpha,
                max_iter=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )
        return HistGradientBoostingRegressor(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    except Exception:
        from iqrp.app.forecasting.tree_models.base.native import NativeGBM

        return NativeGBM(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            task="classification" if is_cls else "regression",
            random_state=random_state,
            quantile_alpha=quantile_alpha if is_quantile else None,
        )


def _make_rf(
    is_cls: bool,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    n_jobs: int,
    *,
    extra: bool,
) -> Any:
    try:
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            ExtraTreesRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
        )

        if extra:
            cls = ExtraTreesClassifier if is_cls else ExtraTreesRegressor
        else:
            cls = RandomForestClassifier if is_cls else RandomForestRegressor
        return cls(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    except Exception:
        from iqrp.app.forecasting.tree_models.base.native import NativeForest

        return NativeForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            task="classification" if is_cls else "regression",
            extra=extra,
        )


def estimator_predict(est: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(est.predict(X), dtype=np.float64).reshape(-1)


def estimator_predict_proba(est: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        return np.asarray(est.predict_proba(X), dtype=np.float64)
    # sigmoid fallback for binary scores
    scores = estimator_predict(est, X)
    p = 1.0 / (1.0 + np.exp(-scores))
    return np.column_stack([1 - p, p])


def estimator_feature_importances(est: Any, n_features: int) -> np.ndarray:
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=np.float64)
        if imp.size == n_features:
            return imp
    if hasattr(est, "get_booster"):
        try:
            score = est.get_booster().get_score(importance_type="gain")
            out = np.zeros(n_features)
            for k, v in score.items():
                idx = int(k.replace("f", "")) if str(k).startswith("f") else int(k)
                if 0 <= idx < n_features:
                    out[idx] = float(v)
            s = out.sum() or 1.0
            return out / s
        except Exception:
            pass
    return np.ones(n_features) / max(n_features, 1)

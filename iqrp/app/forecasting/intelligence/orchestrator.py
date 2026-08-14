"""Institutional Forecast Intelligence Engine — sole downstream forecast API."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.intelligence.automl import optimize_model
from iqrp.app.forecasting.intelligence.benchmark import BenchmarkResult, benchmark_candidates
from iqrp.app.forecasting.intelligence.calibration import (
    Calibrator,
    apply_calibration,
    fit_calibrator,
)
from iqrp.app.forecasting.intelligence.config import IntelligenceSettings
from iqrp.app.forecasting.intelligence.deployment import DeploymentManager
from iqrp.app.forecasting.intelligence.diagnostics import diagnose_leaderboard, diagnose_residuals
from iqrp.app.forecasting.intelligence.drift import DriftReport, detect_drift
from iqrp.app.forecasting.intelligence.ensemble import build_ensemble
from iqrp.app.forecasting.intelligence.monitoring import ForecastMonitor, MonitorSnapshot
from iqrp.app.forecasting.intelligence.ranking import RankedModel, leaderboard_table, rank_models
from iqrp.app.forecasting.intelligence.registry import (
    create_model,
    list_discovered_models,
    load_discovered_engines,
)
from iqrp.app.forecasting.intelligence.retraining import (
    RetrainDecision,
    checkpoint_model,
    decide_retrain,
    restore_checkpoint,
    retrain_model,
)
from iqrp.app.forecasting.intelligence.routing import RoutingTable, build_routing_table, route_model
from iqrp.app.forecasting.intelligence.selector import SelectionResult, select_best
from iqrp.app.forecasting.intelligence.serializer import IntelligenceSerializer
from iqrp.app.forecasting.intelligence.uncertainty import (
    ensemble_uncertainty,
    forecast_distribution,
    model_agreement,
    prediction_intervals,
)
from iqrp.app.forecasting.intelligence.visualization import (
    drift_chart,
    forecast_chart,
    leaderboard_chart,
)


class ForecastIntelligenceEngine:
    """Production façade: discover → benchmark → select → ensemble → deploy."""

    def __init__(self, settings: IntelligenceSettings | None = None) -> None:
        self.settings = settings or IntelligenceSettings.default()
        self._model: Any | None = None
        self._ensemble_models: dict[str, Any] = {}
        self._feature_columns: list[str] = []
        self._target_column: str = self.settings.columns.target
        self._leaderboard: list[RankedModel] = []
        self._benchmarks: list[BenchmarkResult] = []
        self._selection: SelectionResult | None = None
        self._calibrator: Calibrator | None = None
        self._routing: RoutingTable | None = None
        self._monitor = ForecastMonitor(self.settings.monitoring)
        self._n_updates = 0
        self._residual_std = 1.0
        self._ref_features: np.ndarray | None = None
        self._ref_preds: np.ndarray | None = None
        self._ref_target: np.ndarray | None = None
        self._ref_metric: float | None = None
        self._checkpoint: dict[str, Any] = {}
        self._best_params: dict[str, Any] = {}
        self._serializer = IntelligenceSerializer()
        self._deployment = DeploymentManager()
        self._fitted = False
        load_discovered_engines()

    # ------------------------------------------------------------------ API
    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        candidates: list[str] | None = None,
        run_automl: bool | None = None,
        run_selection: bool = True,
    ) -> ForecastIntelligenceEngine:
        feats = self._resolve_features(frame, feature_columns)
        tgt = target_column or self.settings.columns.target
        if tgt not in frame.columns:
            raise ValueError(f"target column '{tgt}' missing from frame")
        self._feature_columns = feats
        self._target_column = tgt

        if candidates is None:
            discovered = list_discovered_models(self.settings)
            candidates = [m.name for m in discovered]
            if not candidates:
                candidates = ["mock"]

        if run_selection:
            self._selection = select_best(
                frame,
                feature_columns=feats,
                target_column=tgt,
                settings=self.settings,
                candidates=candidates,
            )
            self._leaderboard = self._selection.ranked
            best_name = self._selection.best_model
            feats = self._selection.best_features or feats
            self._feature_columns = feats
        else:
            best_name = (candidates or ["mock"])[0]
            self._benchmarks = benchmark_candidates(
                frame,
                feature_columns=feats,
                target_column=tgt,
                settings=self.settings,
                candidates=candidates,
            )
            self._leaderboard = rank_models(
                [
                    {"name": b.name, "family": b.family, "metrics": b.metrics}
                    for b in self._benchmarks
                ],
                self.settings.ranking,
            )
            if self._leaderboard:
                best_name = self._leaderboard[0].name

        do_automl = self.settings.automl.method != "none" if run_automl is None else run_automl
        if do_automl:
            self._best_params = optimize_model(
                best_name,
                frame,
                feature_columns=feats,
                target_column=tgt,
                settings=self.settings,
            )
        else:
            self._best_params = {}

        self._model = create_model(best_name, **self._best_params)
        self._model.fit(frame, feature_columns=feats, target_column=tgt)

        # ensemble members
        self._ensemble_models = {}
        if self.settings.ensemble.method != "none" and self._leaderboard:
            top = self._leaderboard[: max(self.settings.ensemble.top_k, 1)]
            for r in top:
                try:
                    m = create_model(r.name, **(self._best_params if r.name == best_name else {}))
                    m.fit(frame, feature_columns=feats, target_column=tgt)
                    self._ensemble_models[r.name] = m
                except Exception:
                    continue
            if best_name not in self._ensemble_models and self._model is not None:
                self._ensemble_models[best_name] = self._model

        self._routing = build_routing_table(
            best_name,
            regime_models=None if self._selection is None else self._selection.best_regime_models,
            high_vol_model=(
                None if self._selection is None else self._selection.best_volatility_model
            ),
        )

        preds = self._model.predict(frame, feature_columns=feats)
        y = frame[tgt].to_numpy().astype(np.float64)
        n = min(preds.size, y.size)
        self._residual_std = float(np.std(y[:n] - preds[:n])) or 1e-3
        self._ref_features = frame.select(feats).to_numpy().astype(np.float64)
        self._ref_preds = preds[:n]
        self._ref_target = y[:n]
        self._ref_metric = float(np.mean(np.abs(y[:n] - preds[:n])))
        self._checkpoint = checkpoint_model(self._model)
        self._fitted = True
        self._n_updates = 0
        return self

    def predict(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        t0 = perf_counter()
        feats = feature_columns or self._feature_columns
        model = self._resolve_active_model(frame)
        if self.settings.ensemble.method != "none" and len(self._ensemble_models) > 1:
            preds = {
                name: m.predict(frame, feature_columns=feats)
                for name, m in self._ensemble_models.items()
            }
            scores = {r.name: r.score for r in self._leaderboard}
            out = build_ensemble(preds, config=self.settings.ensemble, scores=scores)
        else:
            out = model.predict(frame, feature_columns=feats)
        latency = (perf_counter() - t0) * 1000.0
        if self.settings.monitoring.enabled:
            self._monitor.record(y_pred=float(out[-1]) if out.size else 0.0, latency_ms=latency)
        return np.asarray(out, dtype=np.float64)

    def predict_proba(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        feats = feature_columns or self._feature_columns
        model = self._resolve_active_model(frame)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(frame, feature_columns=feats)
        else:
            pred = model.predict(frame, feature_columns=feats)
            p = 1.0 / (1.0 + np.exp(-pred))
            proba = np.column_stack([1 - p, p])
        if self._calibrator is not None:
            scores = proba[:, -1] if proba.ndim == 2 else proba
            cal = apply_calibration(self._calibrator, scores)
            if proba.ndim == 2:
                proba = proba.copy()
                proba[:, -1] = cal.reshape(-1)[: proba.shape[0]]
                if proba.shape[1] == 2:
                    proba[:, 0] = 1.0 - proba[:, -1]
            else:
                proba = cal
        return np.asarray(proba, dtype=np.float64)

    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        self._require_fitted()
        t0 = perf_counter()
        feats = feature_columns or self._feature_columns
        h = int(
            horizon
            or (
                self._selection.best_horizon
                if self._selection
                else self.settings.forecast.default_horizon
            )
        )
        model = self._resolve_active_model(frame)
        if self.settings.ensemble.method != "none" and len(self._ensemble_models) > 1:
            paths = {}
            for name, m in self._ensemble_models.items():
                fc = m.forecast(frame, horizon=h, feature_columns=feats)
                paths[name] = fc.path()
            scores = {r.name: r.score for r in self._leaderboard}
            values = build_ensemble(paths, config=self.settings.ensemble, scores=scores)
            unc = ensemble_uncertainty(paths)
            intervals = prediction_intervals(
                values,
                residual_std=float(np.mean(unc["std"]) + self._residual_std),
                level=self.settings.forecast.interval_level,
            )
            fc = Forecast.from_values(
                values,
                horizon=h,
                model_name="ensemble",
                model_version="intelligence",
                features_used=tuple(feats),
                strategy="ensemble",
                intervals=intervals,
                metadata={
                    "agreement": float(model_agreement(paths)),
                    "members": list(paths),
                    "uncertainty": {
                        k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in unc.items()
                    },
                },
            )
        else:
            fc = model.forecast(frame, horizon=h, feature_columns=feats)
        latency = (perf_counter() - t0) * 1000.0
        if self.settings.monitoring.enabled and fc.values.size:
            self._monitor.record(y_pred=float(fc.values[0]), latency_ms=latency)
        return fc

    def forecast_interval(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        level: float | None = None,
        feature_columns: list[str] | None = None,
    ) -> list[Any]:
        fc = self.forecast(frame, horizon=horizon, feature_columns=feature_columns)
        if fc.intervals:
            return fc.intervals
        lvl = float(level or self.settings.forecast.interval_level)
        return prediction_intervals(fc.path(), residual_std=self._residual_std, level=lvl)

    def best_model(self) -> str:
        if self._selection is not None:
            return self._selection.best_model
        if self._leaderboard:
            return self._leaderboard[0].name
        if self._model is not None:
            return getattr(getattr(self._model, "meta", None), "name", "unknown")
        return "none"

    @property
    def best_model_name(self) -> str:
        return self.best_model()

    def leaderboard(
        self,
        *,
        by: str | None = None,
        frame: pl.DataFrame | None = None,
    ) -> list[dict[str, Any]]:
        rows = leaderboard_table(self._leaderboard)
        if by is None or frame is None:
            return rows
        # scoped leaderboards
        if by == "asset" and "asset_id" in frame.columns:
            return [
                {"scope": "asset", "asset": a, "leaderboard": rows}
                for a in frame["asset_id"].unique().to_list()
            ]
        if by == "regime" and self.settings.routing.regime_column in frame.columns:
            col = self.settings.routing.regime_column
            return [
                {"scope": "regime", "regime": r, "leaderboard": rows}
                for r in frame[col].unique().to_list()
            ]
        if by == "timeframe":
            return [{"scope": "timeframe", "timeframe": "default", "leaderboard": rows}]
        if by == "feature_set":
            return [
                {
                    "scope": "feature_set",
                    "features": list(self._feature_columns),
                    "leaderboard": rows,
                }
            ]
        return rows

    def benchmark(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        candidates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        feats = self._resolve_features(frame, feature_columns or self._feature_columns)
        tgt = target_column or self._target_column
        results = benchmark_candidates(
            frame,
            feature_columns=feats,
            target_column=tgt,
            settings=self.settings,
            candidates=candidates,
        )
        self._benchmarks = results
        self._leaderboard = rank_models(
            [
                {"name": r.name, "family": r.family, "metrics": r.metrics, "metadata": r.metadata}
                for r in results
            ],
            self.settings.ranking,
        )
        return [r.to_dict() for r in results]

    def ensemble(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        method: str | None = None,
    ) -> np.ndarray:
        self._require_fitted()
        feats = feature_columns or self._feature_columns
        if not self._ensemble_models:
            return self.predict(frame, feature_columns=feats)
        preds = {
            n: m.predict(frame, feature_columns=feats) for n, m in self._ensemble_models.items()
        }
        cfg = self.settings.ensemble
        if method is not None:
            from iqrp.app.forecasting.intelligence.config import EnsembleConfig

            cfg = EnsembleConfig(
                method=method,  # type: ignore[arg-type]
                top_k=cfg.top_k,
                min_weight=cfg.min_weight,
            )
        scores = {r.name: r.score for r in self._leaderboard}
        return build_ensemble(preds, config=cfg, scores=scores)

    def calibrate(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        method: str | None = None,
    ) -> Calibrator | None:
        self._require_fitted()
        feats = feature_columns or self._feature_columns
        tgt = target_column or self._target_column
        method_name = method or self.settings.calibration.method
        if method_name in {"none", None}:
            method_name = "platt"
        scores = self.predict(frame, feature_columns=feats)
        y = frame[tgt].to_numpy()
        self._calibrator = fit_calibrator(y, scores, method=method_name)  # type: ignore[arg-type]
        return self._calibrator

    def monitor(
        self, *, y_true: float | None = None, y_pred: float | None = None
    ) -> MonitorSnapshot:
        if y_true is not None or y_pred is not None:
            self._monitor.record(y_true=y_true, y_pred=y_pred)
        return self._monitor.snapshot()

    def retrain(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        force: bool = False,
    ) -> RetrainDecision:
        self._require_fitted()
        feats = feature_columns or self._feature_columns
        tgt = target_column or self._target_column
        drift = self._compute_drift(frame, feats, tgt)
        perf_deg = False
        if self._ref_metric is not None and self._model is not None:
            pred = self._model.predict(frame, feature_columns=feats)
            y = frame[tgt].to_numpy().astype(np.float64)
            n = min(pred.size, y.size)
            cur = float(np.mean(np.abs(y[:n] - pred[:n])))
            if (
                self._ref_metric > 1e-12
                and (cur - self._ref_metric) / self._ref_metric
                > self.settings.drift.performance_drop
            ):
                perf_deg = True
        self._n_updates += 1
        decision = decide_retrain(
            n_updates=self._n_updates,
            config=self.settings.retrain,
            drift=drift,
            performance_degraded=perf_deg,
        )
        if force:
            decision = RetrainDecision(True, "force", self.settings.retrain.mode)
        if decision.should_retrain and self._model is not None:
            self._checkpoint = checkpoint_model(self._model)
            self._model = retrain_model(
                self._model,
                frame,
                feature_columns=feats,
                target_column=tgt,
                config=self.settings.retrain,
            )
            for name, m in list(self._ensemble_models.items()):
                try:
                    self._ensemble_models[name] = retrain_model(
                        m,
                        frame,
                        feature_columns=feats,
                        target_column=tgt,
                        config=self.settings.retrain,
                    )
                except Exception:
                    continue
        return decision

    def detect_drift(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
    ) -> DriftReport:
        self._require_fitted()
        return self._compute_drift(
            frame,
            feature_columns or self._feature_columns,
            target_column or self._target_column,
        )

    def diagnose(self, frame: pl.DataFrame) -> dict[str, Any]:
        self._require_fitted()
        pred = self.predict(frame)
        y = frame[self._target_column].to_numpy()
        report = diagnose_residuals(y, pred)
        return {
            "residuals": report.to_dict(),
            "leaderboard": diagnose_leaderboard(self._leaderboard),
        }

    def visualize(self, frame: pl.DataFrame | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "leaderboard": leaderboard_chart(self._leaderboard, config=self.settings.visualization)
        }
        if frame is not None and self._fitted:
            pred = self.predict(frame)
            y = (
                frame[self._target_column].to_numpy()
                if self._target_column in frame.columns
                else None
            )
            ts = (
                frame[self.settings.columns.timestamp].to_list()
                if self.settings.columns.timestamp in frame.columns
                else list(range(len(pred)))
            )
            out["forecast"] = forecast_chart(ts, y, pred)
            drift = self.detect_drift(frame)
            out["drift"] = drift_chart(drift.feature_drift)
        return out

    def distribution(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        n_samples: int = 100,
    ) -> np.ndarray:
        fc = self.forecast(frame, horizon=horizon)
        return forecast_distribution(fc.path(), self._residual_std, n_samples=n_samples)

    def deploy(self, *, name: str = "intelligence") -> dict[str, Any]:
        rec = self._deployment.deploy(self, name=name)
        return rec.to_dict()

    def save(self, path: str | Path) -> Path:
        return self._serializer.save(self, path)

    @classmethod
    def load(
        cls, path: str | Path, settings: IntelligenceSettings | None = None
    ) -> ForecastIntelligenceEngine:
        ser = IntelligenceSerializer()
        payload = ser.load(path)
        engine = cls(settings=settings or IntelligenceSettings.default())
        engine.import_state(payload)
        return engine

    def export_state(self) -> dict[str, Any]:
        return {
            "settings": self.settings.model_dump(),
            "best_model": self.best_model(),
            "feature_columns": list(self._feature_columns),
            "target_column": self._target_column,
            "leaderboard": [r.to_dict() for r in self._leaderboard],
            "selection": None if self._selection is None else self._selection.to_dict(),
            "routing": None if self._routing is None else self._routing.to_dict(),
            "best_params": dict(self._best_params),
            "residual_std": self._residual_std,
            "n_updates": self._n_updates,
            "fitted": self._fitted,
            "checkpoint": dict(self._checkpoint),
            "calibrator": (
                None
                if self._calibrator is None
                else {"method": self._calibrator.method, "params": self._calibrator.params}
            ),
        }

    def import_state(self, payload: dict[str, Any]) -> ForecastIntelligenceEngine:
        if "settings" in payload:
            try:
                self.settings = IntelligenceSettings.from_mapping(payload["settings"])
            except Exception:
                pass
        self._feature_columns = list(payload.get("feature_columns") or [])
        self._target_column = str(payload.get("target_column") or self.settings.columns.target)
        self._residual_std = float(payload.get("residual_std", 1.0))
        self._n_updates = int(payload.get("n_updates", 0))
        self._best_params = dict(payload.get("best_params") or {})
        self._leaderboard = [
            RankedModel(
                name=r["name"],
                metrics=dict(r.get("metrics") or {}),
                score=float(r.get("score", 0.0)),
                rank=int(r.get("rank", 0)),
                family=str(r.get("family", "")),
                metadata=dict(r.get("metadata") or {}),
            )
            for r in payload.get("leaderboard") or []
        ]
        best = str(payload.get("best_model") or "mock")
        try:
            self._model = create_model(best, **self._best_params)
            if payload.get("checkpoint"):
                restore_checkpoint(self._model, payload["checkpoint"])
                self._fitted = True
            elif payload.get("fitted"):
                self._fitted = bool(payload["fitted"])
        except Exception:
            self._model = None
            self._fitted = False
        cal = payload.get("calibrator")
        if cal:
            self._calibrator = Calibrator(
                method=str(cal["method"]), params=dict(cal.get("params") or {})
            )
        rt = payload.get("routing")
        if rt:
            self._routing = RoutingTable(
                by_regime=dict(rt.get("by_regime") or {}),
                by_asset=dict(rt.get("by_asset") or {}),
                by_timeframe=dict(rt.get("by_timeframe") or {}),
                default_model=str(rt.get("default_model") or best),
                high_vol_model=rt.get("high_vol_model"),
                low_confidence_model=rt.get("low_confidence_model"),
            )
        self._checkpoint = dict(payload.get("checkpoint") or {})
        return self

    def discovered_models(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in list_discovered_models(self.settings)]

    # --------------------------------------------------------------- helpers
    def _resolve_features(
        self, frame: pl.DataFrame, feature_columns: list[str] | None
    ) -> list[str]:
        if feature_columns:
            return list(feature_columns)
        if self.settings.columns.feature_columns:
            return list(self.settings.columns.feature_columns)
        if self._feature_columns:
            return list(self._feature_columns)
        skip = {
            self.settings.columns.timestamp,
            self.settings.columns.target,
            "regime",
            "vol_forecast",
            "asset_id",
            "spread",
            "close",
            "open",
            "high",
            "low",
            "volume",
        }
        return [c for c in frame.columns if c not in skip and frame[c].dtype.is_numeric()]

    def _resolve_active_model(self, frame: pl.DataFrame) -> Any:
        assert self._model is not None
        if self._routing is None or not self.settings.routing.enabled:
            return self._model
        name = route_model(frame, self._routing, config=self.settings.routing)
        if name in self._ensemble_models:
            return self._ensemble_models[name]
        return self._model

    def _compute_drift(self, frame: pl.DataFrame, feats: list[str], tgt: str) -> DriftReport:
        cur_f = frame.select(feats).to_numpy().astype(np.float64)
        cur_p = None
        if self._model is not None:
            cur_p = self._model.predict(frame, feature_columns=feats)
        cur_t = frame[tgt].to_numpy() if tgt in frame.columns else None
        cur_metric = None
        if cur_p is not None and cur_t is not None:
            n = min(cur_p.size, cur_t.size)
            cur_metric = float(np.mean(np.abs(cur_t[:n] - cur_p[:n])))
        return detect_drift(
            ref_features=self._ref_features if self._ref_features is not None else cur_f,
            cur_features=cur_f,
            ref_preds=self._ref_preds,
            cur_preds=cur_p,
            ref_target=self._ref_target,
            cur_target=cur_t,
            ref_metric=self._ref_metric,
            cur_metric=cur_metric,
            config=self.settings.drift,
            feature_names=feats,
        )

    def _require_fitted(self) -> None:
        if not self._fitted or self._model is None:
            from iqrp.app.core.exceptions import ModelError

            raise ModelError(
                "ForecastIntelligenceEngine is not fitted",
                code="FI_NOT_FITTED",
            )

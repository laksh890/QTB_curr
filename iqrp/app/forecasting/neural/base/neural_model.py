"""Base class for institutional neural forecasting models."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.base.evaluator import EvaluationReport
from iqrp.app.forecasting.base.forecast import Forecast
from iqrp.app.forecasting.base.forecast_model import ForecastModel
from iqrp.app.forecasting.base.metadata import TrainingMetadata
from iqrp.app.forecasting.base.prediction import PredictionInterval
from iqrp.app.forecasting.explainability.importance import ExplanationResult
from iqrp.app.forecasting.neural.base.callbacks import History
from iqrp.app.forecasting.neural.base.data import (
    make_sequences,
    standardize_apply,
    standardize_fit,
    train_val_split,
)
from iqrp.app.forecasting.neural.base.metrics import evaluate_predictions
from iqrp.app.forecasting.neural.base.torch_utils import (
    count_parameters,
    from_tensor,
    has_torch,
    resolve_device,
    to_tensor,
)
from iqrp.app.forecasting.neural.base.trainer import NeuralTrainer
from iqrp.app.forecasting.neural.config import NeuralSettings
from iqrp.app.forecasting.neural.probabilistic.quantiles import (
    extract_point_forecast,
    interval_from_prediction,
    quantiles_from_prediction,
)
from iqrp.app.forecasting.neural.probabilistic.uncertainty import total_uncertainty
from iqrp.app.forecasting.postprocessing.intervals import residual_intervals


class NeuralForecastModel(ForecastModel):
    """Shared API for MLP / LSTM / GRU / TCN / N-BEATS / N-HiTS / DeepAR / Seq2Seq."""

    architecture_name: str = "neural"

    def __init__(self, settings: NeuralSettings | Any | None = None, **params: Any) -> None:
        if settings is None:
            settings = NeuralSettings.default()
        elif isinstance(settings, dict):
            settings = NeuralSettings.from_mapping(settings)
        super().__init__(settings=settings)
        self._neural_settings: NeuralSettings = settings  # type: ignore[assignment]
        self._params_kw = dict(params)
        self._module: Any = None
        self._history = History()
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._X_seq: np.ndarray | None = None
        self._y_seq: np.ndarray | None = None
        self._y_raw: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self._lookback = int(settings.architecture.lookback)
        self._horizon = int(settings.architecture.horizon or settings.forecast.default_horizon)
        self._device = resolve_device(settings.train.device)
        self._update_count = 0
        self._regime_modules: dict[Any, Any] = {}

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> NeuralForecastModel:
        if not has_torch():
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                "PyTorch is required for neural forecasting", code="NEURAL_NO_TORCH"
            )
        tgt = self._resolve_target(frame, target_column)
        cols = self._resolve_feature_columns(frame, feature_columns)
        cols = [c for c in cols if c != tgt]
        regimes = self._maybe_regime(frame, regime_column)
        if (
            regimes is not None
            and self._neural_settings.regime.enabled
            and self._neural_settings.regime.mode == "feature"
            and self._regime_column
            and self._regime_column not in cols
        ):
            cols = cols + [self._regime_column]
        if not cols:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No feature columns for neural model", code="NEURAL_NO_FEATURES")
        X = frame.select(cols).to_numpy().astype(np.float64)
        y = frame[tgt].to_numpy().astype(np.float64)
        self._mu, self._sd = standardize_fit(X)
        Xs = standardize_apply(X, self._mu, self._sd)
        X_seq, y_seq = make_sequences(Xs, y, lookback=self._lookback, horizon=self._horizon)
        # optional HPO
        if self._neural_settings.optimization.method != "none":
            from iqrp.app.forecasting.neural.optimization.hpo import optimize_neural

            best = optimize_neural(self, X_seq, y_seq, settings=self._neural_settings)
            # apply best into params
            for k, v in best.items():
                if hasattr(self._neural_settings.architecture, k) or k in {
                    "learning_rate",
                    "batch_size",
                    "hidden_size",
                    "num_layers",
                    "dropout",
                }:
                    self._params_kw[k] = v
        X_tr, y_tr, X_va, y_va = train_val_split(
            X_seq, y_seq, val_ratio=self._neural_settings.train.val_ratio
        )
        task = self._neural_settings.task.type
        if (
            regimes is not None
            and self._neural_settings.regime.enabled
            and self._neural_settings.regime.mode in {"separate", "moe"}
        ):
            self._regime_modules = {}
            # align regimes to sequence ends
            reg_seq = regimes[self._lookback - 1 : self._lookback - 1 + X_seq.shape[0]]
            for reg in np.unique(reg_seq):
                mask = reg_seq == reg
                if int(mask.sum()) < 16:
                    continue
                mod = self._build_module(n_features=X.shape[1], task=task)
                trainer = NeuralTrainer(self._neural_settings)
                mod, _ = trainer.fit(mod, X_seq[mask], y_seq[mask])
                self._regime_modules[reg] = mod
        self._module = self._build_module(n_features=X.shape[1], task=task)
        if self._neural_settings.distributed.gradient_checkpointing and hasattr(
            self._module, "gradient_checkpointing_enable"
        ):
            try:
                self._module.gradient_checkpointing_enable()
            except Exception:  # pragma: no cover
                pass
        trainer = NeuralTrainer(self._neural_settings)
        self._module, self._history = trainer.fit(self._module, X_tr, y_tr, X_va, y_va)
        self._device = trainer.device
        # residuals on training windows
        pred = trainer.predict(self._module, X_seq)
        point = extract_point_forecast(
            pred, task=task, alphas=self._neural_settings.task.quantile_alphas
        )
        self._residuals = (
            y_seq.reshape(point.shape[0], -1)[:, 0] - point.reshape(point.shape[0], -1)[:, 0]
        )
        self._X_seq, self._y_seq, self._y_raw = X_seq, y_seq, y
        self._feature_columns = list(cols)
        self._target_column = tgt
        self._training_meta = TrainingMetadata(
            n_samples=int(X_seq.shape[0]),
            n_features=int(X.shape[1]),
            feature_columns=tuple(cols),
            target_column=tgt,
            regime_column=self._regime_column,
            horizon=self._horizon,
            extra={
                "architecture": self.architecture_name,
                "n_parameters": count_parameters(self._module),
                "history": self._history.to_dict(),
                "lookback": self._lookback,
            },
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
    ) -> NeuralForecastModel:
        mode = self._neural_settings.online.mode
        if not self._fitted or mode == "refit" or self._module is None:
            return self.fit(
                frame, feature_columns, target_column=target_column, regime_column=regime_column
            )
        self._update_count += 1
        if (
            self._update_count % max(int(self._neural_settings.online.refresh_every), 1) == 0
            and mode != "finetune"
        ):
            return self.fit(
                frame,
                feature_columns or self._feature_columns,
                target_column=target_column or self._target_column,
                regime_column=regime_column or self._regime_column,
            )
        # finetune / warm_start on latest window
        tgt = target_column or self._target_column or self._neural_settings.columns.target
        cols = list(self._feature_columns)
        X = standardize_apply(frame.select(cols).to_numpy().astype(np.float64), self._mu, self._sd)  # type: ignore[arg-type]
        y = frame[tgt].to_numpy().astype(np.float64)
        X_seq, y_seq = make_sequences(X, y, lookback=self._lookback, horizon=self._horizon)
        w = int(self._neural_settings.online.window)
        X_seq, y_seq = X_seq[-w:], y_seq[-w:]
        # temporarily reduce epochs for finetune

        # pydantic frozen — override via mapping
        s = NeuralSettings.from_mapping(
            {
                **self._neural_settings.model_dump(),
                "train": {
                    **self._neural_settings.train.model_dump(),
                    "epochs": self._neural_settings.online.finetune_epochs,
                },
            }
        )
        trainer = NeuralTrainer(s)
        self._module, hist = trainer.fit(self._module, X_seq, y_seq)
        self._history.train_loss.extend(hist.train_loss)
        self._history.val_loss.extend(hist.val_loss)
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        self._require_fitted()
        pred = self._predict_raw(frame, feature_columns)
        point = extract_point_forecast(
            pred,
            task=self._neural_settings.task.type,
            alphas=self._neural_settings.task.quantile_alphas,
        )
        # return last-horizon step per window, pad to frame length
        last = point[:, -1] if point.ndim == 2 else point.reshape(-1)
        return self._align_to_frame(last, frame.height)

    def predict_proba(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        self._require_fitted()
        if not self.meta.supports_proba:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Model '{self.meta.name}' does not support predict_proba",
                code="NEURAL_NO_PROBA",
            )
        import torch

        pred = self._predict_raw(frame, feature_columns)
        # classification logits (B,H,C) or binary (B,H)
        if pred.ndim == 3:
            logits = pred[:, -1, :]
            proba = torch.softmax(to_tensor(logits), dim=-1)
            arr = from_tensor(proba)
        else:
            logits = pred[:, -1] if pred.ndim == 2 else pred
            p = 1 / (1 + np.exp(-logits))
            arr = np.column_stack([1 - p, p])
        return self._align_proba(arr, frame.height)

    def forecast(
        self,
        frame: pl.DataFrame,
        *,
        horizon: int | None = None,
        feature_columns: list[str] | None = None,
    ) -> Forecast:
        self._require_fitted()
        h = max(int(horizon if horizon is not None else self._horizon), 1)
        pred = self._predict_raw(frame, feature_columns)
        point = extract_point_forecast(
            pred,
            task=self._neural_settings.task.type,
            alphas=self._neural_settings.task.quantile_alphas,
        )
        # use last window's multi-horizon path
        if point.ndim == 1:
            path = np.full(h, float(point[-1]))
        else:
            path = point[-1, :h]
            if path.size < h:
                path = np.pad(
                    path, (0, h - path.size), constant_values=path[-1] if path.size else 0.0
                )
        lo, hi = interval_from_prediction(
            pred[-1:],
            task=self._neural_settings.task.type,
            alphas=self._neural_settings.task.quantile_alphas,
            distribution=self._neural_settings.probabilistic.distribution,
        )
        # build intervals list
        lo_p = lo.reshape(-1)[:h] if lo.size else path - 1e-3
        hi_p = hi.reshape(-1)[:h] if hi.size else path + 1e-3
        if lo_p.size < h:
            lo_p = np.resize(lo_p, h)
            hi_p = np.resize(hi_p, h)
        intervals = [
            PredictionInterval(
                lower=float(lo_p[i]),
                upper=float(hi_p[i]),
                level=self._neural_settings.forecast.interval_level,
            )
            for i in range(h)
        ]
        q = quantiles_from_prediction(
            pred[-1:],
            task=self._neural_settings.task.type,
            alphas=self._neural_settings.task.quantile_alphas,
            distribution=self._neural_settings.probabilistic.distribution,
        )
        meta: dict[str, Any] = {
            "architecture": self.architecture_name,
            "quantiles": q.reshape(-1, q.shape[-1]).tolist() if q.size else [],
            "history": self._history.to_dict(),
        }
        if self._neural_settings.probabilistic.enabled and self._module is not None:
            unc = total_uncertainty(
                self._module,
                self._last_window(frame, feature_columns),
                pred[-1:],
                mc_dropout=self._neural_settings.probabilistic.mc_dropout,
                n_samples=self._neural_settings.probabilistic.n_samples,
                device=self._device,
            )
            meta["uncertainty"] = {k: np.asarray(v).reshape(-1).tolist() for k, v in unc.items()}
        return Forecast.from_values(
            path,
            horizon=h,
            model_name=self.meta.name,
            model_version=self.meta.version,
            features_used=tuple(self._feature_columns),
            regime_used=(
                frame[self._regime_column].to_numpy()[-1]
                if self._regime_column and self._regime_column in frame.columns
                else None
            ),
            strategy="sequence",
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
        return residual_intervals(
            fc.path(),
            level=float(
                level if level is not None else self._neural_settings.forecast.interval_level
            ),
        )

    def evaluate(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        probabilities: np.ndarray | None = None,
    ) -> EvaluationReport:
        self._require_fitted()
        tgt = target_column or self._target_column or self._neural_settings.columns.target
        y_true = frame[tgt].to_numpy().astype(np.float64)
        y_pred = self.predict(frame, feature_columns)
        n = min(y_true.size, y_pred.size)
        proba = probabilities
        if proba is None and self.meta.supports_proba:
            try:
                proba = self.predict_proba(frame, feature_columns)
            except Exception:  # pragma: no cover
                proba = None
        metrics = evaluate_predictions(
            y_true[-n:],
            y_pred[-n:],
            proba=None if proba is None else proba[-n:],
            task=self._neural_settings.task.type,
        )
        return EvaluationReport(
            metrics=metrics, method=f"neural_{self.architecture_name}", n_samples=n
        )

    def explain(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        method: str = "integrated_gradients",
    ) -> ExplanationResult:
        self._require_fitted()
        from iqrp.app.forecasting.neural.explainability.attribution import explain_neural

        cols = list(self._feature_columns)
        X = self._last_window(frame, cols)
        attr = explain_neural(self._module, X, method=method, device=self._device)
        # aggregate over time
        if attr.ndim == 3:
            scores = np.mean(np.abs(attr), axis=(0, 1))
        else:
            scores = np.mean(np.abs(attr), axis=0)
        imp = {cols[i]: float(scores[i]) for i in range(min(len(cols), scores.size))}
        return ExplanationResult(method=method, importances=imp, attributions=attr)

    def export_onnx(self, path: str | Path) -> Path:
        self._require_fitted()
        path = Path(path)
        if not has_torch() or self._module is None:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("ONNX export requires a fitted torch module", code="NEURAL_ONNX")
        import torch

        self._module.eval()
        n_features = len(self._feature_columns)
        dummy = torch.zeros(1, self._lookback, n_features, device="cpu")
        cpu_mod = self._module.to("cpu")
        try:
            try:
                torch.onnx.export(
                    cpu_mod,
                    dummy,
                    str(path),
                    input_names=["x"],
                    output_names=["y"],
                    dynamo=False,
                )
            except TypeError:
                torch.onnx.export(cpu_mod, dummy, str(path), input_names=["x"], output_names=["y"])
        except Exception:
            # fallback: TorchScript or raw state dict
            pt_path = path.with_suffix(".pt")
            try:
                scripted = torch.jit.trace(cpu_mod, dummy)
                scripted.save(str(pt_path))
            except Exception:
                torch.save(
                    {
                        "state_dict": cpu_mod.state_dict(),
                        "lookback": self._lookback,
                        "n_features": n_features,
                    },
                    pt_path,
                )
            path = pt_path
        self._module.to(self._device)
        return path

    def diagnostics(self) -> dict[str, Any]:
        self._require_fitted()
        from iqrp.app.forecasting.neural.diagnostics.report import run_neural_diagnostics

        return run_neural_diagnostics(self).to_dict()

    def _predict_raw(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        assert self._module is not None and self._mu is not None and self._sd is not None
        cols = list(self._feature_columns)
        X = standardize_apply(frame.select(cols).to_numpy().astype(np.float64), self._mu, self._sd)
        y_proxy = (
            frame[self._target_column].to_numpy().astype(np.float64)
            if self._target_column in frame.columns
            else X[:, 0]
        )
        X_seq, _ = make_sequences(X, y_proxy, lookback=self._lookback, horizon=self._horizon)
        # regime routing
        if self._regime_modules and self._regime_column and self._regime_column in frame.columns:
            regimes = frame[self._regime_column].to_numpy()
            reg_seq = regimes[self._lookback - 1 : self._lookback - 1 + X_seq.shape[0]]
            outs = []
            trainer = NeuralTrainer(self._neural_settings)
            for i in range(X_seq.shape[0]):
                mod = self._regime_modules.get(reg_seq[i], self._module)
                outs.append(trainer.predict(mod, X_seq[i : i + 1])[0])
            return np.stack(outs)
        trainer = NeuralTrainer(self._neural_settings)
        trainer.device = self._device
        return trainer.predict(self._module, X_seq)

    def _last_window(
        self, frame: pl.DataFrame, feature_columns: list[str] | None = None
    ) -> np.ndarray:
        cols = list(self._feature_columns)
        X = standardize_apply(frame.select(cols).to_numpy().astype(np.float64), self._mu, self._sd)  # type: ignore[arg-type]
        y_proxy = (
            frame[self._target_column].to_numpy().astype(np.float64)
            if self._target_column in frame.columns
            else X[:, 0]
        )
        X_seq, _ = make_sequences(X, y_proxy, lookback=self._lookback, horizon=self._horizon)
        return X_seq[-1:]

    def _align_to_frame(self, values: np.ndarray, n: int) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        out = np.full(n, v[0] if v.size else 0.0)
        start = max(n - v.size, 0)
        out[start:] = v[-min(v.size, n) :]
        return out

    def _align_proba(self, proba: np.ndarray, n: int) -> np.ndarray:
        p = np.asarray(proba, dtype=np.float64)
        if p.ndim == 1:
            p = np.column_stack([1 - p, p])
        out = np.zeros((n, p.shape[1]))
        start = max(n - p.shape[0], 0)
        out[start:] = p[-min(p.shape[0], n) :]
        if start > 0:
            out[:start] = out[start]
        return out

    def _resolve_target(self, frame: pl.DataFrame, target_column: str | None) -> str:
        if target_column:
            return target_column
        if self._target_column:
            return self._target_column
        cfg = self._neural_settings.columns.target
        if cfg in frame.columns:
            return cfg
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError("Target column required", code="NEURAL_NO_TARGET")

    def _maybe_regime(self, frame: pl.DataFrame, regime_column: str | None) -> np.ndarray | None:
        col = regime_column or (
            self._neural_settings.regime.column if self._neural_settings.regime.enabled else None
        )
        if col and col in frame.columns:
            self._regime_column = col
            return frame[col].to_numpy()
        return None

    def _arch_kwargs(self) -> dict[str, Any]:
        a = self._neural_settings.architecture
        t = self._neural_settings.task
        p = self._neural_settings.probabilistic
        kw = {
            "hidden_size": int(self._params_kw.get("hidden_size", a.hidden_size)),
            "num_layers": int(self._params_kw.get("num_layers", a.num_layers)),
            "dropout": float(self._params_kw.get("dropout", a.dropout)),
            "bidirectional": bool(self._params_kw.get("bidirectional", a.bidirectional)),
            "kernel_size": a.kernel_size,
            "n_blocks": a.n_blocks,
            "task": t.type,
            "n_classes": t.n_classes,
            "n_quantiles": len(t.quantile_alphas),
            "dist": p.enabled and p.distribution in {"gaussian", "student_t"},
            "use_attention": a.attention,
        }
        return kw

    def _algorithm_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "architecture": self.architecture_name,
            "lookback": self._lookback,
            "horizon": self._horizon,
            "feature_columns": list(self._feature_columns),
            "mu": None if self._mu is None else self._mu.tolist(),
            "sd": None if self._sd is None else self._sd.tolist(),
            "history": self._history.to_dict(),
            "residuals": None if self._residuals is None else self._residuals.tolist(),
            "params_kw": dict(self._params_kw),
            "arch_kwargs": self._arch_kwargs(),
            "neural_settings": self._neural_settings.model_dump(),
            "update_count": self._update_count,
            "X_seq": None if self._X_seq is None else self._X_seq.tolist(),
            "y_seq": None if self._y_seq is None else self._y_seq.tolist(),
        }
        if self._module is not None and has_torch():
            state["state_dict"] = {
                k: v.detach().cpu().tolist() for k, v in self._module.state_dict().items()
            }
            state["n_features"] = len(self._feature_columns)
        return state

    def _load_algorithm_state(self, state: dict[str, Any]) -> None:
        if state.get("neural_settings"):
            self._neural_settings = NeuralSettings.from_mapping(state["neural_settings"])
            self._settings = self._neural_settings
        self._lookback = int(state.get("lookback", self._lookback))
        self._horizon = int(state.get("horizon", self._horizon))
        self._feature_columns = list(state.get("feature_columns") or [])
        self._mu = None if state.get("mu") is None else np.asarray(state["mu"], dtype=np.float64)
        self._sd = None if state.get("sd") is None else np.asarray(state["sd"], dtype=np.float64)
        self._history = History.from_dict(state.get("history") or {})
        self._residuals = (
            None
            if state.get("residuals") is None
            else np.asarray(state["residuals"], dtype=np.float64)
        )
        self._params_kw = dict(state.get("params_kw") or {})
        # ensure architecture dims used at train time win over defaults
        for k, v in (state.get("arch_kwargs") or {}).items():
            if k in {
                "hidden_size",
                "num_layers",
                "dropout",
                "bidirectional",
                "kernel_size",
                "n_blocks",
                "use_attention",
            }:
                self._params_kw.setdefault(k, v)
        self._update_count = int(state.get("update_count", 0))
        self._X_seq = (
            None if state.get("X_seq") is None else np.asarray(state["X_seq"], dtype=np.float64)
        )
        self._y_seq = (
            None if state.get("y_seq") is None else np.asarray(state["y_seq"], dtype=np.float64)
        )
        n_features = int(state.get("n_features") or max(len(self._feature_columns), 1))
        if has_torch() and state.get("state_dict") is not None:
            import torch

            self._module = self._build_module(
                n_features=n_features, task=self._neural_settings.task.type
            )
            sd = {k: torch.tensor(v) for k, v in state["state_dict"].items()}
            self._module.load_state_dict(sd)
            self._module.to(self._device)

    @abstractmethod
    def _build_module(self, *, n_features: int, task: str) -> Any: ...

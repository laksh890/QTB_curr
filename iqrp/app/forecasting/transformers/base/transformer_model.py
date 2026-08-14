"""Base class for institutional time-series transformer forecasting models."""

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
from iqrp.app.forecasting.neural.probabilistic.quantiles import (
    extract_point_forecast,
    interval_from_prediction,
    quantiles_from_prediction,
)
from iqrp.app.forecasting.postprocessing.intervals import residual_intervals
from iqrp.app.forecasting.transformers.base.trainer import TransformerTrainer
from iqrp.app.forecasting.transformers.config import TransformerSettings


class TransformerForecastModel(ForecastModel):
    """Shared API for TFT / Informer / Autoformer / PatchTST / ... / MoE Transformer."""

    architecture_name: str = "transformer"

    def __init__(self, settings: TransformerSettings | Any | None = None, **params: Any) -> None:
        if settings is None:
            settings = TransformerSettings.default()
        elif isinstance(settings, dict):
            settings = TransformerSettings.from_mapping(settings)
        super().__init__(settings=settings)
        self._tx_settings: TransformerSettings = settings  # type: ignore[assignment]
        self._params_kw = dict(params)
        self._module: Any = None
        self._history = History()
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._X_seq: np.ndarray | None = None
        self._y_seq: np.ndarray | None = None
        self._residuals: np.ndarray | None = None
        self._lookback = int(settings.architecture.lookback)
        self._horizon = int(settings.architecture.horizon or settings.forecast.default_horizon)
        self._device = resolve_device(settings.train.device)
        self._update_count = 0
        self._last_attn: Any = None
        self._last_embeddings: Any = None

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        target_column: str | None = None,
        regime_column: str | None = None,
    ) -> TransformerForecastModel:
        if not has_torch():
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                "PyTorch is required for transformer forecasting", code="TX_NO_TORCH"
            )
        tgt = self._resolve_target(frame, target_column)
        cols = self._resolve_feature_columns(frame, feature_columns)
        cols = [c for c in cols if c != tgt]
        regimes = self._maybe_regime(frame, regime_column)
        if (
            regimes is not None
            and self._tx_settings.regime.enabled
            and self._tx_settings.regime.mode == "feature"
            and self._regime_column
            and self._regime_column not in cols
        ):
            cols = cols + [self._regime_column]
        if not cols:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("No feature columns for transformer model", code="TX_NO_FEATURES")
        X = frame.select(cols).to_numpy().astype(np.float64)
        y = frame[tgt].to_numpy().astype(np.float64)
        self._mu, self._sd = standardize_fit(X)
        Xs = standardize_apply(X, self._mu, self._sd)
        # sliding / long-context: optionally truncate to max_context for training windows
        max_ctx = int(self._tx_settings.architecture.max_context)
        if Xs.shape[0] > max_ctx and self._tx_settings.forecast.sliding_context:
            Xs, y = Xs[-max_ctx:], y[-max_ctx:]
        X_seq, y_seq = make_sequences(Xs, y, lookback=self._lookback, horizon=self._horizon)
        X_tr, y_tr, X_va, y_va = train_val_split(
            X_seq, y_seq, val_ratio=self._tx_settings.train.val_ratio
        )
        task = self._tx_settings.task.type
        self._module = self._build_module(n_features=X.shape[1], task=task)
        trainer = TransformerTrainer(self._tx_settings)
        self._module, self._history = trainer.fit(self._module, X_tr, y_tr, X_va, y_va)
        self._device = trainer.device
        pred = trainer.predict(self._module, X_seq)
        point = extract_point_forecast(
            pred, task=task, alphas=self._tx_settings.task.quantile_alphas
        )
        self._residuals = (
            y_seq.reshape(point.shape[0], -1)[:, 0] - point.reshape(point.shape[0], -1)[:, 0]
        )
        self._X_seq, self._y_seq = X_seq, y_seq
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
    ) -> TransformerForecastModel:
        mode = self._tx_settings.online.mode
        if not self._fitted or mode == "refit" or self._module is None:
            return self.fit(
                frame, feature_columns, target_column=target_column, regime_column=regime_column
            )
        self._update_count += 1
        if (
            self._update_count % max(int(self._tx_settings.online.refresh_every), 1) == 0
            and mode != "finetune"
        ):
            return self.fit(
                frame,
                feature_columns or self._feature_columns,
                target_column=target_column or self._target_column,
                regime_column=regime_column or self._regime_column,
            )
        tgt = target_column or self._target_column or self._tx_settings.columns.target
        cols = list(self._feature_columns)
        X = standardize_apply(frame.select(cols).to_numpy().astype(np.float64), self._mu, self._sd)  # type: ignore[arg-type]
        y = frame[tgt].to_numpy().astype(np.float64)
        X_seq, y_seq = make_sequences(X, y, lookback=self._lookback, horizon=self._horizon)
        w = int(self._tx_settings.online.window)
        X_seq, y_seq = X_seq[-w:], y_seq[-w:]
        s = TransformerSettings.from_mapping(
            {
                **self._tx_settings.model_dump(),
                "train": {
                    **self._tx_settings.train.model_dump(),
                    "epochs": self._tx_settings.online.finetune_epochs,
                },
            }
        )
        trainer = TransformerTrainer(s)
        self._module, hist = trainer.fit(self._module, X_seq, y_seq)
        self._history.train_loss.extend(hist.train_loss)
        self._history.val_loss.extend(hist.val_loss)
        return self

    def predict(self, frame: pl.DataFrame, feature_columns: list[str] | None = None) -> np.ndarray:
        self._require_fitted()
        pred = self._predict_raw(frame, feature_columns)
        point = extract_point_forecast(
            pred, task=self._tx_settings.task.type, alphas=self._tx_settings.task.quantile_alphas
        )
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
                code="TX_NO_PROBA",
            )
        import torch

        pred = self._predict_raw(frame, feature_columns)
        if pred.ndim == 3:
            logits = pred[:, -1, :]
            arr = from_tensor(torch.softmax(to_tensor(logits), dim=-1))
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
            pred, task=self._tx_settings.task.type, alphas=self._tx_settings.task.quantile_alphas
        )
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
            task=self._tx_settings.task.type,
            alphas=self._tx_settings.task.quantile_alphas,
            distribution=(
                self._tx_settings.probabilistic.distribution
                if self._tx_settings.probabilistic.distribution != "mixture"
                else "gaussian"
            ),
        )
        lo_p = lo.reshape(-1)[:h] if lo.size else path - 1e-3
        hi_p = hi.reshape(-1)[:h] if hi.size else path + 1e-3
        if lo_p.size < h:
            lo_p = np.resize(lo_p, h)
            hi_p = np.resize(hi_p, h)
        intervals = [
            PredictionInterval(
                lower=float(lo_p[i]),
                upper=float(hi_p[i]),
                level=self._tx_settings.forecast.interval_level,
            )
            for i in range(h)
        ]
        q = quantiles_from_prediction(
            pred[-1:],
            task=self._tx_settings.task.type,
            alphas=self._tx_settings.task.quantile_alphas,
            distribution="gaussian",
        )
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
            strategy="transformer",
            intervals=intervals,
            metadata={
                "architecture": self.architecture_name,
                "quantiles": q.reshape(-1, q.shape[-1]).tolist() if q.size else [],
                "history": self._history.to_dict(),
            },
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
            level=float(level if level is not None else self._tx_settings.forecast.interval_level),
        )

    def attention(self, frame: pl.DataFrame | None = None) -> np.ndarray:
        self._require_fitted()
        if frame is not None:
            _ = self._predict_raw(frame)
        if self._last_attn is None:
            return np.zeros((1, 1))
        return np.asarray(self._last_attn, dtype=np.float64)

    def embeddings(self, frame: pl.DataFrame | None = None) -> np.ndarray:
        self._require_fitted()
        if frame is not None:
            _ = self._predict_raw(frame)
        if self._last_embeddings is None and self._X_seq is not None:
            return self._X_seq[-1:]
        if self._last_embeddings is None:
            return np.zeros((1, self._lookback, len(self._feature_columns)))
        return np.asarray(self._last_embeddings, dtype=np.float64)

    def evaluate(
        self,
        frame: pl.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
        probabilities: np.ndarray | None = None,
    ) -> EvaluationReport:
        self._require_fitted()
        tgt = target_column or self._target_column or self._tx_settings.columns.target
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
            task=self._tx_settings.task.type,
        )
        return EvaluationReport(
            metrics=metrics, method=f"transformer_{self.architecture_name}", n_samples=n
        )

    def explain(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        method: str = "attention_rollout",
    ) -> ExplanationResult:
        self._require_fitted()
        from iqrp.app.forecasting.transformers.explainability.attribution import explain_transformer

        cols = list(self._feature_columns)
        X = self._last_window(frame, cols)
        attr = explain_transformer(self._module, X, method=method, device=self._device)
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

            raise ValidationError("ONNX export requires a fitted torch module", code="TX_ONNX")
        import torch

        self._module.eval()
        n_features = len(self._feature_columns)
        dummy = torch.zeros(1, self._lookback, n_features, device="cpu")
        cpu_mod = self._module.to("cpu")
        try:
            try:
                torch.onnx.export(
                    cpu_mod, dummy, str(path), input_names=["x"], output_names=["y"], dynamo=False
                )
            except TypeError:
                torch.onnx.export(cpu_mod, dummy, str(path), input_names=["x"], output_names=["y"])
        except Exception:
            pt = path.with_suffix(".pt")
            try:
                torch.jit.trace(cpu_mod, dummy).save(str(pt))
            except Exception:
                torch.save({"state_dict": cpu_mod.state_dict()}, pt)
            path = pt
        self._module.to(self._device)
        return path

    def diagnostics(self) -> dict[str, Any]:
        self._require_fitted()
        from iqrp.app.forecasting.transformers.diagnostics.report import run_transformer_diagnostics

        return run_transformer_diagnostics(self).to_dict()

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
        trainer = TransformerTrainer(self._tx_settings)
        trainer.device = self._device
        out = trainer.predict(self._module, X_seq)
        # capture attention / embeddings if available
        self._capture_internals(X_seq[-1:])
        return out

    def _capture_internals(self, X_last: np.ndarray) -> None:
        if self._module is None or not has_torch():
            return
        import torch

        self._module.eval()
        with torch.no_grad():
            xb = to_tensor(X_last, self._device)
            if hasattr(self._module, "encode"):
                emb = self._module.encode(xb)
                if isinstance(emb, (tuple, list)):
                    emb = emb[0]
                self._last_embeddings = from_tensor(emb)
            else:
                self._last_embeddings = X_last
            attn = None
            for m in self._module.modules():
                if hasattr(m, "last_attn") and m.last_attn is not None:
                    attn = m.last_attn
            if attn is not None:
                self._last_attn = from_tensor(attn)

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
        cfg = self._tx_settings.columns.target
        if cfg in frame.columns:
            return cfg
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError("Target column required", code="TX_NO_TARGET")

    def _maybe_regime(self, frame: pl.DataFrame, regime_column: str | None) -> np.ndarray | None:
        col = regime_column or (
            self._tx_settings.regime.column if self._tx_settings.regime.enabled else None
        )
        if col and col in frame.columns:
            self._regime_column = col
            return frame[col].to_numpy()
        return None

    def _arch_kwargs(self) -> dict[str, Any]:
        a = self._tx_settings.architecture
        t = self._tx_settings.task
        p = self._tx_settings.probabilistic
        return {
            "d_model": int(self._params_kw.get("d_model", a.d_model)),
            "n_heads": int(self._params_kw.get("n_heads", a.n_heads)),
            "num_layers": int(self._params_kw.get("num_layers", a.num_layers)),
            "dropout": float(self._params_kw.get("dropout", a.dropout)),
            "ffn_dim": a.ffn_dim,
            "patch_len": a.patch_len,
            "stride": a.stride,
            "factor": a.factor,
            "moving_avg": a.moving_avg,
            "attention_type": a.attention_type,
            "positional": a.positional,
            "task": t.type,
            "n_classes": t.n_classes,
            "n_quantiles": len(t.quantile_alphas),
            "n_mixtures": p.n_mixtures,
            "dist": p.enabled and p.distribution in {"gaussian", "student_t"},
            "n_regimes": self._tx_settings.regime.n_regimes,
            "use_regime": self._tx_settings.regime.enabled and a.use_regime_token,
        }

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
            "tx_settings": self._tx_settings.model_dump(),
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
        if state.get("tx_settings"):
            self._tx_settings = TransformerSettings.from_mapping(state["tx_settings"])
            self._settings = self._tx_settings
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
        for k, v in (state.get("arch_kwargs") or {}).items():
            if k in {"d_model", "n_heads", "num_layers", "dropout"}:
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
                n_features=n_features, task=self._tx_settings.task.type
            )
            sd = {k: torch.tensor(v) for k, v in state["state_dict"].items()}
            self._module.load_state_dict(sd)
            self._module.to(self._device)

    @abstractmethod
    def _build_module(self, *, n_features: int, task: str) -> Any: ...

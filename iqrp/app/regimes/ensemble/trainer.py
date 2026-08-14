"""Training orchestration for the ensemble regime engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.regimes.ensemble.calibration import Calibrator, expected_calibration_error
from iqrp.app.regimes.ensemble.combiner import combine
from iqrp.app.regimes.ensemble.config import EnsembleSettings
from iqrp.app.regimes.ensemble.orchestrator import (
    collect_transition,
    fit_members,
    member_log_likelihoods,
    predict_members,
)
from iqrp.app.regimes.ensemble.registry import EnsembleMember, EnsembleRegistry
from iqrp.app.regimes.ensemble.weighting import compute_weights


@dataclass
class TrainResult:
    members: list[EnsembleMember]
    weights: np.ndarray
    calibrator: Calibrator
    transition: np.ndarray
    ensemble_proba: np.ndarray
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EnsembleTrainer:
    def __init__(self, settings: EnsembleSettings | None = None) -> None:
        self.settings = settings or EnsembleSettings.default()

    def build_members(self, **kwargs: Any) -> list[EnsembleMember]:
        return EnsembleRegistry(self.settings).create_members(**kwargs)

    def fit(
        self,
        frame: pl.DataFrame,
        feature_columns: list[str] | None = None,
        *,
        members: list[EnsembleMember] | None = None,
        true_states: np.ndarray | None = None,
        parallel: bool = True,
    ) -> TrainResult:
        s = self.settings
        mems = members or self.build_members()
        mems = fit_members(mems, frame, feature_columns, parallel=parallel)
        active = [m for m in mems if m.metadata.get("fitted", m.model.is_fitted)]
        if not active:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("All ensemble members failed to fit", code="ENS_FIT_FAIL")

        n = s.n_states
        # validation split for weighting / calibration
        t_len = frame.height
        val_n = max(1, int(t_len * s.training.validation_fraction))
        train_end = max(1, t_len - val_n)
        val_frame = frame.slice(train_end, val_n) if t_len > val_n else frame

        mapped, hards, names = predict_members(
            active, val_frame, feature_columns, n_canonical=n, parallel=parallel
        )
        # align lengths defensively
        t_min = min(p.shape[0] for p in mapped)
        mapped = [p[:t_min] for p in mapped]
        hards = [h[:t_min] for h in hards]
        name_to_member = {m.name: m for m in active}
        ordered = [name_to_member[nm] for nm in names]

        # scores for weighting
        pred_dict = {nm: h for nm, h in zip(names, hards, strict=False)}
        proba_dict = {nm: p for nm, p in zip(names, mapped, strict=False)}
        ll = member_log_likelihoods(ordered, val_frame, feature_columns)
        ece: dict[str, float] = {}
        truth = true_states
        if truth is not None:
            truth = np.asarray(truth, dtype=np.int64).reshape(-1)
            truth_val = truth[-val_frame.height :] if truth.size >= val_frame.height else truth
        else:
            # pseudo-labels from soft majority
            stack = np.mean(np.stack(mapped, axis=0), axis=0)
            truth_val = np.argmax(stack, axis=1)
        for nm, p in proba_dict.items():
            y = truth_val[: p.shape[0]]
            ece[nm] = expected_calibration_error(p, y)

        weights = compute_weights(
            s.weighting.method,  # type: ignore[arg-type]
            names=names,
            predictions=pred_dict,
            truth=truth_val,
            log_likes=ll,
            ece_scores=ece,
            proba_histories=proba_dict,
            user=s.weighting.user_weights,
            lookback=s.weighting.lookback,
            adaptive_rate=s.weighting.adaptive_rate,
            min_weight=s.weighting.min_weight,
        )
        for m, w in zip(ordered, weights, strict=False):
            m.weight = float(w)

        # full-frame predictions for ensemble output
        mapped_full, _, names_full = predict_members(
            ordered, frame, feature_columns, n_canonical=n, parallel=parallel
        )
        t_full = min(p.shape[0] for p in mapped_full)
        mapped_full = [p[:t_full] for p in mapped_full]
        # align weights to names_full
        w_map = {nm: float(w) for nm, w in zip(names, weights, strict=False)}
        w_full = np.asarray([w_map.get(nm, 1.0 / len(names_full)) for nm in names_full])
        w_full = w_full / max(float(w_full.sum()), 1e-300)

        log_ev = np.asarray([ll.get(nm, 0.0) for nm in names_full], dtype=np.float64)
        ens = combine(
            mapped_full,
            w_full,
            method=s.combination.method,  # type: ignore[arg-type]
            n_states=n,
            log_evidence=log_ev,
            meta_weights=w_full,
            scores=w_full,
        )

        calibrator = Calibrator(
            method=s.calibration.method if s.calibration.enabled else "none",  # type: ignore[arg-type]
            temperature=s.calibration.temperature,
        )
        if s.calibration.enabled and s.calibration.method != "none":
            y_fit = truth_val[: ens.shape[0]] if truth_val.size else np.argmax(ens, axis=1)
            # calibrate on validation mapped combine
            ens_val = combine(
                mapped,
                weights,
                method=s.combination.method,  # type: ignore[arg-type]
                n_states=n,
                log_evidence=np.asarray([ll.get(nm, 0.0) for nm in names]),
                meta_weights=weights,
                scores=weights,
            )
            calibrator.fit(ens_val, y_fit[: ens_val.shape[0]])
            ens = calibrator.transform(ens)

        transition = collect_transition(ordered, n)
        history = [
            {
                "weights": {nm: float(w) for nm, w in zip(names, weights, strict=False)},
                "ece": ece,
                "log_likelihood": ll,
            }
        ]
        return TrainResult(
            members=ordered,
            weights=w_full,
            calibrator=calibrator,
            transition=transition,
            ensemble_proba=ens,
            history=history,
            metadata={"names": names_full, "method": s.combination.method},
        )

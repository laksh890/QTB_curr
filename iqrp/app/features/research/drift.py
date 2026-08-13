"""Population, concept, covariate, and distribution drift detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.features.research._numeric import (
    information_coefficient,
    ks_statistic,
    population_stability_index,
)
from iqrp.app.features.research.config import ResearchSettings
from iqrp.app.features.research.targets import build_targets


@dataclass
class DriftAlert:
    feature: str
    drift_type: str
    metric: str
    value: float
    threshold: float
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureDriftReport:
    feature: str
    population_drift_psi: float
    covariate_drift_ks: float
    distribution_drift_mean_z: float
    concept_drift_ic_ratio: float
    alerts: list[DriftAlert] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["alerts"] = [a.to_dict() for a in self.alerts]
        return d


class DriftDetector:
    def __init__(self, settings: ResearchSettings | None = None) -> None:
        self.settings = settings or ResearchSettings.default()

    def detect(self, frame: pl.DataFrame, columns: list[str]) -> dict[str, FeatureDriftReport]:
        cfg = self.settings.drift
        n = frame.height
        split = max(10, int(n * cfg.reference_fraction))
        targets = build_targets(frame, self.settings)
        y = targets["future_return"].cast(pl.Float64).to_numpy()
        out: dict[str, FeatureDriftReport] = {}
        for name in columns:
            x = frame[name].cast(pl.Float64).to_numpy()
            ref, cur = x[:split], x[split:]
            y_ref, y_cur = y[:split], y[split:]
            psi = population_stability_index(ref, cur, bins=cfg.psi_bins)
            ks = ks_statistic(ref, cur)
            ref_finite = ref[np.isfinite(ref)]
            cur_finite = cur[np.isfinite(cur)]
            ref_mu = float(np.mean(ref_finite)) if ref_finite.size else float("nan")
            ref_sd = float(np.std(ref_finite)) if ref_finite.size else float("nan")
            cur_mu = float(np.mean(cur_finite)) if cur_finite.size else float("nan")
            mean_z = (
                float(abs(cur_mu - ref_mu) / (ref_sd + 1e-12))
                if np.isfinite(ref_sd)
                else float("nan")
            )
            ic_ref = information_coefficient(ref, y_ref)
            ic_cur = information_coefficient(cur, y_cur)
            if np.isfinite(ic_ref) and abs(ic_ref) > 1e-9:
                concept = float(abs(ic_cur) / abs(ic_ref))
            else:
                concept = float("nan")

            alerts: list[DriftAlert] = []
            if np.isfinite(psi) and psi >= cfg.psi_alert_threshold:
                alerts.append(
                    DriftAlert(
                        feature=name,
                        drift_type="population",
                        metric="psi",
                        value=float(psi),
                        threshold=cfg.psi_alert_threshold,
                        severity="high" if psi >= 2 * cfg.psi_alert_threshold else "medium",
                        message=f"Population drift PSI={psi:.3f}",
                    )
                )
            if np.isfinite(ks) and ks >= cfg.ks_alert_threshold:
                alerts.append(
                    DriftAlert(
                        feature=name,
                        drift_type="covariate",
                        metric="ks",
                        value=float(ks),
                        threshold=cfg.ks_alert_threshold,
                        severity="high" if ks >= 2 * cfg.ks_alert_threshold else "medium",
                        message=f"Covariate drift KS={ks:.3f}",
                    )
                )
            if np.isfinite(mean_z) and mean_z >= cfg.mean_shift_z_threshold:
                alerts.append(
                    DriftAlert(
                        feature=name,
                        drift_type="distribution",
                        metric="mean_z",
                        value=mean_z,
                        threshold=cfg.mean_shift_z_threshold,
                        severity="medium",
                        message=f"Distribution mean shift z={mean_z:.2f}",
                    )
                )
            if np.isfinite(concept) and concept <= cfg.concept_ic_drop_threshold:
                alerts.append(
                    DriftAlert(
                        feature=name,
                        drift_type="concept",
                        metric="ic_ratio",
                        value=concept,
                        threshold=cfg.concept_ic_drop_threshold,
                        severity="high",
                        message=f"Concept drift IC ratio={concept:.3f}",
                    )
                )

            out[name] = FeatureDriftReport(
                feature=name,
                population_drift_psi=float(psi) if np.isfinite(psi) else float("nan"),
                covariate_drift_ks=float(ks) if np.isfinite(ks) else float("nan"),
                distribution_drift_mean_z=mean_z,
                concept_drift_ic_ratio=concept,
                alerts=alerts,
            )
        return out

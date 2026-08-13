"""Institutional Feature Validation orchestrator — scoring, ranking, evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from iqrp.app.features.research._numeric import information_coefficient
from iqrp.app.features.research.cache import ResearchCache
from iqrp.app.features.research.config import ResearchSettings
from iqrp.app.features.research.correlation import CorrelationAnalyzer, CorrelationReport
from iqrp.app.features.research.drift import DriftDetector, FeatureDriftReport
from iqrp.app.features.research.feature_statistics import FeatureStatisticsEngine, FeatureStats
from iqrp.app.features.research.importance import ImportanceAnalyzer, ImportanceReport
from iqrp.app.features.research.predictive_power import (
    FeaturePredictiveReport,
    PredictivePowerEngine,
)
from iqrp.app.features.research.redundancy import RedundancyDetector, RedundancyReport
from iqrp.app.features.research.reports import ReportWriter, ResearchReportDocument
from iqrp.app.features.research.stability import FeatureStabilityReport, StabilityAnalyzer
from iqrp.app.features.research.targets import build_targets, select_feature_columns
from iqrp.app.features.research.visualization import ResearchVisualizer, chart_manifest


@dataclass
class FeatureScore:
    feature: str
    score: float
    predictive_power: float
    stability: float
    redundancy_penalty: float
    computational_cost: float
    interpretability: float
    consistency_assets: float
    consistency_timeframes: float
    decision: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "score": self.score,
            "predictive_power": self.predictive_power,
            "stability": self.stability,
            "redundancy_penalty": self.redundancy_penalty,
            "computational_cost": self.computational_cost,
            "interpretability": self.interpretability,
            "consistency_assets": self.consistency_assets,
            "consistency_timeframes": self.consistency_timeframes,
            "decision": self.decision,
            "reason": self.reason,
        }


@dataclass
class FeatureResearchResult:
    settings: ResearchSettings
    columns: list[str]
    statistics: list[FeatureStats]
    correlation: CorrelationReport
    redundancy: RedundancyReport
    predictive: dict[str, FeaturePredictiveReport]
    stability: dict[str, FeatureStabilityReport]
    drift: dict[str, FeatureDriftReport]
    importance: ImportanceReport
    scores: list[FeatureScore]
    rankings: dict[str, list[Any]]
    report_paths: dict[str, Path] = field(default_factory=dict)
    chart_paths: dict[str, Path] = field(default_factory=dict)

    def accepted(self) -> list[FeatureScore]:
        return [s for s in self.scores if s.decision == "accept"]

    def rejected(self) -> list[FeatureScore]:
        return [s for s in self.scores if s.decision == "reject"]

    def weak(self) -> list[FeatureScore]:
        return [s for s in self.scores if s.decision == "weak"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "statistics": [s.to_dict() for s in self.statistics],
            "correlation": self.correlation.to_dict(),
            "redundancy": self.redundancy.to_dict(),
            "predictive": {k: v.to_dict() for k, v in self.predictive.items()},
            "stability": {k: v.to_dict() for k, v in self.stability.items()},
            "drift": {k: v.to_dict() for k, v in self.drift.items()},
            "importance": self.importance.to_dict(),
            "scores": [s.to_dict() for s in self.scores],
            "rankings": self.rankings,
            "report_paths": {k: str(v) for k, v in self.report_paths.items()},
            "chart_paths": {k: str(v) for k, v in self.chart_paths.items()},
        }


class FeatureResearchValidator:
    """End-to-end institutional feature validation engine."""

    def __init__(
        self,
        settings: ResearchSettings | None = None,
        *,
        cache: ResearchCache | None = None,
    ) -> None:
        self.settings = settings or ResearchSettings.default()
        self.cache = cache or ResearchCache(
            Path(self.settings.cache_dir) if self.settings.cache_enabled else None,
            enabled=self.settings.cache_enabled,
        )
        self.stats_engine = FeatureStatisticsEngine(self.settings)
        self.corr_engine = CorrelationAnalyzer(self.settings)
        self.redundancy_engine = RedundancyDetector(self.settings)
        self.predictive_engine = PredictivePowerEngine(self.settings)
        self.stability_engine = StabilityAnalyzer(self.settings)
        self.drift_engine = DriftDetector(self.settings)
        self.importance_engine = ImportanceAnalyzer(self.settings)
        self.visualizer = ResearchVisualizer(self.settings)
        self.report_writer = ReportWriter()

    def validate(
        self,
        frame: pl.DataFrame,
        columns: list[str] | None = None,
        *,
        asset_consistency: dict[str, float] | None = None,
        timeframe_consistency: dict[str, float] | None = None,
        write_reports: bool = True,
    ) -> FeatureResearchResult:
        if not self.settings.enabled:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Feature research engine disabled by configuration",
                code="RESEARCH_DISABLED",
            )
        cols = columns or select_feature_columns(frame, self.settings)
        if not cols:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                "No feature columns available for research validation",
                code="RESEARCH_NO_FEATURES",
            )

        logger.info("feature_research_start n_features={} n_rows={}", len(cols), frame.height)

        statistics = self.stats_engine.compute(frame, cols)
        correlation = self.corr_engine.analyze(frame, cols)
        redundancy = self.redundancy_engine.detect(frame, cols)
        predictive = self.predictive_engine.evaluate(frame, cols)
        stability = self.stability_engine.analyze(frame, cols)
        drift = self.drift_engine.detect(frame, cols)
        importance = self.importance_engine.analyze(frame, cols)

        scores = self._score_features(
            cols,
            predictive=predictive,
            stability=stability,
            redundancy=redundancy,
            statistics=statistics,
            asset_consistency=asset_consistency or {},
            timeframe_consistency=timeframe_consistency or {},
        )
        rankings = self._rankings(scores, correlation, stability, predictive, redundancy)

        chart_paths: dict[str, Path] = {}
        report_paths: dict[str, Path] = {}
        output_dir = Path(self.settings.output_dir)
        if write_reports:
            chart_paths = self._charts(
                output_dir / "charts",
                frame,
                cols,
                correlation,
                predictive,
                drift,
                importance,
                stability,
            )
            document = self._document(
                scores, rankings, statistics, correlation, redundancy, chart_paths
            )
            report_paths = self.report_writer.write(
                document,
                output_dir,
                write_markdown=self.settings.reports.write_markdown,
                write_json=self.settings.reports.write_json,
            )

        logger.info(
            "feature_research_done accepted={} rejected={} weak={}",
            sum(1 for s in scores if s.decision == "accept"),
            sum(1 for s in scores if s.decision == "reject"),
            sum(1 for s in scores if s.decision == "weak"),
        )
        return FeatureResearchResult(
            settings=self.settings,
            columns=cols,
            statistics=statistics,
            correlation=correlation,
            redundancy=redundancy,
            predictive=predictive,
            stability=stability,
            drift=drift,
            importance=importance,
            scores=scores,
            rankings=rankings,
            report_paths=report_paths,
            chart_paths=chart_paths,
        )

    def _score_features(
        self,
        columns: list[str],
        *,
        predictive: dict[str, FeaturePredictiveReport],
        stability: dict[str, FeatureStabilityReport],
        redundancy: RedundancyReport,
        statistics: list[FeatureStats],
        asset_consistency: dict[str, float],
        timeframe_consistency: dict[str, float],
    ) -> list[FeatureScore]:
        w = self.settings.scoring
        stats_map = {s.name: s for s in statistics}
        removal = set(redundancy.suggested_removals)
        vif = redundancy.vif
        scores: list[FeatureScore] = []

        # Normalize predictive / stability across universe for 0-100 components
        abs_ics = [
            abs(predictive[c].mean_abs_ic)
            for c in columns
            if c in predictive and np.isfinite(predictive[c].mean_abs_ic)
        ]
        max_ic = max(abs_ics) if abs_ics else 1.0
        max_ic = max(max_ic, 1e-9)

        for name in columns:
            pred = predictive.get(name)
            stab = stability.get(name)
            st = stats_map.get(name)

            pred_comp = 0.0
            if pred is not None and np.isfinite(pred.mean_abs_ic):
                pred_comp = 100.0 * abs(pred.mean_abs_ic) / max_ic
                if np.isfinite(pred.mean_auc):
                    pred_comp = 0.7 * pred_comp + 0.3 * 100.0 * float(np.clip(pred.mean_auc, 0, 1))

            stab_comp = stab.stability_score if stab is not None else 50.0
            redund_pen = 0.0
            if name in removal:
                redund_pen += 60.0
            if name in vif and np.isfinite(vif[name]):
                redund_pen += min(40.0, float(vif[name]))
            redund_pen = min(100.0, redund_pen)

            # Computational cost proxy: missingness + unique cardinality entropy
            cost = 10.0
            if st is not None:
                cost += min(40.0, st.missing_pct)
                cost += min(30.0, st.infinite_pct)
                if st.distribution_type in {"constant", "empty"}:
                    cost += 40.0
            cost = min(100.0, cost)

            # Interpretability: prefer simple-looking names / low kurtosis extremes
            interp = 70.0
            if st is not None:
                if st.distribution_type == "normal_like":
                    interp += 20.0
                kurt = abs(st.kurtosis) if np.isfinite(st.kurtosis) else 0.0
                if kurt > 10:
                    interp -= 25.0
            interp = float(np.clip(interp, 0, 100))

            cons_a = float(asset_consistency.get(name, 50.0))
            cons_t = float(timeframe_consistency.get(name, 50.0))

            score = (
                w.weight_predictive_power * pred_comp
                + w.weight_stability * stab_comp
                + w.weight_interpretability * interp
                + w.weight_consistency_assets * cons_a
                + w.weight_consistency_timeframes * cons_t
                - w.weight_redundancy_penalty * redund_pen
                - w.weight_computational_cost * cost
            )
            # Rescale roughly into 0-100
            score = float(np.clip(score, 0.0, 100.0))

            if score >= w.accept_score_threshold and name not in removal:
                decision = "accept"
                reason = "Meets predictive/stability thresholds with acceptable redundancy"
            elif score <= w.reject_score_threshold or (
                st is not None and st.distribution_type in {"constant", "empty"}
            ):
                decision = "reject"
                reason = "Insufficient predictive evidence, constant, or severe redundancy"
            elif score < w.weak_score_threshold or name in removal:
                decision = "weak"
                reason = "Marginal score or marked redundant; needs more evidence"
            else:
                decision = "weak"
                reason = "Below accept threshold"

            # Augment reason with quantitative snippets
            ic_txt = (
                f"IC={pred.mean_abs_ic:.4f}" if pred and np.isfinite(pred.mean_abs_ic) else "IC=n/a"
            )
            reason = f"{reason} ({ic_txt}, score={score:.1f})"

            scores.append(
                FeatureScore(
                    feature=name,
                    score=score,
                    predictive_power=float(pred_comp),
                    stability=float(stab_comp),
                    redundancy_penalty=float(redund_pen),
                    computational_cost=float(cost),
                    interpretability=float(interp),
                    consistency_assets=cons_a,
                    consistency_timeframes=cons_t,
                    decision=decision,
                    reason=reason,
                )
            )
        scores.sort(key=lambda s: s.score, reverse=True)
        return scores

    def _rankings(
        self,
        scores: list[FeatureScore],
        correlation: CorrelationReport,
        stability: dict[str, FeatureStabilityReport],
        predictive: dict[str, FeaturePredictiveReport],
        redundancy: RedundancyReport,
    ) -> dict[str, list[Any]]:
        top = [s.to_dict() for s in scores[:20]]
        weak = [s.to_dict() for s in scores if s.decision == "weak"]
        remove = [{"feature": f, "reason": "redundancy"} for f in redundancy.suggested_removals]
        remove += [s.to_dict() for s in scores if s.decision == "reject"]
        stable_pairs = sorted(
            ((k, v.stability_score) for k, v in stability.items()),
            key=lambda t: t[1],
            reverse=True,
        )[:20]
        pred_pairs = sorted(
            ((k, v.mean_abs_ic) for k, v in predictive.items() if np.isfinite(v.mean_abs_ic)),
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:20]
        return {
            "top_features": top,
            "weak_features": weak,
            "features_to_remove": remove,
            "highly_correlated_groups": correlation.high_correlation_groups,
            "most_stable_features": [{"feature": k, "score": s} for k, s in stable_pairs],
            "most_predictive_features": [{"feature": k, "value": v} for k, v in pred_pairs],
        }

    def _charts(
        self,
        chart_dir: Path,
        frame: pl.DataFrame,
        columns: list[str],
        correlation: CorrelationReport,
        predictive: dict[str, FeaturePredictiveReport],
        drift: dict[str, FeatureDriftReport],
        importance: ImportanceReport,
        stability: dict[str, FeatureStabilityReport],
    ) -> dict[str, Path]:
        if not self.settings.reports.include_charts:
            return {}
        rolling_ic = {
            name: [rep.by_target["future_return"].information_coefficient]
            for name, rep in predictive.items()
            if "future_return" in rep.by_target
        }
        # Expand rolling IC series from stability analyzer proxy: repeat mean with noise-free
        # single-point; visualizer handles short series. For richer charts, compute mini series.
        richer: dict[str, list[float]] = {}
        w = self.settings.stability.rolling_window
        step = self.settings.stability.step
        y = build_targets(frame, self.settings)["future_return"].to_numpy()
        for name in columns[: self.settings.visualization.max_features_in_charts]:
            x = frame[name].cast(pl.Float64).to_numpy()
            series = [
                information_coefficient(x[start : start + w], y[start : start + w])
                for start in range(0, max(0, len(x) - w + 1), step)
            ]
            richer[name] = series if series else rolling_ic.get(name, [])

        distributions = {
            c: frame[c].cast(pl.Float64).to_numpy()
            for c in columns[: self.settings.visualization.max_features_in_charts]
        }
        return self.visualizer.write_all(
            chart_dir,
            corr_pearson=correlation.pearson,
            rolling_ic=richer,
            distributions=distributions,
            drift_psi={k: v.population_drift_psi for k, v in drift.items()},
            mi_ranking=[
                (k, float(v.by_target["future_return"].mutual_information))
                for k, v in predictive.items()
                if "future_return" in v.by_target
                and np.isfinite(v.by_target["future_return"].mutual_information)
            ],
            importance=importance.permutation,
            stability={k: v.stability_score for k, v in stability.items()},
        )

    def _document(
        self,
        scores: list[FeatureScore],
        rankings: dict[str, list[Any]],
        statistics: list[FeatureStats],
        correlation: CorrelationReport,
        redundancy: RedundancyReport,
        chart_paths: dict[str, Path],
    ) -> ResearchReportDocument:
        accepted = [s.to_dict() for s in scores if s.decision == "accept"]
        rejected = [s.to_dict() for s in scores if s.decision == "reject"]
        weak = [s.to_dict() for s in scores if s.decision == "weak"]
        recommendations = [
            "Prefer accepted features with high Rank IC and stability for downstream models.",
            "Remove suggested redundant features before multicollinear estimators.",
            "Re-run validation after regime shifts when drift alerts fire.",
        ]
        if redundancy.suggested_removals:
            recommendations.append(
                "Suggested removals: " + ", ".join(redundancy.suggested_removals[:20])
            )
        return ResearchReportDocument(
            summary={
                "n_features": len(scores),
                "n_accepted": len(accepted),
                "n_rejected": len(rejected),
                "n_weak": len(weak),
                "n_correlated_groups": len(correlation.high_correlation_groups),
                "accept_threshold": self.settings.scoring.accept_score_threshold,
                "reject_threshold": self.settings.scoring.reject_score_threshold,
            },
            statistics=[s.to_dict() for s in statistics],
            rankings=rankings,
            recommendations=recommendations,
            accepted_features=accepted,
            rejected_features=rejected,
            weak_features=weak,
            correlated_groups=correlation.high_correlation_groups,
            charts=chart_manifest(chart_paths),
            reasoning={s.feature: s.reason for s in scores},
        )

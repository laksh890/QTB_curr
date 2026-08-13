"""Alpha Research Engine — institutional pipeline orchestrator.

CRITICAL RULES:
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
- economic_hypothesis required for APPROVED.
- Alpha approval ≠ trading approval (Risk Intelligence is not bypassed).
- Point-in-time: no future leakage in signal helpers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.base.signal_registry import ExperimentRecord, SignalRegistry, get_default_registry
from iqrp.app.alpha.base.signal_result import (
    SignalResearchReport,
    SignalStatistics,
    SignalStatus,
)
from iqrp.app.alpha.backtesting.signal_backtest import signal_backtest
from iqrp.app.alpha.config import AlphaSettings
from iqrp.app.alpha.discovery.candidate_generator import CandidateGenerator
from iqrp.app.alpha.economics.capacity import estimate_capacity
from iqrp.app.alpha.ensemble.correlation import signal_correlation_matrix
from iqrp.app.alpha.ensemble.redundancy import redundancy_report
from iqrp.app.alpha.ranking import rank_candidates
from iqrp.app.alpha.regime.regime_performance import regime_performance
from iqrp.app.alpha.research.decay import analyze_decay
from iqrp.app.alpha.research.decay import forward_returns as build_forward_returns
from iqrp.app.alpha.research.evaluator import SignalEvaluator
from iqrp.app.alpha.serializer import AlphaSerializer
from iqrp.app.alpha.statistical_validation.bootstrap import iid_bootstrap_ci
from iqrp.app.alpha.statistical_validation.deflated_sharpe import deflated_sharpe_ratio
from iqrp.app.alpha.statistical_validation.multiple_testing import (
    get_experiment_tracker,
    multiple_testing_adjustment,
)
from iqrp.app.alpha.statistical_validation.permutation import permutation_ic_test
from iqrp.app.alpha.statistical_validation.probability_backtest_overfitting import (
    probability_backtest_overfitting,
)
from iqrp.app.alpha.statistical_validation.significance import ic_significance


_SHARPE_ONLY_RE = re.compile(
    r"\b(sharpe|sr|net_sharpe|gross_sharpe)\b",
    re.IGNORECASE,
)
_VALIDATION_EVIDENCE_KEYS = (
    "significance",
    "bootstrap",
    "permutation",
    "multiple_testing",
    "deflated_sharpe",
    "pbo",
    "validate",
    "validation",
    "ic_significance",
    "evaluate",
    "evaluation",
)


def _as_signal_values(signal: AlphaSignal | np.ndarray | Any) -> np.ndarray:
    if isinstance(signal, AlphaSignal):
        return np.asarray(signal.values, dtype=np.float64)
    return np.asarray(signal, dtype=np.float64).reshape(-1)


def _report_to_eval_dict(report: SignalResearchReport) -> dict[str, Any]:
    out = report.to_dict()
    ic = float("nan")
    if report.performance is not None:
        ic = float(report.performance.ic_mean)
    out["ic"] = ic
    out["ic_mean"] = ic
    return out


class ApprovalError(ValueError):
    """Raised when approve() refuses a promotion."""


class AlphaResearchEngine:
    """End-to-end alpha research orchestrator (discovery → lifecycle)."""

    def __init__(
        self,
        settings: AlphaSettings | None = None,
        registry: SignalRegistry | None = None,
    ) -> None:
        self.settings = settings or AlphaSettings.default()
        # Identity check: empty SignalRegistry is falsy via __len__.
        self.registry = registry if registry is not None else get_default_registry()
        self.serializer = AlphaSerializer()
        self._evaluator = SignalEvaluator(
            horizons=self.settings.research.horizons,
            stability_window=self.settings.research.stability_window,
            seasonality_period=self.settings.research.seasonality_period,
        )
        self._store: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def discover(
        self,
        returns: np.ndarray | None = None,
        prices: np.ndarray | None = None,
        features: dict[str, np.ndarray] | None = None,
        forecasts: np.ndarray | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Discover research candidates (NOT approved alpha)."""
        gen = CandidateGenerator(
            registry=self.registry,
            owner=kwargs.pop("owner", self.settings.owner_default),
            universe=kwargs.pop("universe", self.settings.universe_default),
            frequency=kwargs.pop("frequency", self.settings.frequency_default),
            auto_register=kwargs.pop(
                "auto_register", self.settings.discovery.auto_register
            ),
        )
        target = kwargs.pop("target", None)
        if features is not None and target is None and returns is not None:
            target = build_forward_returns(np.asarray(returns, dtype=np.float64), 1)

        result = gen.discover_all(
            returns=None if returns is None else np.asarray(returns, dtype=np.float64),
            features=features,
            target=None if target is None else np.asarray(target, dtype=np.float64),
            volume=kwargs.pop("volume", None),
            prices=None if prices is None else np.asarray(prices, dtype=np.float64),
            alt_series=kwargs.pop("alt_series", None),
            event_mask=kwargs.pop("event_mask", None),
            forecast=None if forecasts is None else np.asarray(forecasts, dtype=np.float64),
            forecast_hypothesis=kwargs.pop("forecast_hypothesis", None),
        )
        candidates: list[dict[str, Any]] = []
        for i, sig in enumerate(result.signals):
            definition = (
                result.definitions[i]
                if i < len(result.definitions)
                else None
            )
            eid = result.experiment_ids[i] if i < len(result.experiment_ids) else None
            candidates.append(
                {
                    "name": sig.name,
                    "signal": sig,
                    "values": sig.values,
                    "definition": None if definition is None else definition.to_dict(),
                    "experiment_id": eid,
                    "claims_profitability": False,
                    "notes": list(result.notes),
                }
            )
        return candidates

    # ------------------------------------------------------------------
    # Registry / lifecycle
    # ------------------------------------------------------------------
    def register(
        self,
        definition: SignalDefinition,
        signal: AlphaSignal | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ExperimentRecord:
        alpha_sig: AlphaSignal | None = None
        if isinstance(signal, AlphaSignal):
            alpha_sig = signal
        elif signal is not None:
            alpha_sig = AlphaSignal(
                values=np.asarray(signal, dtype=np.float64),
                name=definition.name,
                definition_id=definition.definition_id,
                metadata={"definition": definition.to_dict()},
            )
        return self.registry.register(
            definition,
            signal=alpha_sig,
            status=kwargs.pop("status", SignalStatus.CANDIDATE),
            experiment_id=kwargs.pop("experiment_id", None),
            tags=kwargs.pop("tags", ()),
            actor=kwargs.pop("actor", "system"),
            reason=kwargs.pop("reason", "engine.register"),
            metadata=kwargs.pop("metadata", None),
        )

    def approve(
        self,
        experiment_id: str,
        *,
        reason: str,
        actor: str = "researcher",
        require_hypothesis: bool = True,
    ) -> ExperimentRecord:
        """Promote to APPROVED with hard governance gates.

        MUST refuse if no economic_hypothesis.
        MUST refuse if approval is based only on Sharpe when
        ``settings.scoring.allow_sharpe_only_approval`` is False.
        Prefer evaluate + validate evidence on the experiment report.
        Alpha approval ≠ trading approval — Risk Intelligence is not bypassed.
        """
        record = self.registry.get(experiment_id)
        hyp = (record.definition.economic_hypothesis or "").strip()
        min_chars = int(self.settings.scoring.min_hypothesis_chars)

        if require_hypothesis:
            if not hyp:
                raise ApprovalError(
                    "Cannot approve without economic_hypothesis. "
                    "Statistical significance alone ≠ alpha. "
                    "Historical Sharpe alone cannot approve."
                )
            if len(hyp) < min_chars:
                raise ApprovalError(
                    f"economic_hypothesis too thin ({len(hyp)} < {min_chars} chars). "
                    "Substantive economic rationale required for APPROVED."
                )

        if not self.settings.scoring.allow_sharpe_only_approval:
            if self._is_sharpe_only_approval(reason, record):
                raise ApprovalError(
                    "Approval refused: Historical Sharpe alone cannot approve. "
                    "Provide validation evidence (IC significance / bootstrap / "
                    "permutation / DSR / PBO) beyond Sharpe."
                )

        if not self._has_validation_evidence(record):
            raise ApprovalError(
                "Approval refused: evaluate+validate evidence required on "
                "experiment report/extras before APPROVED."
            )

        # Advance lifecycle CANDIDATE → … → APPROVED
        path = [
            SignalStatus.RESEARCHING,
            SignalStatus.VALIDATING,
            SignalStatus.PROVISIONAL,
            SignalStatus.APPROVED,
        ]
        note = (
            "Alpha approval ≠ trading approval. "
            "Risk Intelligence is NOT bypassed; trading still requires "
            "independent risk / portfolio gates."
        )
        extras = {
            "approval_reason": reason,
            "risk_intelligence_not_bypassed": True,
            "alpha_approval_is_not_trading_approval": True,
            "note": note,
        }
        current = record.status
        for target in path:
            if current == SignalStatus.APPROVED:
                break
            if current == target:
                continue
            # Skip already-passed states
            order = [
                SignalStatus.CANDIDATE,
                SignalStatus.RESEARCHING,
                SignalStatus.VALIDATING,
                SignalStatus.PROVISIONAL,
                SignalStatus.APPROVED,
            ]
            if current in order and target in order:
                if order.index(target) <= order.index(current):
                    continue
            # Only step one allowed transition at a time
            allowed_next = {
                SignalStatus.CANDIDATE: SignalStatus.RESEARCHING,
                SignalStatus.RESEARCHING: SignalStatus.VALIDATING,
                SignalStatus.VALIDATING: SignalStatus.PROVISIONAL,
                SignalStatus.PROVISIONAL: SignalStatus.APPROVED,
                SignalStatus.DEGRADED: SignalStatus.PROVISIONAL,
            }
            nxt = allowed_next.get(current)
            if nxt is None:
                raise ApprovalError(
                    f"Cannot approve from status {current.value}"
                )
            step_reason = reason if nxt == SignalStatus.APPROVED else f"advance toward APPROVED: {reason}"
            record = self.registry.transition(
                experiment_id,
                nxt,
                reason=step_reason,
                actor=actor,
                extras=extras if nxt == SignalStatus.APPROVED else {"toward": "APPROVED"},
            )
            current = record.status

        # Attach governance note on report
        if record.report is not None:
            warnings = list(record.report.warnings)
            if note not in warnings:
                warnings.append(note)
            record.report.warnings = warnings
            record.report.status = SignalStatus.APPROVED
            self.registry.attach_report(experiment_id, record.report)
        else:
            # Minimal report documenting RI non-bypass
            from iqrp.app.alpha.base.signal_result import SignalScore

            stub = SignalResearchReport(
                signal_name=record.definition.name,
                version=record.definition.version,
                status=SignalStatus.APPROVED,
                economic_hypothesis=hyp,
                diagnostics={"approval_extras": extras},
                warnings=[note],
                score=SignalScore(
                    overall=0.0,
                    predictive=0.0,
                    stability=0.0,
                    persistence=0.0,
                    economic_hypothesis_score=min(100.0, len(hyp) / 1.2),
                    notes="Approved with governance; RI not bypassed.",
                ),
            )
            self.registry.attach_report(experiment_id, stub)
        return self.registry.get(experiment_id)

    def degrade(
        self,
        experiment_id: str,
        reason: str = "performance degradation",
        *,
        actor: str = "system",
    ) -> ExperimentRecord:
        record = self.registry.get(experiment_id)
        if record.status == SignalStatus.DEGRADED:
            return record
        # DEGRADED allowed from PROVISIONAL / APPROVED; otherwise route via PROVISIONAL
        if record.status in {
            SignalStatus.CANDIDATE,
            SignalStatus.RESEARCHING,
            SignalStatus.VALIDATING,
        }:
            # Reject research path rather than degrade
            return self.registry.transition(
                experiment_id,
                SignalStatus.REJECTED,
                reason=f"degrade requested pre-approval: {reason}",
                actor=actor,
            )
        return self.registry.transition(
            experiment_id,
            SignalStatus.DEGRADED,
            reason=reason,
            actor=actor,
        )

    def retire(
        self,
        experiment_id: str,
        reason: str = "retired",
        *,
        actor: str = "system",
    ) -> ExperimentRecord:
        record = self.registry.get(experiment_id)
        if record.status == SignalStatus.RETIRED:
            return record
        # RETIRED allowed from most non-terminal; REJECTED is terminal
        if record.status == SignalStatus.REJECTED:
            raise ApprovalError("Cannot retire a REJECTED experiment")
        # May need intermediate step from CANDIDATE/RESEARCHING/VALIDATING
        if record.status in {
            SignalStatus.CANDIDATE,
            SignalStatus.RESEARCHING,
            SignalStatus.VALIDATING,
            SignalStatus.PROVISIONAL,
            SignalStatus.APPROVED,
            SignalStatus.DEGRADED,
        }:
            # Direct retire is allowed from these per validate_transition
            return self.registry.transition(
                experiment_id,
                SignalStatus.RETIRED,
                reason=reason,
                actor=actor,
            )
        raise ApprovalError(f"Cannot retire from {record.status.value}")

    def research_report(self, experiment_id: str) -> SignalResearchReport:
        record = self.registry.get(experiment_id)
        if record.report is not None:
            return record.report
        if record.signal is None:
            raise KeyError(
                f"No report or signal attached for experiment {experiment_id}"
            )
        # Cannot evaluate without returns — return stub report
        return SignalResearchReport(
            signal_name=record.definition.name,
            version=record.definition.version,
            status=record.status,
            economic_hypothesis=record.definition.economic_hypothesis,
            warnings=[
                "No attached research report; run evaluate/validate and attach.",
                "Statistical significance alone ≠ alpha.",
                "Historical Sharpe alone cannot approve.",
                "Alpha approval ≠ trading approval (Risk Intelligence not bypassed).",
            ],
        )

    # ------------------------------------------------------------------
    # Research analytics
    # ------------------------------------------------------------------
    def evaluate(
        self,
        signal: AlphaSignal | np.ndarray,
        forward_returns: np.ndarray,
        **kwargs: Any,
    ) -> SignalStatistics | dict[str, Any]:
        """Evaluate predictive power; returns report dict with IC fields."""
        definition = kwargs.pop("definition", None)
        report = self._evaluator.evaluate(
            signal,
            np.asarray(forward_returns, dtype=np.float64),
            definition=definition,
            status=kwargs.pop("status", SignalStatus.RESEARCHING),
        )
        experiment_id = kwargs.pop("experiment_id", None)
        if experiment_id is not None:
            self.registry.attach_report(experiment_id, report)
            # stash evaluate marker
            report.diagnostics = dict(report.diagnostics)
            report.diagnostics["evaluate"] = True
            self.registry.attach_report(experiment_id, report)
        return _report_to_eval_dict(report)

    def validate(
        self,
        signal: AlphaSignal | np.ndarray,
        forward_returns: np.ndarray,
        *,
        n_trials: int = 20,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Statistical validation: significance, bootstrap, perm, MT, DSR, PBO."""
        sig = _as_signal_values(signal)
        ret = np.asarray(forward_returns, dtype=np.float64).reshape(-1)
        n = min(sig.size, ret.size)
        sig, ret = sig[:n], ret[:n]
        # Use period returns as labels via 1-bar forward when needed
        fwd = kwargs.pop("forward", None)
        if fwd is None:
            # Treat input as period returns → build 1-step forward target
            # unless caller sets returns_are_forward=True
            if kwargs.pop("returns_are_forward", False):
                fwd = ret
            else:
                fwd = build_forward_returns(ret, 1)

        seed = kwargs.pop("seed", self.settings.seed)
        n_boot = int(kwargs.pop("n_boot", max(50, min(200, int(n_trials) * 5))))
        n_perm = int(kwargs.pop("n_perm", max(50, min(200, int(n_trials) * 5))))

        significance = ic_significance(sig, fwd)
        bootstrap = iid_bootstrap_ci(
            sig, fwd, stat="ic", n_boot=n_boot, seed=seed
        )
        permutation = permutation_ic_test(
            sig, fwd, n_perm=n_perm, seed=seed
        )

        pvals = [float(significance.get("pvalue", float("nan")))]
        # Pad with nulls to reflect trial budget for MT demo
        if int(n_trials) > 1:
            pvals = pvals + [0.5] * (int(n_trials) - 1)
        mt = multiple_testing_adjustment(
            pvals,
            method=kwargs.pop("mt_method", "fdr_bh"),
            tracker=get_experiment_tracker(),
            label=kwargs.pop("label", "alpha_validate"),
        )
        # Convert numpy arrays for JSON-friendliness
        mt_out = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in mt.items()
        }

        # Deflated Sharpe from signed signal * forward return
        m = np.isfinite(sig) & np.isfinite(fwd)
        pnl = np.sign(sig[m]) * fwd[m] if m.sum() else np.asarray([], dtype=np.float64)
        if pnl.size > 2:
            mu, sd = float(np.mean(pnl)), float(np.std(pnl, ddof=1))
            obs_sr = mu / (sd + 1e-12) if sd > 0 else 0.0
            skew = float(0.0)
            kurt = float(3.0)
            try:
                from scipy import stats as sp_stats  # type: ignore[import-untyped]

                if pnl.size > 3:
                    skew = float(sp_stats.skew(pnl))
                    kurt = float(sp_stats.kurtosis(pnl) + 3.0)
            except Exception:  # noqa: BLE001
                pass
            dsr = deflated_sharpe_ratio(
                obs_sr,
                n_trials=int(n_trials),
                n_obs=int(pnl.size),
                skew=skew,
                kurtosis=kurt,
                return_details=True,
            )
        else:
            dsr = {"dsr": float("nan"), "n_obs": int(pnl.size)}

        # Strategy returns for PBO
        bt = signal_backtest(sig, ret, cost_bps=0.0, returns_are_forward=False)
        pbo = probability_backtest_overfitting(
            bt.get("net_returns", pnl),
            n_groups=min(8, max(4, n // 50 * 2)),
        )

        out = {
            "significance": significance,
            "bootstrap": bootstrap,
            "permutation": permutation,
            "multiple_testing": mt_out,
            "deflated_sharpe": dsr,
            "pbo": pbo,
            "n_trials": int(n_trials),
            "disclaimer": (
                "Validation diagnostics inform research. "
                "Statistical significance alone ≠ alpha. "
                "Historical Sharpe alone cannot approve."
            ),
        }
        experiment_id = kwargs.pop("experiment_id", None)
        if experiment_id is not None:
            rec = self.registry.get(experiment_id)
            report = rec.report
            if report is None:
                report = SignalResearchReport(
                    signal_name=rec.definition.name,
                    version=rec.definition.version,
                    status=SignalStatus.VALIDATING,
                    economic_hypothesis=rec.definition.economic_hypothesis,
                )
            report.diagnostics = dict(report.diagnostics)
            report.diagnostics["validation"] = out
            report.diagnostics["validate"] = True
            self.registry.attach_report(experiment_id, report)
            if rec.status in {SignalStatus.CANDIDATE, SignalStatus.RESEARCHING}:
                try:
                    self.registry.transition(
                        experiment_id,
                        SignalStatus.VALIDATING
                        if rec.status == SignalStatus.RESEARCHING
                        else SignalStatus.RESEARCHING,
                        reason="engine.validate attached evidence",
                        actor="system",
                    )
                except ValueError:
                    pass
        return out

    def backtest(
        self,
        signal: AlphaSignal | np.ndarray,
        returns: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        sig = _as_signal_values(signal)
        ret = np.asarray(returns, dtype=np.float64)
        cost_bps = float(kwargs.pop("cost_bps", 0.0))
        mode = kwargs.pop("mode", "long_short")
        returns_are_forward = bool(kwargs.pop("returns_are_forward", False))
        result = signal_backtest(
            sig,
            ret,
            cost_bps=cost_bps,
            mode=mode,
            returns_are_forward=returns_are_forward,
            **{k: v for k, v in kwargs.items() if k in {"periods_per_year", "weights"}},
        )
        # JSON-friendly scalars preferred by callers
        out = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in result.items()
        }
        out["net_sharpe"] = result.get("net_sharpe")
        out["gross_sharpe"] = result.get("gross_sharpe")
        return out

    def stress_test(
        self,
        signal: AlphaSignal | np.ndarray,
        returns: np.ndarray,
        regimes: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        sig = _as_signal_values(signal)
        ret = np.asarray(returns, dtype=np.float64).reshape(-1)
        n = min(sig.size, ret.size)
        sig, ret = sig[:n], ret[:n]
        base = self.backtest(sig, ret, **kwargs)
        shocks = {
            "vol_up_2x": ret * 2.0,
            "vol_down_0_5x": ret * 0.5,
            "sign_flip": -ret,
        }
        shocked: dict[str, Any] = {}
        for name, shocked_ret in shocks.items():
            bt = self.backtest(sig, shocked_ret, **kwargs)
            shocked[name] = {
                "net_sharpe": bt.get("net_sharpe"),
                "net_mean": bt.get("net_mean"),
            }
        regime_block = None
        if regimes is not None:
            fwd = build_forward_returns(ret, 1)
            regime_block = regime_performance(sig, fwd, regimes)
        return {
            "baseline": {
                "net_sharpe": base.get("net_sharpe"),
                "gross_sharpe": base.get("gross_sharpe"),
                "avg_turnover": base.get("avg_turnover"),
            },
            "shocks": shocked,
            "regimes": regime_block,
            "disclaimer": (
                "Stress diagnostics only. "
                "Historical Sharpe alone cannot approve."
            ),
        }

    def analyze_decay(
        self,
        signal: AlphaSignal | np.ndarray,
        returns: np.ndarray,
        horizons: tuple[int, ...] | list[int] | None = None,
    ) -> dict[str, Any]:
        sig = _as_signal_values(signal)
        ret = np.asarray(returns, dtype=np.float64)
        hs = horizons or self.settings.research.horizons
        return analyze_decay(sig, ret, horizons=hs)

    def analyze_regimes(
        self,
        signal: AlphaSignal | np.ndarray,
        returns: np.ndarray,
        regimes: np.ndarray | Any,
    ) -> dict[str, Any]:
        sig = _as_signal_values(signal)
        ret = np.asarray(returns, dtype=np.float64)
        fwd = build_forward_returns(ret, 1)
        return regime_performance(sig, fwd, regimes)

    def analyze_capacity(
        self,
        *,
        turnover: float,
        adv: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return estimate_capacity(turnover=turnover, adv=adv, **kwargs)

    def compare(
        self,
        signals: dict[str, Any],
        returns: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Correlation + redundancy across a book of signals."""
        series = {k: _as_signal_values(v) for k, v in signals.items()}
        corr = signal_correlation_matrix(series, kind="prediction")
        red = redundancy_report(series)
        pairwise_ic: dict[str, float] = {}
        if returns is not None:
            ret = np.asarray(returns, dtype=np.float64)
            fwd = build_forward_returns(ret, 1)
            from iqrp.app.alpha.research.information_coefficient import compute_ic

            for name, sig in series.items():
                pairwise_ic[name] = compute_ic(sig, fwd)
        return {
            "correlation": corr,
            "redundancy": red,
            "signal_ic": pairwise_ic,
            "disclaimer": (
                "Comparison is triage only. "
                "Statistical significance alone ≠ alpha."
            ),
        }

    def rank(self, candidates: list[Any]) -> list[dict[str, Any]]:
        return rank_candidates(candidates)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path, obj: Any | None = None) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if obj is None:
            payload = {
                "settings": self.settings.model_dump(),
                "registry": self.registry.to_dict(),
                "store": self._jsonable(self._store),
            }
            p.write_text(json.dumps(self._jsonable(payload), indent=2), encoding="utf-8")
            return p
        if isinstance(obj, SignalDefinition):
            return self.serializer.save_definition(obj, p)
        if isinstance(obj, AlphaSignal):
            return self.serializer.save_signal(obj, p)
        if isinstance(obj, SignalResearchReport):
            return self.serializer.save_report(obj, p)
        p.write_text(
            json.dumps(self._jsonable(obj), indent=2), encoding="utf-8"
        )
        return p

    def load(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "registry" in data:
            self._store = dict(data.get("store") or {})
        return data

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _jsonable(obj: Any) -> Any:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): AlphaResearchEngine._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [AlphaResearchEngine._jsonable(v) for v in obj]
        if hasattr(obj, "model_dump"):
            return AlphaResearchEngine._jsonable(obj.model_dump())
        if hasattr(obj, "to_dict"):
            return AlphaResearchEngine._jsonable(obj.to_dict())
        return str(obj)

    def _is_sharpe_only_approval(self, reason: str, record: ExperimentRecord) -> bool:
        reason_l = (reason or "").strip().lower()
        mentions_sharpe = bool(_SHARPE_ONLY_RE.search(reason_l))
        has_other_reason = any(
            tok in reason_l
            for tok in (
                "hypothesis",
                "economic",
                "validation",
                "bootstrap",
                "permutation",
                "dsr",
                "pbo",
                "ic",
                "capacity",
                "regime",
            )
        )
        if mentions_sharpe and not has_other_reason and not self._has_validation_evidence(record):
            return True
        # Extras that only cite sharpe
        extras: dict[str, Any] = {}
        if record.report is not None:
            extras.update(record.report.diagnostics or {})
        sharpe_keys = {"sharpe", "sharpe_proxy", "net_sharpe", "gross_sharpe"}
        evidence_keys = set(_VALIDATION_EVIDENCE_KEYS)
        present = {k for k in extras if k in sharpe_keys or k in evidence_keys}
        if present and present <= sharpe_keys:
            return True
        if mentions_sharpe and present <= sharpe_keys and not self._has_validation_evidence(record):
            return True
        return False

    def _has_validation_evidence(self, record: ExperimentRecord) -> bool:
        report = record.report
        if report is None:
            return False
        diag = report.diagnostics or {}
        if diag.get("validate") or diag.get("validation") or diag.get("evaluate"):
            return True
        if any(k in diag for k in _VALIDATION_EVIDENCE_KEYS):
            return True
        if report.performance is not None and np.isfinite(report.performance.ic_mean):
            # evaluate evidence present
            if "decay" in (report.performance.extras or {}) or "decay" in diag:
                return True
            if diag:
                return True
        return False

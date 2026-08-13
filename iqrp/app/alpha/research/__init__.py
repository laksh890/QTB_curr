"""Alpha research metrics: IC, decay, stability, evaluation bundles."""

from iqrp.app.alpha.research.decay import analyze_decay, forward_returns
from iqrp.app.alpha.research.evaluator import SignalEvaluator, evaluate_signal
from iqrp.app.alpha.research.hit_rate import compute_hit_rate, hit_rate_summary
from iqrp.app.alpha.research.information_coefficient import compute_ic, ic_summary, rolling_ic
from iqrp.app.alpha.research.persistence import persistence_summary, signal_half_life
from iqrp.app.alpha.research.predictor import SignalPredictor, predict_forward
from iqrp.app.alpha.research.rank_ic import compute_rank_ic, rank_ic_summary
from iqrp.app.alpha.research.seasonality import analyze_seasonality
from iqrp.app.alpha.research.stability import analyze_stability

__all__ = [
    "SignalEvaluator",
    "SignalPredictor",
    "analyze_decay",
    "analyze_seasonality",
    "analyze_stability",
    "compute_hit_rate",
    "compute_ic",
    "compute_rank_ic",
    "evaluate_signal",
    "forward_returns",
    "hit_rate_summary",
    "ic_summary",
    "persistence_summary",
    "predict_forward",
    "rank_ic_summary",
    "rolling_ic",
    "signal_half_life",
]

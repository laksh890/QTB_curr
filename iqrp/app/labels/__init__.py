"""Institutional Label Engineering Platform.

Defines prediction targets for every downstream model. No trading strategies.
No machine learning. Labels only.
"""

from iqrp.app.labels.barrier import (
    TripleBarrierLabel,
    compute_triple_barrier,
    triple_barrier_frame,
)
from iqrp.app.labels.base import (
    Label,
    LabelMeta,
    LabelPipeline,
    LabelRegistry,
    ensure_labels_loaded,
    get_registry,
    register_label,
)
from iqrp.app.labels.config import LabelSettings
from iqrp.app.labels.custom import (
    ensure_custom_examples_registered,
    next_n_period_return,
    probability_of_atr_move,
    probability_of_move,
    register_custom_label,
)
from iqrp.app.labels.meta import meta_label_frame, secondary_confirmation
from iqrp.app.labels.query import (
    LabelQueryService,
    describe_label,
    get_label,
    get_labels,
    list_labels,
)
from iqrp.app.labels.store import LabelStore
from iqrp.app.labels.validation import LabelValidationReport, LabelValidator
from iqrp.app.labels.visualization import LabelVisualizer

ensure_labels_loaded()
ensure_custom_examples_registered()

__all__ = [
    "Label",
    "LabelMeta",
    "LabelPipeline",
    "LabelQueryService",
    "LabelRegistry",
    "LabelSettings",
    "LabelStore",
    "LabelValidationReport",
    "LabelValidator",
    "LabelVisualizer",
    "TripleBarrierLabel",
    "compute_triple_barrier",
    "describe_label",
    "ensure_custom_examples_registered",
    "ensure_labels_loaded",
    "get_label",
    "get_labels",
    "get_registry",
    "list_labels",
    "meta_label_frame",
    "next_n_period_return",
    "probability_of_atr_move",
    "probability_of_move",
    "register_custom_label",
    "register_label",
    "secondary_confirmation",
    "triple_barrier_frame",
]

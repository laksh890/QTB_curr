"""Label base primitives."""

from iqrp.app.labels.base.label import Label, LabelMeta
from iqrp.app.labels.base.pipeline import LabelPipeline, LabelPipelineBenchmarks
from iqrp.app.labels.base.registry import (
    LabelRegistry,
    ensure_labels_loaded,
    get_registry,
    label_factory,
    register_label,
)

__all__ = [
    "Label",
    "LabelMeta",
    "LabelPipeline",
    "LabelPipelineBenchmarks",
    "LabelRegistry",
    "ensure_labels_loaded",
    "get_registry",
    "label_factory",
    "register_label",
]

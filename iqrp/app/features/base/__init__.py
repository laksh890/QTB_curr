"""Feature base package."""

from iqrp.app.features.base.cache import CacheStats, FeatureCache
from iqrp.app.features.base.feature import Feature, FeatureMeta
from iqrp.app.features.base.pipeline import FeaturePipeline, PipelineBenchmarks
from iqrp.app.features.base.registry import (
    FeatureRegistry,
    ensure_features_loaded,
    get_registry,
    register_feature,
)

__all__ = [
    "CacheStats",
    "Feature",
    "FeatureCache",
    "FeatureMeta",
    "FeaturePipeline",
    "FeatureRegistry",
    "PipelineBenchmarks",
    "ensure_features_loaded",
    "get_registry",
    "register_feature",
]

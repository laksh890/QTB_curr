"""Institutional Feature Engineering Platform.

Downstream models must consume features exclusively through this package
(``FeatureQueryService`` / ``get_feature(s)``), never by re-implementing
indicators ad hoc.
"""

from iqrp.app.features.base import (
    Feature,
    FeatureCache,
    FeatureMeta,
    FeaturePipeline,
    FeatureRegistry,
    ensure_features_loaded,
    get_registry,
    register_feature,
)
from iqrp.app.features.metadata import get_metadata, list_metadata
from iqrp.app.features.query import (
    FeatureQueryService,
    describe_feature,
    feature_dependencies,
    get_feature,
    get_features,
    list_features,
)
from iqrp.app.features.store import FeatureStore
from iqrp.app.features.validation import FeatureValidationReport, FeatureValidator

# Eagerly register all built-in features on import.
ensure_features_loaded()

__all__ = [
    "Feature",
    "FeatureCache",
    "FeatureMeta",
    "FeaturePipeline",
    "FeatureQueryService",
    "FeatureRegistry",
    "FeatureStore",
    "FeatureValidationReport",
    "FeatureValidator",
    "describe_feature",
    "ensure_features_loaded",
    "feature_dependencies",
    "get_feature",
    "get_features",
    "get_metadata",
    "get_registry",
    "list_features",
    "list_metadata",
    "register_feature",
]

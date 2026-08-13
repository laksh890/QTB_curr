"""Model explainability interfaces."""

from iqrp.app.forecasting.explainability.attribution import (
    attribute,
    attribution_matrix,
    compare_attributions,
    top_k_features,
)
from iqrp.app.forecasting.explainability.importance import (
    ExplanationResult,
    attention_visualization,
    builtin_importance,
    explain_model,
    integrated_gradients_interface,
    permutation_importance,
    shap_interface,
)

__all__ = [
    "ExplanationResult",
    "attention_visualization",
    "attribute",
    "attribution_matrix",
    "builtin_importance",
    "compare_attributions",
    "explain_model",
    "integrated_gradients_interface",
    "permutation_importance",
    "shap_interface",
    "top_k_features",
]

"""Model-driven alpha research campaign (Prompt 39)."""

from iqrp.app.backtesting.alpha_research.model_campaign.protocol import (
    DISCLAIMER,
    ModelCampaignConfig,
)
from iqrp.app.backtesting.alpha_research.model_campaign.runner import run_model_driven_campaign

__all__ = ["DISCLAIMER", "ModelCampaignConfig", "run_model_driven_campaign"]

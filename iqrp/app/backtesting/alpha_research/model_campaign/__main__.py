"""python -m iqrp.app.backtesting.alpha_research.model_campaign"""

from __future__ import annotations

import json
import sys

from iqrp.app.backtesting.alpha_research.model_campaign.protocol import ModelCampaignConfig
from iqrp.app.backtesting.alpha_research.model_campaign.runner import run_model_driven_campaign


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    smoke = "--smoke" in argv
    cfg = ModelCampaignConfig(smoke=smoke)
    report = run_model_driven_campaign(cfg, progress=True)
    print(json.dumps({"campaign_status": report.get("campaign_status"), "n_experiments": report.get("n_experiments"), "n_candidates_strict": report.get("n_candidates_strict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

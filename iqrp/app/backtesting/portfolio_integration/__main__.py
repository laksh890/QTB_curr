"""python -m iqrp.app.backtesting.portfolio_integration"""

from __future__ import annotations

import json
import sys

from iqrp.app.backtesting.portfolio_integration.protocol import PortfolioIntegrationConfig
from iqrp.app.backtesting.portfolio_integration.runner import run_portfolio_integration


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    smoke = "--smoke" in argv
    cfg = PortfolioIntegrationConfig(smoke=smoke)
    report = run_portfolio_integration(cfg, progress=True)
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "n_candidates": report.get("n_candidates"),
                "cascade": (report.get("answers") or {}).get("8_full_cascade_operational"),
                "proven_profitability": (report.get("answers") or {}).get("proven_profitability"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""python -m iqrp.app.backtesting.alpha_research.consolidation"""

from __future__ import annotations

import json
import sys

from iqrp.app.backtesting.alpha_research.consolidation.protocol import ConsolidationConfig
from iqrp.app.backtesting.alpha_research.consolidation.runner import run_consolidation


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    smoke = "--smoke" in argv
    cfg = ConsolidationConfig(smoke=smoke)
    report = run_consolidation(cfg, progress=True)
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "n_input": report.get("n_input_candidates"),
                "n_distinct": report.get("answers", {}).get("18_n_distinct_research_candidates"),
                "n_ensemble": report.get("answers", {}).get("19_n_ensemble_candidates_remaining"),
                "proven_profitability": report.get("answers", {}).get("20_proven_profitability"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

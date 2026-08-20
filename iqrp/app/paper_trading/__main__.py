"""CLI: python -m iqrp.app.paper_trading [--smoke]"""

from __future__ import annotations

import argparse

from iqrp.app.paper_trading.protocol import PaperTradingValidationConfig
from iqrp.app.paper_trading.runner import run_paper_trading_validation


def main() -> None:
    p = argparse.ArgumentParser(description="Prompt 43 paper trading validation")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--scenario", default="BASE", choices=["BASE", "MODERATE", "ADVERSE"])
    args = p.parse_args()
    cfg = PaperTradingValidationConfig(smoke=bool(args.smoke), exec_scenario=args.scenario)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_paper_trading_validation(cfg, progress=not args.no_progress)


if __name__ == "__main__":
    main()

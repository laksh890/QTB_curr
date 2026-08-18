"""CLI: python -m iqrp.app.backtesting.final_holdout"""

from __future__ import annotations

import argparse

from iqrp.app.backtesting.final_holdout.protocol import FinalHoldoutConfig
from iqrp.app.backtesting.final_holdout.runner import run_final_holdout


def main() -> None:
    p = argparse.ArgumentParser(description="Independent final holdout validation")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    cfg = FinalHoldoutConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_final_holdout(cfg, progress=not args.no_progress)


if __name__ == "__main__":
    main()

"""CLI: python -m iqrp.app.backtesting.independent_validation"""

from __future__ import annotations

import argparse

from iqrp.app.backtesting.independent_validation.protocol import IndependentValidationConfig
from iqrp.app.backtesting.independent_validation.runner import run_independent_validation


def main() -> None:
    p = argparse.ArgumentParser(description="Independent OOS validation of frozen candidates")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    cfg = IndependentValidationConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_independent_validation(cfg, progress=not args.no_progress)


if __name__ == "__main__":
    main()

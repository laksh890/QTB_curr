"""CLI: python -m iqrp.app.backtesting.final_validation [--smoke]"""

from __future__ import annotations

import argparse

from iqrp.app.backtesting.final_validation.protocol import FinalValidationConfig
from iqrp.app.backtesting.final_validation.runner import run_final_validation


def main() -> None:
    p = argparse.ArgumentParser(description="Prompt 42 final trading validation")
    p.add_argument("--smoke", action="store_true", help="Short subsample / 3 candidates")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    cfg = FinalValidationConfig(smoke=bool(args.smoke), run_predeclared_grid=False)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_final_validation(cfg, progress=not args.no_progress)


if __name__ == "__main__":
    main()

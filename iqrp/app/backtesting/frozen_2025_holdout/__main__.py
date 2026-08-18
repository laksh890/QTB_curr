"""CLI: python -m iqrp.app.backtesting.frozen_2025_holdout [--smoke]"""

from __future__ import annotations

import argparse

from iqrp.app.backtesting.frozen_2025_holdout.protocol import Frozen2025Config
from iqrp.app.backtesting.frozen_2025_holdout.runner import run_frozen_2025


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen 2024 research → 2025 holdout validation")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    cfg = Frozen2025Config(smoke=bool(args.smoke))
    if args.output_dir:
        cfg.output_dir = args.output_dir
    run_frozen_2025(cfg, progress=not args.no_progress)


if __name__ == "__main__":
    main()

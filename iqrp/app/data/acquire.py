"""CLI: acquire historical market data into canonical datasets.

Examples
--------
NIFTY (Yahoo, short development window)::

  python -m iqrp.app.data.acquire --provider yahoo_finance --instrument NIFTY50 \\
      --start 2026-08-06 --end 2026-08-14 --frequency 1m --output data/nifty50 \\
      --derive 5m,15m,30m,1h

BTCUSDT (Binance Vision public archives, no API key)::

  python -m iqrp.app.data.acquire --provider binance --symbol BTCUSDT \\
      --interval 1m --start 2019-01-01 --end 2025-12-31 \\
      --output data/btcusdt --derive 5m,15m,30m,1h
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Acquire historical market data (research/dev)")
    p.add_argument("--provider", default="yahoo_finance")
    p.add_argument("--instrument", default=None, help="Normalized instrument id")
    p.add_argument("--symbol", default=None, help="Alias for --instrument (e.g. BTCUSDT)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--frequency", default=None)
    p.add_argument("--interval", default=None, help="Alias for --frequency (e.g. 1m)")
    p.add_argument("--output", default=None, help="Output directory (default depends on instrument)")
    p.add_argument("--adjustment-policy", default="unadjusted")
    p.add_argument("--dataset-id", default=None)
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--derive", default="", help="Comma-separated derived freqs, e.g. 5m,15m,30m,1h")
    p.add_argument("--registry", default="dataset_registry.json")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-incremental", action="store_true")
    p.add_argument("--no-register", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from iqrp.app.data.historical.pipeline import AcquisitionPipeline

    instrument = args.instrument or args.symbol or "NIFTY50"
    frequency = args.frequency or args.interval or "1m"
    if args.output:
        output = args.output
    elif str(instrument).upper().endswith("USDT") or str(args.provider).lower().startswith("binance"):
        output = f"data/{str(instrument).lower()}"
    else:
        output = "data/nifty50"

    derive = [x.strip() for x in str(args.derive).split(",") if x.strip()]
    pipe = AcquisitionPipeline(output_dir=output, registry_path=args.registry)
    try:
        result = pipe.acquire(
            provider=args.provider,
            instrument=instrument,
            start=args.start,
            end=args.end,
            frequency=frequency,
            adjustment_policy=args.adjustment_policy,
            dataset_id=args.dataset_id,
            version=args.version,
            derive=derive,
            use_cache=not args.no_cache,
            incremental=not args.no_incremental,
            register=not args.no_register,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2, default=str))
    print(f"\nWrote: {result.path}")
    print(f"Checksum: {result.checksum}")
    print(f"Quality OK: {result.quality_report.get('ok')}")
    print(
        "NOTE: DEVELOPMENT/RESEARCH data only — not institutional-grade; "
        "not a profitability claim. SOFTWARE VALIDATION ≠ STATISTICAL/ECONOMIC VALIDATION."
    )
    return 0 if result.quality_report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

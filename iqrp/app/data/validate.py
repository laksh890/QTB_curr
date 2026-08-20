"""CLI: validate a canonical historical dataset.

Usage:
  python -m iqrp.app.data.validate --path data/nifty50/nifty50_intraday_1m.parquet --frequency 1m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate canonical historical OHLCV dataset")
    p.add_argument("--path", required=True)
    p.add_argument("--frequency", required=True)
    p.add_argument("--dataset-id", default="")
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-md", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from iqrp.app.backtesting.data.schema import normalize_frame
    from iqrp.app.data.historical.intraday_validation import (
        build_intraday_quality_report,
        quality_report_markdown,
    )

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1
    frame = normalize_frame(pd.read_parquet(path))
    report = build_intraday_quality_report(
        frame,
        frequency=args.frequency,
        dataset_id=args.dataset_id or path.stem,
    )
    text = quality_report_markdown(report)
    print(text)
    out_json = Path(args.output_json) if args.output_json else path.with_name(
        f"{path.stem}_data_quality.json"
    )
    out_md = Path(args.output_md) if args.output_md else path.with_name(
        f"{path.stem}_data_quality.md"
    )
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text(text, encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())

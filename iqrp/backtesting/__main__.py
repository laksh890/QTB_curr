"""Thin shim: ``python -m iqrp.backtesting`` → app.backtesting.run."""

from __future__ import annotations

from iqrp.app.backtesting.run import main

if __name__ == "__main__":
    raise SystemExit(main())

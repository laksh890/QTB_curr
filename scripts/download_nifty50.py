"""Download free NIFTY 50 daily OHLCV from Yahoo Finance (^NSEI).

Research/development data only — not an institutional vendor feed.
Does not modify the backtesting framework.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

START = "2015-01-01"
END = "2026-01-01"

output = Path("data/nifty50")
output.mkdir(parents=True, exist_ok=True)

print("Downloading NIFTY 50...")

df = yf.download(
    "^NSEI",
    start=START,
    end=END,
    interval="1d",
    auto_adjust=False,
    progress=True,
)

if df.empty:
    raise RuntimeError("No data downloaded")

# Handle yfinance MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

df = df.rename(
    columns={
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
)

df["instrument"] = "NIFTY50"

df = df[
    [
        "timestamp",
        "instrument",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
]

# Convert timestamp to UTC
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df = df.sort_values("timestamp").reset_index(drop=True)

output_file = output / "nifty50_daily.parquet"

df.to_parquet(output_file, index=False)

print()
print("Downloaded:", len(df), "rows")
print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())
print("Output:", output_file)
print()
print(df.head())
print(df.tail())

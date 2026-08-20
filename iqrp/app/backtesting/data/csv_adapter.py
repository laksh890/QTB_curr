"""CSV file adapter for historical OHLCV datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from iqrp.app.backtesting.data.adapter import DataAdapter
from iqrp.app.backtesting.data.dataset_validator import DatasetValidator

__all__ = ["CSVAdapter"]


class CSVAdapter(DataAdapter):
    """Load historical market data from one or more CSV files."""

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_id: str | None = None,
        version: str = "1.0.0",
        source: str = "csv",
        validator: DatasetValidator | None = None,
        normalize: bool = True,
        read_csv_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"CSV path not found: {self.path}")
        super().__init__(
            dataset_id=dataset_id or self.path.stem,
            version=version,
            source=source,
            validator=validator,
            normalize=normalize,
        )
        self.read_csv_kwargs = dict(read_csv_kwargs or {})

    def _read_raw(self) -> pd.DataFrame:
        if self.path.is_dir():
            files = sorted(self.path.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"no CSV files under {self.path}")
            frames = [
                pd.read_csv(f, **self.read_csv_kwargs) for f in files
            ]
            return pd.concat(frames, ignore_index=True)
        return pd.read_csv(self.path, **self.read_csv_kwargs)

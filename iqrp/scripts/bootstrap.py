"""Bootstrap local development directories and verify imports."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

console = Console()

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "iqrp"


def ensure_dirs() -> None:
    for relative in (
        "logs",
        "data/dev",
        "data/test",
        "data/dev/parquet",
        "data/dev/cache",
        "data/test/parquet",
        "data/test/cache",
    ):
        path = ROOT / relative
        path.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]ok[/green] {path}")


def verify_import() -> None:
    sys.path.insert(0, str(ROOT))
    import iqrp

    console.print(f"[green]ok[/green] iqrp {iqrp.__version__}")


def main() -> None:
    console.print("[bold]IQRP bootstrap[/bold]")
    ensure_dirs()
    verify_import()
    console.print("[bold green]Bootstrap complete[/bold green]")


if __name__ == "__main__":
    main()

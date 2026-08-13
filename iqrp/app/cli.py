"""Typer CLI entrypoint for IQRP foundation commands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from iqrp import __version__
from iqrp.app.config import Environment, load_config
from iqrp.app.logging import setup_logging

app = typer.Typer(
    name="iqrp",
    help="Institutional Quantitative Research Platform",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command("version")
def version() -> None:
    """Print the IQRP package version."""
    console.print(f"iqrp {__version__}")


@app.command("info")
def info(
    environment: str = typer.Option(
        "development",
        "--environment",
        "-e",
        help="Target environment: development | testing | production",
    ),
) -> None:
    """Load configuration and display platform settings."""
    env = Environment(environment.lower())
    settings = load_config(env)
    setup_logging(settings.logging)

    table = Table(title="IQRP Configuration", show_header=True, header_style="bold")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("name", settings.name)
    table.add_row("version", settings.version)
    table.add_row("environment", settings.environment.value)
    table.add_row("debug", str(settings.debug))
    table.add_row("seed", str(settings.seed))
    table.add_row("timezone", settings.timezone)
    table.add_row("log_level", settings.logging.level)
    table.add_row("data_dir", str(settings.storage.data_dir))
    table.add_row("duckdb_path", str(settings.storage.duckdb_path))
    console.print(table)


@app.command("doctor")
def doctor() -> None:
    """Verify that core dependencies import cleanly."""
    modules = [
        "polars",
        "numpy",
        "scipy",
        "pydantic",
        "hydra",
        "omegaconf",
        "rich",
        "loguru",
        "typer",
        "duckdb",
        "pyarrow",
    ]
    table = Table(title="Dependency Check", show_header=True, header_style="bold")
    table.add_column("Package")
    table.add_column("Status")
    ok = True
    for name in modules:
        try:
            __import__(name)
            table.add_row(name, "[green]ok[/green]")
        except Exception as exc:
            ok = False
            table.add_row(name, f"[red]FAIL[/red] {exc}")
    console.print(table)
    if not ok:
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":
    main()

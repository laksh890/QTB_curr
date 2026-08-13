"""SVG visualization for simulated markets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqrp.app.simulation.base.market import SimulatedMarket


def _polyline(xs: np.ndarray, ys: np.ndarray, color: str, width: float = 1.4) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=False))
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}"/>'


def _scale(
    series: np.ndarray, x0: float, x1: float, y0: float, y1: float
) -> tuple[np.ndarray, np.ndarray]:
    n = len(series)
    xs = np.linspace(x0, x1, n)
    s = np.asarray(series, dtype=np.float64)
    lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    ys = y1 - (np.nan_to_num(s, nan=lo) - lo) / (hi - lo + 1e-12) * (y1 - y0)
    return xs, ys


def plot_price(market: SimulatedMarket, path: Path, *, max_points: int = 800) -> Path:
    close = market.ohlcv()["close"].to_numpy()
    n = min(len(close), max_points)
    series = close[:n]
    width, height = 720, 240
    xs, ys = _scale(series, 40, width - 10, 30, height - 20)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">Simulated Price</text>',
        _polyline(xs, ys, "#4c78a8"),
        "</svg>",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def plot_returns(market: SimulatedMarket, path: Path, *, max_points: int = 800) -> Path:
    rets = market.returns()
    n = min(len(rets), max_points)
    series = rets[:n]
    width, height = 720, 200
    xs, ys = _scale(series, 40, width - 10, 30, height - 20)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">Returns</text>',
        _polyline(xs, ys, "#e45756"),
        "</svg>",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def plot_regimes(market: SimulatedMarket, path: Path, *, max_points: int = 800) -> Path:
    ids = market.ground_truth.regime_ids
    n = min(len(ids), max_points)
    series = ids[:n].astype(float)
    width, height = 720, 140
    colors = ["#e45756", "#bab0ac", "#54a24b", "#4c78a8", "#f58518"]
    bar_w = (width - 40) / max(n, 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">True Regimes</text>',
    ]
    for i, sid in enumerate(series):
        color = colors[int(sid) % len(colors)]
        parts.append(
            f'<rect x="{40 + i * bar_w:.2f}" y="40" width="{max(bar_w, 0.5):.2f}" '
            f'height="60" fill="{color}"/>'
        )
    parts.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_volatility(market: SimulatedMarket, path: Path, *, max_points: int = 800) -> Path:
    vol = np.asarray(market.ground_truth.volatility, dtype=np.float64).ravel()
    n = min(len(vol), max_points)
    width, height = 720, 200
    xs, ys = _scale(vol[:n], 40, width - 10, 30, height - 20)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">True Volatility</text>',
        _polyline(xs, ys, "#f58518"),
        "</svg>",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def plot_transition_matrix(market: SimulatedMarket, path: Path) -> Path:
    tm = np.asarray(market.ground_truth.transition_matrix, dtype=np.float64)
    k = tm.shape[0]
    cell, margin = 36, 60
    width = margin + k * cell + 20
    height = margin + k * cell + 40
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">True Transition Matrix</text>',
    ]
    for i in range(k):
        for j in range(k):
            val = float(tm[i, j])
            intensity = int(255 * (1 - val))
            color = f"rgb({intensity},{intensity},255)"
            parts.append(
                f'<rect x="{margin + j * cell}" y="{margin + i * cell}" '
                f'width="{cell - 2}" height="{cell - 2}" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{margin + j * cell + 6}" y="{margin + i * cell + 22}" '
                f'font-size="9">{val:.2f}</text>'
            )
    parts.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_distribution(market: SimulatedMarket, path: Path, *, bins: int = 40) -> Path:
    rets = market.returns()
    hist, _edges = np.histogram(rets[np.isfinite(rets)], bins=bins)
    width, height = 720, 220
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">Return Distribution</text>',
    ]
    if hist.size:
        max_h = float(hist.max()) or 1.0
        bar_w = (width - 50) / hist.size
        for i, h in enumerate(hist):
            bh = (h / max_h) * (height - 50)
            parts.append(
                f'<rect x="{40 + i * bar_w:.2f}" y="{height - 20 - bh:.2f}" '
                f'width="{max(bar_w - 1, 0.5):.2f}" height="{bh:.2f}" fill="#54a24b"/>'
            )
    parts.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_autocorrelation(market: SimulatedMarket, path: Path, *, lags: int = 20) -> Path:
    from iqrp.app.simulation.validation.statistical_tests import SimulationValidator

    acf = SimulationValidator(acf_lags=lags).autocorrelation(market.returns(), lags=lags)
    width, height = 720, 200
    xs = np.arange(len(acf))
    bar_w = (width - 50) / max(len(acf), 1)
    mid = height / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<text x="10" y="18" font-size="14">Autocorrelation</text>',
        f'<line x1="40" y1="{mid}" x2="{width - 10}" y2="{mid}" stroke="#ccc"/>',
    ]
    for i, v in enumerate(acf):
        h = float(v) * (height / 2 - 30)
        y = mid - h if h >= 0 else mid
        parts.append(
            f'<rect x="{40 + i * bar_w:.2f}" y="{y:.2f}" '
            f'width="{max(bar_w - 2, 0.5):.2f}" height="{abs(h):.2f}" fill="#4c78a8"/>'
        )
    _ = xs
    parts.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_all_charts(
    market: SimulatedMarket, output_dir: Path, *, max_points: int = 800
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "price": plot_price(market, out / "price.svg", max_points=max_points),
        "returns": plot_returns(market, out / "returns.svg", max_points=max_points),
        "regimes": plot_regimes(market, out / "regimes.svg", max_points=max_points),
        "volatility": plot_volatility(market, out / "volatility.svg", max_points=max_points),
        "transition_matrix": plot_transition_matrix(market, out / "transition_matrix.svg"),
        "distribution": plot_distribution(market, out / "distribution.svg"),
        "autocorrelation": plot_autocorrelation(market, out / "autocorrelation.svg"),
    }

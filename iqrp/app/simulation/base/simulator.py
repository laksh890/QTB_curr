"""Institutional market simulation engine orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger

from iqrp.app.simulation.base.generator import ensure_generators_loaded, get_generator_registry
from iqrp.app.simulation.base.market import GroundTruth, SimulatedMarket
from iqrp.app.simulation.base.scenario import Scenario
from iqrp.app.simulation.config import SimulationSettings
from iqrp.app.simulation.events import apply_event_suite
from iqrp.app.simulation.liquidity.orderbook import OrderBookGenerator
from iqrp.app.simulation.liquidity.spread import SpreadModel
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator
from iqrp.app.simulation.validation.statistical_tests import SimulationValidator, ValidationReport
from iqrp.app.simulation.visualization.charts import write_all_charts


class MarketSimulator:
    """Generate realistic synthetic markets with ground truth for model validation."""

    def __init__(self, settings: SimulationSettings | None = None) -> None:
        ensure_generators_loaded()
        self.settings = settings or SimulationSettings.default()
        if not self.settings.enabled:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Simulation engine disabled by configuration",
                code="SIM_DISABLED",
            )

    def available_models(self) -> list[str]:
        return get_generator_registry().list_names()

    def simulate(
        self,
        scenario: Scenario | None = None,
        *,
        write_charts: bool = False,
        validate: bool = True,
    ) -> SimulatedMarket:
        scenario = scenario or Scenario.from_settings(self.settings)
        rng = np.random.default_rng(scenario.random_seed)
        n_steps = scenario.n_steps
        n_assets = scenario.n_assets
        dt = scenario.dt
        params = dict(scenario.parameters)

        # --- Regimes ---
        if scenario.regime_enabled and scenario.transition_matrix is not None:
            switcher = RegimeSwitchingSimulator(rng)
            regime_path = switcher.simulate(
                n_steps,
                transition_matrix=scenario.transition_matrix,
                state_names=tuple(params.get("regime_names", ("bear", "sideways", "bull"))),
                drifts=params.get("regime_drifts", (-0.15, 0.0, 0.12)),
                volatilities=params.get("regime_volatilities", (0.35, 0.15, 0.22)),
            )
            drift = regime_path.drifts
            vol = regime_path.volatilities
            trend = regime_path.trends
            regime_ids = regime_path.state_ids
            regime_names = regime_path.state_names
            tm = regime_path.transition_matrix
        else:
            drift = np.full(n_steps, float(params.get("drift", 0.05)))
            vol = np.full(n_steps, float(params.get("volatility", 0.2)))
            trend = np.sign(drift)
            regime_ids = np.zeros(n_steps, dtype=np.int64)
            regime_names = ("unspecified",)
            tm = np.ones((1, 1), dtype=np.float64)

        # Broadcast for multi-asset: same regime params across assets (correlated shocks)
        if n_assets > 1:
            drift_m = np.tile(np.asarray(drift).reshape(-1, 1), (1, n_assets))
            vol_m = np.tile(np.asarray(vol).reshape(-1, 1), (1, n_assets))
        else:
            drift_m = np.asarray(drift, dtype=np.float64)
            vol_m = np.asarray(vol, dtype=np.float64)

        x0 = np.full(n_assets, scenario.initial_price, dtype=np.float64)
        noise_kwargs = {
            "df": self.settings.noise.df,
            "mixture_weights": self.settings.noise.mixture_weights,
            "mixture_scales": self.settings.noise.mixture_scales,
        }
        generator = get_generator_registry().create(scenario.model, rng=rng)
        path = generator.generate(
            n_steps,
            x0=x0 if n_assets > 1 else float(x0[0]),
            dt=dt,
            drift=drift_m,
            volatility=vol_m,
            noise=scenario.noise_distribution,
            correlation=scenario.correlation_matrix,
            noise_kwargs=noise_kwargs,
            mean_reversion_speed=params.get("mean_reversion_speed", 1.0),
            mean_reversion_level=params.get("mean_reversion_level", scenario.initial_price),
            jump_intensity=params.get("jump_intensity", 5.0),
            jump_mean=params.get("jump_mean", -0.02),
            jump_std=params.get("jump_std", 0.04),
            heston_kappa=params.get("heston_kappa", 2.0),
            heston_theta=params.get("heston_theta", 0.04),
            heston_xi=params.get("heston_xi", 0.3),
            heston_rho=params.get("heston_rho", -0.7),
            vg_theta=params.get("vg_theta", -0.1),
            vg_sigma=params.get("vg_sigma", 0.2),
            vg_nu=params.get("vg_nu", 0.2),
            cir_kappa=params.get("cir_kappa", 1.5),
            cir_theta=params.get("cir_theta", 0.04),
            cir_sigma=params.get("cir_sigma", 0.1),
        )

        prices = np.asarray(path.prices, dtype=np.float64)
        if prices.ndim == 1:
            prices = prices.reshape(-1, 1)
        # Align vol/drift to price length for microstructure (use per-bar, pad first)
        path_vol = np.asarray(path.volatility, dtype=np.float64)
        if path_vol.ndim == 1:
            path_vol = path_vol.reshape(-1, 1)
        if path_vol.shape[0] == n_steps:
            vol_full = np.vstack([path_vol[0:1], path_vol])
        else:
            vol_full = path_vol

        timestamps = _build_timestamps(
            n_steps + 1,
            timeframe=self.settings.market.timeframe,
            market_hours=self.settings.market.market_hours,
        )

        # Build per-asset candles then concat
        candle_frames: list[pl.DataFrame] = []
        trade_frames: list[pl.DataFrame] = []
        book_frames: list[pl.DataFrame] = []
        event_masks: dict[str, np.ndarray] = {}
        symbols = scenario.symbols
        if len(symbols) < n_assets:
            symbols = tuple(f"{self.settings.market.symbol}_{i}" for i in range(n_assets))

        spread_model = SpreadModel(
            base_spread_bps=self.settings.liquidity.base_spread_bps,
            min_spread_bps=self.settings.liquidity.min_spread_bps,
            rng=rng,
        )
        book_gen = OrderBookGenerator(
            depth_levels=self.settings.liquidity.depth_levels,
            base_depth=self.settings.liquidity.base_depth,
            tick_size=self.settings.market.tick_size,
            rng=rng,
        )

        for a in range(n_assets):
            px = prices[:, a].copy()
            vol_a = vol_full[:, min(a, vol_full.shape[1] - 1)].copy()
            volumes = self.settings.liquidity.volume_scale * (
                1.0 + np.abs(rng.standard_normal(len(px)))
            )
            spreads = spread_model.spreads_bps(px, vol_a)
            if scenario.events_enabled:
                px, volumes, vol_a, spreads, masks = apply_event_suite(
                    px,
                    volumes,
                    vol_a,
                    spreads,
                    rng=rng,
                    flash_crash_prob=self.settings.events.flash_crash_prob,
                    news_shock_prob=self.settings.events.news_shock_prob,
                    gap_open_prob=self.settings.events.gap_open_prob,
                    liquidity_collapse_prob=self.settings.events.liquidity_collapse_prob,
                    outage_prob=self.settings.events.outage_prob,
                    vol_spike_prob=self.settings.events.vol_spike_prob,
                    momentum_burst_prob=self.settings.events.momentum_burst_prob,
                )
                if a == 0:
                    event_masks = masks
            bid, ask = spread_model.bid_ask(px, spreads)
            ohlc = _ohlc_from_close(px, rng)
            sym = symbols[a]
            frame = pl.DataFrame(
                {
                    "open_time": timestamps,
                    "symbol": [sym] * len(px),
                    "exchange": [self.settings.market.exchange] * len(px),
                    "timeframe": [self.settings.market.timeframe] * len(px),
                    "open": ohlc["open"],
                    "high": ohlc["high"],
                    "low": ohlc["low"],
                    "close": ohlc["close"],
                    "volume": volumes.tolist(),
                    "bid": bid.tolist(),
                    "ask": ask.tolist(),
                    "spread_bps": spreads.tolist(),
                }
            )
            candle_frames.append(frame)
            trade_frames.append(
                book_gen.generate_trades(px, volumes, timestamps, symbol=sym, trades_per_bar=2)
            )
            book_frames.append(
                book_gen.generate_frame(
                    px,
                    spreads * px / 10_000.0,
                    timestamps,
                    symbol=sym,
                    stride=max(1, n_steps // 50),
                )
            )
            prices[:, a] = px
            vol_full[:, min(a, vol_full.shape[1] - 1)] = vol_a

        candles = pl.concat(candle_frames, how="vertical_relaxed")
        trades = pl.concat(trade_frames, how="vertical_relaxed") if trade_frames else pl.DataFrame()
        books = pl.concat(book_frames, how="vertical_relaxed") if book_frames else pl.DataFrame()

        # Ground truth aligned to bars (exclude final open-only if needed - use n_steps)
        gt_vol = vol_full[1:, 0] if vol_full.shape[0] == n_steps + 1 else vol_full[:, 0]
        if len(gt_vol) != n_steps:
            gt_vol = np.resize(gt_vol, n_steps)
        gt_drift = np.asarray(drift, dtype=np.float64)
        if len(gt_drift) != n_steps:
            gt_drift = np.resize(gt_drift, n_steps)
        gt_trend = np.asarray(trend, dtype=np.float64)
        if len(gt_trend) != n_steps:
            gt_trend = np.resize(gt_trend, n_steps)

        truth = GroundTruth(
            regime_ids=regime_ids,
            regime_names=regime_names,
            volatility=gt_vol,
            drift=gt_drift,
            trend=gt_trend,
            transition_matrix=tm,
            event_mask={
                k: v[1:] if len(v) == n_steps + 1 else v[:n_steps] for k, v in event_masks.items()
            },
            metadata={"model": scenario.model, "asset_class": scenario.asset_class},
        )

        market = SimulatedMarket(
            scenario_name=scenario.name,
            model=scenario.model,
            asset_class=scenario.asset_class,
            candles=candles,
            trades=trades,
            orderbook_snapshots=books,
            ground_truth=truth,
            timestamps=timestamps,
            symbols=symbols[:n_assets],
            metadata={
                "dt": dt,
                "n_steps": n_steps,
                "random_seed": scenario.random_seed,
                "parameters": params,
            },
        )

        if validate:
            report = self.validate(market)
            market.metadata["validation"] = report.to_dict()
            logger.info(
                "simulation_validated passed={} model={} n={}",
                report.passed,
                scenario.model,
                n_steps,
            )

        if write_charts and self.settings.visualization.enabled:
            out = Path(self.settings.visualization.output_dir) / scenario.name
            charts = write_all_charts(
                market, out, max_points=self.settings.visualization.max_points
            )
            market.metadata["charts"] = {k: str(v) for k, v in charts.items()}
            logger.info("simulation_charts_written dir={}", out)

        return market

    def validate(self, market: SimulatedMarket) -> ValidationReport:
        rets = market.returns()
        expected_vol = float(np.mean(market.ground_truth.volatility))
        expected_drift = float(np.mean(market.ground_truth.drift))
        dt = float(market.metadata.get("dt", self.settings.dt))
        return SimulationValidator(
            significance=self.settings.validation.significance,
            acf_lags=self.settings.validation.acf_lags,
        ).validate_returns(
            rets,
            expected_drift=expected_drift,
            expected_volatility=expected_vol,
            dt=dt,
        )

    def simulate_preset(
        self,
        preset: str,
        *,
        n_steps: int | None = None,
        model: str | None = None,
        write_charts: bool = False,
    ) -> SimulatedMarket:
        """Convenience presets: bull, bear, sideways, high_volatility, mixed."""
        from iqrp.app.simulation.regimes.regime_switching import REGIME_PRESETS

        settings = self.settings
        overrides: dict[str, Any] = {}
        if n_steps is not None:
            overrides["n_steps"] = n_steps
        if model is not None:
            overrides["model"] = model
        overrides.setdefault("n_assets", 1)
        scenario = Scenario.from_settings(settings, name=preset, **overrides)
        if preset in REGIME_PRESETS and preset != "mixed":
            p = REGIME_PRESETS[preset]
            scenario.regime_enabled = False
            scenario.parameters["drift"] = p["drift"]
            scenario.parameters["volatility"] = p["volatility"]
        elif preset == "mixed":
            scenario.regime_enabled = True
        return self.simulate(scenario, write_charts=write_charts)


def _build_timestamps(n: int, *, timeframe: str, market_hours: int) -> list[datetime]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    unit = timeframe.lower()
    if unit.endswith("m"):
        delta = timedelta(minutes=int(unit[:-1] or 1))
    elif unit.endswith("h"):
        delta = timedelta(hours=int(unit[:-1] or 1))
    elif unit.endswith("d"):
        delta = timedelta(days=int(unit[:-1] or 1))
    else:
        delta = timedelta(hours=1)
    # market_hours unused for crypto 24/7; reserved for future session calendars
    _ = market_hours
    return [start + i * delta for i in range(n)]


def _ohlc_from_close(close: np.ndarray, rng: np.random.Generator) -> dict[str, list[float]]:
    c = np.asarray(close, dtype=np.float64)
    n = len(c)
    opens = np.empty(n, dtype=np.float64)
    opens[0] = c[0]
    opens[1:] = c[:-1]
    noise = np.abs(rng.normal(0.0, 0.001, size=n))
    high = np.maximum(opens, c) * (1.0 + noise)
    low = np.minimum(opens, c) * (1.0 - noise)
    return {
        "open": opens.tolist(),
        "high": high.tolist(),
        "low": low.tolist(),
        "close": c.tolist(),
    }

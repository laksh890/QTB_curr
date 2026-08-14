# IQRP Prompt 35 — BTCUSDT Alpha Research Campaign

**Campaign ID:** `alpha_research_btc_full_v1`

> Research evidence is not a profitability guarantee.

## 1. Executive summary

Ran 1260 experiments (420 BASE) across 5 timeframes and 6 holding horizons on registered BTCUSDT datasets. Leakage suite OK. Zero experiments received CANDIDATE status after cost-aware OOS gates (statuses dominated by OOS_FAILED and COST_INEFFICIENT). A diversified near-miss watchlist of 7 families is recorded for future research only. Research evidence is not a profitability guarantee.

## 2. Dataset description

- `btcusdt_intraday_1m@1.0.0` (SOURCE): checksum `eb6c6488167b598b…`, rows=3152391, 2019-01-01T00:00:00+00:00 → 2024-12-31T23:59:00+00:00
- `btcusdt_intraday_5m@1.0.0` (DERIVED): checksum `8743b419c92b661a…`, rows=630482, 2019-01-01T00:00:00+00:00 → 2024-12-31T23:55:00+00:00
- `btcusdt_intraday_15m@1.0.0` (DERIVED): checksum `74cc6c8ea63345b6…`, rows=210163, 2019-01-01T00:00:00+00:00 → 2024-12-31T23:45:00+00:00
- `btcusdt_intraday_30m@1.0.0` (DERIVED): checksum `3c5f41cf8dba5f04…`, rows=105086, 2019-01-01T00:00:00+00:00 → 2024-12-31T23:30:00+00:00
- `btcusdt_intraday_1h@1.0.0` (DERIVED): checksum `f3ecf67059770b4d…`, rows=52549, 2019-01-01T00:00:00+00:00 → 2024-12-31T23:00:00+00:00

## 3. Data-quality limitations

{
  "gap_class": "MINOR_GAPS",
  "gaps_not_filled": true,
  "gap_exclusions_recorded": 420,
  "license_status": "UNKNOWN",
  "note": "OHLCV alone does not support institutional capacity claims."
}

## 4. Research universe

{
  "n_features": 14,
  "n_signals": 7,
  "signal_ids": [
    "breakout_signal",
    "mean_reversion_signal",
    "momentum_signal",
    "price_action_signal",
    "trend_signal",
    "volatility_signal",
    "volume_signal"
  ],
  "families": [
    "breakout",
    "mean_reversion",
    "momentum",
    "price_action",
    "trend",
    "volatility",
    "volume"
  ]
}

## 5. Number of experiments: **1260** (BASE=420)

## 6. Timeframes tested: ['1m', '5m', '15m', '30m', '1h']

## 7. Holding periods tested (bars): [1, 2, 3, 5, 10, 20]

## 8. Leakage validation: ok=True

## 9–20. Analytics artifacts

See companion JSON files: `IC_results.json`, `decay_results.json`, `cost_results.json`, `OOS_results.json`, `walk_forward_results.json`, `regime_results.json`, `robustness_results.json`, `multiple_testing_results.json`.

## 20. Multiple-testing correction

- Method: fdr_bh
- Experiments tested: 420
- Before correction: 14
- Surviving FDR: 237

## 21. Candidate ranking (top final set)


## 22. Rejected candidate categories: {'UNSTABLE': 4, 'OOS_FAILED': 322, 'COST_INEFFICIENT': 94}

## 23. Research limitations

- Research evidence is not a profitability guarantee.
- Single-instrument BTC time-series IC is not cross-sectional IC.
- Multiple-testing p-values approximate and may be optimistic under autocorrelation.
- Capacity/liquidity figures from OHLCV are estimates only.
- No live trading was performed.
- Candidates are not PRODUCTION_READY.
- 1m is SOURCE; 5m/15m/30m/1h are DERIVED via causal session-aware resampling.

## 24. Final candidate set

Count: 0

Trade frequency (BASE): avg=33.227338590545706, median=7.898038321167883

> Research evidence is not a profitability guarantee.

No candidate is PRODUCTION_READY. No live trading was performed.

## Research watchlist (near-misses — NOT candidates)

- `trend_signal` @ 1h status=UNSTABLE score=0.46309978217986303 net_sharpe=0.5512589352511671 oos=0.019479990863862328
- `momentum_signal` @ 1h status=OOS_FAILED score=0.44420022651604824 net_sharpe=0.38287311772141297 oos=-0.33086504989011
- `price_action_signal` @ 1h status=COST_INEFFICIENT score=0.3622459422810268 net_sharpe=0.06247287369823064 oos=-0.15056916036705084
- `volume_signal` @ 1h status=COST_INEFFICIENT score=0.3487549788684432 net_sharpe=0.07477273769170797 oos=-0.4315497574095226
- `volatility_signal` @ 1h status=COST_INEFFICIENT score=0.3307911595386495 net_sharpe=0.058653966439912796 oos=0.0714839926491228
- `breakout_signal` @ 1h status=OOS_FAILED score=0.28220705208667574 net_sharpe=-0.32342059538066753 oos=-1.2469498703830955
- `mean_reversion_signal` @ 1m status=COST_INEFFICIENT score=0.25602696927904656 net_sharpe=-19.481630689928178 oos=-27.047873425911114

> Research evidence is not a profitability guarantee.

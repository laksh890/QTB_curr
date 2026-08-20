# Final Trading Validation (Prompt 42)

Status: **PROFITABILITY_EVIDENCE**

FINAL TRADING VALIDATION — research evidence only. PROFITABILITY_EVIDENCE ≠ PROVEN PROFITABILITY ≠ PAPER-READY ≠ LIVE_READY. Do not p-hack: gates and grids are predefined. Prompt 35/36/39/40 artifacts immutable.

- Extended data: `True` keys={'1m': '[btcusdt_intraday_1m@1.0.1](mailto:btcusdt_intraday_1m@1.0.1)', '5m': '[btcusdt_intraday_5m@1.0.1](mailto:btcusdt_intraday_5m@1.0.1)', '15m': '[btcusdt_intraday_15m@1.0.1](mailto:btcusdt_intraday_15m@1.0.1)', '30m': '[btcusdt_intraday_30m@1.0.1](mailto:btcusdt_intraday_30m@1.0.1)', '1h': '[btcusdt_intraday_1h@1.0.1](mailto:btcusdt_intraday_1h@1.0.1)'}
- Candidates validated: 8
- PROFITABILITY_EVIDENCE count: **3**
- Suitable for live trading: **NO**

## Decision matrix


| Candidate          | Model       | TF  | Holding | Trades/day         | Net Sharpe           | Max DD               | Status                     |
| ------------------ | ----------- | --- | ------- | ------------------ | -------------------- | -------------------- | -------------------------- |
| `mdc_51a60a326484` | MTF         | 5m  | 2       | 5.256              | 6.329245371398349    | 0.06837463568758495  | **RESEARCH_ONLY**          |
| `mdc_99aa952c5d5f` | MTF         | 15m | 2       | 2.64192            | 5.239070799148286    | 0.06662184639601576  | **PROFITABILITY_EVIDENCE** |
| `mdc_6f008c954ea2` | MTF         | 15m | 1       | 2.6342399999999997 | 6.1568118004576275   | 0.044437403096704964 | **PROFITABILITY_EVIDENCE** |
| `mdc_678609c534d6` | MTF         | 5m  | 2       | 5.2704             | 7.4918983930805645   | 0.051034846931874545 | **PROFITABILITY_EVIDENCE** |
| `mdc_801488267ad4` | Ensemble    | 1h  | 10      | 1.1832             | -0.4970142325059521  | 0.25833037899525313  | **OOS_FAILED**             |
| `mdc_3ed979ee7d9d` | Combination | 1h  | 20      | 1.5384             | -0.3658690448479371  | 0.47937300284396434  | **OOS_FAILED**             |
| `mdc_4992167889db` | CatBoost    | 15m | 20      | 2.3500799999999997 | 0.051822819852444336 | 0.20916976493998696  | **FRAGILE**                |
| `mdc_a0f769fc2083` | CatBoost    | 1h  | 20      | 0.4392             | -0.9421675386103209  | 0.5459616044243556   | **OOS_FAILED**             |




## Required answers

1. Profitability evidence? **YES** (3)
2. Best model family (diagnostic median OOS Sharpe): MTF
3. Best timeframe (diagnostic): 5m
4. Best holding (diagnostic): 2
5. Suitable horizon discovered? True (diagnostic only)
6. Trade frequency: per-candidate behavior_class in trading_behavior.json
7. Direction mix: {'LONG_SHORT': 5, 'LONG': 2, 'SHORT': 1}

8–11. Costs/regimes/walk-forward: see cost_analysis.json / regime_analysis.json / walk_forward_results.json
12. Portfolio improves? See portfolio_comparison.json methods vs equal_sleeve_baseline. See Prompt 41 comparison; not re-optimized here. Constraints-only often competitive.
13. Statistically convincing? **True**
14. Reproducible? **True**
15. Paper trading suitable? **True** (only if PROFITABILITY_EVIDENCE)
16. Live trading suitable? **NO**

## Limitations (honest)

- Single market (BTCUSDT); no second-asset transfer test in this run.
- OHLCV-only cost model (spread/slippage assumed, not observed bid/ask).
- Holding periods are **bar-based** (clock equivalent = holding_bars × timeframe minutes).
- `PROFITABILITY_EVIDENCE` is research-gate evidence, not absolute proven profit.
- Portfolio methods: risk_parity / HRP slightly above equal-sleeve on correlated MTF sleeves; do not treat as production sizing.



## Paid data (STOP_BEFORE_PURCHASE)

See `data_provenance.json` → `paid_upgrade_candidates` (Tardis/Kaiko/CryptoTick-class: tick + L2 + authentic spread). Not purchased.

## Stop

STOP — no broker connection, no live orders, no LIVE_READY claim.
Recommend independent paper validation of the three evidence IDs before any further escalation.
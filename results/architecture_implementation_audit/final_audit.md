# IQRP Architecture Implementation Audit

Generated: 2026-08-14T15:18:14.630024+00:00

## 1. Executive summary

IQRP contains a LARGE quantitative modelling layer in source (GARCH-family, ARIMA/VAR/VECM, HMM/Markov/GMM regimes, tree ML, neural nets, transformers, portfolio optimizers, execution algos) with substantial unit/integration tests. However, the WORKING alpha research path used for BTC campaigns (iqrp.app.backtesting.alpha_research) is NOT wired to those models: zero cross-imports were found, and Prompt 35 evaluated only reference OHLCV signals (momentum/mean-reversion/breakout/volatility/volume/trend/price_action). Therefore the broader modelling layer is largely IMPLEMENTED AS PARALLEL PLATFORMS, while the currently operational alpha engine remains REFERENCE-SIGNAL BASED.

## 2–6. Counts

{
  "planned_components_tabulated": 104,
  "implemented_or_unverified": 85,
  "partially_implemented": 14,
  "documentation_only": 1,
  "not_implemented_stub_interface": 4,
  "status_histogram": {
    "IMPLEMENTED": 85,
    "DOCUMENTATION_ONLY": 1,
    "PARTIALLY_IMPLEMENTED": 14,
    "NOT_IMPLEMENTED": 4
  }
}

## Critical question

IQRP contains a LARGE quantitative modelling layer in source (GARCH-family, ARIMA/VAR/VECM, HMM/Markov/GMM regimes, tree ML, neural nets, transformers, portfolio optimizers, execution algos) with substantial unit/integration tests. However, the WORKING alpha research path used for BTC campaigns (iqrp.app.backtesting.alpha_research) is NOT wired to those models: zero cross-imports were found, and Prompt 35 evaluated only reference OHLCV signals (momentum/mean-reversion/breakout/volatility/volume/trend/price_action). Therefore the broader modelling layer is largely IMPLEMENTED AS PARALLEL PLATFORMS, while the currently operational alpha engine remains REFERENCE-SIGNAL BASED.

## Status highlights

- GARCH-family: IMPLEMENTED as forecasting.volatility platform; NOT integrated into alpha_research
- Time-series: IMPLEMENTED as forecasting.statistical (+ state_space); NOT integrated into alpha_research
- Regime: IMPLEMENTED as regimes package (HMM etc.) with tests; default loader is mock-only; alpha uses heuristic regimes diagnostically
- Stat-arb: PARTIAL — cointegration/Johansen utilities IMPLEMENTED; pairs-trading strategy NOT_IMPLEMENTED in alpha path
- ML: IMPLEMENTED as tree_models forecasting; NOT in alpha_research
- Deep learning: IMPLEMENTED as neural forecasting nets/trainers; NOT in alpha_research
- Transformers: IMPLEMENTED as forecasting.transformers architectures; NOT in alpha_research
- Forecasting: IMPLEMENTED platform (point/probabilistic modules present); feeds Forecast objects, not Alpha Research Engine
- Ensembles: PARTIAL/IMPLEMENTED in forecasting intelligence, regimes, risk, alpha.ensemble packages — NOT used in Prompt 35 campaign
- Alpha integration: REFERENCE SIGNALS ONLY for operational campaigns; model adapters NOT_IMPLEMENTED
- Risk: Risk package IMPLEMENTED in isolation; not driven by alpha campaign candidates
- Portfolio: Optimizers IMPLEMENTED; not connected to alpha candidate pool
- Execution: Execution platform IMPLEMENTED; alpha_research uses simplified bps costs
- Data: DEVELOPMENT/RESEARCH OHLCV (NIFTY Yahoo, BTC Binance Vision) — not institutional-grade market-data claim

## Model → layer integration matrix

| MODEL | FCAST | SIGNAL | ALPHA | BT | RISK | PORT | EXEC |
|---|---|---|---|---|---|---|---|
| Reference_momentum_meanrev_breakout_etc | NO | YES | YES | YES | NO | NO | PARTIAL |
| GARCH_family | YES | NO | NO | PARTIAL | NO | NO | NO |
| ARIMA_SARIMA_VAR_VECM | YES | NO | NO | PARTIAL | NO | NO | NO |
| HMM_Markov_GMM_regimes | PARTIAL | NO | NO | PARTIAL | PARTIAL | NO | NO |
| Cointegration_utilities | NO | NO | NO | NO | NO | NO | NO |
| XGBoost_LightGBM_CatBoost | YES | NO | NO | PARTIAL | NO | NO | NO |
| LSTM_GRU_MLP_DeepAR | YES | NO | NO | PARTIAL | NO | NO | NO |
| Transformers_Informer_etc | YES | NO | NO | PARTIAL | NO | NO | NO |
| BlackLitterman_MeanVariance_RiskParity | NO | NO | NO | NO | PARTIAL | YES | NO |
| Execution_algos_OM | NO | NO | NO | PARTIAL | NO | NO | YES |
| Kalman_state_space | YES | NO | NO | PARTIAL | NO | NO | NO |

Finding: Cross-import scan found 0 references between iqrp.app.backtesting.alpha_research and iqrp.app.forecasting / regimes / portfolio model outputs. Working BTC alpha campaigns use reference OHLCV signals only.

## Already complete (do not rebuild)

- backtesting.alpha_research (Feature/Signal registries, leakage, MTF, IC, costs, OOS, campaign)
- backtesting.horizon helpers
- backtesting.runner / event engine / synthetic E2E
- data.historical providers (Yahoo, Binance Vision) + dataset registry
- forecasting.volatility GARCH-family registry (fit/forecast smoke)
- forecasting.statistical ARIMA/VAR/VECM packages
- forecasting.transformers architectures + tests
- regimes HMM/Markov/GMM/etc. (when explicitly loaded)
- portfolio optimization modules (MV, BL, risk parity, HRP)
- execution algorithms / order manager packages
- timeseries dependence cointegration utilities

## Missing before final system

- Adapters: forecasting/regime model → SignalRegistry / AlphaSignalResearchEngine
- End-to-end model→alpha→risk→portfolio→execution campaign path
- Default regime registry loading beyond mock_regime
- Pairs/stat-arb strategy signal generators (beyond cointegration tests)
- Institutional-grade order-book/PIT/continuous-futures data
- Alpha campaign consumption of GARCH/ML/transformer forecasts

## Recommended next step

Do NOT rebuild forecasting/regime/portfolio platforms. Next development should define a thin, explicit adapter contract from existing Forecast/RegimeModel outputs into SignalRegistry (or a model-signal bridge) with leakage/OOS gates — only after product priority is confirmed. Until then, treat model platforms and alpha_research as separate verified silos.

> Audit only. No models implemented by this prompt. Research evidence is not a profitability guarantee.

# Model-Driven Alpha Research Campaign (Prompt 39)

Campaign: `model_driven_alpha_campaign_v1`
Status: **RESEARCH_COMPLETE_CANDIDATES_FOUND**

MODEL-DRIVEN ALPHA RESEARCH — predefined protocol. Research evidence is not a profitability guarantee. MODEL IMPLEMENTED ≠ FORECAST ≠ SIGNAL ≠ OOS ≠ COST-SURVIVAL ≠ ROBUST ≠ PROFITABLE ≠ LIVE-READY.

- Experiments: 4806 (BASE: 1602)
- Strict candidates (ResearchStatus=CANDIDATE): **129**
- Conditional: 1
- Multiple-testing FDR survivors (BASE LONG_SHORT): 54 / 534
- Proven profitability: **NO**
- Statistically conclusive: **NO**

## Required answers

1. **Families successfully evaluated:** GARCH, ARIMA, XGBoost, LightGBM, CatBoost, LSTM, GRU, MLP, Transformer, HMM, Markov
2. **Unavailable:** VAR (Single-asset BTCUSDT — VAR requires multivariate series); VECM (Single-asset BTCUSDT — VECM requires cointegrated multivariate series); GMM (GMM regime model present in package but not registered in default adapter/pipeline path)
3. **Models generating valid adapter signals:** 25 successful adapter fits
4. **OOS survivors:** see `model_family_summary.json` (BASE OOS Sharpe > 0 counts by family)
5. **Cost survivors (BASE):** 168 survive; 569 collapse after costs (MODERATE survives=84; ADVERSE survives=31)
6. **Robustness survivors (BASE CANDIDATE/CONDITIONAL):** 131
7. **Multiple-testing survivors:** 54 (FDR-BH α=0.05; autocorrelation limitations apply — nominal significance may be optimistic)
8. **Frequent trading:** median trades/day (BASE) = 2.787
9. **Excessive turnover / HF cost-inefficient:** 264 BASE experiments
10. **Most cost-sensitive families (lowest cost-survivor rate):** [('ARIMA', 0, 54), ('Markov', 0, 18), ('Reference', 23, 630), ('GARCH', 3, 72), ('Combination', 23, 288)]
11. **Promising research regions (gate-surviving cells):** 20 (see campaign_report.json answers.11_promising_regions — not a universal best timeframe)
12. **Any candidate passed ALL declared research gates to CANDIDATE?** YES
13. **How many strict candidates?** 129
14. **None-passed statement:** N/A — candidates found
15. **Proven profitability?** **NO** — research candidates ≠ profitable strategy
16. **Statistically conclusive?** **NO** — STATISTICAL VALIDITY LIMITED
17. **Limitations:**
   - STATISTICAL VALIDITY LIMITED (autocorrelation / overlapping horizons; FDR optimistic).
   - Research subsample MAX_BARS — not full 1m history for model fits.
   - VAR/VECM/GMM unavailable by protocol/data constraints.
   - Alpha research path uses simplified bps cost model.
   - No post-hoc parameter optimization performed.
   - Not live-ready / not production-ready.

## Status counts (BASE)

```{'COST_INEFFICIENT': 569, 'SAMPLE_INSUFFICIENT': 164, 'OOS_FAILED': 710, 'UNSTABLE': 28, 'CANDIDATE': 129, 'CONDITIONAL': 2}```

## Families evaluated

- GARCH
- ARIMA
- XGBoost
- LightGBM
- CatBoost
- LSTM
- GRU
- MLP
- Transformer
- HMM
- Markov

## Unavailable / failed families

- VAR: Single-asset BTCUSDT — VAR requires multivariate series
- VECM: Single-asset BTCUSDT — VECM requires cointegrated multivariate series
- GMM: GMM regime model present in package but not registered in default adapter/pipeline path

## Claim distinctions

MODEL IMPLEMENTED ≠ FORECAST ≠ SIGNAL ≠ BACKTESTABLE ≠ OOS ≠ COSTS ≠ ROBUST ≠ PROFITABLE ≠ LIVE-READY

CANDIDATE status means the experiment passed the declared Alpha Research gate mapping (including OOS/cost-aware classification → ROBUST_ALPHA → CANDIDATE). It does **not** mean a profitable, production, or live-ready strategy.

## Final status

**RESEARCH_COMPLETE_CANDIDATES_FOUND**

STOP — no portfolio optimization, paper trading, broker integration, or live trading.

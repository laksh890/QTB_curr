# Prompt 36 — Validity Audit of Prompt 35 BTC Alpha Campaign

**Audit ID:** `alpha_research_btc_full_audit_v1`

**Final verdict:** `CONDITIONALLY_VALID`

> Research evidence is not a profitability guarantee.

## Explanation

Implementation audits of data identity, causal timing, cost math, OOS/purge, Sharpe/drawdown, and gate reconstruction are consistent with the stored Prompt 35 outcomes: no experiment satisfied the documented CANDIDATE gates under the research return+cost model. However, statistical strength is limited by autocorrelated bars / overlapping forward returns (FDR IC p-values are optimistic), and the research path is not a full institutional fill ledger. Therefore the zero-candidate conclusion is trustworthy as a statement about this gated research pipeline on the registered reference universe, but not as a general proof that no exploitable BTC short-horizon edge exists outside this design.

## A vs B

- Software implementation: **CORRECT_WITH_DOCUMENTED_LIMITATIONS**
- Statistical evidence for 'no edge': **LIMITED**

## Data

- Data validity: VALID
- Checksums/kinds OK: True

## Leakage / timing

- Leakage verdict: PASS
- Accounting: PASS_SIMPLIFIED_MODEL (SIMPLIFIED_BAR_RETURN_ATTRIBUTION)

## Statistics

- Multiple testing: MATH_CORRECT_ASSUMPTIONS_LIMITED
- Autocorrelation: STATISTICAL_LIMITATION
- BH math OK: True

## Gate reconstruction (why near-misses failed)

| signal | tf | status | failed gate | oos | net Sharpe |
|---|---|---|---|---|---|
| trend_signal | 1h | UNSTABLE | UNSTABLE/FRAGILE | 0.019479990863862328 | 0.5512589352511671 |
| momentum_signal | 1h | OOS_FAILED | OOS_FAILED | -0.33086504989011 | 0.38287311772141297 |
| momentum_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | 0.05355873389174057 | 0.22780072340017968 |
| trend_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | -0.3219638790612501 | 0.13897625157600144 |
| price_action_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | -0.15056916036705084 | 0.06247287369823064 |
| volume_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | -0.4315497574095226 | 0.07477273769170797 |
| momentum_signal | 30m | COST_INEFFICIENT | COST_INEFFICIENT | 0.05276003547493303 | 0.268704496029334 |
| momentum_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | -0.22230611596450126 | 0.1820291209382755 |
| volatility_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | 0.0714839926491228 | 0.058653966439912796 |
| momentum_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | -0.968810318329618 | 0.1288686439360759 |
| trend_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | -1.1988721412852448 | 0.06552360000673027 |
| momentum_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | -0.8275753925521958 | 0.09731412155757686 |
| breakout_signal | 1h | OOS_FAILED | OOS_FAILED | -1.2469498703830955 | -0.32342059538066753 |
| trend_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | -0.5567604871706799 | 0.034773189621177 |
| trend_signal | 1h | COST_INEFFICIENT | COST_INEFFICIENT | -1.6958119516331904 | 0.06271532392803285 |
| mean_reversion_signal | 1m | COST_INEFFICIENT | COST_INEFFICIENT | -27.047873425911114 | -19.481630689928178 |

## Defects

- **STAT_AUTO_CORR** (HIGH): FDR/IC p-values treat bars as IID; autocorrelation + overlapping horizons inflate significance.
- **ACCT_SIMPLIFIED** (MEDIUM): Alpha research path uses lagged position × returns, not a cash/fill equity ledger.
- **HOLD_WINDOW_SEMANTICS** (LOW): apply_holding fills fixed windows and ignores intra-window signal changes/zeros.
- **TRADE_DAY_POSTHOC** (LOW): Initial trades/day used n_days=1 when trade timestamps missing; later corrected in artifacts.
- **REGIME_SKIP_LARGE** (LOW): Regime analytics skipped on >250k-bar frames in base matrix.
- **DEEPDIVE_SCORE_SELECTION** (LOW): Deep-dive robustness/slippage selected top-K by full-sample research score.

## Prompt 35 preservation

- Original artifacts not overwritten.
- Zero-candidate trustworthy under stated gates: True
- General no-edge proof: False

> Research evidence is not a profitability guarantee.

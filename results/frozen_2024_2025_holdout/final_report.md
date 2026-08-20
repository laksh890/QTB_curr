# Frozen 2024 → Independent 2025 Holdout Validation

Status: **PROVEN_RESEARCH_PROFITABILITY**

FROZEN 2024→2025 HOLDOUT — research ≤2024-12-31; validation = calendar 2025. No retuning on 2025. PROVEN_RESEARCH_PROFITABILITY ≠ LIVE_READY. Do not modify Prompt 35–42 artifacts.

- Research: `<= 2024-12-31 23:59:59+00:00`
- Holdout: `2025-01-01 00:00:00+00:00` → `2025-12-31 23:59:59+00:00`
- Firewall: **PASS**
- Complete 2025 data: **True**
- Reproducibility: **PASS**
- LIVE_READY: **NO**

## Decision matrix (evidence focus)

| Candidate | TF | Holding | Dir | Net Sharpe | Max DD | BASE | MOD | ADV | Status |
|---|---|---|---|---|---|---|---|---|---|
| `mdc_99aa952c5d5f` | 15m | 2 | LONG_SHORT | 7.030677628891678 | 0.08335973749600833 | True | True | True | **PROVEN_RESEARCH_PROFITABILITY** |
| `mdc_6f008c954ea2` | 15m | 1 | LONG | 5.368918394697802 | 0.08021471608879993 | True | True | True | **PROVEN_RESEARCH_PROFITABILITY** |
| `mdc_678609c534d6` | 5m | 2 | SHORT | 6.227890047581503 | 0.09567388496923235 | True | True | False | **PAPER_TRADING_CANDIDATE** |

## Required answers

1. Reproduced in 2025 (after costs)? **True** → ['mdc_99aa952c5d5f6ff7', 'mdc_6f008c954ea26bf5', 'mdc_678609c534d68189']
3. Net Sharpe: {'mdc_99aa952c5d5f6ff7': 7.030677628891678, 'mdc_6f008c954ea26bf5': 5.368918394697802, 'mdc_678609c534d68189': 6.227890047581503}
4. Max DD: {'mdc_99aa952c5d5f6ff7': 0.08335973749600833, 'mdc_6f008c954ea26bf5': 0.08021471608879993, 'mdc_678609c534d68189': 0.09567388496923235}
5. Cost survival: {'mdc_99aa952c5d5f6ff7': {'BASE': True, 'MODERATE': True, 'ADVERSE': True}, 'mdc_6f008c954ea26bf5': {'BASE': True, 'MODERATE': True, 'ADVERSE': True}, 'mdc_678609c534d68189': {'BASE': True, 'MODERATE': True, 'ADVERSE': False}}
6. Quarters positive: {'mdc_99aa952c5d5f6ff7': ['Q1', 'Q2', 'Q3', 'Q4'], 'mdc_6f008c954ea26bf5': ['Q1', 'Q2', 'Q3', 'Q4'], 'mdc_678609c534d68189': ['Q1', 'Q2', 'Q3', 'Q4']}
7. Dependence-aware stats: {'mdc_99aa952c5d5f6ff7': True, 'mdc_6f008c954ea26bf5': True, 'mdc_678609c534d68189': True}
8. Sharpe genuine vs inflated: {'mdc_99aa952c5d5f6ff7': {'engine': 7.030677628891678, 'independent': 7.030677628891678, 'not_inflated_flag': True, 'n_eff': 17829.02541045858, 'acf1': -0.00874211967289909}, 'mdc_6f008c954ea26bf5': {'engine': 5.368918394697802, 'independent': 5.368918394697802, 'not_inflated_flag': True, 'n_eff': 34959.45555230064, 'acf1': 0.0011506439166398467}, 'mdc_678609c534d68189': {'engine': 6.227890047581503, 'independent': 6.227890047581503, 'not_inflated_flag': True, 'n_eff': 56080.394916335885, 'acf1': -0.03240410640118667}}
9. Portfolio: see portfolio_comparison.json (weights frozen from pre-2025)
10. Strongest under gate: {'candidate_id': 'mdc_99aa952c5d5f6ff7', 'status': 'PROVEN_RESEARCH_PROFITABILITY', 'net_sharpe': 7.030677628891678}
11. Statistically credible? **True**
12. Paper trading evidence? **True** ids=['mdc_51a60a3264845365', 'mdc_678609c534d68189', 'mdc_99aa952c5d5f6ff7', 'mdc_6f008c954ea26bf5']
13. Live trading? **NO**
14. Unsatisfied gates: {'mdc_99aa952c5d5f6ff7': [], 'mdc_6f008c954ea26bf5': [], 'mdc_678609c534d68189': ['survives_ADVERSE']}

## Stop

STOP — no retuning, no broker, no LIVE_READY, Prompt 35–42 artifacts untouched.

# Phase 11 — Alpha Research Validation

**Status:** PASS

- Components passed: 25/25
- Docs present: 10/10

## Checklist

- [x] Signal Definition
- [x] Discovery
- [x] Statistical Validation
- [x] IC
- [x] Decay
- [x] Regime
- [x] Cross-Section
- [x] Neutralization
- [x] Multiple Testing
- [x] Deflated Sharpe
- [x] PBO
- [x] Purged
- [x] Embargo
- [x] TC
- [x] Capacity
- [x] Correlation
- [x] Redundancy
- [x] Clustering
- [x] Ensemble
- [x] Ranking
- [x] Lifecycle
- [x] Monitoring
- [x] Retirement
- [x] Experiment Registry
- [x] Alpha Research Engine

## Architectural rules

- Statistical significance alone ≠ alpha
- Historical Sharpe alone cannot approve
- economic_hypothesis required for APPROVED
- Alpha approval ≠ trading approval — Risk Intelligence is not bypassed
- Point-in-time only — no future leakage in signal helpers
- Rejected experiments are preserved in the registry
- Discovery emits candidates, not approved alpha

Machine-readable report: `Phase11_AlphaResearch_Validation.json`.

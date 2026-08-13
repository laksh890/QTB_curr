# Weighting

Member weight schemes for the ensemble.

## Methods

| Method | Signal |
|--------|--------|
| `equal` | Uniform |
| `accuracy` | Full-sample hard accuracy |
| `recent_accuracy` | Trailing window accuracy |
| `log_likelihood` | Softmax of proxy LL |
| `calibration` | Inverse ECE |
| `stability` | Inverse total variation of max-proba |
| `user` | Hydra `user_weights` |
| `rolling` | Mean of score matrix |
| `adaptive` | EMA update of instantaneous scores |

## Online

`partial_fit` applies `adaptive_update` when `online.weight_update=true`.

## Location

`iqrp/app/regimes/ensemble/weighting.py`

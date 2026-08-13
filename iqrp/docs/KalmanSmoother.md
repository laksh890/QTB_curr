# Kalman Smoother

Rauch–Tung–Striebel (RTS) fixed-interval smoother.

## Location

`iqrp/app/regimes/kalman/smoothing.py`

## Recursion

Given filtered means/covariances and one-step predictions:

\[
G_t = P_{t|t} F^\top P_{t+1|t}^{-1}
\]
\[
\hat{x}_{t|T} = \hat{x}_{t|t} + G_t(\hat{x}_{t+1|T} - \hat{x}_{t+1|t})
\]
\[
P_{t|T} = P_{t|t} + G_t(P_{t+1|T} - P_{t+1|t})G_t^\top
\]

## Usage

```python
model.fit(y)
result = model.smooth(y)          # soft discrete view
means = model.smoothed_means()    # continuous RTS means
```

Smoothed continuous trajectories are also stored in `SmootherResult.metadata["means"]`.

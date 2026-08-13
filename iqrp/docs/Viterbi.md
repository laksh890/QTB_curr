# Viterbi Decoding

Most likely hidden-state path in log-space.

## Module

`iqrp/app/regimes/hmm/viterbi.py`

```python
from iqrp.app.regimes.hmm import viterbi

result = viterbi(log_emissions, transition, initial=pi0)
print(result.states, result.log_prob, result.confidence[:5])
```

Uses backpointers for O(TK) traceback and per-step softmax confidence from
δ scores. `HiddenMarkovModel.decode` / `predict` expose this API.

# Optimization Engine

`iqrp.app.math.optimization` — numerical methods only.

## Methods

- Newton (scalar)
- BFGS (SciPy)
- Gradient descent + numerical gradients
- Projected gradient descent
- Golden-section search
- Root finding: bisection, secant, Brent

```python
from iqrp.app.math.optimization import gradient_descent, find_root

result = gradient_descent(lambda x: float((x**2).sum()), [1.0, 2.0], lr=0.1)
root = find_root(lambda z: z**2 - 2, 0.0, 2.0, method="brent")
```

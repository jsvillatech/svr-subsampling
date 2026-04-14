# Benchmark: Friedman #1

## Why this dataset?

The Friedman #1 benchmark is a standard synthetic dataset widely used
to evaluate regression algorithms. It is particularly well-suited for
this work because:

- It is **fully reproducible** — generated programmatically, no
  external file needed
- It has **controlled complexity** — 20 features but only 5 are
  relevant, which tests whether the subsampling algorithm can still
  identify informative support vectors in the presence of noise features
- It has a **nonlinear target** — making it a meaningful challenge
  for SVR beyond simple linear problems
- It **scales easily** — we can generate any number of samples,
  allowing us to study scalability from n=1,000 to n=50,000

## Generation

```python
from sklearn.datasets import make_friedman1

X, y = make_friedman1(
    n_samples=n,      # varies: 1,000 to 50,000
    n_features=20,
    noise=1.0,
    random_state=42
)
```

## Description

- **Features:** 20 independent features uniformly distributed on [0, 1]
- **Relevant features:** only 5 of 20 are used to compute y
- **Target formula:**
  `y = 10·sin(π·X₀·X₁) + 20·(X₂ − 0.5)² + 10·X₃ + 5·X₄ + noise·N(0,1)`
- **Noise:** Gaussian, σ = 1.0
- **Dataset sizes evaluated:** 1,000 / 5,000 / 10,000 / 15,000 /
  20,000 / 30,000 / 40,000 / 45,000 / 50,000
- **Split:** 80% training / 20% test (`random_state=42`)

## References

- Friedman, J.H. (1991). *Multivariate Adaptive Regression Splines.*
  The Annals of Statistics, 19(1), 1–67.
  https://doi.org/10.1214/aos/1176347963

- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.*
  JMLR, 12, 2825–2830.
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html

# SVRSubsample

**Residual-based subsampling for scalable Support Vector Regression (SVR)**

Implementation and statistical validation of a subsampling algorithm for SVR that extends the SVM classification framework of [Camelo et al.](https://link.springer.com/article/10.1007/s10479-015-1956-8) to the regression setting, replacing its spatial nearest-neighbor selection with a residual-based criterion — selecting candidate points based on similarity of prediction residuals rather than Euclidean distance. This makes the method more robust in high-dimensional feature spaces where distance metrics become unreliable.

This repository accompanies the paper:

> **Jhoan Delgado, Anibal Sosa, and M. D. Gonzalez-Lima** (2026). *An Implementation of a Subsampling Algorithm for Support Vector Regression.* Applied and Industrial Mathematics in Colombia — Extended Abstracts, Springer.

---

## Key Results

| Method | Max Speedup | vs Traditional SVR |
|--------|-------------|-------------------|
| Residual (ours) | 340× at n=50,000 | p < 0.0001 (Wilcoxon) |
| Spatial (Camelo et al.) | 284× at n=50,000 | p < 0.0001 (Wilcoxon) |

Residual criterion achieves significantly higher speedups than spatial criterion across all dataset sizes (Wilcoxon signed-rank test, p = 0.0081), with comparable predictive accuracy (ΔR² < 0.05 for n ≥ 40,000).

---

## Repository Structure

```
SVRSubsample/
│
├── README.md
│
├── Library/
│   └── svr_residual_subsample.py      # Core algorithm (SVRSubsampleOptimizer)
│
├── Notebooks/
│   ├── multi_run_experiment.ipynb     # 5-run statistical validation
│   └── visualizer.ipynb               # Plot generation (mean ± std bands)
│
├── Results/
│   ├── all_runs_raw.csv               # Raw results — one row per (n, seed, criterion)
│   ├── summary_stats.csv              # Mean ± std per dataset size × criterion
│   └── wilcoxon_results.csv           # 3 Wilcoxon signed-rank tests
│
├── Plots/
│   ├── Plot_1.png                     # Training time (log scale)
│   ├── Plot_2.png                     # Support vector count
│   └── Plot_3.png                     # Speedup comparison with Wilcoxon annotation
│
└── Data/
    └── friedman1_description.md       # Benchmark description and generation code
```

---

## Requirements

```bash
pip install numpy pandas scikit-learn scikit-optimize scipy matplotlib seaborn
```

---

## How to Reproduce

### 1. Run the multi-run experiment

Open `Notebooks/Statistical Tests Friedman SVR Paper.ipynb` in Google Colab or Jupyter.

The notebook runs both subsampling methods (Residual and Spatial) across:
- 9 dataset sizes: n ∈ {1,000, 5,000, 10,000, 15,000, 20,000, 30,000, 40,000, 45,000, 50,000}
- 5 independent seeds: {42, 123, 456, 789, 1011}
- 1 Traditional SVR baseline per n (fixed `random_state=42`)

Results are saved automatically to `Results/`. The notebook includes checkpoint logic — if interrupted, it resumes from the last completed run.

### 2. Generate the plots

Open `Notebooks/Plots Subsampling.ipynb` and run all cells. It reads from `Results/summary_stats.csv` and produces the three figures in `Plots/`.

### 3. Run the Wilcoxon tests

The Wilcoxon signed-rank tests are included at the end of `Statistical Tests Friedman SVR Paper.ipynb` and saved to `Results/wilcoxon_results.csv`. Three tests are performed:

| Test | Question | Result |
|------|----------|--------|
| Test 1 | Is Residual speedup > 1? | p < 0.0001 ✅ |
| Test 2 | Is Spatial speedup > 1? | p < 0.0001 ✅ |
| Test 3 | Is Residual speedup > Spatial speedup? | p = 0.0081 ✅ |

---

## Algorithm Overview

The algorithm extends Camelo et al.'s iterative subsampling framework to SVR, with one key modification: **residual-based neighbor selection**.

Instead of selecting the k nearest neighbors by Euclidean distance, we select the k points from the remaining pool whose prediction residuals are closest to those of the current support vectors:

```
r(x) = |f(x) - y| - ε
```

Points with similar residuals to current support vectors are likely near the ε-tube boundary and thus most informative for the next training iteration.

**Parameters used in experiments:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| δ | 0.01 | Initial subsample fraction |
| ε | 0.1 | SVR tube width |
| ρ | 0.1 | Replenishment fraction |
| k | 5 | Neighbors per support vector |
| Kernel | RBF | — |
| HPO | Bayesian | via scikit-optimize |

---

## Citation

```bibtex
@inproceedings{delgado2026svr,
  title     = {An Implementation of a Subsampling Algorithm for Support Vector Regression},
  author    = {Delgado, Jhoan and Sosa, Anibal and Gonzalez-Lima, M. D.},
  booktitle = {Applied and Industrial Mathematics in Colombia --- Extended Abstracts},
  publisher = {Springer},
  year      = {2026}
}
```

---

## Related

- **Development archive:** [SVRSubsample-archive](https://github.com/jsvillatech/SVRSubsample-archive) — contains the original implementation and early experiments.
- **Camelo et al. (2015):** *Nearest neighbors methods for support vector machines.* The framework this work extends.

---



## Authors

**Jhoan Delgado** — Faculty of Engineering, Design and Applied Sciences, Universidad Icesi, Cali, Colombia
`jhoan.delgado@u.icesi.edu.co`
 
**Anibal Sosa** — Department of Physical, Exact and Energy Sciences, Universidad Icesi, Cali, Colombia
`uasosa@icesi.edu.co`
 
**M. D. Gonzalez-Lima** — Department of Mathematics, University of Puerto Rico at Rio Piedras, San Juan, Puerto Rico
`maria.gonzalez168@upr.edu`

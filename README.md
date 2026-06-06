> Download `mnist.npz` from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) or load via `tf.keras.datasets.mnist` and save with `np.savez`.

---

## Algorithms

### AdaBoost 

- **Weak learner:** Decision stump — threshold on a single PCA feature
- **Weight update:** Standard AdaBoost exponential reweighting
- **Alpha computation:** `α = 0.5 * log((1 - err) / err)`
- **Stopping:** 300 iterations; best iteration chosen by validation accuracy
- **Stump search:** Up to 1000 candidate thresholds per feature (midpoints between unique values)

### Gradient Boosting 

- **Loss:** Absolute loss (L1) — pseudo-residuals are `sign(y - F)`
- **Weak learner:** SSR (Sum of Squared Residuals) regression stump
- **Update rule:** `F ← F + η * h`
- **Stopping:** 300 iterations; best by minimum validation MSE
- **Learning rate sweep:** `η ∈ {0.001, 0.01, 0.1, 0.2, 0.5, 1.0}`

### Perceptron 

- **Update rule:** `w ← w + η * y_i * x_i`, `b ← b + η * y_i` on misclassified samples
- **Learning rate:** `η = 0.01`
- **Max epochs:** 300
- **Convergence check:** Stops early if zero misclassifications in one full epoch
- **Tested on:** Dataset A (`cov = I`) and Dataset B (`cov = 3I`)

---

## Results

### AdaBoost

| Metric | Value |
|---|---|
| Best iteration | 96 |
| Validation accuracy | **81.25%** |
| Test accuracy | **79.96%** |

Validation accuracy plateaus around 80–81% after ~50 iterations, with diminishing alpha values indicating weak stumps near random guessing in later rounds.

---

### Gradient Boosting (Learning Rate Sweep)

| η | Best Iter | Train MSE | Val MSE | Test MSE |
|---|---|---|---|---|
| 0.001 | 300 | 0.9317 | 0.9215 | 0.9289 |
| **0.010** | **115** | **0.8686** | **0.8269** | **0.8571** ✅ |
| 0.100 | 12 | 0.8710 | 0.8273 | 0.8588 |
| 0.200 | 6 | 0.8710 | 0.8274 | 0.8586 |
| 0.500 | 2 | 0.8660 | 0.8311 | 0.8555 |
| 1.000 | 1 | 0.8661 | 0.8311 | 0.8555 |

**Best model:** `η = 0.01` with 115 iterations — achieves lowest validation MSE of `0.8269`.

Small `η` (0.001) underfits within 300 iterations. Large `η` (≥ 0.1) converges in very few iterations but oscillates. `η = 0.01` offers the best bias-variance tradeoff.

---

### Q3 — Perceptron

| Dataset | Covariance | Convergence Epoch | Test Accuracy |
|---|---|---|---|
| A | `I` (tight) | Epoch **2** | **100.00%** |
| B | `3I` (spread) | Did not converge | **97.50%** |

Dataset A is linearly separable — the perceptron converges in 2 epochs with perfect test accuracy. Dataset B's wider spread causes class overlap so the perceptron never fully converges within 300 epochs, though it still achieves 97.5% accuracy.

---

## Key Design Choices

- **PCA (k=5) fitted on train only** — val and test sets are projected using training mean and components to prevent data leakage.
- **Threshold sampling** — stump search samples up to 1000 midpoints per feature to keep runtime feasible on 9791 × 5 data.
- **Absolute loss for gradient boosting** — pseudo-residuals are `sign(y - F)`, making stumps fit `{-1, +1}` targets and scale naturally with the binary label convention.
- **Early stopping via validation** — best iteration selected by minimum val MSE (Q2) or maximum val accuracy (Q1), not by a fixed budget.
- **Perceptron tie-breaking** — `sign(0)` is treated as `-1` to avoid ambiguous updates on zero-score samples.
- **Uniform weight initialization** — AdaBoost starts with weights `1/n`; renormalized after every update to maintain a valid distribution.

# Hyperparameter Selection Guide — LDA Wine Classification

**Purpose:** Document the reasoning behind every hyperparameter choice, backed by experimental evidence from cross-validation.

---

## Executive Summary

| Hyperparameter | Value | Selection Method | Reasoning |
|:--|:--|:--|:--|
| **n_components (K)** | 2 | Rank constraint + Elbow Method | C−1 = 2 captures 100% discriminative variance |
| **shrinkage (α)** | 0.20 | 5-fold CV Grid Search | Best CV accuracy (99.46%) |
| **test_size** | 0.25 | Standard ML practice | 133 train / 45 test is balanced for this dataset |
| **k_folds** | 5 | Bias-variance tradeoff | Standard in ML literature (Kohavi, 1995) |
| **random_seed** | 42 | Reproducibility | Arbitrary but fixed for consistent results |

---

## Evaluation Metrics Rationale

Before discussing individual hyperparameters, it is critical to understand **why** we chose specific metrics to evaluate them:

1. **Mean Cross-Validation (CV) Accuracy:**
   - **Why Accuracy?** The Wine dataset has relatively balanced classes (59, 71, and 48 samples). In balanced datasets, pure accuracy (Total Correct / Total Samples) is an intuitive and robust metric. If the classes were highly imbalanced (e.g., 90% Class 1, 10% Class 2), we would have chosen F1-score or Balanced Accuracy to prevent majority-class bias.
   - **Why Cross-Validation?** Evaluating hyperparameters on a single train/test split can lead to overfitting to that specific split. Stratified 5-fold CV evaluates the model across 5 distinct holdout sets and averages the results, providing a much more robust estimate of how the hyperparameter will generalize to unseen data.

2. **Explained Variance Ratio:**
   - **Why this metric?** Used specifically for tuning the number of components (`K`), this ratio ($\lambda_i / \sum \lambda_j$) directly measures the percentage of discriminative power (between-class separation) captured by each axis. It allows us to mathematically quantify how much information we lose if we drop a component, independently of the classifier we use afterward.

---

## 1. Number of Components (K)

### What It Controls
K determines the dimensionality of the LDA-projected space. LDA reduces data from D features (13 for Wine) to K dimensions.

### Mathematical Constraint
$$K \leq C - 1$$

where C is the number of classes. For Wine dataset (C=3), K ∈ {1, 2}.

**Why this upper bound exists:**
The between-class scatter matrix $S_B = \sum_{c=1}^{C} N_c (\mu_c - \mu)(\mu_c - \mu)^T$ is the sum of C rank-1 matrices. However, because $\sum N_c (\mu_c - \mu) = 0$, only C−1 of these are linearly independent. Therefore rank($S_B$) ≤ C−1, and the generalized eigenvalue problem $S_W^{-1} S_B \mathbf{w} = \lambda \mathbf{w}$ has at most C−1 non-zero eigenvalues.

### Experimental Evidence

| K | Eigenvalue | Variance % | Cumulative % | CV Accuracy |
|:--|:--|:--|:--|:--|
| 1 | 7.1637 | 65.58% | 65.58% | 91.60% |
| **2** | **3.7599** | **34.42%** | **100.00%** | **99.46%** |

### Why K=2 Was Chosen

1. **100% variance captured:** K=2 captures all non-zero eigenvalues, meaning NO discriminative information is lost. This is the mathematical maximum.

2. **Dramatic accuracy improvement:** K=1 gives 91.60% accuracy, while K=2 gives 99.46%. The second discriminant axis (LD2) captures 34.42% of discriminative variance, which is essential for separating classes that overlap along LD1.

3. **Visual interpretability:** K=2 allows 2D scatter plots, which are ideal for presentations and human interpretation. K=1 would only give a 1D histogram, losing the spatial structure between classes.

4. **No overfitting risk:** Since K=2 is the theoretical maximum and we're not adding noise dimensions, there is no overfitting penalty.

### How to Choose K for Other Datasets (Elbow Method)
For datasets with many classes (e.g., 10 classes → K_max = 9):
1. Compute all C−1 eigenvalues
2. Plot cumulative explained variance ratio vs. K
3. Choose K at the "elbow" where marginal gain becomes negligible
4. Typically, K=3–5 captures 95%+ of discriminative variance

---

## 2. Shrinkage Parameter (α)

### What It Controls
Shrinkage regularizes the within-class scatter matrix $S_W$:

$$S_{W,reg} = (1 - \alpha) \cdot S_W + \alpha \cdot \frac{\text{tr}(S_W)}{D} \cdot I$$

This interpolates between:
- **α = 0:** Use the empirical $S_W$ directly (no regularization)
- **α = 1:** Replace $S_W$ with a scaled identity matrix (maximum regularization)

### Why Shrinkage Is Needed
$S_W$ must be positive-definite (all eigenvalues > 0) to compute $S_W^{-1/2}$. When $S_W$ is ill-conditioned (near-singular), small eigenvalues cause $S_W^{-1/2}$ to explode, amplifying noise. Shrinkage prevents this by ensuring a minimum eigenvalue floor.

The target $\frac{\text{tr}(S_W)}{D} \cdot I$ preserves the average variance while making all directions equally weighted, eliminating ill-conditioning.

### Experimental Evidence (5-Fold Stratified CV)

| α | Mean CV Accuracy | Std | Interpretation |
|:--|:--|:--|:--|
| 0.00 | 98.89% | 1.36% | No regularization — already good, S_W is well-conditioned |
| 0.10 | 98.89% | 1.36% | Light regularization — same result |
| **0.20** | **99.46%** | **1.08%** | **★ Best: slight regularization improves stability** |
| 0.30 | 97.81% | 2.69% | Over-regularization starts |
| 0.40–0.50 | 97.81% | 2.69% | Plateau — losing covariance structure |
| 0.60–0.90 | 97.22% | 2.45% | Significant over-regularization |
| 1.00 | 66.75% | 12.53% | ✗ Collapsed — S_W is identity, no feature correlation used |

### Why α=0.20 Was Chosen

1. **Highest mean accuracy (99.46%):** Among all candidates, α=0.20 achieved the best average cross-validation accuracy.

2. **Lowest variance (1.08%):** The standard deviation across folds is the smallest, indicating the most consistent performance across different data splits.

3. **Slight regularization is beneficial:** Even though $S_W$ is well-conditioned for Wine (n=178 >> D=13), a small α=0.20 regularization improves generalization by:
   - Reducing sensitivity to outliers in the training data
   - Smoothing out minor ill-conditioning in less-represented feature directions
   - Adding a slight bias that reduces overall variance

4. **Over-regularization hurts:** At α ≥ 0.30, the model starts discarding valuable covariance information. At α=1.0, S_W becomes identity, and LDA degenerates to a simple class centroid comparison — losing 33% accuracy.

### When to Use Different α Values

| Scenario | Recommended α | Reason |
|:--|:--|:--|
| n >> D (many samples, few features) | 0.0–0.2 | S_W is well-conditioned |
| n ≈ D (samples ≈ features) | 0.3–0.6 | Need moderate regularization |
| n < D (high-dimensional, e.g., genomics) | 0.5–0.9 | S_W is rank-deficient |
| Highly collinear features | 0.4–0.7 | S_W is near-singular |
| **Always:** | Use CV grid search | Data-driven selection |

### Alternative: Ledoit-Wolf Analytical Estimation
Instead of grid search, the Ledoit-Wolf formula provides an analytical estimate of optimal shrinkage:

$$\hat{\alpha}^* = \frac{\sum_{i \neq j} \text{Var}(\hat{\sigma}_{ij})}{\sum_{i \neq j} \sigma_{ij}^2}$$

This estimates the optimal α without cross-validation but requires computing fourth-moment statistics. For educational purposes, we use grid search + CV which is more intuitive and easier to verify.

---

## 3. Test/Train Split Ratio (test_size = 0.25)

### Why 75/25

| Split | Train Samples | Test Samples | Assessment |
|:--|:--|:--|:--|
| 90/10 | 160 | 18 | Too few test samples for reliable accuracy |
| 80/20 | 142 | 36 | Good, but slightly less test data |
| **75/25** | **133** | **45** | **Balanced: enough train data + reliable test evaluation** |
| 70/30 | 125 | 53 | Good alternative |
| 50/50 | 89 | 89 | Too few training samples for LDA |

**Key reasoning:**
- LDA needs at least D+1 = 14 training samples (13 features + 1) to have a non-singular S_W. With 133 training samples, we are ~10× above this minimum.
- 45 test samples across 3 classes (≈15 per class) gives statistically meaningful accuracy estimates.
- The 75/25 split is a standard choice in ML literature, balancing training data volume with test set reliability.

---

## 4. Cross-Validation Folds (k = 5)

### Why 5 Folds

| k | Train/Fold | Bias | Variance | Computation |
|:--|:--|:--|:--|:--|
| 2 | 50% | High | Low | Fast |
| 3 | 67% | Moderate | Moderate | Fast |
| **5** | **80%** | **Low** | **Low** | **Moderate** |
| 10 | 90% | Very low | Higher | Slow |
| n (LOO) | 99.4% | Minimal | High | Very slow |

**Key reasoning:**
- k=5 is the standard recommended by Kohavi (1995) and used widely in ML benchmarks
- Each fold uses 80% of data for training (142 samples), leaving 36 for validation
- Stratification ensures each fold has ~12 Class 1, ~14 Class 2, ~10 Class 3 samples
- 5 folds × 11 shrinkage values = 55 model fits — fast enough in pure Python (~3 seconds)

---

## 5. Random Seed (seed = 42)

### Why 42
The value 42 is arbitrary but widely used as a convention in ML (referencing Douglas Adams' *Hitchhiker's Guide*). What matters is:
1. **Fixed:** Same seed = same train/test split = reproducible results
2. **Documented:** Anyone can reproduce the exact experiment
3. **Not cherry-picked:** The results should hold for other seeds (verified: seeds 0, 1, 7, 42, 123 all give similar accuracy ranges)

---

## 6. Comparison: How LDA Hyperparameter Selection Differs from Other Algorithms

| Algorithm | Key Hyperparameter | Selection Method | Analogy to LDA |
|:--|:--|:--|:--|
| **LDA** | shrinkage (α) | CV Grid Search | Controls S_W regularization |
| **KNN** | k (neighbors) | CV Grid Search | Controls decision boundary smoothness |
| **K-Means** | k (clusters) | Elbow Method (Inertia) | Like LDA's n_components (Elbow on eigenvalues) |
| **SVM** | C, γ | CV Grid Search | C controls regularization (similar to α) |
| **Decision Tree** | max_depth | CV + Pruning | Controls model complexity |

**Key insight:** LDA's shrinkage parameter plays the same role as regularization in ridge regression or the C parameter in SVM — it controls the bias-variance tradeoff. Cross-validation is the standard way to tune it.

---

## Appendix: Full Experiment Output

The complete experiment log with per-fold accuracies is saved at:
`outputs/reports/hyperparameter_experiment.txt`

To reproduce:
```bash
python -m src.evaluation.hyperparameter_experiment
```

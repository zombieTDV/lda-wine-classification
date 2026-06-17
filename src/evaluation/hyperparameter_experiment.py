"""
hyperparameter_experiment.py
============================
Hyperparameter experimentation and cross-validation for LDA.

This module provides systematic methods to select LDA hyperparameters
using stratified k-fold cross-validation, rather than hardcoding values.

Hyperparameters Investigated
----------------------------
1. **Shrinkage (α)**: Regularization of S_W via Ledoit-Wolf style blending.
   - Grid search over α ∈ [0.0, 0.1, 0.2, ..., 1.0]
   - Evaluated by stratified 5-fold cross-validation accuracy

2. **Number of Components (K)**: Dimensionality of the LDA projection.
   - K ∈ {1, 2, ..., C-1} where C = number of classes
   - Evaluated by explained variance ratio and cross-validation accuracy

Why Cross-Validation?
---------------------
A single train/test split can give biased estimates of model performance.
K-fold cross-validation averages over multiple splits, giving a more robust
estimate of generalization accuracy. Stratification ensures each fold
preserves the class distribution of the full dataset.

All operations built from scratch — no scikit-learn, numpy, or scipy.
"""

import sys
import os
import math
import time

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.data_loader import load_wine, standardize, train_test_split
from src.lda.lda_model import LDA
from src.lda.math_utils import stratified_k_fold
from src.evaluation.metrics import accuracy
from src.visualization.plots import line_chart_shrinkage, bar_chart_k_accuracy, bar_variance


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: Grid Search for Shrinkage Parameter (α)
# ═══════════════════════════════════════════════════════════════════════════════

def grid_search_shrinkage(X, y, shrinkage_values=None, k_folds=5, seed=42):
    """
    Find the optimal shrinkage parameter α via stratified k-fold cross-validation.

    For each candidate α, this function:
    1. Splits the data into k stratified folds (preserving class proportions)
    2. For each fold: standardizes using training statistics, fits LDA with α,
       predicts on the validation fold, and computes accuracy
    3. Averages accuracy across folds to get a robust estimate

    Why grid search for shrinkage?
    ------------------------------
    The shrinkage parameter α controls the regularization of the within-class
    scatter matrix S_W:
        S_W_reg = (1 - α) * S_W + α * (tr(S_W) / D) * I

    - α = 0: No regularization — uses the empirical S_W directly.
      Risk: If features are collinear or n_samples is small, S_W may be
      near-singular, causing numerical instability in S_W^{-1/2}.
    - α = 1: Maximum regularization — replaces S_W with a scaled identity.
      Risk: Discards all correlation structure, reducing LDA to a simpler model.
    - Optimal α: Balances bias (from regularization) and variance (from
      using a potentially unstable empirical estimate).

    Unlike KNN (where k has a clear geometric interpretation), the optimal α
    depends on the conditioning of S_W, which varies with the dataset. Hence,
    cross-validation is the standard approach.

    Parameters
    ----------
    X : list[list[float]]
        Feature matrix (n_samples × n_features), already standardized
    y : list[int]
        Class labels
    shrinkage_values : list[float], optional
        Grid of α values to search. Default: [0.0, 0.1, ..., 1.0]
    k_folds : int, default=5
        Number of cross-validation folds
    seed : int, default=42
        Random seed for fold generation

    Returns
    -------
    results : list[dict]
        Each dict contains: {'alpha', 'mean_accuracy', 'std_accuracy', 'fold_accuracies'}
    best_alpha : float
        The α value with highest mean cross-validation accuracy
    """
    if shrinkage_values is None:
        shrinkage_values = [i / 10.0 for i in range(11)]  # [0.0, 0.1, ..., 1.0]

    # Generate stratified folds (same folds for all α values = fair comparison)
    folds = stratified_k_fold(y, k=k_folds, seed=seed)

    results = []
    best_alpha = None
    best_mean_acc = -1.0

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 1: Shrinkage Grid Search (α)")
    print(f"  Method: Stratified {k_folds}-Fold Cross-Validation")
    print(f"{'='*70}")
    print(f"  {'α':>6}  {'Mean Acc':>9}  {'Std':>7}  {'Fold Accuracies'}")
    print(f"  {'-'*64}")

    for alpha in shrinkage_values:
        fold_accs = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            # Extract train/val data for this fold
            X_train_fold = [X[i] for i in train_idx]
            y_train_fold = [y[i] for i in train_idx]
            X_val_fold = [X[i] for i in val_idx]
            y_val_fold = [y[i] for i in val_idx]

            # Standardize using training fold statistics only (prevent data leakage)
            X_train_s, X_val_s, _, _ = standardize(X_train_fold, X_val_fold)

            # Fit LDA with this shrinkage value
            model = LDA(n_components=2, shrinkage=alpha)
            model.fit(X_train_s, y_train_fold)

            # Predict on validation fold
            y_pred = model.predict(X_val_s)

            # Compute accuracy
            fold_acc = accuracy(y_val_fold, y_pred)
            fold_accs.append(fold_acc)

        # Compute mean and standard deviation of fold accuracies
        mean_acc = sum(fold_accs) / len(fold_accs)
        std_acc = math.sqrt(sum((a - mean_acc) ** 2 for a in fold_accs) / len(fold_accs))

        fold_str = "  ".join(f"{a:.3f}" for a in fold_accs)
        print(f"  {alpha:>6.2f}  {mean_acc:>9.4f}  {std_acc:>7.4f}  [{fold_str}]")

        results.append({
            'alpha': alpha,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_accuracies': fold_accs
        })

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_alpha = alpha

    print(f"  {'-'*64}")
    print(f"  ★ Best α = {best_alpha:.2f}  (Mean Accuracy = {best_mean_acc:.4f})")
    print(f"{'='*70}\n")

    return results, best_alpha


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: Number of Components (K)
# ═══════════════════════════════════════════════════════════════════════════════

def experiment_n_components(X, y, best_shrinkage=0.0, k_folds=5, seed=42):
    """
    Evaluate the effect of the number of LDA components (K) on performance.

    For each K from 1 to C-1 (where C = number of classes), this function
    computes explained variance ratio and cross-validation accuracy.

    Why K = C-1 is the maximum for LDA?
    ------------------------------------
    The between-class scatter matrix S_B has rank at most C-1 (where C is
    the number of classes). This is because S_B is the sum of C rank-1
    outer products minus a correction, and these C vectors lie in a
    (C-1)-dimensional subspace (they all sum to zero when weighted).

    Therefore, S_W^{-1} S_B has at most C-1 non-zero eigenvalues, and
    extracting more than C-1 components would only add noise dimensions.

    For the Wine dataset (C=3):
    - K=1: Projects to 1D. Only captures the dominant discriminant axis.
    - K=2: Projects to 2D. Captures ALL discriminative information (100%).
    - K=3+: Impossible (rank constraint). No additional information.

    How to choose K in general?
    ----------------------------
    Use the "Elbow Method" on the cumulative explained variance ratio:
    plot the cumulative sum of eigenvalues and choose K at the point
    where adding more components gives diminishing returns.

    Parameters
    ----------
    X : list[list[float]]
        Feature matrix (n_samples × n_features)
    y : list[int]
        Class labels
    best_shrinkage : float
        Shrinkage parameter to use (from grid search result)
    k_folds : int
        Number of cross-validation folds
    seed : int
        Random seed

    Returns
    -------
    results : list[dict]
        Each dict: {'k', 'explained_variance', 'cumulative_variance', 'cv_accuracy'}
    """
    n_classes = len(set(y))
    max_k = n_classes - 1

    folds = stratified_k_fold(y, k=k_folds, seed=seed)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 2: Number of Components (K)")
    print(f"  C = {n_classes} classes → K_max = C - 1 = {max_k}")
    print(f"  Using best shrinkage α = {best_shrinkage:.2f}")
    print(f"{'='*70}")

    # First, fit one model with max K to get all eigenvalues.
    # To match main.py and prevent data leakage, we evaluate this on a 75% 
    # training split rather than the full dataset.
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, seed=seed)
    X_train_s, _, _ = standardize(X_train)
    
    full_model = LDA(n_components=max_k, shrinkage=best_shrinkage)
    full_model.fit(X_train_s, y_train)
    all_eigenvalues = full_model._eigenvalues
    all_evr = full_model.explained_variance_ratio_

    print(f"\n  Eigenvalue Analysis:")
    print(f"  {'K':>4}  {'Eigenvalue':>12}  {'Var. Ratio':>11}  {'Cumulative':>11}")
    print(f"  {'-'*44}")
    cum_var = 0.0
    for k in range(max_k):
        cum_var += all_evr[k]
        print(f"  {k+1:>4}  {all_eigenvalues[k]:>12.4f}  {all_evr[k]*100:>10.2f}%  {cum_var*100:>10.2f}%")

    # Now evaluate CV accuracy for each K
    print(f"\n  Cross-Validation Accuracy by K:")
    print(f"  {'K':>4}  {'CV Accuracy':>12}  {'Std':>7}")
    print(f"  {'-'*28}")

    results = []
    cum_var = 0.0
    for k in range(1, max_k + 1):
        cum_var += all_evr[k - 1]
        fold_accs = []

        for train_idx, val_idx in folds:
            X_train_fold = [X[i] for i in train_idx]
            y_train_fold = [y[i] for i in train_idx]
            X_val_fold = [X[i] for i in val_idx]
            y_val_fold = [y[i] for i in val_idx]

            X_train_s, X_val_s, _, _ = standardize(X_train_fold, X_val_fold)

            model = LDA(n_components=k, shrinkage=best_shrinkage)
            model.fit(X_train_s, y_train_fold)
            y_pred = model.predict(X_val_s)
            fold_accs.append(accuracy(y_val_fold, y_pred))

        mean_acc = sum(fold_accs) / len(fold_accs)
        std_acc = math.sqrt(sum((a - mean_acc) ** 2 for a in fold_accs) / len(fold_accs))

        print(f"  {k:>4}  {mean_acc:>12.4f}  {std_acc:>7.4f}")

        results.append({
            'k': k,
            'eigenvalue': all_eigenvalues[k - 1],
            'explained_variance': all_evr[k - 1],
            'cumulative_variance': cum_var,
            'cv_accuracy': mean_acc,
            'cv_std': std_acc
        })

    print(f"{'='*70}\n")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN: Run Full Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_experiment():
    """
    Execute the complete hyperparameter experiment and save results.

    This function:
    1. Loads the Wine dataset
    2. Runs shrinkage grid search (α ∈ [0.0, 0.1, ..., 1.0])
    3. Runs n_components experiment (K ∈ {1, 2})
    4. Prints reasoning for each hyperparameter choice
    5. Saves comprehensive results to outputs/reports/hyperparameter_experiment.txt
    """
    t0 = time.time()

    print("\n" + "═" * 70)
    print("  LDA HYPERPARAMETER EXPERIMENT")
    print("  Systematic selection with reasoning")
    print("═" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    X, y, feature_names = load_wine("data/processed/wine.csv")
    print(f"  Dataset: {len(X)} samples × {len(X[0])} features × {len(set(y))} classes")

    # ── Experiment 1: Shrinkage Grid Search ──────────────────────────────────
    shrinkage_results, best_alpha = grid_search_shrinkage(
        X, y,
        shrinkage_values=[i / 10.0 for i in range(11)],
        k_folds=5,
        seed=42
    )

    # ── Experiment 2: Number of Components ───────────────────────────────────
    component_results = experiment_n_components(
        X, y,
        best_shrinkage=best_alpha,
        k_folds=5,
        seed=42
    )

    # ── Generate Charts ──────────────────────────────────────────────────────
    line_chart_shrinkage(shrinkage_results, best_alpha)
    
    # Generate charts for n_components
    best_k = len(set(y)) - 1  # As determined by rank constraint C-1
    bar_chart_k_accuracy(component_results, best_k)
    
    # Extract eigenvalues and explained variance from component_results for bar_variance
    eigenvalues = [r['eigenvalue'] for r in component_results]
    explained_ratios = [r['explained_variance'] for r in component_results]
    bar_variance(eigenvalues, explained_ratios, out_path="outputs/figures/k_explained_variance.svg")

    # ── Summary and Reasoning ────────────────────────────────────────────────
    summary_lines = generate_reasoning_summary(
        shrinkage_results, best_alpha, component_results, X, y
    )

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs("outputs/reports", exist_ok=True)
    report_path = "outputs/reports/hyperparameter_experiment.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\n  Results saved to: {report_path}")

    elapsed = time.time() - t0
    print(f"\n  Experiment completed in {elapsed:.2f}s\n")


def generate_reasoning_summary(shrinkage_results, best_alpha, component_results, X, y):
    """Generate a comprehensive summary with mathematical reasoning."""
    n_samples = len(X)
    n_features = len(X[0])
    n_classes = len(set(y))

    lines = [
        "=" * 70,
        "  LDA HYPERPARAMETER EXPERIMENT — COMPREHENSIVE REPORT",
        "=" * 70,
        "",
        f"  Dataset: Wine (UCI) — {n_samples} samples × {n_features} features × {n_classes} classes",
        "",
        "─" * 70,
        "  1. SHRINKAGE PARAMETER (α) — Grid Search Results",
        "─" * 70,
        "",
        "  Mathematical formulation:",
        "    S_W_reg = (1 - α) * S_W + α * (tr(S_W) / D) * I",
        "",
        "  Grid search results (stratified 5-fold CV):",
        f"  {'α':>6}  {'Mean Acc':>9}  {'Std':>7}",
        f"  {'-'*28}",
    ]

    for r in shrinkage_results:
        marker = " ★" if r['alpha'] == best_alpha else ""
        lines.append(
            f"  {r['alpha']:>6.2f}  {r['mean_accuracy']:>9.4f}  {r['std_accuracy']:>7.4f}{marker}"
        )

    lines += [
        "",
        f"  ★ SELECTED: α = {best_alpha:.2f}",
        "",
        "  REASONING WHY THIS α WAS CHOSEN:",
        f"    • α = {best_alpha:.2f} achieved the highest mean CV accuracy",
        "    • Low shrinkage works well for Wine dataset because:",
        f"      - n_samples ({n_samples}) >> n_features ({n_features}): S_W is well-conditioned",
        "      - The 13 chemical features have low multicollinearity after standardization",
        "      - Minimal regularization preserves the rich covariance structure",
        "    • Higher α values over-regularize, collapsing S_W toward identity",
        "      and discarding valuable correlation information between features",
        "    • Very high α (≥ 0.8) reduces LDA essentially to class centroid",
        "      differences in an isotropic space, losing discriminative power",
        "",
        "  WHEN TO USE HIGHER SHRINKAGE:",
        "    • When n_samples ≈ n_features (high-dimensional data)",
        "    • When features are highly collinear (e.g., spectroscopy data)",
        "    • When S_W is near-singular (det(S_W) ≈ 0)",
        "    • Rule of thumb: start with Ledoit-Wolf analytical estimate,",
        "      then validate with cross-validation",
        "",
    ]

    lines += [
        "─" * 70,
        "  2. NUMBER OF COMPONENTS (K) — Analysis",
        "─" * 70,
        "",
        "  Mathematical constraint: K ≤ C - 1",
        f"  For Wine dataset: K ≤ {n_classes} - 1 = {n_classes - 1}",
        "",
        "  Results:",
        f"  {'K':>4}  {'Eigenvalue':>12}  {'Var %':>8}  {'Cumul %':>9}  {'CV Acc':>8}",
        f"  {'-'*48}",
    ]

    for r in component_results:
        lines.append(
            f"  {r['k']:>4}  {r['eigenvalue']:>12.4f}  {r['explained_variance']*100:>7.2f}%  "
            f"{r['cumulative_variance']*100:>8.2f}%  {r['cv_accuracy']:>8.4f}"
        )

    lines += [
        "",
        f"  ★ SELECTED: K = {n_classes - 1}",
        "",
        "  REASONING WHY K = 2 (ALL COMPONENTS) WAS CHOSEN:",
        "    • With 3 classes, S_B has rank ≤ 2, giving exactly 2 useful directions",
        "    • K=2 captures 100% of the discriminative variance (all non-zero eigenvalues)",
        "    • K=1 loses the second discriminant, reducing separation between",
        "      classes that are similar along the first axis",
        "    • K=2 also enables 2D scatter plot visualization, which is ideal",
        "      for human interpretation and presentation",
        "",
        "  HOW TO CHOOSE K IN GENERAL (Elbow Method):",
        "    • Plot cumulative explained variance vs. K",
        "    • Choose K at the 'elbow' where marginal gain becomes negligible",
        "    • For datasets with many classes (e.g., 10 classes → K_max = 9),",
        "      often K=3-5 captures 95%+ of discriminative variance",
        "    • Balance: higher K preserves more information but increases",
        "      model complexity and risk of overfitting with small datasets",
        "",
    ]

    lines += [
        "─" * 70,
        "  3. OTHER HYPERPARAMETERS",
        "─" * 70,
        "",
        "  TEST_SIZE = 0.25 (75/25 train/test split)",
        "    REASONING:",
        "    • 75% training gives LDA sufficient data to estimate S_W and S_B",
        f"    • With {n_samples} samples and {n_features} features, 75% = {int(n_samples*0.75)}",
        f"      training samples is well above the minimum needed ({n_features}+1 = {n_features+1})",
        "    • 25% test set (≈45 samples) is large enough for reliable accuracy",
        "    • Common choices: 70/30, 75/25, 80/20 — all work well here",
        "    • For very small datasets, use k-fold CV instead of a held-out test set",
        "",
        "  RANDOM_SEED = 42",
        "    REASONING:",
        "    • Fixed seed ensures reproducibility across runs",
        "    • The value 42 is arbitrary (a widely used convention)",
        "    • What matters is consistency: same seed = same results",
        "",
        "  K_FOLDS = 5 (in cross-validation)",
        "    REASONING:",
        "    • k=5 is the standard choice in ML literature (Kohavi, 1995)",
        "    • Balances bias-variance tradeoff of the CV estimator:",
        "      - k=2 (50/50): high bias, each fold uses only half the data",
        f"      - k=5: each fold uses {int(n_samples*0.8)}/{n_samples} ≈ 80% of data",
        "      - k=n (LOO): low bias but high variance and computation cost",
        "    • With stratification, k=5 ensures each fold has representative",
        "      samples from all 3 wine classes",
        "",
    ]

    lines += [
        "─" * 70,
        "  4. SUMMARY TABLE — ALL HYPERPARAMETERS",
        "─" * 70,
        "",
        "  ┌──────────────────┬─────────┬────────────────────────────────────┐",
        "  │ Hyperparameter   │  Value  │  Selection Method                  │",
        "  ├──────────────────┼─────────┼────────────────────────────────────┤",
        f"  │ n_components (K) │    {n_classes-1}    │  Rank constraint + Elbow Method    │",
        f"  │ shrinkage (α)    │  {best_alpha:.2f}   │  5-fold CV Grid Search             │",
        "  │ test_size        │  0.25   │  Standard split (75/25)            │",
        "  │ k_folds (CV)     │    5    │  Bias-variance tradeoff (standard) │",
        "  │ random_seed      │   42    │  Reproducibility convention         │",
        "  └──────────────────┴─────────┴────────────────────────────────────┘",
        "",
        "=" * 70,
    ]

    # Print the summary
    for line in lines:
        print(line)

    return lines


if __name__ == '__main__':
    run_full_experiment()

# Professional Critique and Technical Report: Linear Discriminant Analysis (LDA)

**Project Name:** LDA Wine Classification from Scratch  
**Target Audience:** UTH Machine Learning Course / Presentation Committee  
**Prepared by:** Senior Machine Learning Engineer  
**Date:** June 12, 2026  
**Status:** ⚠️ **CRITICAL MATHEMATICAL & ARCHITECTURAL FLAW IDENTIFIED**

---

## Executive Summary

While this codebase is highly organized, modular, and implements data preprocessing, model fitting, and SVG visualizations without external dependencies (such as `scikit-learn` or `numpy`), it contains a **fundamental mathematical error** in how it solves the generalized eigenvalue problem. Specifically, the method of "symmetrizing" the non-symmetric matrix $S_W^{-1} S_B$ alters the eigenvectors and eigenvalues, meaning the model is **not** optimizing the Fisher discriminant criterion. 

Additionally, the codebase lacks professional standards regarding **numerical stability (using Gauss-Jordan explicit inversion and Power Iteration)**, **hyperparameter tuning**, and **cross-validation**.

This report highlights these flaws, provides the correct mathematical formulations, details how LDA compares to other algorithms, explains how to choose hyperparameters professionally, and provides a slide-by-slide presentation outline.

---

## 1. What is Wrong and Lacking (Professional Critique)

### 1.1. Critical Mathematical Error: Invalid Symmetrization of $S_W^{-1} S_B$
* **The Flaw:** In `src/lda/lda_model.py` (lines 189–191) and documented in `docs/MATHEMATICS.md` (lines 55–58), the codebase attempts to solve $M \mathbf{w} = \lambda \mathbf{w}$ where $M = S_W^{-1} S_B$. Because $M$ is non-symmetric, the code symmetrizes it using:
  $$M_{sym} = \frac{1}{2} (M + M^T)$$
  The report claims this "preserves the eigenvectors that maximize variance directions." **This is mathematically false.**
* **Why it is Wrong:** Symmetrizing a matrix in this way changes both its eigenvalues and eigenvectors. The eigenvectors of $M_{sym}$ do not solve the generalized eigenvalue problem $S_B \mathbf{w} = \lambda S_W \mathbf{w}$. Thus, the projected coordinates do not maximize the Fisher criterion, and the class separation is sub-optimal.
* **Proof by Counterexample:**
  Let $S_W = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$ (symmetric, positive-definite) and $S_B = \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}$ (symmetric).
  $$S_W^{-1} = \begin{pmatrix} 1 & 0 \\ 0 & 0.5 \end{pmatrix}$$
  $$M = S_W^{-1} S_B = \begin{pmatrix} 2 & 1 \\ 0.5 & 1.5 \end{pmatrix}$$
  The true eigenvalues of $M$ are $\lambda_1 \approx 2.56, \lambda_2 \approx 0.94$, with eigenvectors $v_1 \approx \begin{pmatrix} 1 \\ 0.56 \end{pmatrix}, v_2 \approx \begin{pmatrix} 1 \\ -1.06 \end{pmatrix}$.
  
  If we apply the code's symmetrization:
  $$M_{sym} = \frac{1}{2}(M + M^T) = \begin{pmatrix} 2 & 0.75 \\ 0.75 & 1.5 \end{pmatrix}$$
  The eigenvalues of $M_{sym}$ are $\lambda_1 \approx 2.54, \lambda_2 \approx 0.96$, and the eigenvectors are $v_{sym, 1} \approx \begin{pmatrix} 1 \\ 0.72 \end{pmatrix}, v_{sym, 2} \approx \begin{pmatrix} 1 \\ -1.39 \end{pmatrix}$.
  
  The principal projection direction has shifted from $\begin{pmatrix} 1 \\ 0.56 \end{pmatrix}$ to $\begin{pmatrix} 1 \\ 0.72 \end{pmatrix}$. In higher dimensions (like the 13 features of the Wine dataset), this error propagates, leading to incorrect projection directions.

* **The Professional Solution:**
  To solve $S_B \mathbf{w} = \lambda S_W \mathbf{w}$ while maintaining a symmetric eigenvalue problem, we must perform a change of variables using the **symmetric square root** of $S_W$:
  1. Compute the eigendecomposition of the symmetric matrix $S_W$:
     $$S_W = P \Sigma P^T$$
  2. Compute $S_W^{-1/2}$:
     $$S_W^{-1/2} = P \Sigma^{-1/2} P^T$$
  3. Transform the generalized eigenvalue problem into a standard symmetric eigenvalue problem:
     $$A = S_W^{-1/2} S_B S_W^{-1/2}$$
     *(Note: $A$ is symmetric because $S_B$ and $S_W^{-1/2}$ are symmetric.)*
  4. Find the eigenvalues and eigenvectors $\mathbf{u}$ of $A$ (which is symmetric, making power iteration/deflation mathematically valid).
  5. Transform the eigenvectors back to obtain the generalized eigenvectors $\mathbf{w}$:
     $$\mathbf{w} = S_W^{-1/2} \mathbf{u}$$

---

### 1.2. Numerical Instability in Explicit Inversion (Gauss-Jordan)
* **The Flaw:** In `src/lda/math_utils.py` (lines 485–521), the code uses Gauss-Jordan elimination to compute $S_W^{-1}$ explicitly.
* **Why it is Wrong:** Explicitly inverting a matrix is computationally expensive ($O(D^3)$) and numerically unstable. If the features are highly collinear (common in chemical datasets), $S_W$ becomes near-singular (ill-conditioned). Gauss-Jordan will produce massive rounding errors, causing numerical overflow or garbage outputs without raising a `ValueError`.
* **The Professional Standard:** Professional ML libraries (e.g., `scikit-learn`'s `LinearDiscriminantAnalysis`) **never** compute $S_W^{-1}$ explicitly. Instead, they use:
  1. **Singular Value Decomposition (SVD):** Projects the data onto the singular vectors of the centered data matrix. This avoids computing scatter matrices altogether, making it highly robust to collinearity.
  2. **Cholesky Solver:** Computes the Cholesky decomposition of $S_W$ ($S_W = L L^T$) and solves the system using back-substitution, which is far more stable than explicit inversion.

---

### 1.3. Crude Regularization (Shrinkage)
* **The Flaw:** In `src/lda/lda_model.py` (lines 175–182), regularization is applied only as a fallback if Gauss-Jordan fails:
  ```python
  try:
      S_W_inv = mat_inv(S_W)
  except ValueError:
      eps = 1e-4
      for i in range(n_features):
          S_W[i][i] += eps
  ```
* **Why it is Wrong:** 
  1. Near-singularity causes numerical issues *before* Gauss-Jordan raises an error.
  2. Adding a hardcoded $\epsilon = 10^{-4}$ is ad-hoc.
  3. It does not scale with the variance of individual features (it adds the same value regardless of whether feature values are in the range of $[0.1, 1.0]$ or $[100, 1000]$).
* **The Professional Standard:** Regularization (shrinkage) should be parameter-controlled and always applied to ensure stability, or analytically estimated using the **Ledoit-Wolf lemma**:
  $$S_W \leftarrow (1 - \alpha) S_W + \alpha \frac{\text{tr}(S_W)}{D} I$$
  where $\alpha \in [0, 1]$ is a tunable hyperparameter.

---

### 1.4. Limitations of Power Iteration and Deflation
* **The Flaw:** Eigendecomposition is computed using Power Iteration + Hotelling Deflation.
* **Why it is Wrong:**
  1. **Deflation Error Propagation:** Hotelling deflation ($A_{new} = A - \lambda_1 \mathbf{v}_1 \mathbf{v}_1^T$) propagates numerical errors. The second eigenvector ($\mathbf{v}_2$) accumulates the rounding errors of the first, making it less accurate.
  2. **Convergence Failure:** Power iteration converges at a rate proportional to $\left|\frac{\lambda_2}{\lambda_1}\right|$. If the two largest eigenvalues are close in magnitude, power iteration will fail to converge within the 1000 iteration limit.
  3. **Deterministic Initialization Vulnerability:** The code initializes the power iteration vector deterministically: $\mathbf{v}_0 = [1, 1/2, \dots, 1/D]$. If this vector happens to be orthogonal to the principal eigenvector, the power iteration will fail.
* **The Professional Standard:** Use QR decomposition or Jacobi rotations to compute all eigenvalues and eigenvectors simultaneously, which is numerically stable and avoids deflation errors.

---

### 1.5. Lack of Hyperparameter Tuning and Cross-Validation
* **The Flaw:** The project hardcodes `n_components = 2` and has no method for hyperparameter selection.
* **Why it is Wrong:** In a professional project, parameters like the regularization shrinkage ($\alpha$) and the number of components ($K$) should be selected systematically.
* **The Professional Standard:** Use $k$-fold cross-validation or grid search to select the shrinkage parameter $\alpha$ that yields the best generalization accuracy on validation folds.

---

## 2. Hyperparameters in LDA and How to Choose Them

Yes, LDA has hyperparameters. A professional implementation must provide methods to tune them, similar to how K-Means uses the Elbow Method and KNN uses Cross-Validation.

| Hyperparameter | Description | How to Choose It Professionally |
| :--- | :--- | :--- |
| **Number of Components ($K$)** | The dimensionality of the projected space ($K \le C - 1$). | **Cumulative Explained Variance Ratio (Elbow Method):** Plot the cumulative sum of the explained variance ratios of the sorted eigenvalues. Choose $K$ at the "elbow" where adding more components yields diminishing returns. |
| **Shrinkage Parameter ($\alpha$)** | Regularizes the within-class scatter matrix: $S_W \leftarrow (1-\alpha)S_W + \alpha I$. | **Grid Search with Cross-Validation:** Evaluate a range of values (e.g., $\alpha \in [0, 1]$ in steps of 0.1) using stratified $k$-fold cross-validation, selecting the one that maximizes validation accuracy. Alternatively, use the **Ledoit-Wolf formula** for analytical shrinkage estimation. |

---

## 3. LDA Specialties and Comparisons

To present LDA professionally, we must highlight its specialties compared to other standard ML algorithms.

### 3.1. LDA vs. PCA (Principal Component Analysis)
* **Objective:** PCA is **unsupervised**; it finds directions of maximum variance without looking at class labels. LDA is **supervised**; it finds directions that maximize class separability (between-class variance relative to within-class variance).
* **Demonstration via Visualization:**
  * If the direction of maximum variance in the dataset does not align with the direction of class separation, PCA will project the classes on top of each other (mixing them). 
  * LDA will ignore the global variance and align the projection axis directly with the separation boundary, showing distinct, separated clusters.

### 3.2. LDA vs. QDA (Quadratic Discriminant Analysis)
* **Covariance Assumption:** LDA assumes all classes share the **same covariance matrix** ($\Sigma_c = \Sigma \quad \forall c$), leading to **linear decision boundaries**. QDA allows each class to estimate its own covariance matrix ($\Sigma_c$), resulting in **quadratic decision boundaries**.
* **Trade-off:** LDA has fewer parameters to estimate ($O(D)$ parameters per class centroid) and is highly robust to overfitting, especially on small datasets like Wine ($N=178$). QDA has $O(D^2)$ parameters per class, making it prone to overfitting on small datasets but more flexible for complex distributions.

### 3.3. LDA vs. KNN (K-Nearest Neighbors)
* **Parametric vs. Non-parametric:** LDA is **parametric** (assumes a Gaussian distribution for each class). KNN is **non-parametric** (makes no assumptions about data distribution, adapting to arbitrary shapes).
* **Inference Complexity:** KNN requires storing all training data and computing distances to every sample during test time ($O(N \cdot D)$). LDA computes a projection matrix once; classification requires projecting a test sample and computing distances to only $C$ centroids ($O(K \cdot D + C \cdot K)$), making it orders of magnitude faster.

### 3.4. LDA vs. K-Means
* **Task:** K-Means is an **unsupervised clustering** algorithm that groups unlabeled data. LDA is a **supervised classification and dimensionality reduction** algorithm. K-Means finds spherical clusters based purely on spatial distance, while LDA uses class labels to distort the space (by scaling by $S_W^{-1}$) to separate known classes.

---

## 4. Presentation Outline (Slide-by-Slide)

Below is the structured outline for the presentation. This outline directly addresses the requirements of the task, using the visualization outputs to demonstrate key concepts.

### Slide 1: Title Slide
* **Title:** Demystifying Linear Discriminant Analysis (LDA)
* **Subtitle:** Supervised Dimensionality Reduction and Classification on the UCI Wine Dataset
* **Presenter Names:** [Student Names]
* **Key Visual:** Diagram of the 8-stage pipeline.

### Slide 2: Introduction & The Classification Challenge
* **Content:**
  * **The Goal:** Classify 178 wine samples into 3 classes based on 13 chemical features (e.g., alcohol, proline).
  * **The Problem:** 13 dimensions are impossible to visualize and analyze directly.
  * **The Solution:** Dimensionality reduction to 2D for visualization and classification.
* **Key Takeaway:** High-dimensional data requires mathematical projection to extract discriminative features.

### Slide 3: The Core Mathematics of LDA
* **Content:**
  * **Within-Class Scatter ($S_W$):** Measures the spread of samples within each class:
    $$S_W = \sum_{c=1}^{C} \sum_{i \in c} (\mathbf{x}_i - \boldsymbol{\mu}_c)(\mathbf{x}_i - \boldsymbol{\mu}_c)^T$$
  * **Between-Class Scatter ($S_B$):** Measures the spread of class centroids around the global mean:
    $$S_B = \sum_{c=1}^{C} N_c (\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T$$
  * **Fisher's Criterion:** Maximize $J(W) = \frac{\det(W^T S_B W)}{\det(W^T S_W W)}$, leading to:
    $$(S_W^{-1} S_B) \mathbf{w} = \lambda \mathbf{w}$$

### Slide 4: Specialties: LDA vs. PCA (Supervised vs. Unsupervised)
* **Content:**
  * **PCA:** Finds directions of maximum variance. If class separation is along a low-variance direction, PCA fails to separate classes.
  * **LDA:** Specifically maximizes class separability.
* **Visual Demonstration:** Show the comparison of PCA projection vs. LDA projection. 
  *(Explain: PCA scatter plot mixes the classes, while the generated [lda_train.svg](file:///C:/document/Study%20documents/UTH-machine-learning/lda-wine-classification/outputs/figures/lda_train.svg) shows three clearly isolated clusters.)*

### Slide 5: Specialties: LDA vs. QDA vs. KNN
* **Content:**
  * **Boundary Types:** LDA produces linear boundaries (highly robust, low variance). QDA produces quadratic boundaries (more flexible, high variance). KNN produces non-linear boundaries.
  * **Resource Efficiency:** Compare the inference speed of LDA (fast matrix multiplication) vs. KNN (slow distance search over all training data).
* **Key Takeaway:** LDA strikes a balance by providing high accuracy (93.3% test accuracy on Wine) with minimal computational footprint.

### Slide 6: Hyperparameters in LDA (And How to Choose Them)
* **Content:**
  * **Does LDA have hyperparameters? Yes!**
  * **1. Number of Components ($K$):** Capped at $C-1$. 
    * *Selection Method:* Show the [explained_variance.svg](file:///C:/document/Study%20documents/UTH-machine-learning/lda-wine-classification/outputs/figures/explained_variance.svg) bar chart. Point out the "Elbow" where LD1 and LD2 capture 100% of the variance, proving $K=2$ is mathematically sufficient.
  * **2. Shrinkage Parameter ($\alpha$):** 
    * *Selection Method:* Stratified $k$-fold cross-validation grid search to find the optimal trade-off between empirical and diagonal covariance.

### Slide 7: Critical Code Review & Technical Refinements
* **Content:**
  * **Honest Evaluation:** Our project's "from-scratch" code has a critical math flaw in `lda_model.py`.
  * **The Flaw:** Symmetrizing $S_W^{-1} S_B$ as $0.5(M + M^T)$ changes eigenvectors and invalidates the optimization.
  * **The Fix:** We must compute the symmetric square root of $S_W$ ($S_W^{-1/2} S_B S_W^{-1/2}$) to keep the problem symmetric without altering the eigenvectors.
  * **Numerical Improvement:** Replace Gauss-Jordan inversion with Singular Value Decomposition (SVD) or Cholesky solvers to prevent numerical instability in collinear features.

### Slide 8: Experimental Results & Visualizations
* **Content:**
  * **Model Metrics:**
    * Training Accuracy: **99.25%**
    * Test Accuracy: **93.33%**
  * **Visual Proofs:**
    * Refer to the generated confusion matrix heatmap ([confusion_matrix.svg](file:///C:/document/Study%20documents/UTH-machine-learning/lda-wine-classification/outputs/figures/confusion_matrix.svg)).
    * Show the [lda_test.svg](file:///C:/document/Study%20documents/UTH-machine-learning/lda-wine-classification/outputs/figures/lda_test.svg) to demonstrate how unseen samples are successfully projected into distinct, separate class zones.

### Slide 9: Conclusion
* **Summary:**
  * LDA is a powerful linear technique for classification and dimensionality reduction.
  * Standard machine learning relies on robust numerical solvers (SVD, Cholesky) and cross-validation for hyperparameter tuning.
  * Building algorithms from scratch is educational, but exposes critical edge cases (symmetrization errors, solver instability) that highlight the value of professional library engineering.
* **Q&A Session**

---

## 5. Summary of Recommended Enhancements for Codebase

To transition this project from an "educational" grade to a "professional standard," the following changes are recommended:

1. **Refactor the Eigenvalue Solver:** Modify `src/lda/lda_model.py` to perform the symmetric square root transformation of $S_W$ rather than direct symmetrization of $S_W^{-1} S_B$.
2. **Implement Regularized LDA (Shrinkage):** Add a user-configurable parameter `shrinkage` (range `[0, 1]`) and apply it directly to $S_W$ before inversion:
   $$S_W \leftarrow (1 - \text{shrinkage}) \cdot S_W + \text{shrinkage} \cdot I$$
3. **Upgrade Matrix Inversion:** Implement a Cholesky decomposition solver (`mat_cholesky`) in `math_utils.py` to solve $S_W \mathbf{w} = S_B \mathbf{w}$ without explicit inversion.
4. **Implement Hyperparameter Optimization:** Add a grid-search script in a new file `src/evaluation/grid_search.py` that uses cross-validation to select the optimal `shrinkage` parameter.

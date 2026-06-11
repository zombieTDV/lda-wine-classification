# Linear Discriminant Analysis (LDA) Mathematical Theory

This document provides a deep dive into the mathematical formulations and algorithms implemented from scratch in this project. By avoiding external dependencies, the codebase reveals the raw linear algebra and statistical computations required to perform Linear Discriminant Analysis (LDA) and classification.

---

## 1. Fundamentals of Linear Discriminant Analysis

Linear Discriminant Analysis is a supervised dimensionality reduction technique. While Principal Component Analysis (PCA) seeks axes of maximum variance to represent data, LDA seeks axes that maximize class separability.

For a dataset containing $N$ samples, $D$ features, and $C$ classes, LDA maps the feature space $\mathbb{R}^D$ to a lower-dimensional space $\mathbb{R}^K$ (where $K \leq C - 1$) by using a projection matrix $W \in \mathbb{R}^{D \times K}$.

The transformation is defined as:
$$\mathbf{y}_i = W^T \mathbf{x}_i$$
where $\mathbf{x}_i \in \mathbb{R}^D$ is a standardized input vector and $\mathbf{y}_i \in \mathbb{R}^K$ is the projected sample in the LDA space.

### Fisher's Criterion
To find the optimal transformation matrix $W$, we maximize Fisher's criterion, defined as the ratio of between-class scatter to within-class scatter:
$$J(W) = \frac{\det(W^T S_B W)}{\det(W^T S_W W)}$$

---

## 2. Scatter Matrices
The core statistics calculated in [lda_model.py](../src/lda/lda_model.py) are the Within-Class Scatter Matrix ($S_W$) and the Between-Class Scatter Matrix ($S_B$).

### Within-Class Scatter Matrix ($S_W$)
$S_W$ measures the spread of samples around their respective class centroids:
$$S_W = \sum_{c=1}^{C} \sum_{i \in \text{Class } c} (\mathbf{x}_i - \boldsymbol{\mu}_c)(\mathbf{x}_i - \boldsymbol{\mu}_c)^T$$
where:
* $\boldsymbol{\mu}_c \in \mathbb{R}^D$ is the mean vector of class $c$:
  $$\boldsymbol{\mu}_c = \frac{1}{N_c} \sum_{i \in \text{Class } c} \mathbf{x}_i$$
* $N_c$ is the number of samples in class $c$.

### Between-Class Scatter Matrix ($S_B$)
$S_B$ measures the spread of class centroids around the global dataset mean:
$$S_B = \sum_{c=1}^{C} N_c (\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T$$
where $\boldsymbol{\mu} \in \mathbb{R}^D$ is the overall mean vector:
  $$\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i$$

---

## 3. The Generalized Eigenvalue Problem

Maximizing Fisher's criterion leads to the generalized eigenvalue problem:
$$S_B \mathbf{w} = \lambda S_W \mathbf{w}$$

If $S_W$ is invertible, this simplifies to a standard eigenvalue problem:
$$(S_W^{-1} S_B) \mathbf{w} = \lambda \mathbf{w}$$

### Regularization of $S_W$
In case features are highly collinear or sample counts are small, $S_W$ may be singular (non-invertible). To guarantee invertibility, we apply $L_2$ regularization (shrinkage):
$$S_W \leftarrow S_W + \epsilon I$$
where $\epsilon = 10^{-4}$ and $I$ is the $D \times D$ identity matrix.

### Symmetrization for Numerical Stability
The matrix product $M = S_W^{-1} S_B$ is generally non-symmetric. Eigendecomposition of non-symmetric matrices using power iteration can be unstable. To ensure stable convergence, we symmetrize $M$:
$$M_{sym} = \frac{1}{2} (M + M^T)$$
This transformation preserves the eigenvectors that maximize variance directions for symmetric components.

---

## 4. Algorithms Implemented from Scratch

All matrix operations are located in [math_utils.py](../src/lda/math_utils.py).

### Gauss-Jordan Matrix Inversion
To compute $S_W^{-1}$, the system uses Gauss-Jordan elimination:
1. **Augmented Matrix Construction:** Set up $[A \mid I]$ where $A = S_W$ and $I$ is the identity matrix.
2. **Partial Pivoting:** Select the row with the largest absolute value in the active column to avoid division by zero and minimize numerical error.
3. **Forward and Backward Elimination:** Use elementary row operations to transform $A$ into identity matrix $I$.
4. **Extraction:** The right half of the augmented matrix becomes $A^{-1}$.

### Eigendecomposition: Power Iteration
To find the eigenvalues $\lambda$ and eigenvectors $\mathbf{v}$ of $M_{sym}$, we implement **Power Iteration**:
1. Initialize a vector $\mathbf{v}_0$ deterministically: $\mathbf{v}_0 = [1, 1/2, 1/3, \dots, 1/D]^T$.
2. Repeatedly multiply by matrix $A$:
   $$\mathbf{x}_{k+1} = A \mathbf{v}_k$$
3. Compute the Rayleigh quotient (estimate of the eigenvalue):
   $$\lambda_{k+1} = \mathbf{v}_k^T \mathbf{x}_{k+1}$$
4. Normalize the vector:
   $$\mathbf{v}_{k+1} = \frac{\mathbf{x}_{k+1}}{\|\mathbf{x}_{k+1}\|}$$
5. Terminate when $|\lambda_{k+1} - \lambda_k| < 10^{-10}$ or when max iterations (1000) are reached.

### Deflation (Hotelling's Method)
Since we need $K$ eigenvectors ($K = 2$ in this project), we extract eigenvectors sequentially. Once we find the dominant eigenvalue $\lambda_1$ and eigenvector $\mathbf{v}_1$, we remove its influence from the matrix:
$$A_{new} = A - \lambda_1 \mathbf{v}_1 \mathbf{v}_1^T$$
Applying Power Iteration to $A_{new}$ yields the second-largest eigenvalue $\lambda_2$ and eigenvector $\mathbf{v}_2$.

---

## 5. Classification in the LDA Space

Classification is performed in the low-dimensional projected space $\mathbb{R}^K$.

1. **Centroid Projection:** Class mean vectors $\boldsymbol{\mu}_c$ are projected into the LDA space:
   $$\boldsymbol{\mu}_c^{lda} = W^T \boldsymbol{\mu}_c \quad \forall c \in \{1, 2, 3\}$$
2. **Sample Projection:** Query sample $\mathbf{x}$ is projected:
   $$\mathbf{x}^{lda} = W^T \mathbf{x}$$
3. **Nearest Centroid Assignment:** The predicted class $\hat{y}$ is the one that minimizes the squared Euclidean distance to the projected centroids:
   $$\hat{y} = \arg\min_{c} \|\mathbf{x}^{lda} - \boldsymbol{\mu}_c^{lda}\|^2 = \arg\min_{c} \sum_{j=1}^{K} (x_j^{lda} - \mu_{c,j}^{lda})^2$$

Because the LDA space has maximized class separation, the simple Nearest Centroid classifier yields high classification performance (approx. 93% test accuracy) while remaining computationally lightweight.

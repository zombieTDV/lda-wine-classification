"""
lda_model.py
============
Linear Discriminant Analysis implementation from scratch.

What is LDA?
-----------
Linear Discriminant Analysis is a supervised dimensionality reduction algorithm that:
1. Finds a lower-dimensional subspace that best separates different classes
2. Maximizes between-class variance while minimizing within-class variance
3. Works well for classification when classes are roughly normally distributed

Key Idea
--------
Rather than finding directions of maximum variance (like PCA), LDA finds directions
that best separate the different classes. This makes it more suitable for classification.

For a dataset with C classes, LDA produces at most C-1 linear discriminants (LD).
For wine dataset (3 classes): we get 2 LDs maximum.

Mathematical Approach
---------------------
1. Compute S_W (Within-class scatter): how spread out each class is internally
2. Compute S_B (Between-class scatter): how separated the class means are
3. Solve generalized eigenvalue problem: S_W^-1 S_B w = λ w
4. The top C-1 eigenvectors become the transformation matrix W
5. Project data onto these eigenvectors to get the LDA space

Advantages
----------
- Produces interpretable results (clear separation between classes)
- Efficient for dimensionality reduction to k dimensions
- Works well with small datasets
- Provides discriminative projection for classification

All math operations are built from scratch (no NumPy/SciPy).
"""

from src.lda.math_utils import (
    mat_zeros, mat_add, mat_scale, mat_mul, mat_inv,
    mat_vec_mul, mat_transpose, mat_sym_eig,
    mat_sym_sqrt_inv, mat_eye,
    mean_vector, outer, vec_sub, dot, norm
)


class LDA:
    """
    Linear Discriminant Analysis classifier and dimensionality reducer.
    
    This class learns a linear transformation that projects data into a 
    lower-dimensional space where class separation is maximized.
    
    Typical Usage
    ------
    1. Create model: model = LDA(n_components=2)
    2. Fit on training data: model.fit(X_train, y_train)
    3. Transform data: X_lda = model.transform(X_test)
    4. Predict labels: y_pred = model.predict(X_test)
    
    Parameters
    ----------
    n_components : int or None, default=None
        Number of discriminants (dimensions after projection).
        For C classes, maximum is C-1.
        If None, defaults to n_classes - 1.
        Example: For 3-class problem, n_components=2
        
    shrinkage : float or None, default=None
        Ledoit-Wolf style regularization parameter in [0, 1].
        Controls the trade-off between the empirical within-class
        covariance and a diagonal (spherical) estimate:
        
            S_W_reg = (1 - α) * S_W + α * (tr(S_W) / D) * I
        
        Why shrinkage matters:
        - Regularizes ill-conditioned S_W matrices, which occur when
          the number of features is close to the number of samples
        - Prevents numerical instability with collinear features
          (near-zero eigenvalues cause S_W^{-1/2} to explode)
        - Trades off the empirical covariance (low bias, high variance)
          vs a diagonal estimate (high bias, low variance)
        - α=0 uses the empirical S_W (no regularization)
        - α=1 replaces S_W with a scaled identity (maximum regularization)
        - If None, a small default regularization eps * I is applied
          (eps = 1e-6 * trace(S_W) / D) to prevent singular matrices
        
    Attributes (after fitting)
    ----------
    scalings_ : list[list[float]]
        Projection matrix W of shape (n_features, n_components)
        Used to transform data: X_new = X @ W^T
        
    means_ : dict
        Mean vector for each class (for nearest centroid prediction)
        
    classes_ : list
        Unique class labels sorted in ascending order
        
    explained_variance_ratio_ : list[float]
        Proportion of variance explained by each discriminant
        
    _eigenvalues : list[float]
        Raw eigenvalues from S_W^-1 S_B (for reference)
    """

    def __init__(self, n_components=None, shrinkage=None):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.scalings_ = None        # Projection matrix W  [n_features × n_components]
        self.means_ = None           # Class means {label: [n_features]}
        self.overall_mean_ = None    # Global mean vector
        self.classes_ = None         # Unique class labels sorted
        self.explained_variance_ratio_ = None
        self._eigenvalues = None

    # ────────────────────────────────────────────────────────────────────────
    # FIT METHOD — Learn the projection from training data
    # ────────────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Learn LDA transformation from training data.
        
        This method computes the projection matrix W that maximizes class
        separation. After fitting, use transform() to project new data.
        
        Steps of the algorithm:
        1. Compute overall mean (center of all data)
        2. Compute per-class means (center of each class)
        3. Calculate S_W: within-class scatter matrix
        4. Calculate S_B: between-class scatter matrix
        5. Apply shrinkage regularization to S_W (if configured)
        6. Compute S_W^{-1/2} via eigendecomposition
        7. Transform to symmetric problem: A = S_W^{-1/2} S_B S_W^{-1/2}
        8. Solve standard eigenvalue problem on A (power iteration)
        9. Back-transform eigenvectors: w = S_W^{-1/2} u
        10. Take top C-1 eigenvectors as projection matrix
        
        Parameters
        ----------
        X : list[list[float]]
            Training features of shape (n_samples, n_features)
            
        y : list
            Training labels of shape (n_samples,)
            Can be any hashable type (int, str, etc.)
            
        Returns
        -------
        self : LDA
            Returns self for method chaining (fit().transform())
            
        Notes
        -----
        If ``shrinkage`` is set, Ledoit-Wolf style regularization is applied
        to S_W before computing the square root inverse. Otherwise, a small
        default regularization (eps * I, eps = 1e-6 * trace(S_W) / D) is
        applied to ensure S_W is positive-definite.
            
        Example
        -------
        model = LDA(n_components=2, shrinkage=0.1)
        model.fit(X_train, y_train)
        # Now model.scalings_ contains the projection matrix
        """
        n_samples = len(X)
        n_features = len(X[0])
        self.classes_ = sorted(set(y))  # Get unique classes
        n_classes = len(self.classes_)

        # ── Set default n_components ────────────────────────────────────────
        if self.n_components is None:
            self.n_components = n_classes - 1

        # ── Step 1: Compute overall mean (center of all data) ──────────────
        self.overall_mean_ = mean_vector(X)

        # ── Step 2: Group data by class and compute per-class means ──────
        class_data = {c: [] for c in self.classes_}
        for xi, yi in zip(X, y):
            class_data[yi].append(xi)

        self.means_ = {c: mean_vector(class_data[c]) for c in self.classes_}

        # ── Step 3: Calculate S_W (Within-class scatter matrix) ──────────
        # S_W measures total spread within each class
        # For each sample x in class c: S_W += (x - μ_c) * (x - μ_c)^T
        S_W = mat_zeros(n_features, n_features)
        for c in self.classes_:
            mu_c = self.means_[c]  # Mean of class c
            for xi in class_data[c]:
                diff = vec_sub(xi, mu_c)  # Deviation from class mean
                S_W = mat_add(S_W, outer(diff, diff))  # Add outer product

        # ── Step 4: Calculate S_B (Between-class scatter matrix) ──────────
        # S_B measures spread of class means from global mean
        # For each class c: S_B += n_c * (μ_c - μ) * (μ_c - μ)^T
        S_B = mat_zeros(n_features, n_features)
        for c in self.classes_:
            n_c = len(class_data[c])  # Number of samples in class c
            diff = vec_sub(self.means_[c], self.overall_mean_)  # Class mean - global mean
            S_B = mat_add(S_B, mat_scale(outer(diff, diff), n_c))  # Weighted outer product

        # ── Step 5: Regularize S_W to ensure positive-definiteness ─────────
        #
        # S_W must be positive-definite to compute S_W^{-1/2}. This can fail
        # when: (a) features are collinear, (b) n_samples < n_features, or
        # (c) classes have very low variance in some direction.
        #
        # Two regularization strategies are supported:
        #   1. Shrinkage (if self.shrinkage is set): Ledoit-Wolf style blend
        #      of empirical S_W with a scaled identity matrix
        #   2. Default (if self.shrinkage is None): add a tiny eps * I
        #      proportional to the feature magnitude

        trace_sw = sum(S_W[i][i] for i in range(n_features))

        if self.shrinkage is not None:
            # ── Ledoit-Wolf style shrinkage ────────────────────────────────
            # S_W_reg = (1 - α) * S_W + α * (tr(S_W) / D) * I
            #
            # This interpolates between:
            #   α=0: pure empirical covariance (no regularization)
            #   α=1: scaled identity (maximum regularization, all directions equal)
            #
            # The target (tr(S_W)/D) * I preserves the average variance while
            # making all directions equally weighted, which eliminates
            # ill-conditioning from collinear features.
            alpha = self.shrinkage
            target_var = trace_sw / n_features  # Average per-feature variance
            for i in range(n_features):
                for j in range(n_features):
                    S_W[i][j] = (1 - alpha) * S_W[i][j]
                    if i == j:
                        S_W[i][j] += alpha * target_var
        else:
            # ── Default minimal regularization ────────────────────────────
            # Add eps * I where eps is scaled to the data magnitude.
            # Using trace(S_W)/D as scale factor ensures the regularization
            # is proportional to the average feature variance, not an
            # arbitrary fixed constant like the old eps=1e-4.
            eps = 1e-6 * trace_sw / n_features if trace_sw > 0 else 1e-10
            for i in range(n_features):
                S_W[i][i] += eps

        # ── Step 6: Transform to symmetric eigenvalue problem ─────────────
        #
        # MATHEMATICAL BACKGROUND:
        # The generalized eigenvalue problem is:  S_B w = λ S_W w
        # which is equivalent to:  S_W^{-1} S_B w = λ w
        #
        # The OLD (WRONG) approach was:
        #   M = S_W^{-1} S_B
        #   M_sym = 0.5 * (M + M^T)   ← INVALID symmetrization!
        #
        # Why that was wrong:
        #   M = S_W^{-1} S_B is NOT symmetric in general (even though
        #   S_W and S_B individually are symmetric). Forcing symmetry via
        #   0.5*(M + M^T) changes the eigenvectors of M, so the resulting
        #   projection directions are NOT the true LDA solution. The
        #   eigenvalues are also altered. This is not just a numerical
        #   issue — it's a fundamental mathematical error.
        #
        # The CORRECT approach (used here):
        #   1. Compute S_W^{-1/2} (symmetric square root inverse)
        #   2. Form A = S_W^{-1/2} S_B S_W^{-1/2}
        #   3. A IS symmetric (proof: A^T = (S_W^{-1/2})^T S_B^T (S_W^{-1/2})^T
        #                                = S_W^{-1/2} S_B S_W^{-1/2} = A)
        #   4. Solve A u = λ u (standard symmetric eigenproblem)
        #   5. Recover w = S_W^{-1/2} u
        #
        # This preserves the generalized eigenvalue solution exactly:
        #   S_B w = λ S_W w  ↔  S_W^{-1/2} S_B S_W^{-1/2} u = λ u
        #   with  w = S_W^{-1/2} u
        #
        # The shrinkage/regularization is applied BEFORE computing S_W^{-1/2}
        # to ensure S_W is positive-definite (all eigenvalues > 0), which is
        # required for the square root inverse to exist.

        # Compute S_W^{-1/2} via eigendecomposition of regularized S_W
        S_W_inv_sqrt = mat_sym_sqrt_inv(S_W)

        # Form the symmetric matrix A = S_W^{-1/2} S_B S_W^{-1/2}
        temp = mat_mul(S_W_inv_sqrt, S_B)
        A = mat_mul(temp, S_W_inv_sqrt)

        # ── Step 7: Extract top n_components eigenvectors via power iteration
        # Because A is truly symmetric, power iteration with deflation is
        # mathematically valid and will converge to the correct eigenvectors.
        eigenvalues, eigenvectors_u = mat_sym_eig(
            A, n_components=self.n_components
        )

        # ── Step 8: Back-transform eigenvectors to original space ─────────
        # The eigenvectors u of A correspond to the transformed space.
        # To get the LDA projection directions w in the original feature
        # space, we compute: w = S_W^{-1/2} u
        # Then normalize to unit length for numerical consistency.
        eigenvectors = []
        for u in eigenvectors_u:
            w = mat_vec_mul(S_W_inv_sqrt, u)
            w_norm = norm(w)
            if w_norm > 1e-15:
                w = [x / w_norm for x in w]
            eigenvectors.append(w)

        self._eigenvalues = eigenvalues

        # ── Compute explained variance ratio ──────────────────────────────
        # Shows how much discriminative power each LD captures
        total = sum(abs(v) for v in eigenvalues) + 1e-15
        self.explained_variance_ratio_ = [abs(v) / total for v in eigenvalues]

        # ── Step 7: Build projection matrix W [n_features × n_components] ─
        # Each column is an eigenvector (discriminant direction)
        # We transpose eigenvectors to make them rows for efficient projection
        self.scalings_ = mat_transpose(eigenvectors)   # [n_features × n_components]

        return self

    # ────────────────────────────────────────────────────────────────────────
    # TRANSFORM METHOD — Project data into LDA space
    # ────────────────────────────────────────────────────────────────────────

    def transform(self, X):
        """
        Project data into LDA space.
        
        Applies the learned transformation to new data:
        X_lda = X @ W  (for each sample x, compute x @ W)
        
        This reduces dimensionality: (n_samples, n_features) → (n_samples, n_components)
        
        Parameters
        ----------
        X : list[list[float]]
            Data to project, shape (n_samples, n_features)
            
        Returns
        -------
        X_lda : list[list[float]]
            Projected data in LDA space, shape (n_samples, n_components)
            Each row = one projected sample
            Each column = one discriminant axis
            
        Example
        -------
        X_test_lda = model.transform(X_test)  # (100, 13) → (100, 2) for 2 components
        
        Note
        ----
        Must call fit() before transform()
        """
        assert self.scalings_ is not None, "Model not fitted yet! Call fit() first."
        
        W = self.scalings_                     # [n_features × n_components]
        W_T = mat_transpose(W)                 # [n_components × n_features]
        
        # Project each sample: x_lda = x @ W^T
        return [mat_vec_mul(W_T, xi) for xi in X]

    def fit_transform(self, X, y):
        """
        Fit model and immediately transform data (convenience method).
        
        Equivalent to: fit(X, y); transform(X)
        
        Parameters
        ----------
        X : list[list[float]]
            Training features
        y : list
            Training labels
            
        Returns
        -------
        X_lda : list[list[float]]
            Transformed training data in LDA space
        """
        self.fit(X, y)
        return self.transform(X)

    # ────────────────────────────────────────────────────────────────────────
    # PREDICT METHOD — Classify using nearest centroid
    # ────────────────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Classify new samples using nearest centroid classifier.
        
        Algorithm:
        1. Project class centroids into LDA space
        2. For each new sample, project it into LDA space
        3. Find which centroid it's closest to (Euclidean distance)
        4. Assign the corresponding class label
        
        This is a simple but effective classifier in LDA space where
        classes are already well-separated.
        
        Parameters
        ----------
        X : list[list[float]]
            Data to classify, shape (n_samples, n_features)
            
        Returns
        -------
        predictions : list
            Predicted class labels for each sample
            Same order as input X
            
        Example
        -------
        y_pred = model.predict(X_test)  # Predicted labels
        
        Note
        ----
        Must call fit() before predict()
        Uses Euclidean distance in LDA space for classification
        """
        assert self.scalings_ is not None, "Model not fitted yet! Call fit() first."

        # ── Project class centroids to LDA space ─────────────────────────
        W_T = mat_transpose(self.scalings_)
        centroids = {
            c: mat_vec_mul(W_T, self.means_[c])
            for c in self.classes_
        }

        # ── Project test data to LDA space ───────────────────────────────
        X_lda = self.transform(X)
        
        # ── Find nearest centroid for each sample ────────────────────────
        predictions = []
        for xi_lda in X_lda:
            best_class = None
            best_dist = float("inf")
            
            # Compute distance to each class centroid
            for c, mu_lda in centroids.items():
                # Euclidean distance: ||x_lda - μ_lda||^2
                dist = sum((xi_lda[j] - mu_lda[j])**2
                           for j in range(len(xi_lda)))
                
                # Keep track of closest centroid
                if dist < best_dist:
                    best_dist = dist
                    best_class = c
            
            predictions.append(best_class)
        
        return predictions

    # ────────────────────────────────────────────────────────────────────────
    # SUMMARY METHOD — Display model information
    # ────────────────────────────────────────────────────────────────────────

    def summary(self):
        """
        Print a summary of the fitted LDA model.
        
        Displays:
        - Classes and number of discriminants
        - Eigenvalues and explained variance for each discriminant
        - Overall model configuration
        
        Returns
        -------
        str
            Formatted summary as a string
            
        Example Output
        -------
        ==================================================
          LDA Model Summary
        ==================================================
          Classes       : [1, 2, 3]
          n_components  : 2
          n_features    : 13
          LD1  eigenvalue=0.1234  explained=67.5%
          LD2  eigenvalue=0.0564  explained=32.5%
        ==================================================
        """
        lines = [
            "=" * 50,
            "  LDA Model Summary",
            "=" * 50,
            f"  Classes       : {self.classes_}",
            f"  n_components  : {self.n_components}",
            f"  n_features    : {len(self.scalings_)}",
        ]
        
        # Add eigenvalue and variance info for each discriminant
        if self.explained_variance_ratio_:
            for i, (ev, ratio) in enumerate(
                zip(self._eigenvalues, self.explained_variance_ratio_)
            ):
                lines.append(
                    f"  LD{i+1}  eigenvalue={ev:.4f}  "
                    f"explained={ratio*100:.1f}%"
                )
        
        lines.append("=" * 50)
        return "\n".join(lines)

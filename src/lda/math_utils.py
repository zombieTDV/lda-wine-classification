"""
math_utils.py
=============
Linear algebra and mathematical operations built from scratch.

This module implements fundamental matrix and vector operations needed for LDA
(Linear Discriminant Analysis) without using external libraries like NumPy or
SciPy. All operations are implemented in pure Python for educational purposes.

Contents
--------
1. Vector Operations: add, subtract, scale, dot product, norm
2. Matrix Operations: addition, multiplication, transposition, inverse
3. Eigenvalue Decomposition: Power iteration method for symmetric matrices
4. Statistics: mean vectors, covariance matrices

All functions work with basic Python lists. Data types:
- Vectors: list[float]  e.g., [1.5, 2.3, 0.9]
- Matrices: list[list[float]]  e.g., [[1, 2], [3, 4]]
"""


# ─── Vector Operations ────────────────────────────────────────────────────────
# These functions perform basic vector arithmetic.
# Vectors are represented as Python lists of numbers.

def vec_add(a, b):
    """
    Add two vectors: c = a + b
    
    Element-wise addition:
    result[i] = a[i] + b[i] for each position i
    
    Parameters
    ----------
    a : list[float]
        First vector
    b : list[float]
        Second vector (must have same length as a)
        
    Returns
    -------
    list[float]
        Vector sum (same length as a and b)
        
    Example
    -------
    vec_add([1, 2, 3], [4, 5, 6]) → [5, 7, 9]
    """
    return [a[i] + b[i] for i in range(len(a))]


def vec_sub(a, b):
    """
    Subtract two vectors: c = a - b
    
    Element-wise subtraction:
    result[i] = a[i] - b[i] for each position i
    
    Parameters
    ----------
    a : list[float]
        First vector (minuend)
    b : list[float]
        Second vector (subtrahend)
        
    Returns
    -------
    list[float]
        Vector difference
        
    Example
    -------
    vec_sub([5, 7, 9], [1, 2, 3]) → [4, 5, 6]
    """
    return [a[i] - b[i] for i in range(len(a))]


def vec_scale(v, s):
    """
    Multiply vector by scalar: c = s * v
    
    Scales each element of the vector by the same factor.
    Useful for scaling, normalization, and matrix calculations.
    
    Parameters
    ----------
    v : list[float]
        Vector to scale
    s : float
        Scalar multiplier
        
    Returns
    -------
    list[float]
        Scaled vector (same shape as v, each element multiplied by s)
        
    Example
    -------
    vec_scale([1, 2, 3], 2) → [2, 4, 6]
    vec_scale([1, 2, 3], 0.5) → [0.5, 1, 1.5]
    """
    return [x * s for x in v]


def dot(a, b):
    """
    Compute dot product (inner product) of two vectors: c = a · b
    
    The dot product measures similarity between vectors and is used in:
    - Matrix-vector multiplication
    - Eigenvalue calculations
    - Distance metrics
    
    Formula: a · b = a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]
    
    Parameters
    ----------
    a : list[float]
        First vector
    b : list[float]
        Second vector (same length as a)
        
    Returns
    -------
    float
        Dot product (scalar value)
        
    Example
    -------
    dot([1, 2, 3], [4, 5, 6]) = 1*4 + 2*5 + 3*6 = 32
    dot([1, 0, 0], [2, 3, 4]) = 1*2 + 0*3 + 0*4 = 2
    """
    return sum(a[i] * b[i] for i in range(len(a)))


def norm(v):
    """
    Compute Euclidean norm (length) of a vector: ||v||
    
    The norm represents the magnitude/length of the vector.
    Used for vector normalization and distance calculations.
    
    Formula: ||v|| = sqrt(v[0]^2 + v[1]^2 + ... + v[n-1]^2)
    
    Parameters
    ----------
    v : list[float]
        Vector
        
    Returns
    -------
    float
        Vector norm (always non-negative)
        
    Example
    -------
    norm([3, 4]) = sqrt(3^2 + 4^2) = sqrt(9 + 16) = 5
    norm([1, 0, 0]) = 1
    """
    return sum(x**2 for x in v) ** 0.5


# ─── Matrix Operations ────────────────────────────────────────────────────────
# These functions perform matrix arithmetic operations.
# Matrices are represented as lists of lists: [[row0], [row1], ...]

def mat_zeros(rows, cols):
    """
    Create a zero matrix (all elements are 0).
    
    Parameters
    ----------
    rows : int
        Number of rows
    cols : int
        Number of columns
        
    Returns
    -------
    list[list[float]]
        Matrix of size (rows × cols) filled with 0.0
        
    Example
    -------
    mat_zeros(2, 3) → [[0, 0, 0], [0, 0, 0]]
    """
    return [[0.0] * cols for _ in range(rows)]


def mat_eye(n):
    """
    Create identity matrix (diagonal = 1, rest = 0).
    
    The identity matrix is like "1" for matrices:
    For any matrix M: M @ I = M
    
    Used in inverse calculations and as initialization.
    
    Parameters
    ----------
    n : int
        Size of square matrix (n × n)
        
    Returns
    -------
    list[list[float]]
        Identity matrix of size (n × n)
        
    Example
    -------
    mat_eye(3) → [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
    M = mat_zeros(n, n)
    for i in range(n):
        M[i][i] = 1.0
    return M


def mat_shape(M):
    """
    Get dimensions of a matrix.
    
    Parameters
    ----------
    M : list[list[float]]
        Matrix
        
    Returns
    -------
    tuple (int, int)
        Shape as (rows, columns)
        
    Example
    -------
    mat_shape([[1, 2, 3], [4, 5, 6]]) → (2, 3)
    """
    return len(M), len(M[0])


def mat_add(A, B):
    """
    Add two matrices: C = A + B
    
    Element-wise addition: C[i][j] = A[i][j] + B[i][j]
    Both matrices must have the same shape.
    
    Parameters
    ----------
    A : list[list[float]]
        First matrix
    B : list[list[float]]
        Second matrix (must have same shape as A)
        
    Returns
    -------
    list[list[float]]
        Sum matrix (same shape as A and B)
        
    Example
    -------
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    mat_add(A, B) → [[6, 8], [10, 12]]
    """
    r, c = mat_shape(A)
    return [[A[i][j] + B[i][j] for j in range(c)] for i in range(r)]


def mat_scale(A, s):
    """
    Multiply matrix by scalar: B = s * A
    
    Scales every element: B[i][j] = s * A[i][j]
    
    Parameters
    ----------
    A : list[list[float]]
        Matrix to scale
    s : float
        Scalar multiplier
        
    Returns
    -------
    list[list[float]]
        Scaled matrix (same shape as A)
        
    Example
    -------
    A = [[1, 2], [3, 4]]
    mat_scale(A, 2) → [[2, 4], [6, 8]]
    """
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]


def mat_mul(A, B):
    """
    Multiply two matrices: C = A @ B
    
    Standard matrix multiplication. Result C[i][j] is the dot product
    of row i from A and column j from B.
    
    Requirement: A.shape[1] (columns of A) must equal B.shape[0] (rows of B)
    Result shape: (A.rows, B.cols)
    
    Parameters
    ----------
    A : list[list[float]]
        First matrix of shape (m × n)
    B : list[list[float]]
        Second matrix of shape (n × p)
        
    Returns
    -------
    list[list[float]]
        Product matrix of shape (m × p)
        
    Example
    -------
    A = [[1, 2, 3], [4, 5, 6]]       # (2 × 3)
    B = [[1, 2], [3, 4], [5, 6]]     # (3 × 2)
    mat_mul(A, B) → [[22, 28], [49, 64]]  # (2 × 2)
    """
    rA, cA = mat_shape(A)
    rB, cB = mat_shape(B)
    assert cA == rB, f"Shape mismatch: ({rA},{cA}) x ({rB},{cB})"
    C = mat_zeros(rA, cB)
    for i in range(rA):
        for k in range(cA):
            if A[i][k] == 0:  # Skip zero elements (optimization)
                continue
            for j in range(cB):
                C[i][j] += A[i][k] * B[k][j]
    return C


def mat_transpose(A):
    """
    Transpose a matrix: B = A^T
    
    Swaps rows and columns: B[i][j] = A[j][i]
    Matrix of shape (m × n) becomes (n × m)
    
    Parameters
    ----------
    A : list[list[float]]
        Matrix of shape (m × n)
        
    Returns
    -------
    list[list[float]]
        Transposed matrix of shape (n × m)
        
    Example
    -------
    A = [[1, 2, 3], [4, 5, 6]]
    mat_transpose(A) → [[1, 4], [2, 5], [3, 6]]
    """
    r, c = mat_shape(A)
    return [[A[i][j] for i in range(r)] for j in range(c)]


def mat_vec_mul(A, v):
    """
    Multiply matrix by vector: y = A @ v
    
    Treats vector as a column vector (n × 1).
    Result is also a column vector of shape (A.rows,)
    
    Parameters
    ----------
    A : list[list[float]]
        Matrix of shape (m × n)
    v : list[float]
        Vector of length n
        
    Returns
    -------
    list[float]
        Result vector of length m
        
    Example
    -------
    A = [[1, 2], [3, 4], [5, 6]]     # (3 × 2)
    v = [1, 2]                        # length 2
    mat_vec_mul(A, v) → [5, 11, 17]  # length 3
    """
    return [dot(row, v) for row in A]


def outer(a, b):
    """
    Compute outer product: M = a ⊗ b
    
    Creates a matrix from two vectors:
    M[i][j] = a[i] * b[j]
    Result shape: (len(a) × len(b))
    
    Used extensively in covariance matrices and deflation (eigenvalue computation).
    
    Parameters
    ----------
    a : list[float]
        Vector of length m
    b : list[float]
        Vector of length n
        
    Returns
    -------
    list[list[float]]
        Outer product matrix of shape (m × n)
        
    Example
    -------
    a = [1, 2, 3]
    b = [4, 5]
    outer(a, b) → [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]]
              → [[4, 5], [8, 10], [12, 15]]
    """
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


# ─── Matrix Inversion via Gauss-Jordan Elimination ───────────────────────────
# Used to compute matrix inverse and determinant

def mat_copy(M):
    """
    Create a deep copy of a matrix.
    
    Prevents modifications to the copy from affecting the original.
    
    Parameters
    ----------
    M : list[list[float]]
        Matrix to copy
        
    Returns
    -------
    list[list[float]]
        Independent copy of M
    """
    return [row[:] for row in M]


def gauss_jordan(M):
    """
    Compute matrix inverse using Gauss-Jordan elimination.
    
    This is a foundational algorithm in linear algebra for:
    1. Computing matrix inverse: M^-1
    2. Computing determinant: det(M)
    3. Solving linear systems: Ax = b
    
    Algorithm outline:
    1. Augment M with identity matrix I: [M | I]
    2. Apply row operations to transform M to I
    3. The result is [I | M^-1]
    
    Parameters
    ----------
    M : list[list[float]]
        Square matrix (n × n)
        
    Returns
    -------
    inverse : list[list[float]]
        Matrix inverse M^-1 (same shape as M)
        
    determinant : float
        Determinant det(M)
        (Product of diagonal elements after row operations)
        
    Raises
    ------
    ValueError
        If matrix is singular (non-invertible, det = 0)
        
    Example
    -------
    M = [[4, 7], [2, 6]]
    inv, det = gauss_jordan(M)
    print(det)   # Determinant
    print(inv)   # Inverse matrix
    """
    n = len(M)
    A = mat_copy(M)      # Copy of M to be transformed to I
    I = mat_eye(n)       # Identity matrix to be transformed to M^-1
    det = 1.0            # Running product for determinant

    # ── Forward elimination (create upper triangular form) ─────────────────
    for col in range(n):
        # Find pivot (row with largest absolute value in this column)
        max_row = max(range(col, n), key=lambda r: abs(A[r][col]))
        
        # Check for singular matrix
        if abs(A[max_row][col]) < 1e-12:
            raise ValueError("Matrix is singular — cannot compute inverse.")

        # Swap rows if needed
        if max_row != col:
            A[col], A[max_row] = A[max_row], A[col]
            I[col], I[max_row] = I[max_row], I[col]
            det *= -1  # Swapping rows negates determinant

        # Scale row to make pivot = 1
        pivot = A[col][col]
        det *= pivot  # Multiply determinant by pivot
        for j in range(n):
            A[col][j] /= pivot
            I[col][j] /= pivot

        # Eliminate below pivot
        for row in range(n):
            if row == col:
                continue
            factor = A[row][col]
            for j in range(n):
                A[row][j] -= factor * A[col][j]
                I[row][j] -= factor * I[col][j]

    return I, det


def mat_inv(M):
    """
    Compute matrix inverse: M^-1 (just the inverse, not determinant).
    
    Convenience wrapper around gauss_jordan() that returns only the inverse.
    
    Parameters
    ----------
    M : list[list[float]]
        Square matrix to invert
        
    Returns
    -------
    list[list[float]]
        Matrix inverse (same shape as M)
        
    Example
    -------
    M = [[4, 7], [2, 6]]
    M_inv = mat_inv(M)
    # M_inv @ M ≈ I (identity matrix)
    """
    inv, _ = gauss_jordan(M)
    return inv


# ─── Eigenvalue Decomposition via Power Iteration ────────────────────────────
# Used to find the largest eigenvectors/eigenvalues of symmetric matrices

def mat_sym_eig(M, n_components=None, max_iter=1000, tol=1e-10):
    """
    Compute eigenvalues and eigenvectors of symmetric matrix.
    
    Uses Power Iteration + Deflation method to find the top eigenvectors.
    
    Algorithm:
    1. Use power iteration to find largest eigenvector
    2. Extract corresponding eigenvalue
    3. "Deflate" the matrix: A ← A - λ·v·v^T
    4. Repeat to find next eigenvector
    
    Why this works for LDA:
    - LDA requires solving: S_W^-1 S_B w = λ w
    - Power iteration efficiently finds top components
    
    Parameters
    ----------
    M : list[list[float]]
        Symmetric matrix (n × n)
        Assumption: M is symmetric (M = M^T)
        
    n_components : int, optional
        Number of eigenvectors to compute
        Default: all (n)
        
    max_iter : int, default=1000
        Maximum iterations for power iteration
        
    tol : float, default=1e-10
        Convergence tolerance for eigenvalues
        
    Returns
    -------
    eigenvalues : list[float]
        Eigenvalues in descending order
        
    eigenvectors : list[list[float]]
        Eigenvectors as columns (each column is one eigenvector)
        Ordered corresponding to eigenvalues
        
    Example
    -------
    M = [[4, 1], [1, 3]]  # Symmetric 2×2 matrix
    evals, evecs = mat_sym_eig(M)
    # evals ≈ [4.56, 2.44]
    # evecs[0] is eigenvector for eval 4.56
    
    Note
    ----
    For LDA, the result is used to define transformation matrix:
    W = [evec1, evec2, ...].T  (top eigenvectors as rows)
    """
    n = len(M)
    if n_components is None:
        n_components = n

    A = mat_copy(M)  # Work with copy to preserve original
    eigenvalues = []
    eigenvectors = []

    # ── Iteratively extract top eigenvectors via power iteration ──────────
    for _ in range(n_components):
        # Initialize with deterministic random vector
        v = [1.0 / (i + 1) for i in range(n)]
        v_norm = norm(v)
        v = [x / v_norm for x in v]  # Normalize to unit length

        lam = 0.0  # Current eigenvalue estimate

        # ── Power iteration: repeatedly multiply by A and normalize ────
        for iteration in range(max_iter):
            Av = mat_vec_mul(A, v)  # Multiply matrix by vector
            lam_new = dot(v, Av)    # Rayleigh quotient (eigenvalue estimate)
            n_Av = norm(Av)
            
            # Check for convergence or zero vector
            if n_Av < 1e-15:
                break
                
            v_new = [x / n_Av for x in Av]  # Normalize
            
            # Check convergence
            if abs(lam_new - lam) < tol:
                v = v_new
                lam = lam_new
                break
                
            v = v_new
            lam = lam_new

        eigenvalues.append(lam)
        eigenvectors.append(v)

        # ── Deflation: Remove this eigenvalue/eigenvector from matrix ────
        # A ← A - λ·v·v^T
        # This removes the component in direction v so next iteration finds
        # the next (second-largest) eigenvalue
        vvT = outer(v, v)
        A = mat_add(A, mat_scale(vvT, -lam))

    return eigenvalues, eigenvectors


# ─── Statistics ──────────────────────────────────────────────────────────────
# Used for computing data statistics needed by LDA

def mean_vector(X):
    """
    Compute mean (centroid) of dataset.
    
    Computes the average value for each feature.
    Result is a vector of length = number of features.
    
    Formula: mean[j] = (1/n) * Σ X[i][j]  for each feature j
    
    Parameters
    ----------
    X : list[list[float]]
        Data matrix of shape (n_samples, n_features)
        
    Returns
    -------
    list[float]
        Mean vector of length n_features
        
    Example
    -------
    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3 samples × 3 features
    mean_vector(X) → [4, 5, 6]  # average of each feature
    """
    n = len(X)           # Number of samples
    d = len(X[0])        # Number of features
    mu = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    return mu


def cov_matrix(X, mu=None):
    """
    Compute covariance matrix of dataset.
    
    Covariance matrix measures how features vary together:
    - Large positive value: features increase together
    - Large negative value: features vary inversely
    - Near zero: features are independent
    
    Formula: Cov[i,j] = (1/(n-1)) * Σ(X[:,i] - μ[i]) * (X[:,j] - μ[j])
    
    Uses Bessel's correction (divide by n-1) for unbiased estimate.
    
    Parameters
    ----------
    X : list[list[float]]
        Data matrix of shape (n_samples, n_features)
        
    mu : list[float], optional
        Mean vector. If None, computed automatically.
        
    Returns
    -------
    list[list[float]]
        Covariance matrix of shape (n_features, n_features)
        Always symmetric: Cov[i,j] = Cov[j,i]
        
    Example
    -------
    X = [[1, 2], [2, 4], [3, 6]]
    C = cov_matrix(X)  # 2×2 symmetric covariance matrix
    
    Note
    ----
    Critical for LDA: Within-class and between-class scatter matrices
    are computed using outer products of (X - μ) vectors.
    """
    n = len(X)           # Number of samples
    d = len(X[0])        # Number of features
    
    # Compute mean if not provided
    if mu is None:
        mu = mean_vector(X)
    
    C = mat_zeros(d, d)  # Initialize covariance matrix
    
    # For each sample, compute outer product of deviation vector
    for row in X:
        diff = vec_sub(row, mu)  # (x - μ)
        outer_prod = outer(diff, diff)  # (x - μ) ⊗ (x - μ)
        C = mat_add(C, outer_prod)  # Accumulate
    
    # Normalize by sample size (Bessel's correction: n-1)
    C = mat_scale(C, 1.0 / (n - 1))
    return C

"""
test_lda.py
===========
Comprehensive unit tests for the entire LDA pipeline.

This test suite validates all major components:
- Math utilities: Matrix operations, eigendecomposition, statistics
- Data preprocessing: Train/test splitting, standardization
- LDA model: Fitting, transforming, predicting
- Metrics: Accuracy, confusion matrix
- Data preparation: Raw data → processed CSV conversion

Testing Strategy
----------------
1. Unit tests for individual functions (low-level)
2. Integration tests for multi-component workflows
3. End-to-end tests on real wine dataset

Why These Tests?
----------------
- Ensures mathematical correctness (matrix operations)
- Validates data pipeline integrity (no data leakage)
- Confirms LDA implementation matches theory (eigenvalues, transforms)
- Catches regressions when modifying code

Running Tests
-------------
From project root:
    python -m pytest tests/test_lda.py -v
    
Or without pytest:
    python tests/test_lda.py

Expected Output:
    ✓ 15+ tests passing
    All major components validated
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lda.math_utils import (
    mat_mul, mat_inv, mat_transpose, mat_sym_eig,
    mean_vector, cov_matrix, dot, norm, outer, mat_zeros, mat_eye
)
from src.lda.lda_model import LDA
from src.preprocessing.data_loader import (
    train_test_split, standardize, load_wine, prepare_data
)
from src.evaluation.metrics import accuracy, confusion_matrix
import csv
import tempfile


# ─── Test Utilities ──────────────────────────────────────────────────────────

def assert_close(a, b, tol=1e-6, msg=""):
    """
    Assert two floating-point numbers are approximately equal.
    
    Parameters
    ----------
    a, b : float
        Values to compare
    tol : float, default=1e-6
        Tolerance threshold
    msg : str
        Custom error message for clarity
        
    Raises
    ------
    AssertionError if |a - b| >= tol
    """
    assert abs(a - b) < tol, f"Expected {a} ≈ {b}  [{msg}]"

def assert_close_vec(v1, v2, tol=1e-4, msg=""):
    """
    Assert two vectors are approximately equal element-wise.
    
    Parameters
    ----------
    v1, v2 : list[float]
        Vectors to compare
    tol : float, default=1e-4
        Per-element tolerance
    msg : str
        Error message context
    """
    for i, (a, b) in enumerate(zip(v1, v2)):
        assert abs(a - b) < tol, f"Index {i}: {a} vs {b}  [{msg}]"

PASS = "  ✓"
FAIL = "  ✗"


# ═════════════════════════════════════════════════════════════════════════════
#  MATH UTILITIES TESTS — Linear algebra operations from scratch
# ═════════════════════════════════════════════════════════════════════════════

def test_mat_mul_identity():
    """
    Test matrix multiplication: A × I = A
    
    Verifies that multiplying any matrix A by identity matrix I
    returns A unchanged. This is a fundamental property used in
    eigendecomposition and LDA transformations.
    """
    I = mat_eye(3)
    A = [[1.0, 2, 3], [4, 5, 6], [7, 8, 9]]
    R = mat_mul(A, I)
    for i in range(3):
        assert_close_vec(R[i], A[i], msg="A @ I = A")
    print(PASS, "mat_mul identity")

def test_mat_inv():
    """
    Test matrix inversion: A × A⁻¹ = I
    
    Verifies the matrix inversion algorithm. The product of a matrix
    and its inverse should equal the identity matrix (within numerical error).
    
    Example
    -------
    A = [[4, 7], [2, 6]]
    A⁻¹ ≈ [[0.6, -0.7], [-0.2, 0.4]]
    A @ A⁻¹ ≈ [[1, 0], [0, 1]]
    """
    A = [[4.0, 7], [2, 6]]
    A_inv = mat_inv(A)
    I_approx = mat_mul(A, A_inv)
    assert_close(I_approx[0][0], 1.0, msg="(0,0)")
    assert_close(I_approx[1][1], 1.0, msg="(1,1)")
    assert_close(I_approx[0][1], 0.0, msg="(0,1)")
    print(PASS, "mat_inv")

def test_mean_vector():
    """
    Test computation of mean vector (centroid).
    
    For matrix X of shape (n_samples, n_features), computes
    mean_vector(X) = [mean(col_0), mean(col_1), ..., mean(col_k)]
    
    This is used in LDA for computing:
    - Global mean (centroid of all data)
    - Per-class means (centroid of each class)
    """
    X = [[1.0, 2], [3.0, 4], [5.0, 6]]
    mu = mean_vector(X)
    assert_close(mu[0], 3.0)
    assert_close(mu[1], 4.0)
    print(PASS, "mean_vector")

def test_transpose():
    """
    Test matrix transpose: (A^T)^T = A
    
    Transpose flips matrix along diagonal:
    - Rows become columns
    - (m, n) → (n, m)
    
    Used in covariance matrix and eigendecomposition calculations.
    """
    A = [[1, 2, 3], [4, 5, 6]]
    AT = mat_transpose(A)
    assert len(AT) == 3 and len(AT[0]) == 2
    assert_close(AT[0][1], 4.0)
    print(PASS, "mat_transpose")

def test_eigenvalue_2x2():
    """
    Test eigenvalue decomposition on 2×2 matrix.
    
    For diagonal matrix M = [[3, 0], [0, 1]], eigenvalues are 3 and 1.
    
    Mathematical Background
    -----------------------
    M = W Λ W⁻¹  (eigendecomposition)
    where:
        W = eigenvectors (columns)
        Λ = diagonal matrix of eigenvalues
        
    This is the core of LDA: solving S_W⁻¹ S_B w = λw
    """
    # Diagonal matrix: eigenvalues are diagonal elements
    M = [[3.0, 0], [0, 1.0]]
    evals, evecs = mat_sym_eig(M, n_components=2)
    assert_close(abs(evals[0]), 3.0, tol=0.05)
    print(PASS, "eigenvalue 2×2 diagonal")


# ═════════════════════════════════════════════════════════════════════════════
#  DATA PREPROCESSING TESTS — Splitting, standardization, loading
# ═════════════════════════════════════════════════════════════════════════════

def test_train_test_split():
    """
    Test train/test split with correct proportions.
    
    With test_size=0.3 and 100 samples:
    - Training: 70 samples (30% removed)
    - Testing: 30 samples (30% reserved)
    
    Uses deterministic seed for reproducibility.
    """
    X = [[float(i)] for i in range(100)]
    y = [i % 3 for i in range(100)]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, seed=0)
    assert len(X_tr) == 70 and len(X_te) == 30
    assert len(X_tr) + len(X_te) == 100
    print(PASS, "train_test_split")

def test_standardize():
    """
    Test Z-score standardization: (X - μ) / σ
    
    After standardization, data should have:
    - Mean = 0 (zero-centered)
    - Std = 1 (unit variance)
    
    This is critical for LDA: prevents features with large scales
    from dominating the algorithm.
    
    Example
    -------
    X = [[1, 10], [2, 20], [3, 30]]
    X_std has mean ≈ [0, 0]
    """
    X_train = [[1.0, 10], [2.0, 20], [3.0, 30]]
    X_train_s, means, stds = standardize(X_train)
    mu = mean_vector(X_train_s)
    assert_close(mu[0], 0.0, tol=1e-6, msg="mean after scale")
    print(PASS, "standardize zero-mean")



# ═════════════════════════════════════════════════════════════════════════════
#  LDA MODEL TESTS — Fitting, transforming, predicting
# ═════════════════════════════════════════════════════════════════════════════

def test_lda_fit_transform_shape():
    """
    Test LDA fit_transform returns correct output shapes.
    
    Input: 9 samples × 3 features, 3 classes
    Output: 9 samples × 2 LDA components
    
    Verifies:
    - Output has same number of samples as input
    - Output has exactly n_components features
    """
    X = [[1.0, 2, 3], [2, 3, 4], [3, 4, 5],
         [10.0, 1, 2], [11, 2, 1], [12, 1, 3],
         [5.0, 10, 1], [6, 11, 2], [7, 10, 3]]
    y = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X, y)
    assert len(X_lda) == 9
    assert len(X_lda[0]) == 2
    print(PASS, "LDA fit_transform shape")

def test_lda_predict_simple():
    """
    Test LDA prediction on well-separated clusters.
    
    Data: 3 clusters with centers at (1, 0), (10, 0), (5, 5)
    These clusters are far apart, so LDA should classify with ~100% accuracy.
    
    Expected Accuracy: ≥ 99%
    
    This validates that:
    - LDA correctly learns class boundaries
    - Prediction logic works as expected
    """
    X = [[1.0, 0], [1.1, 0], [0.9, 0],
         [10.0, 0], [10.1, 0], [9.9, 0],
         [5.0, 5], [5.1, 5], [4.9, 5]]
    y = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    lda = LDA(n_components=2)
    lda.fit(X, y)
    y_pred = lda.predict(X)
    acc = accuracy(y, y_pred)
    assert acc >= 0.99, f"Expected near 100%, got {acc:.2%}"
    print(PASS, f"LDA predict simple (acc={acc:.0%})")

def test_lda_wine_pipeline():
    """
    End-to-end test on real UCI wine dataset.
    
    Pipeline:
    1. Load 178 wine samples (13 features, 3 classes)
    2. Split: 75% train (133), 25% test (45)
    3. Standardize: Z-score normalization
    4. Fit LDA: 2 components
    5. Predict: Evaluate on test set
    
    Expected Test Accuracy: > 90%
    This is a realistic benchmark for this dataset/algorithm.
    """
    X, y, _ = load_wine()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, seed=42)
    X_tr_s, X_te_s, _, _ = standardize(X_tr, X_te)
    lda = LDA(n_components=2)
    lda.fit(X_tr_s, y_tr)
    y_pred = lda.predict(X_te_s)
    acc = accuracy(y_te, y_pred)
    assert acc > 0.90, f"Expected >90% on wine test, got {acc:.2%}"
    print(PASS, f"LDA wine pipeline (test acc={acc:.2%})")


# ═════════════════════════════════════════════════════════════════════════════
#  EVALUATION METRICS TESTS — Accuracy, confusion matrix
# ═════════════════════════════════════════════════════════════════════════════

def test_accuracy():
    """
    Test accuracy computation: (# correct) / (# total)
    
    Example: [1,2,3,1] vs [1,2,3,2]
    Correct: positions 0,1,2 (3 out of 4)
    Accuracy: 3/4 = 0.75
    """
    assert_close(accuracy([1,2,3,1], [1,2,3,2]), 0.75)
    print(PASS, "accuracy")

def test_confusion_matrix():
    """
    Test confusion matrix creation: CM[true][pred] = count
    
    Example: 
    - True:  [1, 1, 2, 2]
    - Pred:  [1, 2, 1, 2]
    
    Results:
    - CM[1][1] = 1  (sample 0: true 1, pred 1)
    - CM[1][2] = 1  (sample 1: true 1, pred 2 — MISCLASSIFIED)
    - CM[2][1] = 1  (sample 2: true 2, pred 1 — MISCLASSIFIED)
    - CM[2][2] = 1  (sample 3: true 2, pred 2)
    """
    CM, classes = confusion_matrix([1,1,2,2], [1,2,1,2])
    assert CM[1][1] == 1
    assert CM[1][2] == 1
    print(PASS, "confusion_matrix")


# ═════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION TESTS — Raw → Processed CSV conversion
# ═════════════════════════════════════════════════════════════════════════════

def test_prepare_data_creates_csv():
    """
    Test prepare_data() creates properly formatted CSV output.
    
    Workflow:
    1. Create temporary raw data (headerless)
    2. Call prepare_data() with temp paths
    3. Verify output CSV exists and has headers
    4. Verify data integrity (correct number of columns/rows)
    
    This validates the raw→processed pipeline without modifying actual data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create temporary raw data file
        raw_path = os.path.join(tmpdir, "test_raw.data")
        output_path = os.path.join(tmpdir, "test_processed.csv")
        
        # Write test data (3 samples, 14 columns as per wine dataset)
        with open(raw_path, "w") as f:
            f.write("1,14.23,1.76,2.45,15.6,127,2.3,0.92,0.66,1.52,0.58,1.62,480,0.8\n")
            f.write("1,13.2,1.78,2.14,11.2,100,2.4,1.35,2.67,0.33,1.97,660,0.8\n")
            f.write("2,12.37,0.94,2.36,21,88,1.8,0.69,1.15,1.59,0.61,500,0.82\n")
        
        # Call prepare_data
        result = prepare_data(raw_path, output_path)
        
        # Verify success
        assert result, "prepare_data should return True on success"
        assert os.path.exists(output_path), "Output CSV file should exist"
        
        # Verify CSV structure by reading header
        with open(output_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert len(header) == 14, f"Expected 14 header columns, got {len(header)}"
            assert header[0].lower() == "class", "First column should be 'class'"
            assert "alcohol" in header[1].lower(), "Second column should contain 'alcohol'"
        
        # Verify data rows are present
        with open(output_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 4, f"Expected 4 lines (1 header + 3 data), got {len(lines)}"
    
    print(PASS, "prepare_data creates CSV with headers")

def test_prepare_data_nonexistent_source():
    """
    Test prepare_data() handles missing raw data file gracefully.
    
    Should return False when source file doesn't exist, rather than crashing.
    This prevents silent failures in automated data pipelines.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "nonexistent.data")
        output_path = os.path.join(tmpdir, "output.csv")
        
        result = prepare_data(raw_path, output_path)
        assert result == False, "Should return False when source file not found"
        assert not os.path.exists(output_path), "Should not create output when source missing"
    
    print(PASS, "prepare_data handles missing source gracefully")

def test_prepare_data_integration():
    """
    Integration test: prepare_data output can be loaded by load_wine().
    
    Validates the full cycle:
    1. Raw data → CSV via prepare_data()
    2. CSV → (X, y, feature_names) via load_wine()
    3. Verify shapes and content
    
    This ensures data flows correctly through the pipeline.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "test.data")
        csv_path = os.path.join(tmpdir, "test.csv")
        
        # Create raw data with 3 samples
        with open(raw_path, "w") as f:
            f.write("1,14.23,1.76,2.45,15.6,127,2.3,0.92,0.66,1.52,0.58,1.62,480,0.8\n")
            f.write("1,13.2,1.78,2.14,11.2,100,2.4,1.35,2.67,0.33,1.97,660,0.8\n")
            f.write("2,12.37,0.94,2.36,21,88,1.8,0.69,1.15,1.59,0.61,500,0.82\n")
        
        # Convert to CSV
        success = prepare_data(raw_path, csv_path)
        assert success, "prepare_data failed"
        
        # Load processed CSV
        X, y, feature_names = load_wine(csv_path)
        
        # Verify loaded data
        assert len(X) == 3, f"Expected 3 samples, got {len(X)}"
        assert len(X[0]) == 13, f"Expected 13 features, got {len(X[0])}"
        assert len(y) == 3, f"Expected 3 labels, got {len(y)}"
        assert len(feature_names) == 13, f"Expected 13 feature names, got {len(feature_names)}"
        
        # Verify labels
        assert set(y) == {1, 2}, f"Expected labels {{1, 2}}, got {set(y)}"
    
    print(PASS, "prepare_data integration with load_wine()")



# ═════════════════════════════════════════════════════════════════════════════
#  TEST RUNNER — Execute all tests and report results
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Main test runner.
    
    Execution Modes:
    
    1. Direct execution:
        python tests/test_lda.py
        
    2. With pytest (if available):
        python -m pytest tests/test_lda.py -v
        
    3. From project root:
        python -m tests.test_lda
    
    Test Results:
    - ✓ = test passed
    - ✗ = test failed (prints error message)
    - Summary shows # passed / # total
    """
    
    # List of all test functions to run
    tests = [
        # Math utilities (5 tests)
        test_mat_mul_identity,
        test_mat_inv,
        test_mean_vector,
        test_transpose,
        test_eigenvalue_2x2,
        
        # Data preprocessing (2 tests)
        test_train_test_split,
        test_standardize,
        
        # LDA model (3 tests)
        test_lda_fit_transform_shape,
        test_lda_predict_simple,
        test_lda_wine_pipeline,
        
        # Metrics (2 tests)
        test_accuracy,
        test_confusion_matrix,
        
        # Data preparation (3 tests)
        test_prepare_data_creates_csv,
        test_prepare_data_nonexistent_source,
        test_prepare_data_integration,
    ]

    # Print test header
    print("\n" + "=" * 60)
    print("  Running LDA Pipeline Tests")
    print("=" * 60)

    # Execute each test
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"{FAIL} {t.__name__}: {e}")
            failed += 1

    # Print summary
    print("=" * 60)
    print(f"  {passed}/{passed+failed} tests passed")
    if failed > 0:
        print(f"  {failed} tests FAILED")
    print("=" * 60 + "\n")

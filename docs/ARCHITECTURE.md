# Codebase Architecture

This document describes the software architecture, modular structure, file responsibilities, and directory layout of the Linear Discriminant Analysis (LDA) wine classification project.

---

## Directory Structure

The project is structured logically into components separating data, source code, tests, and outputs:

```
lda-wine-classification/
├── data/
│   ├── raw/                  # Raw UCI dataset files
│   │   ├── wine.data         # Headerless comma-separated data
│   │   └── wine.names        # Dataset source details
│   └── processed/            # Structured, standardized data
│       └── wine.csv          # Processed data with column headers
├── docs/                     # Documentation files
│   ├── ARCHITECTURE.md       # (This file) Architecture overview
│   ├── MATHEMATICS.md        # Math formulas & algorithm theories
│   ├── PIPELINE.md           # Step-by-step pipeline execution flow
│   └── PIPELINE_VERIFICATION.md # Verified runs & configuration audits
├── src/                      # Project source modules
│   ├── __init__.py           # Package initializer
│   ├── lda/                  # LDA algorithm modules
│   │   ├── __init__.py
│   │   ├── lda_model.py      # Core LDA model class
│   │   └── math_utils.py     # Pure Python linear algebra helpers
│   ├── preprocessing/        # Data handling & normalization
│   │   ├── __init__.py
│   │   ├── data_loader.py    # Parsers, splitters, standardizers
│   │   └── prepare_raw_data.py # Conversion script raw -> processed CSV
│   ├── evaluation/           # Performance monitoring
│   │   ├── __init__.py
│   │   └── metrics.py        # Accuracy, F1, Precision, Recall, CM
│   └── visualization/        # Graphical renderers
│       ├── __init__.py
│       └── plots.py          # SVG graphics generators
├── tests/                    # Verification suite
│   ├── __init__.py
│   └── test_lda.py           # Unit and integration tests
├── outputs/                  # Created during execution (ignored by git)
│   ├── figures/              # Generated SVG vector graphics
│   ├── reports/              # Text classification reports
│   └── models/               # Serialized model coefficients
├── README.md                 # Main workspace introduction
└── main.py                   # Master orchestrator script
```

---

## Module Responsibilities

### 1. Orchestration
* **File:** [main.py](../main.py)
* **Role:** Acts as the entry point. It imports components, loads configurations (hyperparameters), executes the 8 sequential pipeline stages, tracks completion timing, and handles top-level directory creation.

### 2. Math & Linear Algebra
* **File:** [src/lda/math_utils.py](../src/lda/math_utils.py)
* **Role:** Provides low-level mathematical operations on vectors (lists of floats) and matrices (lists of lists of floats). Highlights include matrix inversion via Gauss-Jordan elimination, covariance calculation, and symmetric eigendecomposition via power iteration.
* **API Examples:**
  - `mat_mul(A, B)` $\to A \times B$
  - `mat_inv(M)` $\to M^{-1}$
  - `mat_sym_eig(M, n_components)` $\to (\text{eigenvalues}, \text{eigenvectors})$

### 3. Model Logic
* **File:** [src/lda/lda_model.py](../src/lda/lda_model.py)
* **Role:** Implements the `LDA` class containing learning, transformation, and prediction methods. 
* **State variables:**
  - `scalings_`: The projection matrix $W$.
  - `means_`: Class means in features space.
  - `explained_variance_ratio_`: Proportion of discriminative info per component.

### 4. Data Preprocessing
* **File:** [src/preprocessing/data_loader.py](../src/preprocessing/data_loader.py)
* **Role:** Handles ingestion, splitting, and scaling. It reads CSV datasets or falls back to an internal hardcoded string representation.
* **Standardization:** Scales training features to zero mean and unit variance. Standardizes test sets using training averages to prevent leakage.
* **Deterministic Splitting:** Implements a custom Fisher-Yates shuffle with a Linear Congruential Generator (LCG) to make train/test splits repeatable without relying on external seed APIs.

### 5. Evaluation Metrics
* **File:** [src/evaluation/metrics.py](../src/evaluation/metrics.py)
* **Role:** Computes predictive performance metrics from predicted and actual label vectors.
* **Formulas:** Outputs Accuracy, Confusion Matrices, Precision, Recall, and harmonic F1-scores. Features a `classification_report()` formatting function mimicking `scikit-learn` outputs.

### 6. Visualization
* **File:** [src/visualization/plots.py](../src/visualization/plots.py)
* **Role:** Translates numerical vectors into visual elements. Standardizes a custom dark theme palette (Red, Teal, Gold) matched to classes.
* **Format:** Generates Scalable Vector Graphics (SVG) directly using raw XML tag strings (no dependency on `matplotlib`). Coordinates are mapped to canvas pixels using custom scaling logic.

---

## Data Flow Diagram

```
[wine.data] ---> (prepare_raw_data.py) ---> [wine.csv] (or Built-in Fallback)
                                                  │
                                                  ▼
                                            (load_wine)
                                                  │
                                                  ▼
                                         (train_test_split)
                                            /          \
                                           ▼            ▼
                                      [X_train]      [X_test]
                                           │            │
                                           ▼            ▼
                                       (standardize statistics)
                                           │            │
                                           ▼            ▼
                                     [X_train_s]    [X_test_s]
                                           │            │
                                           ▼            │
                                        (fit)           │
                                       /     \          │
                                      ▼       ▼         ▼
                       [lda_weights.txt]  (transform & predict)
                                                  │
                                                  ▼
                                            [Evaluation]
                                           /           \
                                          ▼             ▼
                                   [results.txt]   (plots.py)
                                                        │
                                                        ▼
                                                   [SVG Files]
```

---

## Test Infrastructure

The codebase uses standard library tests located in [tests/test_lda.py](../tests/test_lda.py).

### How to Run Tests
Tests can be executed in two ways:
1. **Directly using Python (No external dependencies):**
   ```bash
   python tests/test_lda.py
   ```
2. **Using pytest (if installed in virtual environment):**
   ```bash
   python -m pytest tests/test_lda.py -v
   ```

### Test Coverage Breakdown
The test suite contains **15 assertions/test scenarios** divided into groups:
* **Math Utilities:** Validates multiplication identity ($A \times I = A$), inversion correctness ($A \times A^{-1} = I$), transpose symmetry, mean computation, and symmetric 2D matrix eigendecomposition.
* **Preprocessing:** Checks deterministic splitting bounds, train/test ratios, and Z-score standardization properties (confirming scaled training mean is exactly 0).
* **LDA Model:** Validates the fit/transform output dimension shapes ($N \times D \to N \times 2$) and checks prediction accuracy on synthetic, well-separated clusters (asserting accuracy $\ge 99\%$).
* **Pipeline Validation:** Executes an integration test of the full data cycle (shuffling, standardizing, training, predicting) on the UCI Wine Dataset, asserting that accuracy is higher than a baseline of $90\%$ (actual is $93.33\%$).
* **Data Preparation:** Verifies that raw-to-processed CSV conversion outputs correct file types, headers, and column shapes, handling missing raw files gracefully.

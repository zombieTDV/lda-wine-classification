# Linear Discriminant Analysis (LDA) Wine Classification from Scratch

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/tests-15%2F15%20passed-success.svg)](#running-tests)

A clean, modular, and dependency-free implementation of a **Linear Discriminant Analysis (LDA)** machine learning classification pipeline. Written entirely in pure Python, it requires no external libraries such as `numpy`, `pandas`, `scipy`, `scikit-learn`, or `matplotlib`.

This codebase showcases how the underlying linear algebra and statistical computations of LDA function under the hood—making it ideal for educational purposes, lightweight deployments, and custom machine learning algorithm exploration.

---

## 📖 Table of Contents
1. [Key Features](#-key-features)
2. [Project Architecture](#-project-architecture)
3. [The UCI Wine Dataset](#-the-uci-wine-dataset)
4. [Installation & Setup](#-installation--setup)
5. [Usage Guide](#-usage-guide)
6. [API Code Example](#-api-code-example)
7. [Mathematical Background](#-mathematical-background)
8. [Performance Metrics](#-performance-metrics)
9. [Detailed Documentation](#-detailed-documentation)
10. [License](#-license)

---

## ✨ Key Features

* **Pure Python implementation:** Z-score standardization, seeded train/test splits, matrix arithmetic, Gauss-Jordan matrix inversion, power iteration, deflation, and centroid distance classification are all built from scratch.
* **Auto-Fallback Data Loader:** The pipeline tries to read processed data from a CSV, but falls back automatically to an embedded copy of the UCI dataset if files are missing.
* **SVG Vector Graphics Engine:** Generates publication-ready, dark-themed plots (training scatter, testing scatter, explained variance bars, and confusion heatmaps) directly as SVG files without relying on Matplotlib.
* **Comprehensive Test Coverage:** Contains 15 unit and integration tests verifying math operations, pipeline integrity, data split statistics, and model precision.

---

## 📂 Project Architecture

The codebase separates concerns across preprocessing, model training, evaluation metrics, and visualization:

```
├── data/                  # Wine datasets
│   ├── raw/               # Raw UCI source dataset (wine.data, wine.names)
│   └── processed/         # Processed CSV format (wine.csv) with headers
├── docs/                  # Detailed design and theory documents
│   ├── ARCHITECTURE.md    # Software architecture design
│   ├── MATHEMATICS.md     # Eigendecomposition and LDA theory
│   ├── PIPELINE.md        # Step-by-step pipeline execution guide
│   └── PIPELINE_VERIFICATION.md # Configuration and sanity check audits
├── src/                   # Package source code
│   ├── lda/               # Core LDA math engine and class logic
│   │   ├── lda_model.py   # Main LDA model class
│   │   └── math_utils.py  # Custom linear algebra functions
│   ├── preprocessing/     # Data preparer, loader, and normalizer
│   │   ├── data_loader.py # Splitting, standardization, and EDA utilities
│   │   └── prepare_raw_data.py # Script converting raw files to CSV
│   ├── evaluation/        # Performance measurements
│   │   └── metrics.py     # Accuracy, F1, Recall, Precision, and CM
│   └── visualization/     # Custom SVG plot generators
│       └── plots.py       # Renderers for scatters, bars, and heatmaps
├── tests/                 # Unit and integration test suite
│   └── test_lda.py        # 15 tests covering all components
├── outputs/               # Created at runtime (ignored by Git)
│   ├── figures/           # SVG scatter plots and heatmaps
│   ├── reports/           # Detailed text results summary
│   └── models/            # Serialized model scalings and centroids
├── main.py                # Pipeline entrypoint orchestrator
└── README.md              # Project home page (this file)
```

For a thorough breakdown of modules and interfaces, see the [Codebase Architecture Guide](docs/ARCHITECTURE.md).

---

## 🍇 The UCI Wine Dataset

The pipeline uses the **UCI Wine Recognition Dataset**, which contains the results of a chemical analysis of wines grown in the same region in Italy but derived from three different cultivars (Class 1, 2, and 3). 
* **Samples:** 178
* **Classes:** 3 (Class 1: 59, Class 2: 71, Class 3: 48)
* **Chemical Attributes (13 Features):**
  1. *Alcohol*
  2. *Malic acid*
  3. *Ash*
  4. *Alcalinity of ash*
  5. *Magnesium*
  6. *Total phenols*
  7. *Flavanoids*
  8. *Nonflavanoid phenols*
  9. *Proanthocyanins*
  10. *Color intensity*
  11. *Hue*
  12. *OD280/OD315 of diluted wines*
  13. *Proline*

---

## ⚙️ Installation & Setup

1. Clone the repository and navigate to the directory:
   ```bash
   git clone <repository_url>
   cd lda-wine-classification
   ```

2. Create a virtual environment (recommended) and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. (Optional) Install `pytest` if you want to run tests with detailed test report formatting:
   ```bash
   pip install pytest
   ```

---

## 🚀 Usage Guide

### 1. Data Preparation
To regenerate the processed CSV file from the raw data:
```bash
python src/preprocessing/prepare_raw_data.py
```
This converts the raw `data/raw/wine.data` file to a standardized CSV at `data/processed/wine.csv` complete with appropriate headers.

### 2. Run the Classification Pipeline
Execute the complete classification workflow:
```bash
python main.py
```
This runs the entire ML pipeline (Loading $\to$ EDA $\to$ Split $\to$ Normalization $\to$ Fitting $\to$ Transform $\to$ Evaluate $\to$ Plot $\to$ Save) in less than **0.01 seconds**.

### 3. Running Tests
Run the comprehensive test suite to verify math utilities and model routines:
```bash
python tests/test_lda.py
```
Or run with `pytest` for advanced reports:
```bash
pytest tests/test_lda.py -v
```

---

## 💻 API Code Example

You can import and use the custom LDA model, data loader, and metrics in your own Python projects:

```python
from src.preprocessing.data_loader import load_wine, train_test_split, standardize
from src.lda.lda_model import LDA
from src.evaluation.metrics import accuracy, classification_report

# 1. Load and prepare dataset
X, y, feature_names = load_wine("data/processed/wine.csv")

# 2. Split into training and testing partitions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, seed=42)

# 3. Standardize training features (preventing data leakage to test set)
X_train_s, X_test_s, means, stds = standardize(X_train, X_test)

# 4. Instantiate and fit the LDA model
model = LDA(n_components=2)
model.fit(X_train_s, y_train)

# 5. Transform high-dimensional data to 2D space
X_train_lda = model.transform(X_train_s)
X_test_lda = model.transform(X_test_s)

# 6. Predict class labels
predictions = model.predict(X_test_s)

# 7. Evaluate performance
acc = accuracy(y_test, predictions)
print(f"Test Accuracy: {acc:.4%}")
print(classification_report(y_test, predictions))
```

---

## 🧮 Mathematical Background

LDA aims to find projection directions $W$ that maximize Fisher's Criterion:
$$J(W) = \frac{\det(W^T S_B W)}{\det(W^T S_W W)}$$

Where:
* **$S_W$ (Within-class Scatter):** Measures the spread of samples inside each class centroid:
  $$S_W = \sum_{c=1}^{C} \sum_{i \in \text{Class } c} (\mathbf{x}_i - \boldsymbol{\mu}_c)(\mathbf{x}_i - \boldsymbol{\mu}_c)^T$$
* **$S_B$ (Between-class Scatter):** Measures the spread of class centroids around the overall mean:
  $$S_B = \sum_{c=1}^{C} N_c (\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T$$

To maximize $J(W)$, we solve the generalized eigenvalue problem:
$$(S_W^{-1} S_B) \mathbf{w} = \lambda \mathbf{w}$$

The eigenvectors corresponding to the largest eigenvalues define the transformation axes in $W$. The codebase handles this by regularizing $S_W$, computing the inverse via Gauss-Jordan elimination, and performing symmetric power iteration with Hotelling deflation.

For a thorough review of the math details, see the [Mathematical Theory of LDA](docs/MATHEMATICS.md).

---

## 📊 Performance Metrics

Running the pipeline yields the following outcomes on the test set (25% split, seed 42):

* **Training Accuracy:** `99.25%` (132/133 samples correctly classified)
* **Test Accuracy:** `93.33%` (42/45 samples correctly classified)
* **Explained Variance:**
  - **LD1:** `69.6%` variance explained (Eigenvalue: 11.0962)
  - **LD2:** `30.4%` variance explained (Eigenvalue: 4.8364)
  - **Total:** `100.0%` of between-class variance retained in a 2-dimensional projection.

### Classification Report (Test Set)
```
  Class       Precision     Recall         F1    Support
  ----------------------------------------------------
  1              0.8462     1.0000     0.9167         11
  2              1.0000     0.8235     0.9032         17
  3              0.9444     1.0000     0.9714         17
  ----------------------------------------------------
  macro avg      0.9302     0.9412     0.9304
```

---

## 📚 Detailed Documentation

* **[Machine Learning Pipeline Guide](docs/PIPELINE.md):** Detailed guide on the 8 stages of the execution flow, configurations, and serializations.
* **[Mathematical Theory of LDA](docs/MATHEMATICS.md):** Equations, covariance calculations, and eigensolver derivation.
* **[Codebase Architecture Guide](docs/ARCHITECTURE.md):** Modular file layout, API design, class diagrams, and test setups.
* **[Pipeline Verification Report](docs/PIPELINE_VERIFICATION.md):** Auditing notes and checks.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
"""
algorithm_comparison.py
=======================
Compares LDA with other fundamental classification algorithms:
1. K-Nearest Neighbors (KNN) - Non-parametric, distance-based.
2. Gaussian Naive Bayes (GNB) - Parametric, assumes feature independence.
3. Logistic Regression (OvR) - Linear classifier using gradient descent.
4. Linear Discriminant Analysis (LDA) - Parametric, models covariance structure.

This script implements all algorithms from scratch to ensure a fair, 
dependency-free comparison, and generates decision boundary visualizations.
"""

import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.data_loader import load_wine, standardize
from src.lda.lda_model import LDA
from src.lda.math_utils import stratified_k_fold, mat_vec_mul, mat_transpose
from src.evaluation.metrics import accuracy

# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 1: K-Nearest Neighbors (KNN)
# ═══════════════════════════════════════════════════════════════════════════════

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = sum((x[j] - x_train[j]) ** 2 for j in range(len(x)))
                distances.append((dist, self.y_train[i]))
            
            distances.sort(key=lambda item: item[0])
            top_k_labels = [label for _, label in distances[:self.k]]
            
            vote_counts = {}
            for label in top_k_labels:
                vote_counts[label] = vote_counts.get(label, 0) + 1
                
            best_label = max(vote_counts.keys(), key=lambda l: vote_counts[l])
            predictions.append(best_label)
        return predictions

# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 2: Gaussian Naive Bayes (GNB)
# ═══════════════════════════════════════════════════════════════════════════════

class GaussianNB:
    def __init__(self):
        self.classes = []
        self.class_priors = {}
        self.class_means = {}
        self.class_vars = {}

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.classes = list(set(y))
        
        for c in self.classes:
            X_c = [X[i] for i in range(n_samples) if y[i] == c]
            n_c = len(X_c)
            self.class_priors[c] = n_c / n_samples
            
            means = [sum(x[j] for x in X_c) / n_c for j in range(n_features)]
            self.class_means[c] = means
            
            eps = 1e-9
            vars_ = [sum((x[j] - means[j]) ** 2 for x in X_c) / n_c + eps for j in range(n_features)]
            self.class_vars[c] = vars_
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            best_class = None
            best_log_prob = -float('inf')
            for c in self.classes:
                log_prob = math.log(self.class_priors[c])
                for j in range(len(x)):
                    v = self.class_vars[c][j]
                    m = self.class_means[c][j]
                    log_prob += -0.5 * math.log(2 * math.pi * v) - 0.5 * ((x[j] - m) ** 2) / v
                if log_prob > best_log_prob:
                    best_log_prob = log_prob
                    best_class = c
            predictions.append(best_class)
        return predictions

# ═══════════════════════════════════════════════════════════════════════════════
#  ALGORITHM 3: Logistic Regression (One-vs-Rest)
# ═══════════════════════════════════════════════════════════════════════════════

class LogisticRegressionOvR:
    """Logistic Regression using One-vs-Rest for multiclass classification."""
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.classes = []
        self.models = {}

    def fit(self, X, y):
        self.classes = list(set(y))
        n_features = len(X[0])
        n_samples = len(X)
        
        for c in self.classes:
            y_bin = [1 if label == c else 0 for label in y]
            w = [0.0] * n_features
            b = 0.0
            
            # Gradient Descent
            for _ in range(self.epochs):
                dw = [0.0] * n_features
                db = 0.0
                
                for i in range(n_samples):
                    z = sum(w[j] * X[i][j] for j in range(n_features)) + b
                    z = max(-20, min(20, z)) # Prevent overflow
                    y_pred = 1.0 / (1.0 + math.exp(-z))
                    
                    dz = y_pred - y_bin[i]
                    for j in range(n_features):
                        dw[j] += dz * X[i][j]
                    db += dz
                
                for j in range(n_features):
                    w[j] -= self.lr * (dw[j] / n_samples)
                b -= self.lr * (db / n_samples)
                
            self.models[c] = (w, b)
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            best_class = None
            best_prob = -1.0
            for c in self.classes:
                w, b = self.models[c]
                z = sum(w[j] * x[j] for j in range(len(x))) + b
                z = max(-20, min(20, z))
                prob = 1.0 / (1.0 + math.exp(-z))
                if prob > best_prob:
                    best_prob = prob
                    best_class = c
            predictions.append(best_class)
        return predictions

# ═══════════════════════════════════════════════════════════════════════════════
#  EVALUATION AND COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def compare_algorithms():
    X, y, feature_names = load_wine("data/processed/wine.csv")
    folds = stratified_k_fold(y, k=5, seed=42)
    
    algorithms = {
        "KNN (k=5)": KNN(k=5),
        "Gaussian NB": GaussianNB(),
        "Logistic Reg": LogisticRegressionOvR(lr=0.1, epochs=1000),
        "LDA (shrink=None)": LDA(n_components=2, shrinkage=None)
    }
    
    results = {name: [] for name in algorithms}
    times = {name: [] for name in algorithms}
    
    print("\n" + "="*60)
    print("  ALGORITHM COMPARISON (5-Fold Cross Validation)")
    print("="*60)
    
    for name, model in algorithms.items():
        fold_accs = []
        fold_times = []
        
        for train_idx, val_idx in folds:
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]
            
            X_train_s, X_val_s, _, _ = standardize(X_train, X_val)
            
            t0 = time.time()
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)
            t1 = time.time()
            
            fold_accs.append(accuracy(y_val, y_pred))
            fold_times.append((t1 - t0) * 1000)
            
        mean_acc = sum(fold_accs) / len(fold_accs)
        std_acc = math.sqrt(sum((a - mean_acc)**2 for a in fold_accs) / len(fold_accs))
        mean_time = sum(fold_times) / len(fold_times)
        
        results[name] = {"acc": mean_acc, "std": std_acc, "time_ms": mean_time}
        
        print(f"  {name:<20} | Accuracy: {mean_acc*100:>6.2f}% ± {std_acc*100:>4.2f}% | Time: {mean_time:>5.2f} ms")
        
    print("="*60)
    
    plot_comparison(results)
    
    # Generate decision boundary visualizations
    print("\n  Generating 2D Decision Boundaries (this may take a few seconds)...")
    plot_decision_boundaries(X, y)


def plot_comparison(results, out_path="outputs/figures/algorithm_comparison.svg"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    W, H = 700, 400
    margin = {"top": 60, "right": 40, "bottom": 60, "left": 80}
    plot_w = W - margin["left"] - margin["right"]
    plot_h = H - margin["top"] - margin["bottom"]
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'  <rect width="{W}" height="{H}" fill="#1a1a2e"/>',
        f'  <text x="{W/2}" y="35" fill="#ffffff" font-family="sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Algorithm Comparison (Wine Dataset)</text>'
    ]
    
    names = list(results.keys())
    accs = [r["acc"] for r in results.values()]
    
    min_acc_plot = 0.90
    acc_range_plot = 1.0 - min_acc_plot
    
    svg.append(f'  <g stroke="#ffffff" stroke-opacity="0.2" stroke-width="1">')
    for i in range(6):
        tick_acc = min_acc_plot + i * 0.02
        y = margin["top"] + plot_h - (i * 0.02 / acc_range_plot) * plot_h
        svg.append(f'    <line x1="{margin["left"]}" y1="{y}" x2="{W - margin["right"]}" y2="{y}"/>')
        svg.append(f'    <text x="{margin["left"] - 10}" y="{y + 5}" fill="#a0a0b0" font-family="sans-serif" font-size="14" text-anchor="end" stroke="none">{tick_acc*100:.0f}%</text>')
    svg.append(f'  </g>')

    svg.append(f'  <text x="25" y="{H/2}" fill="#ffffff" font-family="sans-serif" font-size="16" text-anchor="middle" transform="rotate(-90 25 {H/2})">Cross-Validation Accuracy</text>')
    
    bar_width = min(80, plot_w / (len(names) + 1))
    spacing = (plot_w - (bar_width * len(names))) / (len(names) + 1)
    colors = ["#f4a261", "#e76f51", "#8ab17d", "#2a9d8f"]
    
    for i, name in enumerate(names):
        acc = results[name]["acc"]
        x = margin["left"] + spacing + i * (bar_width + spacing)
        bar_h = ((acc - min_acc_plot) / acc_range_plot) * plot_h
        if bar_h < 0: bar_h = 5
        y = margin["top"] + plot_h - bar_h
        
        svg.append(f'  <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="{colors[i]}" rx="4" ry="4"/>')
        svg.append(f'  <text x="{x + bar_width/2}" y="{y - 10}" fill="#ffffff" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="middle">{acc*100:.2f}%</text>')
        svg.append(f'  <text x="{x + bar_width/2}" y="{margin["top"] + plot_h + 20}" fill="#a0a0b0" font-family="sans-serif" font-size="14" text-anchor="middle">{name}</text>')

    svg.append('</svg>')
    
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"\n  [Plot] Saved comparison chart to: {out_path}")

def plot_decision_boundaries(X, y, out_path="outputs/figures/decision_boundaries.svg"):
    """
    To visualize all algorithms in 2D, we first project the data using LDA.
    Then we train all algorithms on the 2D projected data and plot their 
    decision boundaries.
    """
    X_s, _, _ = standardize(X)
    
    # 1. Project data to 2D using LDA
    lda_proj = LDA(n_components=2, shrinkage=None)
    lda_proj.fit(X_s, y)
    X_2d = lda_proj.transform(X_s)
    
    # 2. Train models on 2D data
    models = {
        "LDA": LDA(n_components=2, shrinkage=0.0), # No shrinkage needed for 2D
        "KNN (k=5)": KNN(k=5),
        "Gaussian NB": GaussianNB(),
        "Logistic Reg": LogisticRegressionOvR(lr=0.1, epochs=500)
    }
    
    for model in models.values():
        model.fit(X_2d, y)
        
    # 3. Create grid
    x_min = min(row[0] for row in X_2d) - 1.0
    x_max = max(row[0] for row in X_2d) + 1.0
    y_min = min(row[1] for row in X_2d) - 1.0
    y_max = max(row[1] for row in X_2d) + 1.0
    
    grid_size = 60 # 60x60 grid
    dx = (x_max - x_min) / grid_size
    dy = (y_max - y_min) / grid_size
    
    grid_points = []
    for j in range(grid_size):
        for i in range(grid_size):
            grid_points.append([x_min + i * dx, y_min + j * dy])
            
    # 4. Predict grid for all models
    grid_preds = {}
    for name, model in models.items():
        grid_preds[name] = model.predict(grid_points)
        
    # 5. Draw SVG (2x2 grid)
    W, H = 800, 840
    margin_x = 60
    margin_y = 60
    top_margin = 100
    plot_w = (W - 3 * margin_x) / 2
    plot_h = (H - top_margin - 2 * margin_y) / 2
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'  <rect width="{W}" height="{H}" fill="#1a1a2e"/>',
        f'  <text x="{W/2}" y="45" fill="#ffffff" font-family="sans-serif" font-size="28" font-weight="bold" text-anchor="middle">2D Decision Boundaries Comparison</text>'
    ]
    
    # Colors for regions and points
    colors = {
        1: ("#ff6b7a", "#a31d2a"), # Red (Light for bg, Dark for point)
        2: ("#52d9cb", "#127065"), # Teal
        3: ("#f4d792", "#a6822b")  # Gold
    }
    
    positions = [
        (margin_x, top_margin, "LDA"),
        (2*margin_x + plot_w, top_margin, "Logistic Reg"),
        (margin_x, top_margin + margin_y + plot_h, "Gaussian NB"),
        (2*margin_x + plot_w, top_margin + margin_y + plot_h, "KNN (k=5)")
    ]
    
    for px_start, py_start, name in positions:
        svg.append(f'  <g transform="translate({px_start}, {py_start})">')
        svg.append(f'    <text x="{plot_w/2}" y="-15" fill="#ffffff" font-family="sans-serif" font-size="18" font-weight="bold" text-anchor="middle">{name}</text>')
        svg.append(f'    <rect x="0" y="0" width="{plot_w}" height="{plot_h}" fill="#2a2a40" stroke="#ffffff" stroke-opacity="0.3"/>')
        
        # Draw background regions
        preds = grid_preds[name]
        cell_w = plot_w / grid_size
        cell_h = plot_h / grid_size
        
        for idx, pred in enumerate(preds):
            i = idx % grid_size
            j = idx // grid_size
            # Invert Y axis for drawing
            rect_x = i * cell_w
            rect_y = plot_h - (j + 1) * cell_h
            bg_color = colors[pred][0]
            svg.append(f'    <rect x="{rect_x}" y="{rect_y}" width="{cell_w+0.5}" height="{cell_h+0.5}" fill="{bg_color}" opacity="0.3"/>')
            
        # Draw scatter points
        for i in range(len(X_2d)):
            x_val = X_2d[i][0]
            y_val = X_2d[i][1]
            label = y[i]
            
            cx = ((x_val - x_min) / (x_max - x_min)) * plot_w
            cy = plot_h - ((y_val - y_min) / (y_max - y_min)) * plot_h
            
            pt_color = colors[label][1]
            svg.append(f'    <circle cx="{cx}" cy="{cy}" r="3" fill="{pt_color}" stroke="#ffffff" stroke-width="0.5"/>')
            
        svg.append('  </g>')
        
    svg.append('</svg>')
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"  [Plot] Saved decision boundaries to: {out_path}")

if __name__ == '__main__':
    compare_algorithms()


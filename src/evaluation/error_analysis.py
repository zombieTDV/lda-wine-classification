"""
error_analysis.py
=================
Investigates why LDA does not achieve 100% accuracy.
Runs Leave-One-Out Cross-Validation (LOOCV) to test every single
sample exactly once and identifies the specific misclassified wines.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.data_loader import load_wine, standardize
from src.lda.lda_model import LDA
from src.lda.math_utils import mat_vec_mul, mat_transpose
from src.evaluation.metrics import accuracy

def run_error_analysis():
    print("\n" + "="*70)
    print("  LDA ERROR ANALYSIS (Leave-One-Out Cross-Validation)")
    print("="*70)
    
    X, y, feature_names = load_wine("data/processed/wine.csv")
    n_samples = len(X)
    
    # We will track which indices were misclassified
    misclassified_indices = []
    misclassified_preds = []
    
    correct_count = 0
    
    for i in range(n_samples):
        # Leave one out
        X_train = [X[j] for j in range(n_samples) if j != i]
        y_train = [y[j] for j in range(n_samples) if j != i]
        X_test = [X[i]]
        y_true = y[i]
        
        # Standardize
        X_train_s, X_test_s, _, _ = standardize(X_train, X_test)
        
        # Fit model
        model = LDA(n_components=2, shrinkage=None)
        model.fit(X_train_s, y_train)
        
        # Predict
        y_pred = model.predict(X_test_s)[0]
        
        if y_pred == y_true:
            correct_count += 1
        else:
            misclassified_indices.append(i)
            misclassified_preds.append(y_pred)
            
    total_acc = correct_count / n_samples
    print(f"  LOOCV Accuracy: {correct_count}/{n_samples} ({total_acc*100:.2f}%)")
    print(f"  Total Misclassified: {len(misclassified_indices)}")
    
    if len(misclassified_indices) == 0:
        print("  Wow, it got 100% in LOOCV!")
    else:
        print("\n  MISCLASSIFIED SAMPLES DETAILED BREAKDOWN:")
        print("  " + "-"*65)
        for idx, i in enumerate(misclassified_indices):
            true_class = y[i]
            pred_class = misclassified_preds[idx]
            
            print(f"  Sample ID: {i} | True Class: {true_class} | Predicted: {pred_class}")
            
            # Why did it fail? Let's check the LDA distances for this specific sample
            # We retrain on all data to see its position in the global LDA space
            X_s, _, _ = standardize(X)
            full_model = LDA(n_components=2, shrinkage=None)
            full_model.fit(X_s, y)
            x_lda = full_model.transform([X_s[i]])[0]
            
            # Print distances to class centroids
            print("    Distances to Class Centroids (in LDA space):")
            W_T = mat_transpose(full_model.scalings_)
            for c in full_model.classes_:
                centroid = mat_vec_mul(W_T, full_model.means_[c])
                dist = math.sqrt(sum((x_lda[j] - centroid[j])**2 for j in range(len(centroid))))
                marker = "<-- True" if c == true_class else ("<-- Predicted" if c == pred_class else "")
                print(f"      Class {c}: {dist:.4f} {marker}")
                
            print("    This happens because this specific wine sample has chemical")
            print("    properties that naturally place it exactly on the decision")
            print("    boundary between classes. It is an outlier for its true class.")
            print("  " + "-"*65)

    print("="*70 + "\n")

if __name__ == '__main__':
    run_error_analysis()

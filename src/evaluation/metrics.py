"""
metrics.py
==========
Model evaluation metrics — built from scratch without external libraries.
Includes: accuracy, confusion matrix, precision, recall, F1-score, and reports.
"""


def accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Accuracy = (Number of correct predictions) / (Total predictions)
    
    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        
    Returns:
        float: Accuracy score between 0 and 1
        
    Example:
        accuracy([1, 1, 2, 2], [1, 2, 2, 2]) = 0.75  (3 out of 4 correct)
    """
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, classes=None):
    """
    Build a confusion matrix as nested dictionaries.
    
    A confusion matrix shows how many samples of each true class
    were predicted as each predicted class.
    
    Structure: CM[true_class][predicted_class] = count
    
    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        classes (list, optional): List of class labels. If None, automatically detect from data.
        
    Returns:
        tuple: (CM dict, classes list)
               CM is structured as {true_class: {pred_class: count}}
               
    Example:
        y_true = [1, 1, 2, 2]
        y_pred = [1, 2, 2, 2]
        CM = {1: {1: 1, 2: 1},  <- 1 sample of class 1 predicted as 1, 1 as 2
              2: {1: 0, 2: 2}}  <- 0 samples of class 2 predicted as 1, 2 as 2
    """
    if classes is None:
        # Automatically find all unique classes from both true and predicted
        classes = sorted(set(y_true) | set(y_pred))
    
    # Initialize matrix with zeros
    CM = {c: {c2: 0 for c2 in classes} for c in classes}
    
    # Fill the matrix by counting predictions
    for t, p in zip(y_true, y_pred):
        CM[t][p] += 1
    
    return CM, classes


def precision_recall_f1(y_true, y_pred, classes=None):
    """
    Calculate per-class precision, recall, F1-score and their macro-averages.
    
    Definitions:
    - Precision: Of samples predicted as this class, how many were correct?
      Precision = TP / (TP + FP)
    - Recall: Of samples that truly belong to this class, how many did we find?
      Recall = TP / (TP + FN)
    - F1-score: Harmonic mean of precision and recall (balance between both)
      F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Where:
    - TP (True Positive): Correctly predicted as this class
    - FP (False Positive): Incorrectly predicted as this class (but belonged to other)
    - FN (False Negative): Missed this class (predicted as something else)
    
    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        classes (list, optional): List of class labels
        
    Returns:
        dict: Metrics for each class + macro-average
              Format: {class: {precision, recall, f1}, ..., "macro": {precision, recall, f1}}
    """
    CM, classes = confusion_matrix(y_true, y_pred, classes)
    results = {}
    
    # Calculate metrics for each class
    for c in classes:
        # Get counts from confusion matrix
        TP = CM[c][c]  # True Positives: correctly predicted as class c
        FP = sum(CM[r][c] for r in classes if r != c)  # False Positives: predicted as c but aren't
        FN = sum(CM[c][r] for r in classes if r != c)  # False Negatives: actually c but predicted as something else
        
        # Calculate precision, recall, F1 with safety check for division by zero
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        results[c] = {"precision": prec, "recall": rec, "f1": f1}

    # Calculate macro-averages (simple average across all classes)
    macro_prec = sum(v["precision"] for v in results.values()) / len(classes)
    macro_rec  = sum(v["recall"]    for v in results.values()) / len(classes)
    macro_f1   = sum(v["f1"]        for v in results.values()) / len(classes)
    results["macro"] = {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1}
    
    return results


def classification_report(y_true, y_pred):
    """
    Generate a formatted classification report with all metrics.
    
    Includes:
    - Per-class precision, recall, F1-score, and support (count)
    - Macro-average metrics (simple average across classes)
    - Overall accuracy
    
    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        
    Returns:
        str: Formatted report as a string suitable for printing
    """
    acc = accuracy(y_true, y_pred)
    CM, classes = confusion_matrix(y_true, y_pred)
    metrics = precision_recall_f1(y_true, y_pred, classes)

    # Build report lines
    lines = [
        "",
        "  Classification Report",
        "  " + "-" * 52,
        f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "  " + "-" * 52,
    ]
    
    # Add per-class metrics
    for c in classes:
        support = sum(1 for t in y_true if t == c)  # How many samples of this class in data?
        m = metrics[c]
        lines.append(
            f"  {str(c):<10} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {support:>10}"
        )
    
    # Add macro-average and accuracy
    lines += [
        "  " + "-" * 52,
        f"  {'macro avg':<10} {metrics['macro']['precision']:>10.4f} "
        f"{metrics['macro']['recall']:>10.4f} {metrics['macro']['f1']:>10.4f}",
        "  " + "-" * 52,
        f"\n  Accuracy: {acc:.4f}  ({sum(1 for a,b in zip(y_true,y_pred) if a==b)}"
        f"/{len(y_true)} correct)",
        "",
    ]
    return "\n".join(lines)


def print_confusion_matrix(y_true, y_pred):
    """
    Print confusion matrix in a nicely formatted table.
    
    Rows = True labels (what they actually are)
    Columns = Predicted labels (what the model predicted)
    Diagonal = Correct predictions
    Off-diagonal = Incorrect predictions (misclassifications)
    
    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
    """
    CM, classes = confusion_matrix(y_true, y_pred)
    w = 8  # Column width
    
    # Print header
    print("\n  Confusion Matrix")
    print("  " + "-" * (w * (len(classes) + 1) + 4))
    print("  " + " " * w + "".join(f"{f'Pred {c}':>{w}}" for c in classes))
    
    # Print each row
    for c in classes:
        row = f"  {'True '+str(c):<{w}}"  # Row label
        for c2 in classes:
            row += f"{CM[c][c2]:>{w}}"  # Add count for each predicted class
        print(row)
    print()

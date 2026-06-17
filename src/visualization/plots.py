"""
plots.py
========
Data visualization module — generates SVG plots without external libraries.

This module creates publication-quality visualizations by directly generating
Scalable Vector Graphics (SVG) without depending on matplotlib or other
visualization libraries. All plots are rendered to SVG files that can be
viewed in any web browser or vector graphics editor.

Why SVG?
--------
- Independent: No need to install matplotlib, seaborn, plotly, etc.
- Scalable: Resolution-independent (zoom in/out without pixelation)
- Lightweight: Text-based format (can be edited in any text editor)
- Web-ready: Can be embedded directly in HTML
- Offline: No need for interactive backends

Visualizations Included
------------------------
1. scatter_lda: 2D scatter plot of LDA projections with axes, grid, legend
2. bar_variance: Bar chart showing explained variance ratio + cumulative line
3. heatmap_confusion: Color-coded confusion matrix heatmap

All plots follow a dark theme (dark background, light text) for professional appearance.
"""

import os
import math


# ─── Color Palette for Wine Classes ─────────────────────────────────────────
# Each class is assigned a pair of colors: (main_color, accent_color)
# Main color = primary fill, Accent = stroke/highlight for emphasis

PALETTE = {
    1: ("#e63946", "#ff6b7a"),   # Red   — Wine class 1
    2: ("#2a9d8f", "#52d9cb"),   # Teal  — Wine class 2
    3: ("#e9c46a", "#f4d792"),   # Gold  — Wine class 3
}


def _minmax(vals):
    """
    Find minimum, maximum, and range of values.
    
    This utility function normalizes data to [0, 1] range for plotting.
    Used to map data coordinates to pixel coordinates in SVG.
    
    Parameters
    ----------
    vals : list[float]
        List of numeric values
        
    Returns
    -------
    lo : float
        Minimum value
        
    hi : float
        Maximum value
        
    r : float
        Range (hi - lo). If range is too small (< 1e-10), returns 1.0
        to avoid division by zero in normalization.
        
    Example
    -------
    _minmax([1, 3, 2, 5, 4]) → (1, 5, 4)
    _minmax([1.0, 1.0, 1.0]) → (1.0, 1.0, 1.0)  all same value
    """
    lo, hi = min(vals), max(vals)
    r = hi - lo
    # Avoid division by zero when all values are the same
    if r < 1e-10:
        r = 1.0
    return lo, hi, r




def scatter_lda(X_lda, y, title="LDA Projection",
                out_path="outputs/figures/lda_scatter.svg"):
    """
    Create 2D scatter plot of LDA projection and save as SVG.
    
    Visualizes the lower-dimensional LDA space showing how well classes
    are separated. Each point represents one data sample, colored by class.
    Includes axes, grid, and legend for easy interpretation.
    
    SVG Structure
    ------
    - Dark background (#1a1a2e) with light text for professional appearance
    - X and Y axes with tick marks at regular intervals
    - Faint grid lines for reference
    - Semi-transparent points to show overlaps
    - Color-coded legend identifying each class
    
    Parameters
    ----------
    X_lda : list[list[float]]
        LDA-transformed data, shape (n_samples, n_components)
        Must have at least 2 components (LD1, LD2)
        Each row = one sample's position in LDA space
        
    y : list
        Class labels for each sample (e.g., [1, 2, 3, 1, 2, ...])
        Used for color assignment
        
    title : str, default="LDA Projection"
        Title displayed at top of plot
        
    out_path : str, default="outputs/figures/lda_scatter.svg"
        Output file path for SVG file
        Parent directories created automatically if needed
        
    Returns
    -------
    str
        Path to the generated SVG file
        
    Example
    -------
    X_lda = model.transform(X_test)
    scatter_lda(X_lda, y_test, title="Test Set LDA", 
                out_path="outputs/plots/test_lda.svg")
    
    Note
    ----
    - Requires at least 2D data (LD1, LD2)
    - Uses first two components only
    - Auto-scales to fit all data points
    """
    assert len(X_lda[0]) >= 2, "Need at least 2 LDs for 2D plot"

    # ── Canvas settings ──────────────────────────────────────────────────
    W, H = 700, 520
    pad = 70  # Padding from edges for axes

    # ── Extract X and Y coordinates from LDA space ────────────────────
    x_vals = [p[0] for p in X_lda]  # LD1 (first discriminant)
    y_vals = [p[1] for p in X_lda]  # LD2 (second discriminant)
    
    # ── Find data range for normalization ────────────────────────────
    x_lo, x_hi, x_r = _minmax(x_vals)
    y_lo, y_hi, y_r = _minmax(y_vals)

    # ── Coordinate conversion function (data → pixels) ──────────────
    def to_px(xv, yv):
        """Convert data coordinates to pixel coordinates in SVG"""
        px = pad + (xv - x_lo) / x_r * (W - 2 * pad)
        py = H - pad - (yv - y_lo) / y_r * (H - 2 * pad)  # Flip Y (SVG goes down)
        return px, py

    classes = sorted(set(y))
    
    # ── Build SVG ────────────────────────────────────────────────────
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'style="background:#1a1a2e;font-family:monospace;">',

        # Title
        f'<text x="{W//2}" y="28" text-anchor="middle" '
        f'fill="#e0e0e0" font-size="16" font-weight="bold">{title}</text>',

        # Axes (X and Y)
        f'<line x1="{pad}" y1="{H-pad}" x2="{W-pad}" y2="{H-pad}" '
        f'stroke="#555" stroke-width="1"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{H-pad}" '
        f'stroke="#555" stroke-width="1"/>',

        # Axis labels
        f'<text x="{W//2}" y="{H-8}" text-anchor="middle" '
        f'fill="#aaa" font-size="12">LD1</text>',
        f'<text x="14" y="{H//2}" text-anchor="middle" '
        f'fill="#aaa" font-size="12" '
        f'transform="rotate(-90,14,{H//2})">LD2</text>',
    ]

    # ── Add grid lines ───────────────────────────────────────────────
    # Helps read values from plot
    for i in range(5):
        gy = pad + i * (H - 2*pad) // 4
        svg.append(f'<line x1="{pad}" y1="{gy}" x2="{W-pad}" y2="{gy}" '
                   f'stroke="#333" stroke-width="0.5" stroke-dasharray="4,4"/>')
        gx = pad + i * (W - 2*pad) // 4
        svg.append(f'<line x1="{gx}" y1="{pad}" x2="{gx}" y2="{H-pad}" '
                   f'stroke="#333" stroke-width="0.5" stroke-dasharray="4,4"/>')

    # ── Plot points ──────────────────────────────────────────────────
    for xi, yi in zip(X_lda, y):
        px, py = to_px(xi[0], xi[1])
        fill, stroke = PALETTE.get(yi, ("#888", "#aaa"))
        svg.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="0.8" opacity="0.85"/>'
        )

    # ── Add legend ───────────────────────────────────────────────────
    lx, ly = W - pad - 90, pad + 20
    svg.append(f'<rect x="{lx-8}" y="{ly-16}" width="100" '
               f'height="{len(classes)*24+12}" rx="6" '
               f'fill="#16213e" stroke="#444" stroke-width="1"/>')
    for i, c in enumerate(classes):
        fill, _ = PALETTE.get(c, ("#888", "#aaa"))
        cy = ly + i * 24
        svg.append(f'<circle cx="{lx+6}" cy="{cy}" r="5" fill="{fill}"/>')
        svg.append(f'<text x="{lx+16}" y="{cy+4}" fill="#ccc" '
                   f'font-size="11">Class {c}</text>')

    svg.append("</svg>")
    content = "\n".join(svg)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"  [Plot] Saved: {out_path}")
    return out_path





def bar_variance(eigenvalues, explained_ratios,
                 out_path="outputs/figures/explained_variance.svg"):
    """
    Create bar chart showing explained variance ratio of each LDA component.
    
    Shows how much of the discriminative power comes from each linear
    discriminant. Includes a cumulative curve to show cumulative explained
    variance as more components are added.
    
    Chart Components
    ----------------
    - Bars: Explained variance % for each LD component
    - Cumulative Line: Running total of variance explained
    
    Mathematical Background
    -----------------------
    For eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λₖ from the generalized eigenvalue
    problem S_W⁻¹ S_B w = λw, the explained variance ratio is:
    
        explained_ratio[i] = λᵢ / (Σⱼ λⱼ)
    
    This measures how much of the between-class scatter is captured by
    each discriminant axis.
    
    Parameters
    ----------
    eigenvalues : list[float]
        Eigenvalues from LDA fit, sorted in descending order
        Larger eigenvalue = more discriminative power
        
    explained_ratios : list[float]
        Normalized explained variance for each eigenvalue
        Sum should be 1.0 (100%)
        Each value in [0, 1]
        
    out_path : str, default="outputs/figures/explained_variance.svg"
        Output file path for SVG
        
    Returns
    -------
    str
        Path to the generated SVG file
        
    Example
    -------
    model = LDA(n_components=2)
    model.fit(X_train, y_train)
    bar_variance(model.eigenvalues_, model.explained_variance_ratio_,
                 out_path="outputs/plots/variance.svg")
    
    Interpretation
    ---------------
    - If first bar is 70% and second is 30%, the first LD captures
      most of the class separation
    - Cumulative line reaching 100% shows that K components are sufficient
    """
    W, H = 560, 380
    pad_l, pad_r, pad_t, pad_b = 70, 40, 50, 60
    k = len(eigenvalues)
    bar_w = (W - pad_l - pad_r) / (k + 1)
    # Cumulative sum: cum[i] = explained_ratios[0] + ... + explained_ratios[i]
    cum = [sum(explained_ratios[:i+1]) for i in range(k)]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'style="background:#1a1a2e;font-family:monospace;">',
        f'<text x="{W//2}" y="30" text-anchor="middle" fill="#e0e0e0" '
        f'font-size="14" font-weight="bold">Explained Variance Ratio</text>',
    ]

    chart_h = H - pad_t - pad_b
    chart_w = W - pad_l - pad_r

    # ── Draw axes ────────────────────────────────────────────────────
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" '
               f'y2="{H-pad_b}" stroke="#555" stroke-width="1"/>')
    svg.append(f'<line x1="{pad_l}" y1="{H-pad_b}" x2="{W-pad_r}" '
               f'y2="{H-pad_b}" stroke="#555" stroke-width="1"/>')

    # ── Y-axis ticks and grid ────────────────────────────────────────
    # Tick marks at 0%, 25%, 50%, 75%, 100%
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        ty = pad_t + (1 - tick) * chart_h
        svg.append(f'<line x1="{pad_l-4}" y1="{ty:.1f}" x2="{W-pad_r}" '
                   f'y2="{ty:.1f}" stroke="#333" stroke-width="0.5" stroke-dasharray="3,3"/>')
        svg.append(f'<text x="{pad_l-8}" y="{ty+4:.1f}" text-anchor="end" '
                   f'fill="#999" font-size="10">{tick:.0%}</text>')

    # ── Draw bars for each component ─────────────────────────────────
    for i, (ev, ratio) in enumerate(zip(eigenvalues, explained_ratios)):
        bx = pad_l + (i + 0.5) * bar_w  # Bar center X position
        bh = ratio * chart_h             # Bar height proportional to variance ratio
        by = H - pad_b - bh              # Bar top Y position
        svg.append(f'<rect x="{bx - bar_w*0.35:.1f}" y="{by:.1f}" '
                   f'width="{bar_w*0.7:.1f}" height="{bh:.1f}" '
                   f'fill="#e63946" rx="3" opacity="0.85"/>')
        # Percentage label on top of bar
        svg.append(f'<text x="{bx:.1f}" y="{by-6:.1f}" text-anchor="middle" '
                   f'fill="#e0e0e0" font-size="10">{ratio:.1%}</text>')
        # Component label below
        svg.append(f'<text x="{bx:.1f}" y="{H-pad_b+16:.1f}" '
                   f'text-anchor="middle" fill="#aaa" font-size="11">LD{i+1}</text>')

    # ── Draw cumulative line ─────────────────────────────────────────
    # Shows running total: LD1 alone, LD1+LD2, etc.
    pts = []
    for i, c in enumerate(cum):
        cx = pad_l + (i + 0.5) * bar_w
        cy = pad_t + (1 - c) * chart_h
        pts.append(f"{cx:.1f},{cy:.1f}")
    svg.append(f'<polyline points="{" ".join(pts)}" fill="none" '
               f'stroke="#2a9d8f" stroke-width="2" stroke-dasharray="5,3"/>')
    # Markers on cumulative line
    for cx_s, cy_s in (p.split(",") for p in pts):
        svg.append(f'<circle cx="{cx_s}" cy="{cy_s}" r="4" fill="#2a9d8f"/>')

    svg.append("</svg>")
    content = "\n".join(svg)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"  [Plot] Saved: {out_path}")
    return out_path





def heatmap_confusion(CM_dict, classes,
                      out_path="outputs/figures/confusion_matrix.svg"):
    """
    Create color-coded heatmap of confusion matrix and save as SVG.
    
    Visualizes classification performance by showing:
    - Correct predictions: Green shades (diagonal elements)
    - Incorrect predictions: Red shades (off-diagonal)
    - Intensity: Darker = more count
    
    Confusion Matrix Structure
    ---------------------------
    CM[true_label][predicted_label] = count
    
    Example 3×3 matrix:
                 Predicted
                 1    2    3
    True    1    [a]  [b]  [c]
            2    [d]  [e]  [f]
            3    [g]  [h]  [i]
    
    Perfect classifier: Only diagonal elements (a, e, i) have counts.
    
    Color Scheme
    -----------
    Diagonal (Correct predictions):
        - Gradient from light to dark green
        - Darker = more correct in this class
        
    Off-diagonal (Misclassifications):
        - Gradient from light to dark red
        - Darker = more common misclassification
    
    Parameters
    ----------
    CM_dict : dict[dict[int]]
        Nested dictionary confusion matrix
        CM_dict[true_label][predicted_label] = count
        Example: CM_dict[1][2] = 5 means 5 samples from class 1
                                  were predicted as class 2
        
    classes : list
        List of unique class labels (e.g., [1, 2, 3])
        Used to order rows and columns
        
    out_path : str, default="outputs/figures/confusion_matrix.svg"
        Output file path for SVG
        
    Returns
    -------
    str
        Path to the generated SVG file
        
    Example
    -------
    from src.evaluation.metrics import confusion_matrix
    
    CM = confusion_matrix(y_test, y_pred, classes=[1, 2, 3])
    heatmap_confusion(CM, [1, 2, 3],
                      out_path="outputs/plots/cm.svg")
    
    Interpretation
    ---------------
    - Read rows as "True Label", columns as "Predicted Label"
    - Strong green diagonal = good classifier (correct predictions)
    - Red off-diagonal cells = areas where model confuses classes
    - Can identify which classes are hard to distinguish
    """
    n = len(classes)
    cell = 70  # Size of each cell in pixels
    pad_l, pad_t = 80, 80
    W = pad_l + cell * n + 40
    H = pad_t + cell * n + 60

    # ── Find maximum count for color normalization ───────────────────
    # Helps determine color intensity (darker = higher count)
    vals = [CM_dict[r][c] for r in classes for c in classes]
    max_v = max(vals) if vals else 1

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'style="background:#1a1a2e;font-family:monospace;">',
        f'<text x="{W//2}" y="28" text-anchor="middle" fill="#e0e0e0" '
        f'font-size="14" font-weight="bold">Confusion Matrix</text>',
        f'<text x="{W//2}" y="{H-10}" text-anchor="middle" '
        f'fill="#aaa" font-size="11">Predicted Label</text>',
        f'<text x="12" y="{H//2}" text-anchor="middle" fill="#aaa" '
        f'font-size="11" transform="rotate(-90,12,{H//2})">True Label</text>',
    ]

    # ── Add column headers (predicted labels) ─────────────────────────
    for ci, c_pred in enumerate(classes):
        x = pad_l + ci * cell + cell // 2
        svg.append(f'<text x="{x}" y="{pad_t-10}" text-anchor="middle" '
                   f'fill="#ccc" font-size="12">Class {c_pred}</text>')
    
    # ── Add row headers (true labels) ─────────────────────────────────
    for ri, c_true in enumerate(classes):
        y = pad_t + ri * cell + cell // 2 + 5
        svg.append(f'<text x="{pad_l-10}" y="{y}" text-anchor="end" '
                   f'fill="#ccc" font-size="12">Class {c_true}</text>')

    # ── Draw cells with color coding ─────────────────────────────────
    for ri, c_true in enumerate(classes):
        for ci, c_pred in enumerate(classes):
            v = CM_dict[c_true][c_pred]
            intensity = v / max_v  # Normalized to [0, 1]
            
            # Color depends on whether cell is on diagonal (correct) or not
            if c_true == c_pred:
                # Diagonal (correct predictions): gradient to green
                # Formula: Light → Dark Green as intensity increases
                r_ = int(42 * (1 - intensity) + 42)      # RGB: (42, 157±, 143±) → darker
                g_ = int(157 * intensity + 50 * (1 - intensity))
                b_ = int(143 * intensity + 62 * (1 - intensity))
            else:
                # Off-diagonal (misclassifications): gradient to red
                # Formula: Light → Dark Red as intensity increases
                r_ = int(230 * intensity + 26 * (1 - intensity))  # RGB: (230±, 57±, 70±) → darker
                g_ = int(57 * intensity + 26 * (1 - intensity))
                b_ = int(70 * intensity + 46 * (1 - intensity))
            
            color = f"rgb({r_},{g_},{b_})"
            x = pad_l + ci * cell
            y = pad_t + ri * cell
            
            # Draw cell
            svg.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                       f'fill="{color}" stroke="#1a1a2e" stroke-width="2"/>')
            
            # Add count text
            # Choose white text for dark cells, light gray for bright cells
            txt_fill = "#fff" if intensity > 0.4 else "#ccc"
            svg.append(f'<text x="{x+cell//2}" y="{y+cell//2+6}" '
                       f'text-anchor="middle" fill="{txt_fill}" '
                       f'font-size="18" font-weight="bold">{v}</text>')

    svg.append("</svg>")
    content = "\n".join(svg)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"  [Plot] Saved: {out_path}")
    return out_path


def line_chart_shrinkage(shrinkage_results, best_alpha, 
                         title="Shrinkage (α) vs Accuracy", 
                         out_path="outputs/figures/shrinkage_accuracy.svg"):
    """
    Create a line chart showing accuracy across different shrinkage values.
    
    Parameters
    ----------
    shrinkage_results : list[dict]
        Results from grid_search_shrinkage
    best_alpha : float
        The optimal shrinkage value to highlight
    title : str
        Plot title
    out_path : str
        File path to save the SVG
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    W, H = 800, 500
    margin = {"top": 60, "right": 40, "bottom": 60, "left": 80}
    plot_w = W - margin["left"] - margin["right"]
    plot_h = H - margin["top"] - margin["bottom"]
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'  <rect width="{W}" height="{H}" fill="#1a1a2e"/>',
        f'  <text x="{W/2}" y="35" fill="#ffffff" font-family="sans-serif" font-size="24" font-weight="bold" text-anchor="middle">{title}</text>'
    ]
    
    alphas = [r['alpha'] for r in shrinkage_results]
    accs = [r['mean_accuracy'] for r in shrinkage_results]
    
    min_alpha, max_alpha = min(alphas), max(alphas)
    min_acc, max_acc = min(accs), max(accs)
    
    # Add padding to accuracy range
    acc_range = max_acc - min_acc
    if acc_range < 1e-5: acc_range = 1.0
    min_acc_plot = max(0, min_acc - acc_range * 0.1)
    max_acc_plot = min(1.0, max_acc + acc_range * 0.1)
    acc_range_plot = max_acc_plot - min_acc_plot
    
    def to_px(a, acc):
        px = margin["left"] + ((a - min_alpha) / (max_alpha - min_alpha)) * plot_w
        py = margin["top"] + plot_h - ((acc - min_acc_plot) / acc_range_plot) * plot_h
        return px, py

    # Draw grid and axes
    svg.append(f'  <g stroke="#ffffff" stroke-opacity="0.2" stroke-width="1">')
    # Y-axis ticks (Accuracy)
    num_ticks = 5
    for i in range(num_ticks + 1):
        tick_acc = min_acc_plot + i * (acc_range_plot / num_ticks)
        y = margin["top"] + plot_h - i * (plot_h / num_ticks)
        svg.append(f'    <line x1="{margin["left"]}" y1="{y}" x2="{W - margin["right"]}" y2="{y}"/>')
        svg.append(f'    <text x="{margin["left"] - 10}" y="{y + 5}" fill="#a0a0b0" font-family="sans-serif" font-size="14" text-anchor="end">{tick_acc:.3f}</text>')
    
    # X-axis ticks (Alpha)
    for a in alphas:
        px, _ = to_px(a, 0)
        y = margin["top"] + plot_h
        svg.append(f'    <line x1="{px}" y1="{margin["top"]}" x2="{px}" y2="{y}"/>')
        svg.append(f'    <text x="{px}" y="{y + 25}" fill="#a0a0b0" font-family="sans-serif" font-size="14" text-anchor="middle">{a:.1f}</text>')
    svg.append(f'  </g>')

    # Axis Labels
    svg.append(f'  <text x="{W/2}" y="{H - 15}" fill="#ffffff" font-family="sans-serif" font-size="16" text-anchor="middle">Shrinkage (α)</text>')
    svg.append(f'  <text x="25" y="{H/2}" fill="#ffffff" font-family="sans-serif" font-size="16" text-anchor="middle" transform="rotate(-90 25 {H/2})">Cross-Validation Accuracy</text>')
    
    # Draw line
    path_d = []
    points_svg = []
    for r in shrinkage_results:
        px, py = to_px(r['alpha'], r['mean_accuracy'])
        if not path_d:
            path_d.append(f"M {px} {py}")
        else:
            path_d.append(f"L {px} {py}")
            
        color = "#ff6b7a" if r['alpha'] == best_alpha else "#52d9cb"
        r_size = 6 if r['alpha'] == best_alpha else 4
        points_svg.append(f'  <circle cx="{px}" cy="{py}" r="{r_size}" fill="{color}"/>')

    svg.append(f'  <path d="{" ".join(path_d)}" fill="none" stroke="#52d9cb" stroke-width="3"/>')
    svg.extend(points_svg)
    
    svg.append('</svg>')
    
    
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"  [Plot] Saved: {out_path}")
    return out_path


def bar_chart_k_accuracy(k_results, best_k, 
                         title="Number of Components (K) vs Accuracy", 
                         out_path="outputs/figures/k_accuracy.svg"):
    """
    Create a bar chart showing cross-validation accuracy for each K.
    
    Parameters
    ----------
    k_results : list[dict]
        Results from experiment_n_components
    best_k : int
        The optimal K value to highlight
    title : str
        Plot title
    out_path : str
        File path to save the SVG
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    W, H = 600, 400
    margin = {"top": 60, "right": 40, "bottom": 60, "left": 80}
    plot_w = W - margin["left"] - margin["right"]
    plot_h = H - margin["top"] - margin["bottom"]
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'  <rect width="{W}" height="{H}" fill="#1a1a2e"/>',
        f'  <text x="{W/2}" y="35" fill="#ffffff" font-family="sans-serif" font-size="20" font-weight="bold" text-anchor="middle">{title}</text>'
    ]
    
    ks = [r['k'] for r in k_results]
    accs = [r['cv_accuracy'] for r in k_results]
    
    min_acc, max_acc = min(accs), max(accs)
    # Give some headroom and bottom room
    acc_range = max_acc - min_acc
    if acc_range < 1e-5: acc_range = 0.5
    min_acc_plot = max(0, min_acc - acc_range * 0.2)
    max_acc_plot = min(1.0, max_acc + acc_range * 0.2)
    acc_range_plot = max_acc_plot - min_acc_plot
    
    # Draw grid and axes
    svg.append(f'  <g stroke="#ffffff" stroke-opacity="0.2" stroke-width="1">')
    # Y-axis ticks
    num_ticks = 5
    for i in range(num_ticks + 1):
        tick_acc = min_acc_plot + i * (acc_range_plot / num_ticks)
        y = margin["top"] + plot_h - i * (plot_h / num_ticks)
        svg.append(f'    <line x1="{margin["left"]}" y1="{y}" x2="{W - margin["right"]}" y2="{y}"/>')
        svg.append(f'    <text x="{margin["left"] - 10}" y="{y + 5}" fill="#a0a0b0" font-family="sans-serif" font-size="14" text-anchor="end" stroke="none">{tick_acc:.3f}</text>')
    svg.append(f'  </g>')

    # Axis Labels
    svg.append(f'  <text x="{W/2}" y="{H - 15}" fill="#ffffff" font-family="sans-serif" font-size="16" text-anchor="middle">Number of Components (K)</text>')
    svg.append(f'  <text x="25" y="{H/2}" fill="#ffffff" font-family="sans-serif" font-size="16" text-anchor="middle" transform="rotate(-90 25 {H/2})">Cross-Validation Accuracy</text>')
    
    # Draw bars
    bar_width = min(80, plot_w / (len(ks) + 1))
    spacing = (plot_w - (bar_width * len(ks))) / (len(ks) + 1)
    
    for i, r in enumerate(k_results):
        k = r['k']
        acc = r['cv_accuracy']
        
        x = margin["left"] + spacing + i * (bar_width + spacing)
        bar_h = ((acc - min_acc_plot) / acc_range_plot) * plot_h
        y = margin["top"] + plot_h - bar_h
        
        color = "#e9c46a" if k == best_k else "#2a9d8f"
        
        svg.append(f'  <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_h}" fill="{color}" rx="4" ry="4"/>')
        svg.append(f'  <text x="{x + bar_width/2}" y="{y - 10}" fill="#ffffff" font-family="sans-serif" font-size="14" font-weight="bold" text-anchor="middle">{acc*100:.1f}%</text>')
        
        # X-axis label
        svg.append(f'  <text x="{x + bar_width/2}" y="{margin["top"] + plot_h + 20}" fill="#a0a0b0" font-family="sans-serif" font-size="14" text-anchor="middle">{k}</text>')

    svg.append('</svg>')
    
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"  [Plot] Saved: {out_path}")
    return out_path

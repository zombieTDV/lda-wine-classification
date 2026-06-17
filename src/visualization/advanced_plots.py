"""
advanced_plots.py
=================
Generates advanced visualizations for the Wine classification dataset
without using any external libraries (no matplotlib, seaborn, etc).

Visualizations:
1. Correlation Matrix Heatmap (to check for multicollinearity)
2. Feature Distribution (to check Gaussian assumptions of LDA/GNB)
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.data_loader import load_wine, standardize

def pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient between two lists."""
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))
    
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)

def plot_correlation_heatmap(X, feature_names, out_path="outputs/figures/correlation_heatmap.svg"):
    """Generate SVG heatmap of feature correlations."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    n_features = len(feature_names)
    
    # Calculate correlation matrix
    corr_matrix = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    for i in range(n_features):
        feature_i = [row[i] for row in X]
        for j in range(n_features):
            if i == j:
                corr_matrix[i][j] = 1.0
            elif i < j:
                feature_j = [row[j] for row in X]
                corr = pearson_correlation(feature_i, feature_j)
                corr_matrix[i][j] = corr
                corr_matrix[j][i] = corr
                
    # SVG Setup
    cell_size = 40
    margin_left = 180
    margin_top = 180
    margin_right = 20
    margin_bottom = 20
    
    W = margin_left + n_features * cell_size + margin_right
    H = margin_top + n_features * cell_size + margin_bottom
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'  <rect width="{W}" height="{H}" fill="#1a1a2e"/>',
        f'  <text x="{W/2}" y="40" fill="#ffffff" font-family="sans-serif" font-size="24" font-weight="bold" text-anchor="middle">Feature Correlation Heatmap</text>'
    ]
    
    # Draw heatmap
    for i in range(n_features):
        # Y-axis label
        svg.append(f'  <text x="{margin_left - 10}" y="{margin_top + i*cell_size + cell_size/2 + 5}" fill="#a0a0b0" font-family="sans-serif" font-size="14" text-anchor="end">{feature_names[i]}</text>')
        
        # X-axis label (rotated)
        x_label = margin_left + i*cell_size + cell_size/2
        y_label = margin_top - 10
        svg.append(f'  <text x="{x_label}" y="{y_label}" fill="#a0a0b0" font-family="sans-serif" font-size="14" text-anchor="start" transform="rotate(-45 {x_label} {y_label})">{feature_names[i]}</text>')
        
        for j in range(n_features):
            corr = corr_matrix[i][j]
            x = margin_left + j * cell_size
            y = margin_top + i * cell_size
            
            # Map correlation [-1, 1] to color
            # Red for positive (+1), Blue for negative (-1), Dark for 0
            if corr > 0:
                intensity = int(corr * 255)
                color = f"rgb({intensity}, {max(50, int(50*(1-corr)))}, {max(50, int(50*(1-corr)))})"
            else:
                intensity = int(abs(corr) * 255)
                color = f"rgb({max(50, int(50*(1-abs(corr))))}, {max(50, int(50*(1-abs(corr))))}, {intensity})"
                
            svg.append(f'  <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{color}" stroke="#1a1a2e" stroke-width="1"/>')
            
            # Text inside cell
            text_color = "#ffffff" if abs(corr) > 0.4 else "#888888"
            svg.append(f'  <text x="{x + cell_size/2}" y="{y + cell_size/2 + 4}" fill="{text_color}" font-family="sans-serif" font-size="11" text-anchor="middle">{corr:.2f}</text>')

    svg.append('</svg>')
    
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"  [Plot] Saved: {out_path}")


def plot_feature_distributions(X, y, feature_names, out_path="outputs/figures/feature_distributions.svg"):
    """
    Generate an SVG grid of histograms (simplified as density bars) 
    for the first 4 features to check Gaussian distributions.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    n_plots = min(4, len(feature_names))
    classes = sorted(list(set(y)))
    colors = ["#ff6b7a", "#52d9cb", "#f4d792"] # Class 1, 2, 3 colors
    
    plot_w, plot_h = 300, 200
    margin = {"top": 80, "right": 20, "bottom": 40, "left": 40}
    cols = 2
    rows = (n_plots + cols - 1) // cols
    
    W = cols * (plot_w + margin["left"] + margin["right"])
    H = rows * (plot_h + margin["top"] + margin["bottom"])
    
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        f'  <rect width="{W}" height="{H}" fill="#1a1a2e"/>',
        f'  <text x="{W/2}" y="40" fill="#ffffff" font-family="sans-serif" font-size="24" font-weight="bold" text-anchor="middle">Feature Distributions by Class</text>'
    ]
    
    # Legend
    for i, c in enumerate(classes):
        svg.append(f'  <rect x="{W/2 - 100 + i*80}" y="60" width="15" height="15" fill="{colors[i]}" rx="2"/>')
        svg.append(f'  <text x="{W/2 - 80 + i*80}" y="73" fill="#ffffff" font-family="sans-serif" font-size="14">Class {c}</text>')
    
    for i in range(n_plots):
        row = i // cols
        col = i % cols
        
        ox = col * (plot_w + margin["left"] + margin["right"]) + margin["left"]
        oy = row * (plot_h + margin["top"] + margin["bottom"]) + margin["top"] + 40
        
        feature_vals = [r[i] for r in X]
        min_v = min(feature_vals)
        max_v = max(feature_vals)
        v_range = max_v - min_v if max_v > min_v else 1.0
        
        svg.append(f'  <text x="{ox + plot_w/2}" y="{oy - 10}" fill="#ffffff" font-family="sans-serif" font-size="16" text-anchor="middle">{feature_names[i]}</text>')
        
        # Axes
        svg.append(f'  <line x1="{ox}" y1="{oy+plot_h}" x2="{ox+plot_w}" y2="{oy+plot_h}" stroke="#ffffff" stroke-opacity="0.5" stroke-width="2"/>')
        svg.append(f'  <line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+plot_h}" stroke="#ffffff" stroke-opacity="0.5" stroke-width="2"/>')
        
        # Ticks
        svg.append(f'  <text x="{ox}" y="{oy+plot_h+20}" fill="#a0a0b0" font-family="sans-serif" font-size="12" text-anchor="middle">{min_v:.1f}</text>')
        svg.append(f'  <text x="{ox+plot_w}" y="{oy+plot_h+20}" fill="#a0a0b0" font-family="sans-serif" font-size="12" text-anchor="middle">{max_v:.1f}</text>')
        
        # Simple Histogram (15 bins)
        n_bins = 15
        bin_width = v_range / n_bins
        
        for c_idx, c in enumerate(classes):
            c_vals = [X[j][i] for j in range(len(X)) if y[j] == c]
            
            bins = [0] * n_bins
            for v in c_vals:
                b = min(n_bins - 1, int((v - min_v) / bin_width))
                bins[b] += 1
                
            max_count = max(bins) if max(bins) > 0 else 1
            
            # Draw polygon/line for density
            path_d = []
            for b in range(n_bins):
                x = ox + (b + 0.5) * (plot_w / n_bins)
                y_pos = oy + plot_h - (bins[b] / max_count) * (plot_h * 0.8) # 80% max height
                path_d.append(f"{'M' if b == 0 else 'L'} {x} {y_pos}")
                
            svg.append(f'  <path d="{" ".join(path_d)}" fill="none" stroke="{colors[c_idx]}" stroke-width="3" stroke-opacity="0.8"/>')

    svg.append('</svg>')
    
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"  [Plot] Saved: {out_path}")

def generate_all_advanced_plots():
    print("\n" + "="*60)
    print("  GENERATING ADVANCED VISUALIZATIONS")
    print("="*60)
    
    X, y, feature_names = load_wine("data/processed/wine.csv")
    X_s, _, _ = standardize(X) # Standardize to remove scale effects on correlation
    
    plot_correlation_heatmap(X_s, feature_names)
    plot_feature_distributions(X, y, feature_names)
    
    print("="*60 + "\n")

if __name__ == "__main__":
    generate_all_advanced_plots()

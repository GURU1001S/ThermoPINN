"""
paper_figure_generator.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Publication-Quality Vector Graphic Generator.
Adheres to Nature Machine Intelligence & IEEE formatting specifications.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ─── Journal Formatting Specs ───
MM_TO_INCH = 1 / 25.4
SINGLE_COL = 88 * MM_TO_INCH
DOUBLE_COL = 180 * MM_TO_INCH

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
    'pdf.fonttype': 42,  # Ensures text is editable in Illustrator
    'ps.fonttype': 42
})

OUT_DIR = "paper_pdfs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    if not os.path.exists("ablation_results.json"):
        print("Error: ablation_results.json not found. Run ablation_suite.py first.")
        return None
    with open("ablation_results.json", "r") as f:
        return json.load(f)

def plot_pareto_adaptation(data):
    """Generates Figure 2: The Meta-Learning Pareto Curve."""
    if "experiment_K" not in data: return
    
    k_data = data["experiment_K"]["results"]
    k_shots = sorted([int(k) for k in k_data.keys()])
    rmses = [k_data[str(k)]["rmse"] for k in k_shots]
    
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.75))
    
    ax.plot(k_shots, rmses, marker='o', markersize=4, color='#185FA5', label='RMSE')
    
    # Highlight Pareto optimal
    pareto_k = data["experiment_K"].get("pareto_k", 2)
    pareto_idx = k_shots.index(int(pareto_k))
    ax.scatter([pareto_k], [rmses[pareto_idx]], color='#E24B4A', s=60, zorder=5, label=f'Pareto Optima (k={pareto_k})')
    
    ax.set_xlabel('MAML Adaptation Steps (k-shot)')
    ax.set_ylabel('RMSE (Cycles)')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "Fig2_MAML_Adaptation.pdf")
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"[Generated] {out_path}")
    plt.close()

def plot_feature_importance(data):
    """Generates Figure 4: Dimensionality Stress Test."""
    if "experiment_S" not in data: return
    
    s_data = data["experiment_S"]
    dims = []
    rmses = []
    for label, res in s_data.items():
        dim = int(label.split('D')[0])
        dims.append(dim)
        rmses.append(res['rmse'])
        
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.6))
    
    ax.plot(dims, rmses, marker='s', markersize=4, color='#534AB7')
    ax.axvline(22, color='#E24B4A', linestyle='--', linewidth=1, label='N-CMAPSS Eqv.')
    
    ax.set_xlabel('Active Input Sensor Dimensions')
    ax.set_ylabel('RMSE (Cycles)')
    ax.invert_xaxis()  # 55 down to 18
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "Fig4_Dimensionality_Stress.pdf")
    fig.savefig(out_path, format='pdf', bbox_inches='tight')
    print(f"[Generated] {out_path}")
    plt.close()

if __name__ == "__main__":
    d = load_data()
    if d:
        plot_pareto_adaptation(d)
        plot_feature_importance(d)
        print("\n[Success] Vector graphics generated for publication.")
"""
sensitivity_analysis.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Variance-Based Sobol Sensitivity Analysis.
Quantifies the exact contribution of each sensor and latent physics state 
to the RUL prediction variance, proving feature importance for the manuscript.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
from tqdm import tqdm

from pinn_model import PINNModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt")
TOTAL_FEAT = 55
WINDOW_SIZE = 30

# Generate generic names for the 55 features (0-13 sensors, 20-23 env, etc.)
FEATURE_NAMES = [f"F_{i}" for i in range(TOTAL_FEAT)]
FEATURE_NAMES[0:14] = [f"Sensor_{i}" for i in range(14)]
FEATURE_NAMES[20:24] = [f"Env_{i}" for i in range(4)]
FEATURE_NAMES[36:55] = [f"Physics_{i}" for i in range(19)]

def run_sobol_analysis():
    print(f"\n{'='*78}")
    print(f"{'ThermoPINN · Global Sobol Sensitivity Analysis':^78}")
    print(f"{'='*78}")

    # 1. Load Model
    model = PINNModel(max_rul=150.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()

    # 2. Define the Problem for SALib
    # Assuming normalized features roughly in the [-3, 3] z-score range
    problem = {
        'num_vars': TOTAL_FEAT,
        'names': FEATURE_NAMES,
        'bounds': [[-3.0, 3.0]] * TOTAL_FEAT
    }

    # 3. Generate Saltelli Samples
    # N=512 yields 512 * (2 * 55 + 2) = 57,344 model evaluations
    print("[1/3] Generating Saltelli Samples (N=512)...")
    param_values = saltelli.sample(problem, 512)
    n_samples = param_values.shape[0]
    print(f"      Created {n_samples:,} evaluation points.")

    # 4. Evaluate the Model
    print("\n[2/3] Executing Neural Network Forward Passes...")
    Y = np.zeros(n_samples)
    batch_size = 1024

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Inferring"):
            end = min(i + batch_size, n_samples)
            batch_np = param_values[i:end]
            
            # Broadcast the 1D sample across the 30-step temporal window
            # Shape: [Batch, 30, 55]
            x_t = torch.tensor(batch_np, dtype=torch.float32).unsqueeze(1).expand(-1, WINDOW_SIZE, -1).to(DEVICE)
            
            # Dummy operating settings for baseline analysis
            op = torch.zeros(x_t.size(0), dtype=torch.long, device=DEVICE)
            ev = torch.zeros(x_t.size(0), dtype=torch.long, device=DEVICE)
            
            out = model(x_t, op_setting=op, event_flag=ev)
            # Convert log-RUL back to cycle space
            pred_cy = torch.expm1(out["rul_log"].squeeze(-1)).cpu().numpy()
            Y[i:end] = pred_cy

    # 5. Compute Sobol Indices
    print("\n[3/3] Calculating Variance-Based Sobol Indices...")
    Si = sobol.analyze(problem, Y, print_to_console=False)

    # 6. Extract and Plot Top 15 Features
    S1 = Si['S1']    # First-order sensitivity
    ST = Si['ST']    # Total-order sensitivity (includes interactions)
    
    top_indices = np.argsort(ST)[::-1][:15]
    
    print(f"\n{'Top 15 Drivers of RUL Variance':^40}")
    print(f"{'-'*40}")
    print(f"{'Feature':<15} | {'S1 (Direct)':<10} | {'ST (Total)':<10}")
    print(f"{'-'*40}")
    for idx in top_indices:
        print(f"{FEATURE_NAMES[idx]:<15} | {S1[idx]:>10.4f} | {ST[idx]:>10.4f}")

    # Generate Figure
    plt.figure(figsize=(8, 5))
    bar_width = 0.35
    x = np.arange(len(top_indices))
    
    plt.bar(x - bar_width/2, S1[top_indices], bar_width, label='S1 (Direct Effect)', color='#185FA5')
    plt.bar(x + bar_width/2, ST[top_indices], bar_width, label='ST (Total Effect)', color='#E24B4A')
    
    plt.xticks(x, [FEATURE_NAMES[i] for i in top_indices], rotation=45, ha='right')
    plt.ylabel('Sobol Sensitivity Index')
    plt.title('Feature Sensitivity Analysis (Variance Contribution)')
    plt.legend(frameon=False)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    out_path = "paper_pdfs/Fig5_Sobol_Sensitivity.pdf"
    os.makedirs("paper_pdfs", exist_ok=True)
    plt.savefig(out_path, format='pdf')
    print(f"\n[Saved] High-res vector plot exported to {out_path}")

if __name__ == "__main__":
    run_sobol_analysis()
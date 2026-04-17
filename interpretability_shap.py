"""
interpretability_shap.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Explainable AI (XAI) Diagnostics for MRO Operators.
Utilizes Kernel SHAP to compute the marginal contribution of each physical 
sensor to the final RUL prediction, outputting localized diagnostic charts.
"""

import os
import torch
import numpy as np
import h5py
import shap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.lib.stride_tricks import sliding_window_view

from pinn_model import PINNModel

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser("~/nasa_research/data/")
MODEL_PATH = os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt")
TEST_FILE  = "N-CMAPSS_DS01-005.h5" 

WINDOW_SIZE = 30
TOTAL_FEAT  = 55
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Human-readable sensor mapping for N-CMAPSS (Cols 0-13)
SENSOR_NAMES = [
    "Altitude (alt)", "Mach Number (Mach)", "Throttle (TRA)", "T2: Fan Inlet Temp", 
    "T24: LPC Outlet Temp", "T30: HPC Outlet Temp", "T48: HPT Outlet Temp", 
    "T50: LPT Outlet Temp", "P15: Bypass Duct Press", "P2: Fan Inlet Press", 
    "P24: LPC Outlet Press", "Ps30: HPC Static Press", "P40: Burner Press", 
    "P50: LPT Outlet Press"
]
# Pad the rest of the 55 dimensions with generic names
FEATURE_NAMES = SENSOR_NAMES + [f"Latent_Dim_{i}" for i in range(14, TOTAL_FEAT)]

def compute_sensor_shap():
    print(f"\n{'='*78}")
    print(f"{'ThermoPINN · MRO Diagnostic SHAP Explainer':^78}")
    print(f"{'='*78}")

    # 1. Load Model
    model = PINNModel(max_rul=150.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()

    # 2. Extract Background and Test Data
    print("[1/3] Extracting background distribution and test instances...")
    ds_path = os.path.join(DATA_DIR, TEST_FILE)
    
    with h5py.File(ds_path, "r") as f:
        X_s = f["X_s_dev"][:2000].astype(np.float32)
        X_mean = X_s.mean(axis=0, keepdims=True)
        X_std  = X_s.std(axis=0, keepdims=True) + 1e-6
        X_s = (X_s - X_mean) / X_std

        X_view = sliding_window_view(X_s, WINDOW_SIZE, axis=0).swapaxes(1, 2)
        
        # Build full 55D tensor (we will only explain the 14 physical sensors for clarity)
        batch_cpu = np.zeros((len(X_view), WINDOW_SIZE, TOTAL_FEAT), dtype=np.float32)
        batch_cpu[:, :, 0:14] = X_view[:, :, 0:14]

        # Background: 50 random healthy flights
        X_background = batch_cpu[np.random.choice(1000, 50, replace=False)]
        
        # Test: 10 critical End-of-Life flights
        X_test = batch_cpu[-30:-20] 

    # 3. Model Wrapper for SHAP
    # SHAP needs a function that takes a 2D array [N, features] and returns 1D predictions.
    # We flatten the temporal dimension by taking the mean of the window to keep SHAP fast and interpretable.
    def model_wrapper(x_flat_np):
        # x_flat_np is [N, TOTAL_FEAT]. Expand to [N, 30, TOTAL_FEAT]
        x_t = torch.tensor(x_flat_np, dtype=torch.float32).unsqueeze(1).expand(-1, WINDOW_SIZE, -1).to(DEVICE)
        op = torch.zeros(x_t.size(0), dtype=torch.long, device=DEVICE)
        ev = torch.zeros(x_t.size(0), dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            out = model(x_t, op_setting=op, event_flag=ev)
            
        return torch.expm1(out["rul_log"].squeeze(-1)).cpu().numpy()

    # Flatten the temporal dimension for the explainer
    bg_flat = X_background.mean(axis=1)
    test_flat = X_test.mean(axis=1)

    print(f"[2/3] Computing Kernel SHAP values for {len(test_flat)} critical flights...")
    # Initialize the KernelExplainer
    explainer = shap.KernelExplainer(model_wrapper, bg_flat)
    
    # Compute SHAP values (nsamples controls the accuracy/speed tradeoff)
    shap_vals = explainer.shap_values(test_flat, nsamples=200, l1_reg="num_features(10)")

    print("[3/3] Generating diagnostic PDF reports...")
    out_path = "paper_pdfs/Fig7_SHAP_Diagnostics.pdf"
    os.makedirs("paper_pdfs", exist_ok=True)
    
    with PdfPages(out_path) as pdf:
        # Plot 1: Global Summary Plot (Which sensors generally drive failures?)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_vals[:, :14], test_flat[:, :14], feature_names=SENSOR_NAMES, show=False)
        plt.title("Global MRO Sensor Importance (End of Life)")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Plot 2: Local Force Plot (Deep dive into the very last flight cycle)
        # Force plots require initializing JS, so we use a waterfall or decision plot for PDF
        plt.figure(figsize=(8, 5))
        shap.decision_plot(explainer.expected_value, shap_vals[0, :14], test_flat[0, :14], feature_names=SENSOR_NAMES, show=False)
        plt.title("Local Diagnostic: Single Engine Failure Profile")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print(f"✅ PASSED: MRO Interpretability artifacts exported to {out_path}")
    print(f"{'='*78}\n")

if __name__ == "__main__":
    compute_sensor_shap()
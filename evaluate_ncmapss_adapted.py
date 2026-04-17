import os
import time
import warnings
import h5py
import numpy as np
import torch
import pandas as pd
from torch.amp import autocast
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Attempting to import the model
try:
    from pinn_model import PINNModel
except ImportError:
    exit("❌ Error: 'pinn_model.py' not found in current directory.")

# ─── Masterclass Config ──────────────────────────────────────────────────────
DATA_DIR    = os.path.expanduser("~/nasa_research/data/")
MODEL_PATH  = os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt")

TOTAL_FEAT  = 55
WINDOW_SIZE = 30
MC_PASSES   = 8
# Aggregation factor: Collapses 500 rows of 1Hz data into 1 "Health Epoch"
AGGR_FACTOR = 500 

def align_and_map_physics(X_s, W):
    """
    Complex Alignment: Bridges the Sim-to-Real gap by anchoring sensors 
    to their initial healthy state.
    """
    # 1. Temporal Smoothing (handles the high-frequency 850k row noise)
    df_x = pd.DataFrame(X_s)
    df_w = pd.DataFrame(W)
    
    # Use moving average to extract the clean degradation signal
    X_smooth = df_x.rolling(window=1000, min_periods=1).mean().values
    W_smooth = df_w.rolling(window=1000, min_periods=1).mean().values

    # 2. Healthy-State Anchoring (Standardization relative to Start-of-Life)
    # We assume the first 2000 cycles represent the "Gold Standard" health.
    X_ref_mean = np.mean(X_smooth[:2000], axis=0)
    X_ref_std  = np.std(X_smooth[:2000], axis=0) + 1e-6
    
    # Perform Delta-Z scaling: (Current - Healthy_Mean) / Healthy_Std
    X_aligned = (X_smooth - X_ref_mean) / X_ref_std
    W_aligned = (W_smooth - np.mean(W_smooth, axis=0)) / (np.std(W_smooth, axis=0) + 1e-6)

    # 3. Virtual Cycle Aggregation
    # Reduces the 850k data points into meaningful physics snapshots
    X_final = X_aligned[::AGGR_FACTOR]
    W_final = W_aligned[::AGGR_FACTOR]

    # 4. Feature Manifold Construction (55-Feature Vector)
    full_data = np.zeros((X_final.shape[0], TOTAL_FEAT), dtype=np.float32)
    # Placing the 14 sensors and 4 flight conditions into the expected slots
    full_input = np.zeros((X_final.shape[0], TOTAL_FEAT), dtype=np.float32)
    full_input[:, 0:14]  = np.clip(X_final, -4, 4) # Sensors
    full_input[:, 20:24] = np.clip(W_final[:, :4], -4, 4) # Ops
    
    return full_input

def predict_with_uncertainty(model, data, device):
    n_rows = data.shape[0]
    n_win  = n_rows - WINDOW_SIZE + 1
    if n_win <= 0: return None
    
    # Generate windows
    views = sliding_window_view(data, WINDOW_SIZE, axis=0).transpose(0, 2, 1)
    results = []

    model.train() # Enable Dropout
    with torch.no_grad():
        for i in tqdm(range(n_win), desc="[Physics Inference]"):
            # Individual window for maximum precision
            win_tensor = torch.from_numpy(views[i]).float().unsqueeze(0).to(device)
            mc_batch   = win_tensor.repeat(MC_PASSES, 1, 1)
            
            # Placeholders for auxiliary inputs
            dummy = torch.zeros(MC_PASSES, dtype=torch.long, device=device)

            with autocast("cuda"):
                out = model(mc_batch, op_setting=dummy, event_flag=dummy)
                
                # Check for output key (rul_log is standard for v20 models)
                res_key = "rul_log" if "rul_log" in out else "rul"
                raw_pred = out[res_key].mean().item()
                
                # Inverse Transform: exp(x) - 1
                # If the model predicts a raw log-value of 4.14, result is ~62
                results.append(np.expm1(raw_pred))

    return np.array(results)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ThermoPINN] Loading Model on {device}...")
    
    # ── FIXED LOADER ──
    model = None
    # Iterating through common argument names to find the right one
    for arg_name in ['input_dim', 'in_channels', 'in_features', 'd_in', '']:
        try:
            if arg_name == '':
                model = PINNModel().to(device)
            else:
                model = PINNModel(**{arg_name: TOTAL_FEAT}).to(device)
            print(f"✅ PINNModel initialized using: '{arg_name}'")
            break
        except TypeError:
            continue

    if model is None:
        exit("❌ Critical Error: Could not instantiate PINNModel. Check pinn_model.py.")

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"✅ Weights loaded: {os.path.basename(MODEL_PATH)}")
    except Exception as e:
        exit(f"❌ Weight Error: {e}")

    # ── DATA PROCESSING ──
    target_path = os.path.join(DATA_DIR, "N-CMAPSS_DS02-006.h5")
    with h5py.File(target_path, 'r') as hdf:
        A = np.array(hdf.get('A_dev'))
        mask = (A[:, 0] == 2)
        X_s = np.array(hdf.get('X_s_dev'))[mask]
        W   = np.array(hdf.get('W_dev'))[mask]
        Y   = np.array(hdf.get('Y_dev'))[mask]

    print(f"[Prep] Aligning Domain for {len(Y)} time-steps...")
    processed_input = align_and_map_physics(X_s, W)
    
    preds_agg = predict_with_uncertainty(model, processed_input, device)
    
    if preds_agg is not None:
        # Interpolate the aggregated predictions back to the original 850k cycle count
        interp_preds = np.interp(np.arange(len(Y)), 
                                 np.linspace(0, len(Y), len(preds_agg)), 
                                 preds_agg)
        
        os.makedirs('data', exist_ok=True)
        pd.DataFrame({
            'cycle': np.arange(len(Y)),
            'true_rul': Y.flatten(),
            'predicted_rul': interp_preds
        }).to_csv('data/ncmapss_predictions.csv', index=False)
        
        print("\n✅ SUCCESS: Result exported to data/ncmapss_predictions.csv")
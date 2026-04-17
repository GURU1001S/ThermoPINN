"""
sensor_failure_sim.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Experiment 4: Systematic Sensor Failure & DQI Uncertainty Tracking.
Injects adversarial sensor degradation (drift, dropout, stuck) into UTDTB v5.
Uses Monte Carlo (MC) Dropout as a Data Quality Index (DQI) to track 
uncertainty inflation before prediction collapse.
"""

import os
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

# ─── Configuration ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30
TOTAL_FEAT = 55
MC_PASSES = 5  # Number of forward passes for DQI uncertainty estimation
DQI_MULTIPLIER = 1.5  # Alarm triggers if uncertainty is 1.5x the healthy baseline

# ─── Sensor Failure Logic ─────────────────────────────────────────────────────

def apply_sensor_failures(X_clean, n_failed, mode):
    """
    Injects physical sensor failures into the sliding window tensor.
    X_clean shape: (N_windows, Window_Size, 55_Features)
    """
    if n_failed == 0: return X_clean.copy()
    
    X_deg = X_clean.copy()
    n_windows, win_size, n_features = X_deg.shape
    
    # Randomly pick which sensors to destroy (we restrict to the first 21 sensor columns)
    failed_idx = np.random.choice(min(21, n_features), n_failed, replace=False)
    
    for idx in failed_idx:
        if mode == "gradual_drift":
            # Sensor calibration drifts linearly over the 30-step window
            drift = np.linspace(0, 3.0, win_size)
            X_deg[:, :, idx] += drift
            
        elif mode == "random_dropout":
            # Loose wire: 10% chance a reading drops to 0 at any timestep
            dropout_mask = np.random.rand(n_windows, win_size) > 0.90
            X_deg[:, :, idx][dropout_mask] = 0.0
            
        elif mode == "stuck_at_last":
            # Frozen sensor: repeats its very first reading for the whole window
            # Broadcasting the first timestep across the window dimension
            first_vals = X_deg[:, 0:1, idx] 
            X_deg[:, :, idx] = np.repeat(first_vals, win_size, axis=1)
            
        elif mode == "spike_then_drop":
            # Massive electrical spike halfway through the window, then dead
            mid = win_size // 2
            X_deg[:, mid, idx] += 10.0
            X_deg[:, mid+1:, idx] = 0.0
            
    return X_deg

# ─── Data Extractor ───────────────────────────────────────────────────────────

def extract_test_windows(h5_path, n_engines=20):
    """Extracts ground-truth test trajectories for failure injection."""
    print("Extracting clean test engines from UTDTB...")
    with h5py.File(h5_path, "r") as f:
        grp = f["test"]
        eng_ids = grp["engine_id"][:]
        ruls = grp["RUL"][:]
        
        unique_engines = np.unique(eng_ids)[:n_engines]
        X_all, Y_all = [], []
        
        for eng in unique_engines:
            idx = np.where(eng_ids == eng)[0]
            if len(idx) < WINDOW_SIZE: continue
            
            X_raw = np.concatenate([
                np.nan_to_num(grp["sensors"][idx], nan=0.0),
                np.nan_to_num(grp["env"][idx], nan=0.0),
                np.nan_to_num(grp["causal_state"][idx], nan=0.0)
            ], axis=1).astype(np.float32)
            
            X_norm = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)
            X_norm = X_norm[np.argsort(ruls[idx])[::-1]] 
            
            X_win = sliding_window_view(X_norm, WINDOW_SIZE, axis=0).swapaxes(1, 2)
            X_all.append(X_win)
            Y_all.append(np.arange(len(X_win) - 1, -1, -1).astype(np.float32))
            
    return np.concatenate(X_all), np.concatenate(Y_all)

# ─── Inference & DQI Monitor ──────────────────────────────────────────────────

def evaluate_with_dqi(model, X_np, baseline_uncertainty=None):
    """
    Runs MC Dropout inference to get RUL predictions and DQI uncertainty.
    """
    model.train() # Force dropout to remain active for Monte Carlo sampling
    X_tens = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    op = torch.zeros(len(X_tens), dtype=torch.long, device=DEVICE)
    ev = torch.zeros(len(X_tens), dtype=torch.long, device=DEVICE)
    
    mc_preds = []
    with torch.no_grad():
        for _ in range(MC_PASSES):
            out = model(X_tens, op_setting=op, event_flag=ev)
            mc_preds.append(torch.expm1(out["rul_log"].squeeze(-1)).cpu().numpy())
            
    mc_preds = np.array(mc_preds)
    mean_preds = np.mean(mc_preds, axis=0)
    uncertainty = np.std(mc_preds, axis=0) # DQI Metric
    
    if baseline_uncertainty is None:
        alarm_rate = 0.0
    else:
        # Trigger alarm if uncertainty exceeds normal baseline by multiplier
        alarms = uncertainty > (baseline_uncertainty * DQI_MULTIPLIER)
        alarm_rate = np.mean(alarms) * 100
        
    return mean_preds, np.mean(uncertainty), alarm_rate

# ─── Main Execution ───────────────────────────────────────────────────────────

def run_experiment(args):
    print(f"\n{'='*80}\n{'Experiment 4: Sensor Failure & DQI Uncertainty Simulator':^80}\n{'='*80}")
    
    # 1. Load Model & Data
    try:
        from pinn_model import PINNModel
        model = PINNModel(max_rul=500.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0).to(DEVICE)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True).get("model_state", torch.load(args.model_path, map_location=DEVICE, weights_only=True)), strict=False)
    except Exception as e:
        print(f"❌ Error loading model: {e}"); return

    X_test, Y_test = extract_test_windows(args.data_path, n_engines=20)
    if X_test is None: return

    # 2. Establish Healthy Baseline
    print("\nEstablishing Healthy Baseline (0 Failed Sensors)...")
    baseline_preds, baseline_unc_mean, _ = evaluate_with_dqi(model, X_test)
    baseline_rmse = np.sqrt(np.mean((baseline_preds - Y_test)**2))
    
    print(f"  Baseline RMSE       : {baseline_rmse:.2f} cycles")
    print(f"  Baseline Uncertainty: ±{baseline_unc_mean:.2f} cycles")
    
    # We will use the raw array of healthy uncertainties as our DQI threshold map
    # Recalculate to get the array instead of the mean
    model.train()
    with torch.no_grad():
        X_tens = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        op = torch.zeros(len(X_tens), dtype=torch.long, device=DEVICE)
        ev = torch.zeros(len(X_tens), dtype=torch.long, device=DEVICE)
        mc_base = [torch.expm1(model(X_tens, op_setting=op, event_flag=ev)["rul_log"].squeeze(-1)).cpu().numpy() for _ in range(MC_PASSES)]
        baseline_unc_array = np.std(mc_base, axis=0) + 1e-6 # Add epsilon to prevent div/0

    # 3. Simulate Failures
    failure_counts = [1, 2, 3, 5, 7, 10, 14]
    modes = ["gradual_drift", "random_dropout", "stuck_at_last", "spike_then_drop"]
    
    results = []
    
    for mode in modes:
        print(f"\n[Injecting Mode: {mode.upper()}]")
        for n_failed in failure_counts:
            # 1. Degrade Data
            X_deg = apply_sensor_failures(X_test, n_failed, mode)
            
            # 2. Evaluate with DQI
            preds, unc_mean, dqi_rate = evaluate_with_dqi(model, X_deg, baseline_unc_array)
            rmse = np.sqrt(np.mean((preds - Y_test)**2))
            
            unc_inflation = unc_mean / baseline_unc_mean
            
            print(f"  {n_failed:>2} Sensors Failed | RMSE: {rmse:>6.2f} | DQI Alarms: {dqi_rate:>5.1f}% | Unc Inflation: {unc_inflation:>4.2f}x")
            results.append({"mode": mode, "n_failed": n_failed, "rmse": rmse, "alarms": dqi_rate, "inflation": unc_inflation})

    print(f"\n{'='*80}")
    print(f"Experiment Complete. Check the printed log to analyze architecture robustness.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/nasa_research/data/utdtb_v5.h5"))
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt"))
    args = parser.parse_args()
    
    run_experiment(args)
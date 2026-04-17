"""
regime_eval.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Experiment 6: Flight Regime Sensitivity Study.
Evaluates model accuracy across different operational conditions 
(Altitude and Mach bands) to identify physical regime weaknesses.
"""

import os
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30

ALTITUDE_BANDS = [(0, 10000, "Takeoff/Low"), (10000, 25000, "Climb/Mid"), (25000, 45000, "Cruise/High")]
MACH_BANDS = [(0.0, 0.4, "Low Mach"), (0.4, 0.7, "Transonic"), (0.7, 1.0, "High Mach")]

def run_regime_eval(args):
    print(f"\n{'='*80}\n{'Experiment 6: Flight Regime Sensitivity Analysis':^80}\n{'='*80}")
    
    try:
        from pinn_model import PINNModel
        model = PINNModel(max_rul=500.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0).to(DEVICE)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True).get("model_state", torch.load(args.model_path, map_location=DEVICE, weights_only=True)), strict=False)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}"); return

    all_preds, all_trues, all_alts, all_machs = [], [], [], []
    
    with h5py.File(args.data_path, "r") as f:
        grp = f["test"]
        eng_ids = grp["engine_id"][:]
        ruls = grp["RUL"][:]
        
        # In UTDTB, 'env' usually contains Altitude (ft) as col 0, Mach as col 1
        raw_env = grp["env"][:] 
        
        for eng in tqdm(np.unique(eng_ids)[:20], desc="Processing Regimes", ncols=80, colour="yellow"):
            idx = np.where(eng_ids == eng)[0]
            if len(idx) < WINDOW_SIZE: continue
            
            X_raw = np.concatenate([
                np.nan_to_num(grp["sensors"][idx], nan=0.0),
                np.nan_to_num(raw_env[idx], nan=0.0),
                np.nan_to_num(grp["causal_state"][idx], nan=0.0)
            ], axis=1).astype(np.float32)
            
            # Extract raw unnormalized Env features for regime masking
            env_raw_eng = raw_env[idx]
            altitudes = env_raw_eng[:, 0]
            machs = env_raw_eng[:, 1]
            
            X_norm = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)
            order = np.argsort(ruls[idx])[::-1]
            
            X_win = sliding_window_view(X_norm[order], WINDOW_SIZE, axis=0).swapaxes(1, 2)
            Y_tgt = np.arange(len(X_win) - 1, -1, -1).astype(np.float32)
            
            # Record the env variables at the END of the window (most recent state)
            alt_win = altitudes[order][WINDOW_SIZE-1:]
            mach_win = machs[order][WINDOW_SIZE-1:]
            
            with torch.no_grad():
                out = model(torch.tensor(X_win).to(DEVICE), op_setting=torch.zeros(len(X_win), dtype=torch.long, device=DEVICE), event_flag=torch.zeros(len(X_win), dtype=torch.long, device=DEVICE))
                preds = torch.expm1(out["rul_log"].squeeze(-1)).cpu().numpy()
            
            all_preds.extend(preds)
            all_trues.extend(Y_tgt)
            all_alts.extend(alt_win)
            all_machs.extend(mach_win)

    all_preds, all_trues, all_alts, all_machs = map(np.array, [all_preds, all_trues, all_alts, all_machs])

    print("\n  ── Flight Regime Metrics ───────────────────────────────────")
    for alt_lo, alt_hi, alt_name in ALTITUDE_BANDS:
        for ma_lo, ma_hi, ma_name in MACH_BANDS:
            mask = (all_alts >= alt_lo) & (all_alts < alt_hi) & (all_machs >= ma_lo) & (all_machs < ma_hi)
            n_samples = np.sum(mask)
            
            if n_samples > 10:
                rmse = np.sqrt(np.mean((all_preds[mask] - all_trues[mask])**2))
                bias = np.mean(all_preds[mask] - all_trues[mask])
                print(f"  [{alt_name:12} | {ma_name:10}] RMSE: {rmse:5.1f} | Bias: {bias:+5.1f} | N={n_samples}")

    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/nasa_research/data/utdtb_v5.h5"))
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt"))
    run_regime_eval(parser.parse_args())
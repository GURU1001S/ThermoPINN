"""
evaluate_ncmapss_adapted.py  ·  ThermoPINN  ·  v2.6 (Zero-Copy Vectorized Streaming)
══════════════════════════════════════════════════════════════════════════════════
Zero-shot sim-to-real transfer: UTDTB v5 → NASA N-CMAPSS DS01–DS05.
Fixes: Replaced slow Python JIT loops with instantaneous C-level sliding_window_view.
"""

import os
import math
import time
import h5py
import numpy as np
import torch
import gc
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

from pinn_model import PINNModel

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser("~/nasa_research/data/")
MODEL_PATH = os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt")

SENSOR_START = 0
SENSOR_END   = 14
ENV_START    = 20
ENV_END      = 24
TOTAL_FEAT   = 55
WINDOW_SIZE  = 30  
BATCH_SIZE   = 512 
MC_PASSES    = 10  

DATASETS = ["N-CMAPSS_DS01-005.h5", "N-CMAPSS_DS02-006.h5", "N-CMAPSS_DS03-012.h5", "N-CMAPSS_DS04.h5", "N-CMAPSS_DS05.h5"]
CALIBRATION_DATASET = "N-CMAPSS_DS01-005.h5"

# ─── Vectorized Inference Engine ──────────────────────────────────────────────
def predict_streaming(model, X_s_u, W_u, Y_u, device):
    N = len(Y_u)
    n_windows = N - WINDOW_SIZE + 1
    if n_windows <= 0: return [], [], []
    
    all_preds, all_stds = [], []
    
    # 🚀 THE FIX: Zero-Copy C-level memory views. Instantaneous. 0 extra RAM.
    X_view = sliding_window_view(X_s_u, WINDOW_SIZE, axis=0).swapaxes(1, 2)
    W_view = sliding_window_view(W_u, WINDOW_SIZE, axis=0).swapaxes(1, 2)
    Y_target = Y_u[WINDOW_SIZE - 1:]
    
    op_gpu = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
    ev_gpu = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
    
    if MC_PASSES > 1:
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm1d): m.eval()
    else:
        model.eval()

    with torch.no_grad():
        for start_idx in range(0, n_windows, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, n_windows)
            current_bs = end_idx - start_idx
            
            # Instantiates only the 512-batch directly from the C-view
            X_batch = X_view[start_idx:end_idx]
            W_batch = W_view[start_idx:end_idx]
            
            # Assemble the 55D tensor instantly
            batch_cpu = np.zeros((current_bs, WINDOW_SIZE, TOTAL_FEAT), dtype=np.float32)
            batch_cpu[:, :, SENSOR_START:SENSOR_END] = X_batch
            batch_cpu[:, :, ENV_START:ENV_END]       = W_batch

            gpu_batch = torch.tensor(batch_cpu, device=device)
            gpu_op = op_gpu[:current_bs]
            gpu_ev = ev_gpu[:current_bs]

            if MC_PASSES > 1:
                pass_preds = []
                for _ in range(MC_PASSES):
                    out = model(gpu_batch, op_setting=gpu_op, event_flag=gpu_ev)
                    pass_preds.append(out["rul_log"].detach())
                stack = torch.stack(pass_preds, dim=0)
                mean_log = stack.mean(0).squeeze(-1)
                std_log  = stack.std(0).squeeze(-1)
                aleat    = torch.exp(0.5 * out["rul_log_var"].detach().squeeze(-1))
                total_std_log = torch.sqrt(std_log**2 + aleat**2)
            else:
                out = model(gpu_batch, op_setting=gpu_op, event_flag=gpu_ev)
                mean_log = out["rul_log"].detach().squeeze(-1)
                total_std_log = torch.exp(0.5 * out["rul_log_var"].detach().squeeze(-1))

            preds_cy = torch.expm1(mean_log).cpu().numpy()
            stds_cy  = (total_std_log * torch.expm1(mean_log)).cpu().numpy()
            stds_cy  = np.clip(stds_cy, 1.0, None)

            all_preds.extend(preds_cy)
            all_stds.extend(stds_cy)

    model.eval()
    return np.array(all_preds), np.array(all_stds), Y_target

# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_nasa(preds, trues, max_rul):
    clamp = max_rul * 0.5
    errors = np.clip(preds - trues, -clamp, clamp)
    raw = np.where(errors < 0, np.exp(-errors / 13.0) - 1.0, np.exp(errors / 10.0) - 1.0)
    return float(np.mean(raw))

def conformal_calibrate(preds, stds, trues, target_coverage=0.90):
    scores = np.abs(preds - trues) / (stds + 1e-6)
    n = len(scores)
    q_level = min(1.0, math.ceil(target_coverage * (n + 1)) / n)
    return float(np.quantile(scores, q_level))

def compute_coverage(preds, stds, trues, q_hat):
    lower, upper = preds - stds * q_hat, preds + stds * q_hat
    return float(np.mean((trues >= lower) & (trues <= upper)) * 100)

# ─── Main Execution ───────────────────────────────────────────────────────────
def main():
    global_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    print(f"\n{'='*72}")
    print(f"{'ThermoPINN · Zero-Copy Vectorized Streaming Transfer':^72}")
    print(f"{'='*72}")

    try:
        model = PINNModel(max_rul=150.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
    except Exception as e:
        print(f"❌ ERROR: {e}"); return

    # 1. Alignment & Calibration
    print(f"\n[Alignment] Computing global sensor alignment from DS01...")
    cal_path = os.path.join(DATA_DIR, CALIBRATION_DATASET)
    
    with h5py.File(cal_path, "r") as f:
        X_sample = f["X_s_dev"][:200000]
        X_mean = X_sample.mean(axis=0, keepdims=True)
        X_std  = X_sample.std(axis=0, keepdims=True) + 1e-6
        del X_sample; gc.collect()

        unit_id = f["A_dev"][:, 0].astype(int)
        unique_engines = np.unique(unit_id)
        n_cal_engines  = max(20, int(len(unique_engines) * 0.30))
        cal_engines    = unique_engines[-n_cal_engines:]
        
        cal_p, cal_s, cal_y = [], [], []
        
        for uid in tqdm(cal_engines, desc="Fitting Calibrator", unit="engine"):
            idx = np.where(unit_id == uid)[0]
            start, end = idx[0], idx[-1] + 1
            X_s_u = (f["X_s_dev"][start:end] - X_mean) / X_std
            W_u   = f["W_dev"][start:end]
            Y_u   = f["Y_dev"][start:end].flatten()
            
            p, s, y = predict_streaming(model, X_s_u, W_u, Y_u, device)
            cal_p.extend(p); cal_s.extend(s); cal_y.extend(y)
            del X_s_u, W_u, Y_u; gc.collect()

        q_hat = conformal_calibrate(np.array(cal_p), np.array(cal_s), np.array(cal_y))
        print(f"[Calibration] CS-E 1550 Multiplier (q_hat) computed: {q_hat:.4f}")

    # 2. Fleet Evaluation
    print(f"\n[Eval] Running Vectorized Inference on DS01–DS05...")
    results = []

    for ds_name in DATASETS:
        ds_path = os.path.join(DATA_DIR, ds_name)
        if not os.path.exists(ds_path): continue
        
        label = ds_name.replace("N-CMAPSS_", "").replace(".h5", "")
        all_p, all_s, all_y = [], [], []
        n_win_total = 0

        with h5py.File(ds_path, "r") as f:
            unit_id = f["A_dev"][:, 0].astype(int)
            unique_engines = np.unique(unit_id)
            max_rul = float(f["Y_dev"][:].max())

            for uid in tqdm(unique_engines, desc=f"Eval {label}", unit="engine"):
                idx = np.where(unit_id == uid)[0]
                start, end = idx[0], idx[-1] + 1
                
                X_s_u = (f["X_s_dev"][start:end] - X_mean) / X_std
                W_u   = f["W_dev"][start:end]
                Y_u   = f["Y_dev"][start:end].flatten()
                
                p, s, y = predict_streaming(model, X_s_u, W_u, Y_u, device)
                if len(p) > 0:
                    all_p.extend(p); all_s.extend(s); all_y.extend(y)
                    n_win_total += len(y)
                del X_s_u, W_u, Y_u; gc.collect()

        if len(all_p) > 0:
            preds, stds, trues = np.array(all_p), np.array(all_s), np.array(all_y)
            rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
            mae = float(np.mean(np.abs(preds - trues)))
            nasa = compute_nasa(preds, trues, max_rul)
            coverage = compute_coverage(preds, stds, trues, q_hat)
            results.append({"label": label, "rmse": rmse, "mae": mae, "nasa": nasa, "cov": coverage, "wins": n_win_total})

    # 3. Final Output
    total_time = (time.time() - global_start_time) / 60.0
    
    print(f"\n{'='*80}")
    print(f"{'Zero-Shot Transfer · CS-E 1550 Coverage Compliance':^80}")
    print(f"{'='*80}")
    print(f"  {'Dataset':<12} | {'RMSE':>7} | {'MAE':>7} | {'NASA':>8} | {'Coverage':>10} | {'#Windows':>9}")
    print(f"  {'-'*72}")

    ar = am = an = ac = 0.0
    for r in results:
        cf = "✓" if r["cov"] >= 90.0 else "✗"
        print(f"  {r['label']:<12} | {r['rmse']:>7.2f} | {r['mae']:>7.2f} | {r['nasa']:>8.2f} | {r['cov']:>8.1f}% {cf} | {r['wins']:>9,}")
        ar += r["rmse"]; am += r["mae"]; an += r["nasa"]; ac += r["cov"]

    if len(results) > 0:
        print(f"  {'-'*72}")
        n = len(results)
        cf = "✓" if ac/n >= 90.0 else "✗"
        print(f"  {'AVERAGE':<12} | {ar/n:>7.2f} | {am/n:>7.2f} | {an/n:>8.2f} | {ac/n:>8.1f}% {cf} | ")
    print(f"{'='*80}")
    print(f"  Total Execution Time: {total_time:.2f} minutes")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
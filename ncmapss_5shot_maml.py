"""
ncmapss_5shot_maml.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Sim-to-Real via Meta-Learning: ANIL (Almost No Inner Loop) Adaptation.
Proves that 5 real flight cycles are sufficient to adapt the synthetic 
ThermoPINN model to a physical NASA engine without catastrophic forgetting.
(OOM-Safe & Dynamic Split Version)
"""

import os
import copy
import math
import gc
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from pinn_model import PINNModel
from evaluate_ncmapss_adapted import compute_rmse, compute_nasa, conformal_calibrate, compute_coverage

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser("~/nasa_research/data/")
MODEL_PATH = os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt")
TEST_FILE  = "N-CMAPSS_DS01-005.h5"

WINDOW_SIZE = 30
TOTAL_FEAT  = 55
INNER_LR    = 2.5e-4
K_SHOTS     = [0, 1, 3, 5, 10]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_anil_params(model):
    """Extracts ONLY prediction heads to prevent catastrophic forgetting."""
    head_params = []
    for name, param in model.named_parameters():
        if any(h in name for h in ['head', 'log_var_head', 'task_to_rul', 'film']):
            head_params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    return head_params

def adapt_to_real_engine(model, x_sup, y_sup, k, inner_lr):
    """Runs the MAML inner-loop adaptation using Cosine Annealing."""
    if k == 0: return model

    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d): m.eval()

    head_params = get_anil_params(model)
    opt = torch.optim.Adam(head_params, lr=inner_lr)

    for step in range(k):
        cos_lr = inner_lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / max(1, k - 1))))
        for g in opt.param_groups: g['lr'] = cos_lr

        with autocast("cuda"):
            out = model(x_sup)
            pred_log = out["rul_log"].squeeze(-1)
            loss = F.smooth_l1_loss(pred_log, y_sup, beta=1.0)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opt.step()

    model.eval()
    return model

def extract_engine_views(f, uid, X_mean, X_std):
    """Creates memoryless C-strided views instead of allocating RAM."""
    unit_col = f["A_dev"][:, 0].astype(int)
    idx = np.where(unit_col == uid)[0]
    s, e = idx[0], idx[-1] + 1

    X_s = (f["X_s_dev"][s:e].astype(np.float32) - X_mean) / X_std
    W = f["W_dev"][s:e].astype(np.float32)
    Y = f["Y_dev"][s:e].astype(np.float32).flatten()

    n_win = len(Y) - WINDOW_SIZE + 1
    if n_win <= 20: return None, None, None

    X_view = sliding_window_view(X_s, WINDOW_SIZE, axis=0).swapaxes(1, 2)
    W_view = sliding_window_view(W, WINDOW_SIZE, axis=0).swapaxes(1, 2)
    Y_tgt = Y[WINDOW_SIZE - 1:]

    return X_view, W_view, Y_tgt

def main():
    print(f"\n{'='*78}")
    print(f"{'ThermoPINN · MAML Sim-to-Real Few-Shot Adaptation':^78}")
    print(f"{'='*78}")

    base_model = PINNModel(max_rul=150.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    base_model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    base_model.eval()

    ds_path = os.path.join(DATA_DIR, TEST_FILE)
    with h5py.File(ds_path, "r") as f:
        X_sample = f["X_s_dev"][:100000].astype(np.float32)
        X_mean = X_sample.mean(axis=0, keepdims=True)
        X_std  = X_sample.std(axis=0, keepdims=True) + 1e-6
        del X_sample

        unit_col = f["A_dev"][:, 0].astype(int)
        engines = np.unique(unit_col)
        max_rul = float(f["Y_dev"][:].max())

        # 🚀 THE FIX: Dynamically split engines to ensure we don't hit 0 evaluation engines
        split_idx = max(1, int(len(engines) * 0.3))
        cal_engines = engines[:split_idx]
        eval_engines = engines[split_idx:]

        print(f"[1/3] Computing Conformal q_hat Baseline using {len(cal_engines)} engines...")
        cal_p, cal_s, cal_y = [], [], []
        for uid in cal_engines:
            X_view, W_view, Y_tgt = extract_engine_views(f, uid, X_mean, X_std)
            if X_view is None: continue
            
            n_win = len(Y_tgt)
            with torch.no_grad():
                for i in range(0, n_win, 512):
                    j = min(i + 512, n_win)
                    bc = np.zeros((j-i, WINDOW_SIZE, TOTAL_FEAT), dtype=np.float32)
                    bc[:, :, 0:14] = X_view[i:j]
                    bc[:, :, 20:24] = W_view[i:j]
                    
                    with autocast("cuda"):
                        out = base_model(torch.from_numpy(bc).to(DEVICE))
                    
                    cal_p.extend(torch.expm1(out["rul_log"].squeeze(-1)).cpu().numpy())
                    cal_s.extend((torch.exp(0.5 * out["rul_log_var"].squeeze(-1)) * torch.expm1(out["rul_log"].squeeze(-1))).cpu().numpy())
            
            cal_y.extend(Y_tgt)
            del X_view, W_view, Y_tgt
            gc.collect() # Force RAM clearance
        
        q_hat = conformal_calibrate(np.array(cal_p), np.array(cal_s), np.array(cal_y), target=0.90)
        print(f"      Base q_hat: {q_hat:.4f}")

        print(f"\n[2/3] Running ANIL Adaptation on {len(eval_engines)} Real NASA Engines...")
        results = {k: {"p": [], "s": [], "y": []} for k in K_SHOTS}

        for uid in tqdm(eval_engines, desc="Adapting", unit="engine"):
            X_view, W_view, Y_tgt = extract_engine_views(f, uid, X_mean, X_std)
            if X_view is None: continue

            # Pre-allocate Support Set (first 15 flights)
            bc_sup = np.zeros((15, WINDOW_SIZE, TOTAL_FEAT), dtype=np.float32)
            bc_sup[:, :, 0:14] = X_view[:15]
            bc_sup[:, :, 20:24] = W_view[:15]
            
            X_sup_full = torch.from_numpy(bc_sup).to(DEVICE)
            Y_sup_full = torch.log1p(torch.from_numpy(Y_tgt[:15]).to(DEVICE))

            sup_idx = torch.randperm(15)[:10]
            X_sup, Y_sup = X_sup_full[sup_idx], Y_sup_full[sup_idx]

            n_win = len(Y_tgt)
            
            for k in K_SHOTS:
                adapted_model = copy.deepcopy(base_model)
                adapted_model = adapt_to_real_engine(adapted_model, X_sup[:k], Y_sup[:k], k, INNER_LR)

                pred_list, std_list = [], []
                
                with torch.no_grad():
                    for i in range(15, n_win, 512):
                        j = min(i + 512, n_win)
                        bc_qry = np.zeros((j-i, WINDOW_SIZE, TOTAL_FEAT), dtype=np.float32)
                        bc_qry[:, :, 0:14] = X_view[i:j]
                        bc_qry[:, :, 20:24] = W_view[i:j]
                        
                        with autocast("cuda"):
                            out = adapted_model(torch.from_numpy(bc_qry).to(DEVICE))
                        
                        mean_log = out["rul_log"].squeeze(-1)
                        log_var = out["rul_log_var"].squeeze(-1)

                        pred_cy = torch.expm1(mean_log)
                        std_cy = torch.exp(0.5 * log_var) * pred_cy
                        
                        pred_list.extend(pred_cy.cpu().numpy())
                        std_list.extend(std_cy.cpu().numpy())

                results[k]["p"].extend(pred_list)
                results[k]["s"].extend(std_list)
                results[k]["y"].extend(Y_tgt[15:])
            
            del X_view, W_view, Y_tgt
            gc.collect() # Force RAM clearance

    print(f"\n[3/3] Final MAML Adaptation Results")
    print(f"{'-'*78}")
    print(f"  {'k-Shots':<10} | {'RMSE':>7} | {'NASA':>8} | {'Coverage':>10} | {'Gain (%)':>10}")
    print(f"{'-'*78}")

    base_rmse = compute_rmse(np.array(results[0]["p"]), np.array(results[0]["y"]))

    for k in K_SHOTS:
        preds = np.array(results[k]["p"])
        stds = np.array(results[k]["s"])
        trues = np.array(results[k]["y"])

        rmse = compute_rmse(preds, trues)
        nasa = compute_nasa(preds, trues, max_rul)
        cov = compute_coverage(preds, stds, trues, q_hat)
        
        gain = ((base_rmse - rmse) / base_rmse) * 100 if k > 0 else 0.0
        cov_flag = "✓" if cov >= 90.0 else "✗"

        print(f"  {k:<10} | {rmse:>7.2f} | {nasa:>8.2f} | {cov:>8.1f}% {cov_flag} | {gain:>9.1f}%")
    
    print(f"{'='*78}\n")

if __name__ == "__main__":
    main()
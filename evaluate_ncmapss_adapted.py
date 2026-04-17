"""
evaluate_ncmapss_adapted.py  ·  ThermoPINN  ·  v3.0 (Masterclass Edition - Ice Cold)
══════════════════════════════════════════════════════════════════════════════
Zero-Shot Sim-to-Real Transfer: UTDTB v5 → NASA N-CMAPSS DS01–DS05
"""

import os, math, time, gc, warnings
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings("ignore")

from pinn_model import PINNModel

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.expanduser("~/nasa_research/data/")
MODEL_PATH = os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt")

SENSOR_START, SENSOR_END = 0, 14
ENV_START,    ENV_END    = 20, 24
TOTAL_FEAT   = 55
WINDOW_SIZE  = 30

BASE_BATCH   = 512
MC_PASSES    = 10
USE_AMP      = True

# ── Thermal governor (Ice-Cold Settings) ─────────────────────────────────────
TEMP_SOFT_LIMIT = 55   # °C — start reducing batch size
TEMP_HARD_LIMIT = 60   # °C — pause inference for 5 seconds
TEMP_PAUSE_SECS = 5.0  # cool-down pause duration

# ── Calibration ──────────────────────────────────────────────────────────────
CALIBRATION_DATASET = "N-CMAPSS_DS01-005.h5"
CAL_FRACTION        = 0.30
CAL_MIN_ENGINES     = 20
TARGET_COVERAGE     = 0.90

DATASETS = [
    "N-CMAPSS_DS01-005.h5", "N-CMAPSS_DS02-006.h5", 
    "N-CMAPSS_DS03-012.h5", "N-CMAPSS_DS04.h5", "N-CMAPSS_DS05.h5"
]

def gpu_temp() -> float:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=1
        )
        return float(out.decode().strip())
    except Exception:
        return -1.0

def predict_engine_batched_mc(model, X_s_u, W_u, Y_u, device):
    n_rows = len(Y_u)
    n_win  = n_rows - WINDOW_SIZE + 1
    if n_win <= 0: return np.array([]), np.array([]), np.array([])

    X_view   = sliding_window_view(X_s_u, WINDOW_SIZE, axis=0).swapaxes(1, 2)
    W_view   = sliding_window_view(W_u,   WINDOW_SIZE, axis=0).swapaxes(1, 2)
    Y_target = Y_u[WINDOW_SIZE - 1:]

    all_means = np.empty(n_win, dtype=np.float32)
    all_stds  = np.empty(n_win, dtype=np.float32)

    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d): m.eval()

    op_zeros = torch.zeros(BASE_BATCH * MC_PASSES, dtype=torch.long, device=device)
    ev_zeros = torch.zeros(BASE_BATCH * MC_PASSES, dtype=torch.long, device=device)

    with torch.no_grad():
        for start_idx in range(0, n_win, BASE_BATCH):
            t = gpu_temp()
            if t >= TEMP_HARD_LIMIT: time.sleep(TEMP_PAUSE_SECS)

            end_idx    = min(start_idx + BASE_BATCH, n_win)
            current_bs = end_idx - start_idx

            batch_cpu = np.zeros((current_bs, WINDOW_SIZE, TOTAL_FEAT), dtype=np.float32)
            batch_cpu[:, :, SENSOR_START:SENSOR_END] = X_view[start_idx:end_idx]
            batch_cpu[:, :, ENV_START:ENV_END]       = W_view[start_idx:end_idx]

            mc_batch_cpu = np.repeat(batch_cpu, MC_PASSES, axis=0)

            gpu_batch = torch.from_numpy(mc_batch_cpu).pin_memory().to(device, non_blocking=True)
            n_super   = current_bs * MC_PASSES
            gpu_op    = op_zeros[:n_super]
            gpu_ev    = ev_zeros[:n_super]

            with autocast("cuda", enabled=USE_AMP):
                out = model(gpu_batch, op_setting=gpu_op, event_flag=gpu_ev)
                rul_log = out["rul_log"].squeeze(-1).float()
                log_var = out["rul_log_var"].squeeze(-1).float()

            rul_log = rul_log.reshape(current_bs, MC_PASSES)
            log_var = log_var.reshape(current_bs, MC_PASSES)

            mean_log  = rul_log.mean(dim=1)
            epis_std  = rul_log.std(dim=1)
            alea_std  = torch.exp(0.5 * log_var.mean(dim=1))
            total_std = torch.sqrt(epis_std**2 + alea_std**2)

            pred_cy  = torch.expm1(mean_log)
            std_cy   = torch.clamp(total_std * pred_cy, min=1.0)

            all_means[start_idx:end_idx] = pred_cy.cpu().numpy()
            all_stds[start_idx:end_idx]  = std_cy.cpu().numpy()
            
            time.sleep(0.05) # Thermal micro-stutter

    model.eval()
    return all_means, all_stds, Y_target.astype(np.float32)

def compute_rmse(preds, trues): return float(np.sqrt(np.mean((preds - trues) ** 2)))
def compute_mae(preds, trues): return float(np.mean(np.abs(preds - trues)))

def compute_nasa(preds, trues, max_rul):
    clamp  = max_rul * 0.5
    errors = np.clip(preds - trues, -clamp, clamp)
    scores = np.where(errors < 0, np.exp(-errors / 13.0) - 1.0, np.exp( errors / 10.0) - 1.0)
    return float(np.mean(scores))

def conformal_calibrate(preds, stds, trues, target=TARGET_COVERAGE):
    scores  = np.abs(preds - trues) / (stds + 1e-6)
    n       = len(scores)
    q_level = min(1.0, math.ceil(target * (n + 1)) / n)
    return float(np.quantile(scores, q_level))

def compute_coverage(preds, stds, trues, q_hat):
    lower, upper = preds - stds * q_hat, preds + stds * q_hat
    return float(np.mean((trues >= lower) & (trues <= upper)) * 100.0)

def compute_sharpness(stds, q_hat): return float(np.mean(2.0 * stds * q_hat))
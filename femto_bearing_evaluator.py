"""
femto_bearing_evaluator.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Cross-Domain Independent Validation: FEMTO-PRONOSTIA (IEEE PHM 2012).
"""

import os, glob
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from torch.amp import autocast
from numpy.lib.stride_tricks import sliding_window_view
from pinn_model import PINNModel

BASE_DATA_DIR = os.path.expanduser("~/nasa_research/data/FEMTOBearingDataSet/")
MODEL_PATH = os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt")
WINDOW_SIZE, TOTAL_FEAT = 30, 55
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_statistical_features(signal):
    rms = np.sqrt(np.mean(signal**2)) + 1e-8
    return [np.mean(signal), rms, stats.kurtosis(signal, fisher=False), stats.skew(signal), np.max(np.abs(signal)) / rms, np.max(signal) - np.min(signal)]

def extract_bearing_features(bearing_folder, ema_alpha=0.1):
    files = sorted(glob.glob(os.path.join(bearing_folder, "*.csv")) + glob.glob(os.path.join(bearing_folder, "*.xlsx")))
    if not files: return None, None

    raw_features = []
    for f in files:
        try:
            df = pd.read_csv(f, header=None, float_precision='high') if f.endswith('.csv') else pd.read_excel(f, header=None)
            horiz_acc, vert_acc = df.iloc[:, 4].values.astype(np.float32), df.iloc[:, 5].values.astype(np.float32)
            raw_features.append(compute_statistical_features(horiz_acc) + compute_statistical_features(vert_acc))
        except: continue

    if not raw_features: return None, None
    smoothed_features = pd.DataFrame(raw_features).ewm(alpha=ema_alpha, adjust=False).mean().values
    return smoothed_features, np.arange(len(smoothed_features) - 1, -1, -1).astype(np.float32)

def evaluate_bearing_dataset(model, X_feat, Y_true, dataset_name):
    X_norm = (X_feat - X_feat.mean(axis=0, keepdims=True)) / (X_feat.std(axis=0, keepdims=True) + 1e-6)
    n_win = len(Y_true) - WINDOW_SIZE + 1
    if n_win <= 0: return

    batch_cpu = np.zeros((n_win, WINDOW_SIZE, TOTAL_FEAT), dtype=np.float32)
    batch_cpu[:, :, 0:12] = sliding_window_view(X_norm, WINDOW_SIZE, axis=0).swapaxes(1, 2)
    Y_tgt = Y_true[WINDOW_SIZE - 1:]

    pred_list = []
    with torch.no_grad():
        for i in range(0, n_win, 512):
            out = model(torch.from_numpy(batch_cpu[i:min(i + 512, n_win)]).to(DEVICE))
            pred_list.extend(torch.expm1(out["rul_log"].squeeze(-1)).cpu().numpy())

    preds = np.array(pred_list)
    print(f"     [Metrics] RMSE: {np.sqrt(np.mean((preds - Y_tgt)**2)):.2f} | MAE: {np.mean(np.abs(preds - Y_tgt)):.2f}")

    plt.figure(figsize=(8, 4))
    plt.plot(Y_tgt, label="True RUL", color="black", linestyle="--")
    plt.plot(preds, label="Prediction", color="#E24B4A", alpha=0.8)
    plt.title(f"Zero-Shot Bearing Degradation: {dataset_name}")
    plt.legend(frameon=False); plt.grid(True, alpha=0.3); plt.tight_layout()
    os.makedirs("paper_pdfs", exist_ok=True)
    out_path = f"paper_pdfs/Fig8_{dataset_name}.pdf"
    plt.savefig(out_path, format="pdf"); plt.close()
    print(f"     [Saved] Plot exported to {out_path}")

def main():
    print(f"\n{'='*80}\n{'ThermoPINN · IEEE PHM 2012 Bearing Evaluation (Zero-Shot)':^80}\n{'='*80}")
    model = PINNModel(max_rul=150.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True).get("model_state", torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)), strict=False)
    except Exception as e: print(f"❌ ERROR: Could not load model. {e}"); return
    model.eval()

    bearing_folders = sorted([root for root, _, _ in os.walk(BASE_DATA_DIR) if "Bearing" in os.path.basename(root)])
    if not bearing_folders: print("❌ ERROR: No 'Bearing' folders found."); return

    for b_folder in bearing_folders:
        dataset_name = f"{os.path.basename(os.path.dirname(b_folder))}_{os.path.basename(b_folder)}"
        print(f"\n[{dataset_name}]\n  -> Extracting features...")
        X_feat, Y_true = extract_bearing_features(b_folder)
        if X_feat is not None: evaluate_bearing_dataset(model, X_feat, Y_true, dataset_name)
    print(f"\n{'='*80}\n✅ Multi-domain evaluation complete. All PDFs saved to paper_pdfs/")

if __name__ == "__main__": main()
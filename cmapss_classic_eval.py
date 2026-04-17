"""
cmapss_classic_eval.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Experiment 1: Classic N-CMAPSS Generalization & Few-Shot Adaptation.
Parses the original 2008 PHM Data Challenge raw .txt files (FD001-FD004).
Maps the 24-D physical feature space (3 OpSettings + 21 Sensors) into the 
55-D ThermoPINN latent manifold via zero-padding.
"""

import os
import copy
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

# ─── Configuration & Metrics ──────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30
TOTAL_FEAT = 55
BASE_CALIB_SIZE = 15  # Scaled down for classic C-MAPSS short trajectories

def calculate_nasa_score(pred_rul, true_rul):
    """Asymmetric scoring function from the 2008 PHM Data Challenge."""
    d = pred_rul - true_rul
    score = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return np.sum(score)

# ─── Data Extractor ───────────────────────────────────────────────────────────

def extract_classic_cmapss(data_dir):
    """
    Parses space-delimited text files for FD001 through FD004.
    Reconstructs the true RUL for every cycle using the final RUL anchor.
    """
    if not os.path.exists(data_dir): 
        print(f"❌ Error: Could not find directory at {data_dir}")
        return None
    
    engines_data = []
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    
    print(f"Scanning {data_dir} for classic C-MAPSS text files...")
    
    for fd in subsets:
        test_file = os.path.join(data_dir, f"test_{fd}.txt")
        rul_file = os.path.join(data_dir, f"RUL_{fd}.txt")
        
        # Windows hides extensions sometimes, try without .txt if standard fails
        if not os.path.exists(test_file): test_file = os.path.join(data_dir, f"test_{fd}")
        if not os.path.exists(rul_file): rul_file = os.path.join(data_dir, f"RUL_{fd}")
            
        if not os.path.exists(test_file) or not os.path.exists(rul_file):
            print(f"  [Skip] Missing files for {fd}")
            continue
            
        # Parse space-delimited NASA schema
        df_test = pd.read_csv(test_file, sep=r'\s+', header=None)
        df_rul = pd.read_csv(rul_file, sep=r'\s+', header=None)
        
        eng_ids = df_test[0].values
        cycles = df_test[1].values
        # Cols 2,3,4 = OpSettings | Cols 5-25 = Sensors (Total 24 features)
        features = df_test.iloc[:, 2:26].values.astype(np.float32)
        
        unique_engines = np.unique(eng_ids)
        print(f"  -> Extracted {len(unique_engines)} physical engines from {fd}")
        
        for eng in unique_engines:
            idx = np.where(eng_ids == eng)[0]
            
            # The true RUL file gives the RUL at the LAST cycle of the test engine.
            # RUL at any cycle t = (max_cycle - cycle[t]) + rul_last
            rul_last = df_rul.iloc[int(eng) - 1, 0]
            max_cycle = cycles[idx][-1]
            ruls = max_cycle - cycles[idx] + rul_last
            
            # Require at least 1 window
            if len(idx) < WINDOW_SIZE + 5: continue
            
            X_raw = features[idx]
            # Standardize across the engine's timeline
            X_norm = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)
            
            # Map 24-D classic features into the 55-D ThermoPINN architecture
            X_55d = np.zeros((len(X_norm), TOTAL_FEAT), dtype=np.float32)
            X_55d[:, :24] = X_norm
            
            X_win = sliding_window_view(X_55d, WINDOW_SIZE, axis=0).swapaxes(1, 2)
            
            # Target is the true RUL at the END of the 30-step window
            Y_tgt = ruls[WINDOW_SIZE - 1:]
            engines_data.append((X_win, Y_tgt))
            
    return engines_data

# ─── Core Evaluation Engine ───────────────────────────────────────────────────

def evaluate_engine_few_shot(base_model, X_win, Y_tgt, k_shots):
    """
    Performs true PyTorch gradient updates for k-shot adaptation, 
    calibrates Conformal q_hat dynamically, and computes test metrics.
    """
    n_total = len(X_win)
    
    # Dynamic calibration sizing for short C-MAPSS engines
    available_for_test = n_total - k_shots
    if available_for_test <= 2: return None 
    calib_size = min(BASE_CALIB_SIZE, int(available_for_test * 0.5))
    if calib_size < 2: return None

    # 1. Clone model so adaptation doesn't corrupt the base global weights
    model = copy.deepcopy(base_model).to(DEVICE)
    
    X_tens = torch.tensor(X_win, dtype=torch.float32).to(DEVICE)
    Y_tens = torch.tensor(Y_tgt, dtype=torch.float32).to(DEVICE)
    op = torch.zeros(n_total, dtype=torch.long, device=DEVICE)
    ev = torch.zeros(n_total, dtype=torch.long, device=DEVICE)

    # 2. Few-Shot Fine-tuning (MAML style adaptation)
    if k_shots > 0:
        model.train()
        
        # Unfreeze all parameters for Full-Network Adaptation
        for param in model.parameters(): 
            param.requires_grad = True
        
        # Use a smaller learning rate (1e-4) to prevent catastrophic forgetting
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        X_train, Y_train = X_tens[:k_shots], Y_tens[:k_shots]
        
        for _ in range(15):
            optimizer.zero_grad()
            out = model(X_train, op_setting=op[:k_shots], event_flag=ev[:k_shots])
            pred_rul = torch.expm1(out["rul_log"].squeeze(-1))
            loss = criterion(pred_rul, Y_train)
            loss.backward()
            optimizer.step()
    # 3. Conformal Calibration
    model.eval()
    with torch.no_grad():
        calib_out = model(X_tens[k_shots : k_shots+calib_size], op_setting=op[:calib_size], event_flag=ev[:calib_size])
        calib_preds = torch.expm1(calib_out["rul_log"].squeeze(-1)).cpu().numpy()
        calib_trues = Y_tgt[k_shots : k_shots+calib_size]
        
        errors = np.abs(calib_preds - calib_trues)
        q_idx = min(calib_size - 1, int(np.ceil(0.90 * (calib_size + 1))) - 1)
        q_hat = np.sort(errors)[q_idx]

    # 4. Test Evaluation
    test_idx = k_shots + calib_size
    with torch.no_grad():
        test_out = model(X_tens[test_idx:], op_setting=op[test_idx:], event_flag=ev[test_idx:])
        test_preds = torch.expm1(test_out["rul_log"].squeeze(-1)).cpu().numpy()
        test_trues = Y_tgt[test_idx:]

    rmse = np.sqrt(np.mean((test_preds - test_trues)**2))
    nasa = calculate_nasa_score(test_preds, test_trues) / len(test_preds)
    
    lower_bound = test_preds - q_hat
    upper_bound = test_preds + q_hat
    coverage = np.mean((test_trues >= lower_bound) & (test_trues <= upper_bound)) * 100

    return {"rmse": rmse, "coverage": coverage, "nasa": nasa}

# ─── Main Execution ───────────────────────────────────────────────────────────

def run_experiment(args):
    print(f"\n{'='*80}\n{'Classic C-MAPSS Generalization & Few-Shot Adaptation':^80}\n{'='*80}")
    
    try:
        from pinn_model import PINNModel
        base_model = PINNModel(max_rul=500.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0)
        base_model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True).get("model_state", torch.load(args.model_path, map_location=DEVICE, weights_only=True)), strict=False)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    engine_list = extract_classic_cmapss(args.data_path)
    if not engine_list: 
        print("❌ No valid engines extracted. Check data path.")
        return

    dataset_metrics = {"Dataset": "Classic C-MAPSS"}
    
    print(f"\nEvaluating {len(engine_list)} valid test trajectories...")
    for k in args.k_shots:
        k_rmse, k_cov, k_nasa = [], [], []
        for X_win, Y_tgt in tqdm(engine_list, desc=f"  k={k} Adaptation", leave=False, ncols=80, colour="blue"):
            res = evaluate_engine_few_shot(base_model, X_win, Y_tgt, k)
            if res:
                k_rmse.append(res["rmse"])
                k_cov.append(res["coverage"])
                k_nasa.append(res["nasa"])
                
        dataset_metrics[f"k={k} RMSE"] = np.mean(k_rmse)
        dataset_metrics[f"k={k} Cov %"] = np.mean(k_cov)
        dataset_metrics[f"k={k} NASA"] = np.mean(k_nasa)

    print(f"\n{'='*80}")
    print(f"{'Final Classic C-MAPSS Test Metrics':^80}")
    print(f"{'-'*80}")
    
    headers = ["Dataset", "k=0 RMSE", f"k={args.k_shots[-1]} RMSE", "Coverage", "NASA Score"]
    row_fmt = "{:<20} | {:>9} | {:>9} | {:>9} | {:>10}"
    print(row_fmt.format(*headers))
    print(f"{'-'*80}")
    
    k_last = args.k_shots[-1]
    print(row_fmt.format(
        dataset_metrics["Dataset"], 
        f"{dataset_metrics['k=0 RMSE']:.1f}", 
        f"{dataset_metrics[f'k={k_last} RMSE']:.1f}", 
        f"{dataset_metrics['k=0 Cov %']:.1f}%", 
        f"{dataset_metrics['k=0 NASA']:.2f}"
    ))
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Point this to the directory containing test_FD001.txt, RUL_FD001.txt, etc.
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/nasa_research/data/"))
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt"))
    parser.add_argument("--k_shots", nargs="+", type=int, default=[0, 5, 10], help="List of k-shots for adaptation")
    args = parser.parse_args()
    
    run_experiment(args)
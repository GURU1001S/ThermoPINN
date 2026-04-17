"""
cross_domain_adaptation.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Experiment 2: Cross-Domain Adversarial Adaptation (DANN)
Source Domain : UTDTB v5 (55-D thermodynamic features)
Target Domain : Classic C-MAPSS FD001 (24-D legacy features, zero-padded)
Objective     : Force the encoder to map both datasets into a shared 
                physics manifold, dropping the Target Zero-Shot RMSE.
"""

import os
import h5py
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30
TOTAL_FEAT = 55
BATCH_SIZE = 256

# ─── Adversarial Components ───────────────────────────────────────────────────

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.save_for_backward(torch.tensor(lambda_val))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_val = ctx.saved_tensors[0].item()
        return -lambda_val * grad_output, None  # The Adversarial Strike

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, feat, lambda_val=1.0):
        feat_rev = GradientReversalLayer.apply(feat, lambda_val)
        return self.layers(feat_rev)

# ─── Data Engineering ─────────────────────────────────────────────────────────

def extract_source_utdtb(h5_path, max_engines=50):
    """Extracts the 55-D UTDTB Dataset (Source Domain)."""
    print("Extracting Source Domain [UTDTB v5]...")
    with h5py.File(h5_path, "r") as f:
        grp = f["train"]
        eng_ids = grp["engine_id"][:]
        ruls = grp["RUL"][:]
        
        unique_engines = np.unique(eng_ids)[:max_engines]
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
            
    return torch.tensor(np.concatenate(X_all), dtype=torch.float32), torch.tensor(np.concatenate(Y_all), dtype=torch.float32)

def extract_target_cmapss(data_dir, max_engines=50):
    """Extracts the 24-D Classic C-MAPSS Dataset (Target Domain)."""
    print("Extracting Target Domain [Classic C-MAPSS FD001]...")
    test_file = os.path.join(data_dir, "test_FD001.txt")
    rul_file = os.path.join(data_dir, "RUL_FD001.txt")
    
    if not os.path.exists(test_file): test_file = os.path.join(data_dir, "test_FD001")
    if not os.path.exists(rul_file): rul_file = os.path.join(data_dir, "RUL_FD001")

    df_test = pd.read_csv(test_file, sep=r'\s+', header=None)
    df_rul = pd.read_csv(rul_file, sep=r'\s+', header=None)
    
    eng_ids = df_test[0].values
    cycles = df_test[1].values
    features = df_test.iloc[:, 2:26].values.astype(np.float32)
    
    unique_engines = np.unique(eng_ids)[:max_engines]
    X_all, Y_all = [], []
    
    for eng in unique_engines:
        idx = np.where(eng_ids == eng)[0]
        rul_last = df_rul.iloc[int(eng) - 1, 0]
        max_cycle = cycles[idx][-1]
        ruls = max_cycle - cycles[idx] + rul_last
        
        if len(idx) < WINDOW_SIZE: continue
        
        X_raw = features[idx]
        X_norm = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)
        
        # Zero-pad 24-D to 55-D
        X_55d = np.zeros((len(X_norm), TOTAL_FEAT), dtype=np.float32)
        X_55d[:, :24] = X_norm
        
        X_win = sliding_window_view(X_55d, WINDOW_SIZE, axis=0).swapaxes(1, 2)
        X_all.append(X_win)
        Y_all.append(ruls[WINDOW_SIZE - 1:])
        
    return torch.tensor(np.concatenate(X_all), dtype=torch.float32), torch.tensor(np.concatenate(Y_all), dtype=torch.float32)

# ─── Core Adversarial Training ────────────────────────────────────────────────

def run_adversarial_adaptation(args):
    print(f"\n{'='*80}\n{'Cross-Domain DANN: UTDTB ➔ Classic C-MAPSS':^80}\n{'='*80}")
    
    # 1. Load Data
    try:
        X_src, Y_src = extract_source_utdtb(args.utdtb_path, max_engines=50)
        X_tgt, Y_tgt = extract_target_cmapss(args.cmapss_path, max_engines=50)
    except Exception as e:
        print(f"❌ Error extracting data: {e}"); return

    src_loader = DataLoader(TensorDataset(X_src, Y_src), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(TensorDataset(X_tgt, Y_tgt), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Initialize Models
    from pinn_model import PINNModel
    model = PINNModel(max_rul=500.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True).get("model_state", torch.load(args.model_path, map_location=DEVICE, weights_only=True)), strict=False)
    
    # Evaluate Baseline Target RMSE
    model.eval()
    with torch.no_grad():
        out_base = model(X_tgt.to(DEVICE), op_setting=torch.zeros(len(X_tgt), dtype=torch.long, device=DEVICE), event_flag=torch.zeros(len(X_tgt), dtype=torch.long, device=DEVICE))
        preds_base = torch.expm1(out_base["rul_log"].squeeze(-1)).cpu().numpy()
        base_rmse = np.sqrt(np.mean((preds_base - Y_tgt.numpy())**2))
    print(f"\n[Baseline] Pre-Adversarial C-MAPSS RMSE: {base_rmse:.2f}")

    with torch.no_grad():
        dummy_out = model(X_src[:2].to(DEVICE), op_setting=torch.zeros(2, dtype=torch.long, device=DEVICE), event_flag=torch.zeros(2, dtype=torch.long, device=DEVICE))
        latent_dim = dummy_out.get("latent", dummy_out.get("physics_preds", dummy_out["rul_log"])).shape[-1]

    discriminator = DomainDiscriminator(input_dim=latent_dim).to(DEVICE)
    
    # 3. Optimizers
    # We use a very low learning rate to gently shift the encoder without forgetting the physics
    optimizer = torch.optim.Adam(list(model.parameters()) + list(discriminator.parameters()), lr=5e-5)
    criterion_rul = nn.MSELoss()
    criterion_domain = nn.BCEWithLogitsLoss()
    
    # 4. Training Loop
    epochs = args.epochs
    total_batches = min(len(src_loader), len(tgt_loader))
    
    print("\nInitiating Cross-Domain Adversarial Alignment...")
    for epoch in range(epochs):
        model.train()
        discriminator.train()
        domain_accs = []
        
        # P in [0, 1] dictates the curriculum of the Gradient Reversal
        p = epoch / epochs 
        lambda_val = (2.0 / (1.0 + np.exp(-10 * p))) - 1.0 
        
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        
        for _ in tqdm(range(total_batches), desc=f"Epoch {epoch+1}/{epochs} [λ={lambda_val:.2f}]", leave=False, ncols=80, colour="red"):
            x_s, y_s = next(src_iter)
            x_t, _   = next(tgt_iter)
            x_s, y_s, x_t = x_s.to(DEVICE), y_s.to(DEVICE), x_t.to(DEVICE)
            
            op = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
            ev = torch.zeros(BATCH_SIZE, dtype=torch.long, device=DEVICE)
            
            optimizer.zero_grad()
            
            # Source Domain Forward (Label = 0)
            out_s = model(x_s, op_setting=op, event_flag=ev)
            feat_s = out_s.get("latent", out_s.get("physics_preds", out_s["rul_log"]))
            pred_rul_s = torch.expm1(out_s["rul_log"].squeeze(-1))
            
            loss_rul = criterion_rul(pred_rul_s, y_s)
            domain_pred_s = discriminator(feat_s, lambda_val)
            loss_domain_s = criterion_domain(domain_pred_s, torch.zeros_like(domain_pred_s))
            
            # Target Domain Forward (Label = 1)
            out_t = model(x_t, op_setting=op, event_flag=ev)
            feat_t = out_t.get("latent", out_t.get("physics_preds", out_t["rul_log"]))
            
            domain_pred_t = discriminator(feat_t, lambda_val)
            loss_domain_t = criterion_domain(domain_pred_t, torch.ones_like(domain_pred_t))
            
            # Total Adversarial Loss (Weighing Domain Loss heavily to force alignment)
            loss_domain = 0.5 * (loss_domain_s + loss_domain_t)
            total_loss = loss_rul + (2.0 * loss_domain)
            total_loss.backward()
            optimizer.step()
            
            pred_cls_s = torch.sigmoid(domain_pred_s) > 0.5
            pred_cls_t = torch.sigmoid(domain_pred_t) > 0.5
            acc_s = (pred_cls_s == 0).float().mean().item()
            acc_t = (pred_cls_t == 1).float().mean().item()
            domain_accs.append((acc_s + acc_t) / 2.0)
            
        print(f"Epoch {epoch+1:02d} | Domain Confusion: {np.mean(domain_accs)*100:.1f}% (Target: ~50.0%)")

    # 5. Final Evaluation
    model.eval()
    with torch.no_grad():
        out_final = model(X_tgt.to(DEVICE), op_setting=torch.zeros(len(X_tgt), dtype=torch.long, device=DEVICE), event_flag=torch.zeros(len(X_tgt), dtype=torch.long, device=DEVICE))
        preds_final = torch.expm1(out_final["rul_log"].squeeze(-1)).cpu().numpy()
        final_rmse = np.sqrt(np.mean((preds_final - Y_tgt.numpy())**2))
        
    print(f"\n{'='*80}")
    print(f"Cross-Domain Adaptation Complete")
    print(f"  Pre-DANN C-MAPSS RMSE  : {base_rmse:.2f}")
    print(f"  Post-DANN C-MAPSS RMSE : {final_rmse:.2f}")
    
    improvement = base_rmse - final_rmse
    if improvement > 0:
        print(f"  ✅ DANN SUCCESS: RMSE Improved by {improvement:.2f} cycles!")
    else:
        print(f"  ❌ DANN FAILED: RMSE Degraded by {abs(improvement):.2f} cycles.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--utdtb_path", type=str, default=os.path.expanduser("~/nasa_research/data/utdtb_v5.h5"))
    parser.add_argument("--cmapss_path", type=str, default=os.path.expanduser("~/nasa_research/data/"))
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    
    run_adversarial_adaptation(args)
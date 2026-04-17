"""
domain_adaptation.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Experiment 2: Dynamic Domain Adaptation (Adversarial)
Implements a Gradient Reversal Layer (GRL) and Domain Discriminator to force 
the PINN encoder to learn domain-invariant thermodynamic representations.
Aligns the source training domain with the target evaluation domain.
"""

import os
import h5py
import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30
BATCH_SIZE = 256

# ─── Adversarial Components ───────────────────────────────────────────────────

class GradientReversalLayer(torch.autograd.Function):
    """Reverses the gradient during the backward pass to enforce domain invariance."""
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.save_for_backward(torch.tensor(lambda_val))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_val = ctx.saved_tensors[0].item()
        # The Adversarial Strike: Multiply gradient by negative lambda
        return -lambda_val * grad_output, None

class DomainDiscriminator(nn.Module):
    """Binary classifier: Is this feature from Source (0) or Target (1)?"""
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

def extract_domain_tensors(h5_path, split="train", max_engines=40):
    """Extracts sliding windows for a specific domain split."""
    print(f"Extracting [{split}] domain data...")
    with h5py.File(h5_path, "r") as f:
        if split not in f: return None, None
        grp = f[split]
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
            X_norm = X_norm[np.argsort(ruls[idx])[::-1]] # Sort chronologically
            
            X_win = sliding_window_view(X_norm, WINDOW_SIZE, axis=0).swapaxes(1, 2)
            Y_tgt = np.arange(len(X_win) - 1, -1, -1).astype(np.float32)
            
            X_all.append(X_win)
            Y_all.append(Y_tgt)
            
    X_tensor = torch.tensor(np.concatenate(X_all), dtype=torch.float32)
    Y_tensor = torch.tensor(np.concatenate(Y_all), dtype=torch.float32)
    return X_tensor, Y_tensor

# ─── Core Adversarial Training ────────────────────────────────────────────────

def run_adversarial_adaptation(args):
    print(f"\n{'='*80}\n{'Experiment 2: Dynamic Domain Adaptation (DANN)':^80}\n{'='*80}")
    
    # 1. Load Data
    X_src, Y_src = extract_domain_tensors(args.data_path, split="train", max_engines=50) # Source Domain
    X_tgt, Y_tgt = extract_domain_tensors(args.data_path, split="test", max_engines=50)  # Target Domain
    
    if X_src is None or X_tgt is None:
        print("❌ Could not load domains from HDF5.")
        return

    # Create DataLoaders (Drop last to maintain parallel batch sizes)
    src_loader = DataLoader(TensorDataset(X_src, Y_src), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(TensorDataset(X_tgt, Y_tgt), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Initialize Models
    from pinn_model import PINNModel
    model = PINNModel(max_rul=500.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True).get("model_state", torch.load(args.model_path, map_location=DEVICE, weights_only=True)), strict=False)
    
    # We evaluate Baseline Zero-Shot RMSE first
    model.eval()
    with torch.no_grad():
        out_base = model(X_tgt.to(DEVICE), op_setting=torch.zeros(len(X_tgt), dtype=torch.long, device=DEVICE), event_flag=torch.zeros(len(X_tgt), dtype=torch.long, device=DEVICE))
        preds_base = torch.expm1(out_base["rul_log"].squeeze(-1)).cpu().numpy()
        base_rmse = np.sqrt(np.mean((preds_base - Y_tgt.numpy())**2))
    print(f"\n[Baseline] Pre-Adversarial Target RMSE: {base_rmse:.2f}")

    # Determine Latent Dimension dynamically
    with torch.no_grad():
        dummy_out = model(X_src[:2].to(DEVICE), op_setting=torch.zeros(2, dtype=torch.long, device=DEVICE), event_flag=torch.zeros(2, dtype=torch.long, device=DEVICE))
        latent_dim = dummy_out.get("latent", dummy_out.get("physics_preds", dummy_out["rul_log"])).shape[-1]

    discriminator = DomainDiscriminator(input_dim=latent_dim).to(DEVICE)
    
    # 3. Optimizers & Losses
    optimizer = torch.optim.Adam(list(model.parameters()) + list(discriminator.parameters()), lr=1e-4)
    criterion_rul = nn.MSELoss()
    criterion_domain = nn.BCEWithLogitsLoss()
    
    # 4. Training Loop
    epochs = args.epochs
    total_batches = min(len(src_loader), len(tgt_loader))
    
    print("\nInitiating Adversarial Domain Alignment...")
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
            
            # Total Adversarial Loss
            loss_domain = 0.5 * (loss_domain_s + loss_domain_t)
            total_loss = loss_rul + loss_domain
            total_loss.backward()
            optimizer.step()
            
            # Calculate Domain Confusion (Target ~50%)
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
    print(f"Domain Adaptation Complete")
    print(f"  Pre-DANN Target RMSE  : {base_rmse:.2f}")
    print(f"  Post-DANN Target RMSE : {final_rmse:.2f}")
    print(f"  RMSE Improvement      : {base_rmse - final_rmse:.2f} cycles")
    print(f"{'='*80}\n")
    
    # Save the domain-invariant model
    torch.save({"model_state": model.state_dict()}, "thermoPINN_domain_invariant.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/nasa_research/data/utdtb_v5.h5"))
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt"))
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    
    run_adversarial_adaptation(args)
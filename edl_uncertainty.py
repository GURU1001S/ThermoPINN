"""
edl_uncertainty.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Evidential Deep Learning (EDL) Uncertainty Post-Hoc Wrapper.
Implementation of Amini et al. (NeurIPS 2020) Deep Evidential Regression.
Replaces MC Dropout with a Normal-Inverse-Gamma (NIG) output head to 
analytically decompose Aleatoric and Epistemic uncertainty in a single pass.
"""

import os
import math
import h5py
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30
BATCH_SIZE = 256

# ─── Evidential Deep Learning (NIG) ───────────────────────────────────────────

class EDLHead(nn.Module):
    """4-output NIG head that attaches to the frozen ThermoPINN latent space."""
    def __init__(self, d_model=256, head_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, 4), # Outputs: γ, log_ν, log_α, log_β
        )
        # Initialize virtual observations and shape parameter to be safe
        nn.init.constant_(self.net[-1].bias[1], 0.5) # ν init
        nn.init.constant_(self.net[-1].bias[2], 0.5) # α init > 1

    def forward(self, features):
        out = self.net(features)
        gamma = out[:, 0]                            # Mean prediction (Log RUL)
        nu    = F.softplus(out[:, 1]) + 1e-4         # Virtual obs count > 0
        alpha = F.softplus(out[:, 2]) + 1.0 + 1e-4   # Shape > 1 (Required for variance)
        beta  = F.softplus(out[:, 3]) + 1e-4         # Scale > 0
        return gamma, nu, alpha, beta

def nig_nll_loss(gamma, nu, alpha, beta, y, lam=0.01):
    """NIG Negative Log-Likelihood + Evidence Regularization (Amini 2020)."""
    # NIG-NLL formulation
    nll = (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(beta)
        + (alpha + 0.5) * torch.log((y - gamma)**2 * nu + beta)
        + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    )
    # Evidence regularizer: Penalizes high certainty on high errors (prevents ν → ∞ collapse)
    reg = torch.abs(y - gamma) * (2.0 * nu + alpha)
    return (nll + lam * reg).mean()

def edl_uncertainty(nu, alpha, beta):
    """Analytic uncertainty decomposition (Single Pass)."""
    aleatoric = beta / (alpha - 1.0)               # Data noise
    epistemic = beta / (nu * (alpha - 1.0))        # Model ignorance (OOD detection)
    return aleatoric, epistemic

# ─── Data & Wrapper ───────────────────────────────────────────────────────────

def extract_domain_tensors(h5_path, split="train", max_engines=40):
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
            X_norm = X_norm[np.argsort(ruls[idx])[::-1]] 
            X_win = sliding_window_view(X_norm, WINDOW_SIZE, axis=0).swapaxes(1, 2)
            X_all.append(X_win)
            Y_all.append(np.arange(len(X_win) - 1, -1, -1).astype(np.float32))
    return torch.tensor(np.concatenate(X_all), dtype=torch.float32), torch.tensor(np.concatenate(Y_all), dtype=torch.float32)

class EDLWrapper:
    def __init__(self, base_model, device):
        self.encoder = base_model.eval().to(device)
        # Freeze entire base model
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Dynamically find latent dimension size
        with torch.no_grad():
            dummy = torch.randn(2, WINDOW_SIZE, 55).to(device)
            op = torch.zeros(2, dtype=torch.long, device=device)
            out = self.encoder(dummy, op_setting=op, event_flag=op)
            latent_dim = out.get("latent", out.get("physics_preds", out["rul_log"])).shape[-1]

        self.edl = EDLHead(d_model=latent_dim).to(device)
        self.device = device

    def fit(self, train_loader, epochs=30, lr=1e-3):
        print(f"\nTraining EDL Head on Frozen Encoder ({epochs} Epochs)...")
        opt = torch.optim.Adam(self.edl.parameters(), lr=lr)
        self.edl.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in tqdm(train_loader, desc=f"  Epoch {epoch+1:02d}", leave=False, ncols=80, colour="magenta"):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_log = torch.log1p(y_batch) # Target in log-space for stability
                
                with torch.no_grad():
                    op = torch.zeros(len(x_batch), dtype=torch.long, device=self.device)
                    out = self.encoder(x_batch, op_setting=op, event_flag=op)
                    feat = out.get("latent", out.get("physics_preds", out["rul_log"])).detach()
                    
                gamma, nu, alpha, beta = self.edl(feat)
                loss = nig_nll_loss(gamma, nu, alpha, beta, y_log)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

    def predict(self, x):
        self.edl.eval()
        with torch.no_grad():
            op = torch.zeros(len(x), dtype=torch.long, device=self.device)
            out = self.encoder(x.to(self.device), op_setting=op, event_flag=op)
            feat = out.get("latent", out.get("physics_preds", out["rul_log"]))
            gamma, nu, alpha, beta = self.edl(feat)
            
        # Convert predictions from Log-Space back to Linear RUL Space
        pred_rul = torch.expm1(gamma)
        alea_log, epis_log = edl_uncertainty(nu, alpha, beta)
        
        # Approximate linear-space uncertainty mapping
        alea_lin = alea_log * pred_rul
        epis_lin = epis_log * pred_rul
        
        return pred_rul.cpu().numpy(), alea_lin.cpu().numpy(), epis_lin.cpu().numpy()

# ─── Main Execution ───────────────────────────────────────────────────────────

def run_edl_experiment(args):
    print(f"\n{'='*80}\n{'Evidential Deep Learning (EDL) Uncertainty Decomposition':^80}\n{'='*80}")
    
    # 1. Load Data
    X_train, Y_train = extract_domain_tensors(args.data_path, split="train", max_engines=40)
    X_test, Y_test = extract_domain_tensors(args.data_path, split="test", max_engines=10)
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Load Frozen Model & Train EDL Wrapper
    try:
        from pinn_model import PINNModel
        model = PINNModel(max_rul=500.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True).get("model_state", torch.load(args.model_path, map_location=DEVICE, weights_only=True)), strict=False)
    except Exception as e:
        print(f"❌ Error loading model: {e}"); return

    wrapper = EDLWrapper(model, DEVICE)
    wrapper.fit(train_loader, epochs=30)
    
    # 3. Evaluate on Healthy Data
    print("\nEvaluating Single-Pass EDL on Healthy Test Data...")
    preds_clean, alea_clean, epis_clean = wrapper.predict(X_test)
    rmse_clean = np.sqrt(np.mean((preds_clean - Y_test.numpy())**2))
    mean_alea_clean = np.mean(alea_clean)
    mean_epis_clean = np.mean(epis_clean)
    
    print(f"  RMSE (EDL Head)        : {rmse_clean:.2f} cycles")
    print(f"  Mean Aleatoric (Data)  : {mean_alea_clean:.3f}")
    print(f"  Mean Epistemic (Model) : {mean_epis_clean:.3f}")

    # 4. Out-of-Distribution Stress Test (Simulating Exp 4 Flaw)
    print("\nInitiating OOD Stress Test: Destroying 14 Sensors...")
    X_deg = X_test.numpy().copy()
    X_deg[:, :, :14] = 0.0 # Blind the model to 14 thermodynamic sensors
    X_deg_tensor = torch.tensor(X_deg, dtype=torch.float32)
    
    preds_ood, alea_ood, epis_ood = wrapper.predict(X_deg_tensor)
    mean_alea_ood = np.mean(alea_ood)
    mean_epis_ood = np.mean(epis_ood)
    
    print(f"  Mean Aleatoric (Data)  : {mean_alea_ood:.3f}")
    print(f"  Mean Epistemic (Model) : {mean_epis_ood:.3f}")
    
    epis_inflation = mean_epis_ood / (mean_epis_clean + 1e-8)
    
    print(f"\n  ── Epistemic Safety Check ───────────────────────────")
    if epis_inflation > 2.0:
        print(f"  ✅ SUCCESS: Epistemic ignorance spiked by {epis_inflation:.1f}x!")
        print("  The NIG distribution successfully caught the OOD sensor failure,")
        print("  fixing the MC Dropout overconfidence flaw from Experiment 4.")
    else:
        print(f"  ❌ FAILED: Epistemic ignorance only inflated by {epis_inflation:.1f}x.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/nasa_research/data/utdtb_v5.h5"))
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt"))
    run_edl_experiment(parser.parse_args())
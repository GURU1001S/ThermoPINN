"""
survival_head.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Weibull Survival Analysis Post-Hoc Wrapper.
Trains an Accelerated Failure Time (AFT) model on top of a frozen 
PINN encoder. Outputs P(failure < t) for MRO risk scheduling.
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

# ─── Weibull AFT Survival Model ───────────────────────────────────────────────

class WeibullSurvivalHead(nn.Module):
    """
    Weibull AFT model on frozen ThermoPINN encoder features.
    Outputs: lambda (scale) and k (shape) per sample.
    """
    def __init__(self, d_model=256, head_hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, head_hidden), nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(head_hidden, 2),  # [log_lambda, log_k]
        )
        # Init: k ≈ 2.5 (Typical turbofan wear-out shape)
        nn.init.constant_(self.net[-1].bias[1], math.log(2.5))

    def forward(self, features):
        out = self.net(features)
        lam = F.softplus(out[:, 0]) + 1e-4    # Scale > 0
        k   = F.softplus(out[:, 1]) + 1.0     # Shape > 1 (Strict wear-out regime)
        return lam, k

def weibull_mle_loss(lam, k, true_rul, censored_mask):
    """
    Weibull Maximum Likelihood Estimation with right-censoring support.
    """
    t_over_lam = (true_rul / lam).clamp(min=1e-8)
    
    # Log-survival: applies to ALL engines (failed and censored)
    log_S = -(t_over_lam ** k)
    
    # Log-hazard: applies ONLY to observed failures
    log_h = torch.log(k / lam + 1e-8) + (k - 1) * torch.log(t_over_lam + 1e-8)
    
    # MLE: Maximize log-likelihood = log_h (if failed) + log_S
    ll = torch.where(~censored_mask, log_h + log_S, log_S)
    return -ll.mean() # Minimize negative log-likelihood

def failure_probability(lam, k, t_horizon):
    """P(T <= t_horizon): Probability of failing before horizon."""
    return 1.0 - torch.exp(-((t_horizon / lam) ** k))

def concordance_index(pred_risk, true_times, events):
    """C-index: Fraction of comparable pairs correctly ranked."""
    # Fast vectorized approximation for large batches
    n = len(pred_risk)
    if n < 2: return 0.5
    
    # Matrix of true time differences
    t_diff = true_times.unsqueeze(1) - true_times.unsqueeze(0)
    # Matrix of risk differences
    r_diff = pred_risk.unsqueeze(1) - pred_risk.unsqueeze(0)
    
    # Valid pairs: Engine i failed before Engine j
    valid_mask = (t_diff < 0) & (events.unsqueeze(1) == 1)
    
    concordant = (r_diff > 0) & valid_mask
    discordant = (r_diff < 0) & valid_mask
    
    num_c = concordant.sum().item()
    num_d = discordant.sum().item()
    total = num_c + num_d
    
    return num_c / total if total > 0 else 0.5

# ─── Data Extraction & Wrapper ────────────────────────────────────────────────

def extract_survival_tensors(h5_path, split="train", max_engines=40, censor_prob=0.15):
    """Extracts tensors and generates synthetic right-censoring."""
    with h5py.File(h5_path, "r") as f:
        if split not in f: return None, None, None
        grp = f[split]
        eng_ids = grp["engine_id"][:]
        ruls = grp["RUL"][:]
        unique_engines = np.unique(eng_ids)[:max_engines]
        
        X_all, Y_all, Censor_all = [], [], []
        
        for eng in unique_engines:
            idx = np.where(eng_ids == eng)[0]
            if len(idx) < WINDOW_SIZE: continue
            
            # Simulate censoring: 15% of engines are "pulled from service" early
            is_censored = np.random.rand() < censor_prob
            cutoff_idx = len(idx)
            
            if is_censored:
                # Calculate the random cutoff
                candidate_cutoff = int(len(idx) * np.random.uniform(0.5, 0.9))
                # FIX: Ensure we never drop below the 30-step window requirement
                cutoff_idx = max(WINDOW_SIZE, candidate_cutoff)
                
                # If the engine was too short to actually censor, revert the flag
                if cutoff_idx == len(idx):
                    is_censored = False
                    
            idx = idx[:cutoff_idx]
            
            X_raw = np.concatenate([
                np.nan_to_num(grp["sensors"][idx], nan=0.0),
                np.nan_to_num(grp["env"][idx], nan=0.0),
                np.nan_to_num(grp["causal_state"][idx], nan=0.0)
            ], axis=1).astype(np.float32)
            
            X_norm = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)
            X_norm = X_norm[np.argsort(ruls[idx])[::-1]] 
            X_win = sliding_window_view(X_norm, WINDOW_SIZE, axis=0).swapaxes(1, 2)
            
            # Actual RUL at each window
            Y_tgt = ruls[idx][np.argsort(ruls[idx])[::-1]][WINDOW_SIZE - 1:]
            
            # Mask: True if the engine was censored and hasn't failed yet
            Censor_mask = np.full(len(Y_tgt), is_censored, dtype=bool)
            
            X_all.append(X_win)
            Y_all.append(Y_tgt)
            Censor_all.append(Censor_mask)
            
    return (torch.tensor(np.concatenate(X_all), dtype=torch.float32), 
            torch.tensor(np.concatenate(Y_all), dtype=torch.float32),
            torch.tensor(np.concatenate(Censor_all), dtype=torch.bool))

class SurvivalWrapper:
    def __init__(self, base_model, device):
        self.encoder = base_model.eval().to(device)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        with torch.no_grad():
            dummy = torch.randn(2, WINDOW_SIZE, 55).to(device)
            op = torch.zeros(2, dtype=torch.long, device=device)
            out = self.encoder(dummy, op_setting=op, event_flag=op)
            latent_dim = out.get("latent", out.get("physics_preds", out["rul_log"])).shape[-1]

        self.survival = WeibullSurvivalHead(d_model=latent_dim).to(device)
        self.device = device

    def fit(self, train_loader, epochs=40, lr=5e-4):
        print(f"\nTraining Weibull Survival Head on Frozen Encoder ({epochs} Epochs)...")
        opt = torch.optim.Adam(self.survival.parameters(), lr=lr)
        self.survival.train()
        
        for epoch in range(epochs):
            for x, rul, censored in tqdm(train_loader, desc=f"  Epoch {epoch+1:02d}", leave=False, ncols=80, colour="cyan"):
                x, rul, censored = x.to(self.device), rul.to(self.device), censored.to(self.device)
                
                with torch.no_grad():
                    op = torch.zeros(len(x), dtype=torch.long, device=self.device)
                    out = self.encoder(x, op_setting=op, event_flag=op)
                    feat = out.get("latent", out.get("physics_preds", out["rul_log"])).detach()
                    
                lam, k = self.survival(feat)
                loss = weibull_mle_loss(lam, k, rul, censored)
                
                opt.zero_grad()
                loss.backward()
                opt.step()

    def early_warning(self, x, horizon_cycles=50.0):
        self.survival.eval()
        with torch.no_grad():
            op = torch.zeros(len(x), dtype=torch.long, device=self.device)
            out = self.encoder(x.to(self.device), op_setting=op, event_flag=op)
            feat = out.get("latent", out.get("physics_preds", out["rul_log"]))
            lam, k = self.survival(feat)
            
        p_fail = failure_probability(lam, k, horizon_cycles)
        return p_fail.cpu()

# ─── Main Execution ───────────────────────────────────────────────────────────

def run_survival_experiment(args):
    print(f"\n{'='*80}\n{'Experiment 2 (Post-Hoc): Weibull Survival Analysis & Early Warning':^80}\n{'='*80}")
    
    # 1. Load Data
    X_train, Y_train, C_train = extract_survival_tensors(args.data_path, split="train", max_engines=50)
    X_test, Y_test, C_test = extract_survival_tensors(args.data_path, split="test", max_engines=20)
    train_loader = DataLoader(TensorDataset(X_train, Y_train, C_train), batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Load Frozen Model
    try:
        from pinn_model import PINNModel
        model = PINNModel(max_rul=500.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True).get("model_state", torch.load(args.model_path, map_location=DEVICE, weights_only=True)), strict=False)
    except Exception as e:
        print(f"❌ Error loading model: {e}"); return

    # 3. Train Head
    wrapper = SurvivalWrapper(model, DEVICE)
    wrapper.fit(train_loader, epochs=40)
    
    # 4. Evaluate MRO Metrics
    print("\nEvaluating MRO Concordance Index (C-Index) & Early Warnings...")
    
    horizons = [50.0, 100.0]
    for h in horizons:
        # Predict risk: Probability of failure within 'h' cycles
        p_fail = wrapper.early_warning(X_test, horizon_cycles=h)
        
        # Events = 1 if engine actually failed within horizon AND wasn't censored
        events = ((Y_test <= h) & (~C_test)).float()
        
        c_index = concordance_index(p_fail, Y_test, events)
        
        # Calculate Alert Rates based on probability thresholds
        critical_alarms = (p_fail > 0.90).float().mean().item() * 100
        warning_alarms = (p_fail > 0.70).float().mean().item() * 100
        
        print(f"\n  [Horizon: {int(h)} Cycles]")
        print(f"  Concordance Index (C) : {c_index:.3f} (Target ≥ 0.85)")
        print(f"  CRITICAL Alarms (>90%): {critical_alarms:.1f}% of timeline")
        print(f"  WARNING Alarms (>70%) : {warning_alarms:.1f}% of timeline")
        
        if c_index >= 0.85:
            print(f"  ✅ VERDICT: Meets commercial aviation PHM standard for {int(h)}cy ranking.")
        else:
            print(f"  ❌ VERDICT: Ranking accuracy falls below 0.85 commercial threshold.")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/nasa_research/data/utdtb_v5.h5"))
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt"))
    run_survival_experiment(parser.parse_args())
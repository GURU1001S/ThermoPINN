"""
cert_conformal.py  ·  AeroMRO Digital Twin  ·  CS-E 1550 Certification Evidence
═══════════════════════════════════════════════════════════════════════════════
"""

import math, copy, random, torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch.amp import autocast
from task_sampler import DigitalTwinTaskSampler
from pinn_model import PINNModel
from train_maml_pinn import get_anil_params

def compute_empirical_coverage(nominal_alpha, residuals):
    """Calculates if the empirical coverage meets the nominal target."""
    q_hat = np.quantile(residuals, 1.0 - nominal_alpha)
    empirical_coverage = np.mean(residuals <= q_hat)
    return q_hat, empirical_coverage

def run_conformal_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*60)
    print(" 📜 Conformal Prediction Coverage Audit · CS-E 1550")
    print("="*60)
    
    sampler = DigitalTwinTaskSampler(h5_path="~/nasa_research/data/utdtb_v5.h5", window_size=30, stride=5, support_ratio=0.6, seed=42, device=device)
    _, test_tasks = sampler.held_out_split()
    
    # Isolate Validation tasks purely for Calibration
    val_tasks = [t for t in sampler._registry.keys() if t[0] == 'val']
    
    model = PINNModel(max_rul=sampler.max_rul, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=sampler.mean_rul_log).to(device)
    model.load_state_dict(torch.load(Path("~/nasa_research/checkpoints/best_model_v19.pt").expanduser(), map_location=device, weights_only=True)["model_state"])
    
    print(f"Building Conformal Calibrators from Validation Split (N={len(val_tasks)} tasks)...")
    
    # 1. Gather Calibration Non-Conformity Scores
    calib_residuals = []
    for tid in random.sample(val_tasks, min(40, len(val_tasks))):
        sup, qry = sampler.get_fast_task_tensors(tid)
        if qry is None or sup is None: continue
        
        # 10-Shot Calibration
        calib_model = copy.deepcopy(model)
        calib_model.train()
        adapt_opt = torch.optim.Adam(get_anil_params(calib_model), lr=0.01)
        for _ in range(10):
            idx = torch.randperm(sup["x"].shape[0], device=device)[:min(128, sup["x"].shape[0])]
            with autocast("cuda"):
                loss = F.smooth_l1_loss(calib_model(sup["x"][idx], op_setting=sup["op_setting"][idx], event_flag=sup["event_flag"][idx])["rul_log"].squeeze(-1), sup["rul_log"][idx].squeeze(-1))
            adapt_opt.zero_grad(); loss.backward(); adapt_opt.step()
            
        calib_model.eval()
        with torch.no_grad():
            out = calib_model(qry["x"], op_setting=qry["op_setting"], event_flag=qry["event_flag"])
            pred, true = out["rul_log"].squeeze(-1).cpu().numpy(), qry["rul_log"].cpu().numpy()
            calib_residuals.extend(np.abs(pred - true))
            
    calib_residuals = np.array(calib_residuals)
    
    # 2. Evaluate Empirical Coverage on Held-Out Test Set
    test_audit_tasks = random.sample(test_tasks, min(50, len(test_tasks)))
    print(f"Evaluating Coverage Guarantees on Held-Out Test Fleet (N={len(test_audit_tasks)} engines)...")
    
    test_residuals = []
    for tid in test_audit_tasks:
        sup, qry = sampler.get_fast_task_tensors(tid)
        if qry is None or sup is None: continue
        
        test_model = copy.deepcopy(model)
        test_model.train()
        adapt_opt = torch.optim.Adam(get_anil_params(test_model), lr=0.01)
        for _ in range(10):
            idx = torch.randperm(sup["x"].shape[0], device=device)[:min(128, sup["x"].shape[0])]
            with autocast("cuda"):
                loss = F.smooth_l1_loss(test_model(sup["x"][idx], op_setting=sup["op_setting"][idx], event_flag=sup["event_flag"][idx])["rul_log"].squeeze(-1), sup["rul_log"][idx].squeeze(-1))
            adapt_opt.zero_grad(); loss.backward(); adapt_opt.step()
            
        test_model.eval()
        with torch.no_grad():
            out = test_model(qry["x"], op_setting=qry["op_setting"], event_flag=qry["event_flag"])
            pred, true = out["rul_log"].squeeze(-1).cpu().numpy(), qry["rul_log"].cpu().numpy()
            test_residuals.extend(np.abs(pred - true))
            
    test_residuals = np.array(test_residuals)
    
    print("\n" + "-" * 60)
    print(f"{'Nominal Target':<15} | {'Empirical Coverage':<20} | {'Status'}")
    print("-" * 60)
    
    alphas = [0.20, 0.10, 0.05] # 80%, 90%, 95% confidence
    all_passed = True
    
    for alpha in alphas:
        target_pct = (1.0 - alpha) * 100
        # Get q_hat from Calibration set
        q_hat = np.quantile(calib_residuals, 1.0 - alpha)
        
        # Test if Test set residuals fall within q_hat
        empirical_coverage = np.mean(test_residuals <= q_hat) * 100
        
        margin = empirical_coverage - target_pct
        status = "✅ PASS" if empirical_coverage >= target_pct else "❌ FAIL"
        if empirical_coverage < target_pct: all_passed = False
        
        print(f"{target_pct:>5.1f}%          | {empirical_coverage:>16.1f}% ({margin:+.1f}%) | {status}")

    print("-" * 60)
    if all_passed:
        print("STATUS: CS-E 1550 PHM DISTRIBUTION-FREE COVERAGE CERTIFIED ✅")
    else:
        print("STATUS: COVERAGE FAILED (Model is overconfident) ❌")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_conformal_audit()
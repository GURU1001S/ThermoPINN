import math, copy, torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from torch.amp import autocast
from task_sampler import DigitalTwinTaskSampler
from pinn_model import PINNModel
from physics_loss import NASAAsymmetricScore
from train_maml_pinn import get_anil_params

def run_strata_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[Audit] Initializing Flight Envelope Strata Audit (NASA Score Normalized)...")
    
    sampler = DigitalTwinTaskSampler(h5_path="~/nasa_research/data/utdtb_v5.h5", window_size=30, stride=5, support_ratio=0.6, seed=42, device=device)
    _, test_tasks = sampler.held_out_split()
    
    model = PINNModel(max_rul=sampler.max_rul, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=sampler.mean_rul_log).to(device)
    model.load_state_dict(torch.load(Path("~/nasa_research/checkpoints/best_model_v19.pt").expanduser(), map_location=device, weights_only=True)["model_state"])
    
    nasa_sc = NASAAsymmetricScore(clamp_range=50.0)
    strata_results = defaultdict(list)
    print("Evaluating entire test fleet across Flight Envelope strata...")
    
    for tid in test_tasks:
        sup, qry = sampler.get_fast_task_tensors(tid)
        if qry is None or sup is None: continue
        
        adapted_model = copy.deepcopy(model)
        adapted_model.train()
        adapt_opt = torch.optim.Adam(get_anil_params(adapted_model), lr=0.01)
        
        calibration_shots = 15
        for step_i in range(calibration_shots):
            cos_factor = 0.5 * (1 + math.cos(math.pi * step_i / max(1, calibration_shots - 1)))
            for pg in adapt_opt.param_groups: pg['lr'] = 0.01 * (0.05 + 0.95 * cos_factor)
            
            idx = torch.randperm(sup["x"].shape[0], device=device)[:min(128, sup["x"].shape[0])]
            with autocast("cuda"):
                out = adapted_model(sup["x"][idx], op_setting=sup["op_setting"][idx], event_flag=sup["event_flag"][idx])
                loss = F.smooth_l1_loss(out["rul_log"].squeeze(-1), sup["rul_log"][idx].squeeze(-1))
            adapt_opt.zero_grad(); loss.backward(); adapt_opt.step()
            
        adapted_model.eval()
        op_code = int(qry["op_setting"][0].item())
        with torch.no_grad():
            with torch.autocast("cuda"):
                out = adapted_model(qry["x"], op_setting=qry["op_setting"], event_flag=qry["event_flag"])
            
        pred_cy, true_cy = torch.expm1(out["rul_log"].squeeze(-1)), torch.expm1(qry["rul_log"])
        rmse = float(torch.sqrt(F.mse_loss(pred_cy, true_cy)).item())
        nasa = float(nasa_sc(out["rul_log"].detach(), qry["rul_log"].unsqueeze(-1)).item())
        strata_results[op_code].append({"rmse": rmse, "nasa": nasa, "count": len(pred_cy)})

    print("\n" + "="*75)
    print(f" {'Flight Envelope Code':<25} | {'Engines':<10} | {'RMSE':<10} | {'NASA Score':<10}")
    print("="*75)
    
    all_nasas = []
    for op_code in sorted(strata_results.keys()):
        results = strata_results[op_code]
        avg_rmse, avg_nasa = np.mean([r["rmse"] for r in results]), np.mean([r["nasa"] for r in results])
        total_eng = sum([r["count"] for r in results])
        all_nasas.append(avg_nasa)
        regime = "Standard Cruise" if op_code < 10 else ("High-Alt / High-Mach" if op_code < 20 else "Extreme Sea-Level")
        print(f" Op-Code {op_code:<2} ({regime:<14}) | {total_eng:<10} | {avg_rmse:<10.2f} | {avg_nasa:<10.2f}")
    
    print("-" * 75)
    if not all_nasas: print("❌ STATUS: FAILED (No data)")
    else:
        # 🚨 FIX: Evaluate NASA Score Variance, not raw RMSE variance!
        max_variance = np.max(all_nasas) - np.min(all_nasas)
        if max_variance < 35.0: print(f"✅ STATUS: ENVELOPE UNBIASED (Max NASA Score variance: {max_variance:.1f})")
        else: print(f"❌ STATUS: BIASED (Max NASA Score variance: {max_variance:.1f})")
    print("="*75)

if __name__ == "__main__": run_strata_audit()
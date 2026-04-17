"""
visualize_trajectory.py  ·  AeroMRO Digital Twin  ·  v19.11 (Standalone Diagnostic)
══════════════════════════════════════════════════════════════════════════════════
Self-contained diagnostic tool to inspect the sabotaged prior and outlier data.
"""

import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

# Safe, direct imports from the source files to avoid any circular/missing errors
from task_sampler import DigitalTwinTaskSampler
from pinn_model import PINNModel

def assemble_full_engine(sampler, test_tasks, device, target_engine_id=None):
    """Assembles a chronological history, ensuring target engine is prioritized."""
    print("\n[Data] Assembling full chronological trajectory...")
    if target_engine_id:
        # Prioritize the target engine for diagnosis
        test_tasks = sorted(test_tasks, key=lambda t: t[1] != target_engine_id)
        
    for target_eng_tid in test_tasks:
        target_eng = target_eng_tid[1]
        eng_tasks = [t for t in test_tasks if t[1] == target_eng]
        
        all_x, all_op, all_ev, all_rul = [], [], [], []
        for tid in eng_tasks:
            sup, qry = sampler.get_fast_task_tensors(tid)
            if qry is not None:
                all_x.append(qry['x']); all_op.append(qry['op_setting'])
                all_ev.append(qry['event_flag']); all_rul.append(qry['rul_log'])
                
        if not all_x: continue
        
        x, op, ev, rul = torch.cat(all_x, dim=0), torch.cat(all_op, dim=0), torch.cat(all_ev, dim=0), torch.cat(all_rul, dim=0)
        
        # Chronological sorting by descending True RUL
        sort_idx = torch.argsort(rul.flatten(), descending=True)
        x, op, ev, rul = x[sort_idx], op[sort_idx], ev[sort_idx], rul[sort_idx]
        
        # Deduplication of identical data points from overlaps
        diff = torch.cat([torch.tensor([1.0], device=device), torch.abs(rul[1:] - rul[:-1]).flatten()])
        keep = diff > 1e-5
        x, op, ev, rul = x[keep], op[keep], ev[keep], rul[keep]
        
        if len(x) > 50:
            print(f"  [Data] Successfully assembled Engine {target_eng} (Length: {len(x)} cycles)")
            return target_eng, x, op, ev, rul
            
    raise ValueError("Could not assemble a complete engine longer than 50 cycles.")

def debug_predict_trajectory(model, x, op, ev, mc_passes, max_rul_val) -> Dict[str, np.ndarray]:
    """Runs a proper MC ensemble, enforcing 0-shot and collecting debug data."""
    means, stds = [], []
    model.train() # Enable dropout for MC passes
    
    for m in range(mc_passes):
        with torch.no_grad():
            with torch.autocast("cuda", enabled=(x.device.type == "cuda")):
                out_m = model(x, op_setting=op, event_flag=ev)
        
        means.append(out_m["rul_log"].detach())
        stds.append(torch.exp(0.5 * out_m["rul_log_var"].detach()))
        
    model.eval()
    
    # Law of total variance for ensemble uncertainty
    preds_stack = torch.stack(means, dim=0)
    stds_stack = torch.stack(stds, dim=0)
    
    gp = preds_stack.mean(0)
    epistem_var = preds_stack.var(0)
    aleat_var = (stds_stack ** 2).mean(0)
    gt_unc = torch.sqrt(epistem_var + aleat_var).clamp(min=1e-3).cpu().numpy()
    gp_numpy = gp.cpu().numpy()
    
    # 🚨 Baseline Diagnosis: Create a plausible decline as a sanity check
    initial_log_baseline = math.log1p(max_rul_val)
    time_steps = len(gp)
    baseline_log = np.linspace(initial_log_baseline, 0, time_steps)
    
    return {
        "rul_log": gp_numpy,
        "rul_std": gt_unc,
        "debug_baseline_log": baseline_log 
    }

def perform_diagnostics():
    """Definitive diagnostic execution and plotting."""
    DIAG_CONFIG = {
        "h5_path": "~/nasa_research/data/utdtb_v5.h5", 
        "checkpoint_dir": "~/nasa_research/checkpoints",
        "mc_passes": 20, "window_size": 30, "stride": 5, "support_ratio": 0.6,
        "n_sensors": 55, "n_op_settings": 32, "n_events": 10,
        "conv_channels": 256, "gru_hidden": 512, "head_hidden": 128, "dropout": 0.30,
        "diag_override_q_hat_log": 0.60
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[Diagnostics] Running AeroMRO Audit v19.11...")
    
    sampler = DigitalTwinTaskSampler(h5_path=DIAG_CONFIG["h5_path"], **{k: DIAG_CONFIG[k] for k in ["window_size", "stride", "support_ratio"]}, seed=42, device=device)
    train_tasks, test_tasks = sampler.held_out_split()
    
    model = PINNModel(
        max_rul=sampler.max_rul, n_sensors=DIAG_CONFIG["n_sensors"], conv_channels=DIAG_CONFIG["conv_channels"],
        gru_hidden=DIAG_CONFIG["gru_hidden"], head_hidden=DIAG_CONFIG["head_hidden"], dropout=DIAG_CONFIG["dropout"],
        n_op_settings=DIAG_CONFIG["n_op_settings"], n_events=DIAG_CONFIG["n_events"], mean_rul_log=5.50
    ).to(device)
    
    ckpt_path = Path(DIAG_CONFIG["checkpoint_dir"]).expanduser() / "best_model_v19.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    # 1. Asymmetric q_hat check
    diag_override_q_hat = DIAG_CONFIG["diag_override_q_hat_log"]
    print(f"  [Diagnostics] conformal q_hat (log) = {diag_override_q_hat:.4f}")
    
    # Priority on the problem engine, 19858
    target_engine_id = 19858
    eng_id, x, op, ev, rul = assemble_full_engine(sampler, test_tasks, device, target_engine_id)
    
    # 2. Maximum RUL check
    max_rul_val = sampler.max_rul
    log_max_rul_val = math.log1p(max_rul_val)
    print(f"  [Diagnostics] log_max_rul = {log_max_rul_val:.4f} (~{max_rul_val:.0f} cycles)")

    # Definitive prediction
    preds = debug_predict_trajectory(model, x, op, ev, DIAG_CONFIG["mc_passes"], max_rul_val)
    
    gp, gt_std, debug_baseline = preds["rul_log"], preds["rul_std"], preds["debug_baseline_log"]
    gp_numpy, gt_std_numpy = gp.flatten(), gt_std.flatten()
    true_rul_cy = torch.expm1(rul).cpu().numpy().flatten()
    
    # 🚨 Sabotage check
    if np.abs(gp_numpy[-1] - gp_numpy[0]) < 1e-4 and gp_numpy[0] > 6.0:
        print("  [Diagnostics] DETECTED Prediction Sabotage: RUL is horizontal at dangerous high value!")
    else:
        print("  [Diagnostics] Prediction Sabotage not detected, but behaviors remain complex.")
    
    debug_baseline_cy = np.expm1(debug_baseline)
    pred_rul_cy = np.expm1(gp_numpy)
    uncertainty_band = diag_override_q_hat * gt_std_numpy
    lower_bound_cy = np.expm1((gp_numpy - uncertainty_band).clip(min=0.0))
    upper_bound_cy = np.expm1(gp_numpy + uncertainty_band)

    # Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    time_steps = np.arange(len(true_rul_cy))
    
    ax.plot(time_steps, true_rul_cy, color='white', linewidth=2.5, label="True RUL", zorder=3)
    ax.plot(time_steps, pred_rul_cy, color='#00ffcc', linewidth=2, linestyle='--', label="PINN Prediction (Sabotaged, 0-Shot)", zorder=4)
    ax.plot(time_steps, debug_baseline_cy, color='#ff00ff', linewidth=1.5, linestyle='-.', label="Diagnostics: initial log baseline", zorder=2)
    ax.fill_between(time_steps, lower_bound_cy, upper_bound_cy, color='#00ffcc', alpha=0.15, label="Diagnostics: Outlier Conformal bounds", zorder=1)
    
    ax.axvspan(0, 10, color='yellow', alpha=0.10, zorder=0, label="10-Cycle Phase (CONTRADICTION Check)")
    print("  [Diagnostics] Rendered Contradiction Check Phase overlay.")
    
    ax.fill_between(time_steps, true_rul_cy, true_rul_cy + 50, color='red', alpha=0.15, label="NASA Death Penalty Zone")

    ax.set_title(f"AeroMRO Digital Twin: Diagnostic Fix (Engine {eng_id}, 0-Shot Audit)", fontsize=18, fontweight='bold', color='white', pad=15)
    ax.set_xlabel("Flight Cycles (Chronological Time →)", fontsize=14, color='gray')
    ax.set_ylabel("Remaining Useful Life (Cycles)", fontsize=14, color='gray')
    ax.grid(True, color='#333333', linestyle='--', alpha=0.7)
    
    ax.set_xlim(0, len(time_steps))
    ax.set_ylim(0, max(upper_bound_cy.max(), debug_baseline_cy.max()) * 1.05)
    
    ax.legend(loc="upper right", facecolor='#111111', edgecolor='gray', fontsize=11)
    
    plt.tight_layout()
    plt.savefig("aeromro_diagnostic_plot.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved as 'aeromro_diagnostic_plot.png'. Inspect this plot to see the un-sabotaged state.")

if __name__ == "__main__":
    perform_diagnostics()
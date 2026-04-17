"""
ensemble_evaluate.py
────────────────────
Evaluates a Hybrid Deep Ensemble + MC Dropout Meta-Learned PINN.
Combines Aleatoric (learned variance) and Epistemic (ensemble + dropout disagreement) 
uncertainty using the Law of Total Variance, followed by Isotonic Recalibration.
"""

import torch
import numpy as np
import torch.nn as nn
from pathlib import Path

from task_sampler import NCMAPSSTaskSampler
from pinn_model import PINNModel
from physics_loss import NASAAsymmetricScore
from calibration import CalibrationEvaluator

def enable_mc_dropout(model: nn.Module):
    """Locks normalization layers but enables Dropout for Epistemic sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def evaluate_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Ensemble] Device : {device}")

    ckpt_dir = Path("~/nasa_research/checkpoints").expanduser()
    base_ckpt = torch.load(ckpt_dir / "best_model_seed42.pt", map_location=device, weights_only=False)
    cfg = base_ckpt["config"]

    sampler = NCMAPSSTaskSampler(
        h5_path=cfg["h5_path"], window_size=cfg["window_size"],
        support_ratio=cfg["support_ratio"], batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], seed=42, 
    )
    _, test_tasks = sampler.held_out_split(cfg["holdout_fraction"])

    seeds = [42, 43, 44]
    models = []
    print("\nLoading Hybrid MC-Ensemble Members...")
    for s in seeds:
        model_path = ckpt_dir / f"best_model_seed{s}.pt"
        model = PINNModel(
            n_sensors=cfg["n_sensors"], window_size=cfg["window_size"],
            conv_channels=cfg["conv_channels"], gru_hidden=cfg["gru_hidden"],
            head_hidden=cfg["head_hidden"], dropout=cfg["dropout"], max_rul=cfg["max_rul"],
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state"])
        
        # 🚨 FIX: Enable MC Dropout to create a virtual 30-model ensemble
        enable_mc_dropout(model)
        models.append(model)
        print(f"  ✓ Model Seed {s} loaded.")

    nasa_scorer = NASAAsymmetricScore()
    calibrator = CalibrationEvaluator(n_bins=15)
    mc_passes = 10 # 3 models * 10 passes = 30 virtual models

    print("\n── Hybrid MC-Ensemble 0-Shot Evaluation ──")
    
    all_ensemble_preds = []
    all_ensemble_stds = []
    all_targets = []

    with torch.no_grad():
        for task_id in test_tasks:
            _, query_loader = sampler.get_task_loaders(task_id)
            
            for x_batch, y_batch in query_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                batch_means = []
                batch_vars = []
                
                # Forward pass through all ensemble members AND MC passes
                for model in models:
                    for _ in range(mc_passes):
                        out = model(x_batch)
                        batch_means.append(out["rul"])
                        batch_vars.append(torch.exp(out["rul_log_var"]))
                
                stacked_means = torch.stack(batch_means)
                stacked_vars = torch.stack(batch_vars)
                
                ensemble_mean = stacked_means.mean(dim=0)
                mean_of_vars = stacked_vars.mean(dim=0)
                var_of_means = ((stacked_means - ensemble_mean) ** 2).mean(dim=0)
                
                ensemble_var = mean_of_vars + var_of_means
                ensemble_std = torch.sqrt(ensemble_var)
                
                all_ensemble_preds.append(ensemble_mean.cpu())
                all_ensemble_stds.append(ensemble_std.cpu())
                all_targets.append(y_batch.cpu())

    preds_cat = torch.cat(all_ensemble_preds, dim=0)
    stds_cat = torch.cat(all_ensemble_stds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    nasa_score = nasa_scorer(preds_cat, targets_cat).item() / len(test_tasks)
    calibrator.update(preds_cat, stds_cat, targets_cat)
    
    raw_ece = calibrator.compute_ece(mode="raw")
    
    optimal_T = calibrator.get_optimal_temperature()
    temp_ece = calibrator.compute_ece(mode="temperature", temperature=optimal_T)
    
    # 🚨 FIX: Calculate the Non-Linear Isotonic Regression ECE
    iso_ece = calibrator.compute_ece(mode="isotonic")
    iso_sharpness = calibrator.get_sharpness(mode="isotonic")

    print(f"  NASA Score:                 {nasa_score:>6.2f}")
    print(f"  Raw ECE:                    {raw_ece:>6.4f}")
    print(f"  Temperature ECE (T={optimal_T:.2f}):  {temp_ece:>6.4f}")
    print(f"  Isotonic Calibrated ECE:    {iso_ece:>6.4f} 🔥")
    print(f"  Final Sharpness:            {iso_sharpness:>6.2f}")
    
    calibrator.plot_reliability_diagram(
        mode="isotonic",
        save_path="~/nasa_research/isotonic_reliability.png",
        title=f"Isotonic Hybrid Ensemble Reliability"
    )
    print("\n[Success] Isotonic calibration complete. Check isotonic_reliability.png!")

if __name__ == "__main__":
    evaluate_ensemble()
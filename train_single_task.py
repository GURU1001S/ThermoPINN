"""
train_single_task.py
────────────────────
Phase 2 Validation: Single-Task PINN Backbone Sanity Check.
Isolates the neural network and physics loss from the MAML loop to verify:
  1. VRAM stability (< 6GB) with live monitoring.
  2. PDE Autograd functionality (active in BOTH train and eval).
  3. FP16 stability with automatic NaN tripwires.
  4. Adaptive lambda weighting convergence.
"""

import math
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import random
import numpy as np
from tqdm import tqdm

from task_sampler import NCMAPSSTaskSampler
from pinn_model import PINNModel
from physics_loss import CompositePINNLoss, NASAAsymmetricScore, rank_physics_violations

# ── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    "h5_path": "~/nasa_research/data/N-CMAPSS_DS02-006.h5",
    "seed": 42,
    "window_size": 30,
    "batch_size": 16, 
    "lr": 1e-3,
    "epochs": 50,
    "grad_clip_norm": 1.0,
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"── Phase 2: Single Task PINN Validation ──")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total VRAM: {total_vram:.1f} GB")

    # 1. Initialize Sampler & Extract ONE Task
    sampler = NCMAPSSTaskSampler(
        h5_path=CONFIG["h5_path"],
        window_size=CONFIG["window_size"],
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"]
    )
    
    target_task = sampler.task_ids[0]
    print(f"\n[Data] Isolating Task: Unit {target_task[0]}, Flight Class {target_task[1]}")
    support_loader, query_loader = sampler.get_task_loaders(target_task)

    # 2. Initialize Model & Training Tools
    model = PINNModel(window_size=CONFIG["window_size"]).to(device)
    print(f"[Model] Parameter count: {model.count_parameters():,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    scaler = GradScaler('cuda')
    loss_fn = CompositePINNLoss()
    nasa_scorer = NASAAsymmetricScore()

    # 3. Training Loop
    print("\n[Train] Starting 50-epoch burn-in test...")
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        nan_detected = False
        
        for x_batch, y_batch in support_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            # PDE Requirement
            x_batch.requires_grad_(True) 

            optimizer.zero_grad()

            with autocast(device_type=device.type):
                output = model(x_batch)
                losses = loss_fn.compute(output, y_batch, x_batch)
                loss = losses["total"]

            # 🚨 FIX 3: Immediate NaN/Inf Tripwire
            if not math.isfinite(loss.item()):
                print(f"\n⚠️ FATAL: NaN/Inf loss detected at epoch {epoch}. Stopping training.")
                nan_detected = True
                break

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        if nan_detected:
            break

        avg_train_loss = total_train_loss / len(support_loader)

        # 4. Evaluation Loop (Every 10 epochs)
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            all_preds, all_trues = [], []
            
            # 🚨 FIX 2: Proper tracking of ALL loss components
            eval_losses = {"total": 0.0, "data": 0.0, "brayton": 0.0, "creep": 0.0, "degrad": 0.0}
            
            # 🚨 FIX 4: VRAM tracking
            if device.type == "cuda":
                mem_used = torch.cuda.memory_allocated() / 1e9
            
            with torch.no_grad():
                for x_val, y_val in query_loader:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    
                    # 🚨 FIX 1: Enable physics eval for accurate violation ranking
                    # We use torch.enable_grad() just for the inputs to compute the PDE
                    with torch.enable_grad():
                        x_val.requires_grad_(True)
                        with autocast(device_type=device.type):
                            out_val = model(x_val)
                            val_loss_dict = loss_fn.compute(out_val, y_val, x_val)
                    
                    all_preds.append(out_val["rul"].detach().cpu())
                    all_trues.append(y_val.cpu())
                    
                    for k in eval_losses:
                        eval_losses[k] += val_loss_dict[k].item()

            preds_cat = torch.cat(all_preds)
            trues_cat = torch.cat(all_trues)
            nasa_score = nasa_scorer(preds_cat, trues_cat).item()
            
            # Average the eval losses
            for k in eval_losses:
                eval_losses[k] /= len(query_loader)
                
            lams = loss_fn.get_lambda_log()
            
            print(f"\n[Epoch {epoch:02d}] Train Loss: {avg_train_loss:.4f} | Eval NASA Score: {nasa_score:.2f}")
            if device.type == "cuda":
                print(f"  VRAM Used: {mem_used:.2f} GB (Target: < 6.0 GB)")
                
            # 🚨 FIX 5: Convergence signal tracking
            print(f"  Eval Losses → Data: {eval_losses['data']:.4f} | Brayton: {eval_losses['brayton']:.4f} | Creep: {eval_losses['creep']:.4f} | PDE: {eval_losses['degrad']:.4f}")
            print(f"  Adaptive λ  → Data: {lams.get('lambda_data', 0):.2f} | Brayton: {lams.get('lambda_brayton', 0):.2f} | Creep: {lams.get('lambda_creep', 0):.2f} | PDE: {lams.get('lambda_degrad', 0):.2f}")
            
            # Accurate Physics Violation Ranking
            mock_loss_dict = {k: torch.tensor(v) for k, v in eval_losses.items()}
            mock_loss_dict["lambdas"] = lams
            violations = rank_physics_violations(mock_loss_dict)
            if violations:
                print(f"  Top Physics Violation: {violations[0][0]} ({violations[0][1]:.4f})")

    print("\n[Success] Phase 2 validation script complete.")

if __name__ == "__main__":
    main()
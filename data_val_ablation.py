import copy, random, torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torch.amp import autocast
from task_sampler import DigitalTwinTaskSampler
from pinn_model import PINNModel
from train_maml_pinn import get_anil_params

def run_dataset_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[Validation] Initializing Dataset Feature Ablation Study...")
    
    sampler = DigitalTwinTaskSampler(h5_path="~/nasa_research/data/utdtb_v5.h5", window_size=30, stride=5, support_ratio=0.6, seed=42, device=device)
    _, test_tasks = sampler.held_out_split()
    
    model = PINNModel(max_rul=sampler.max_rul, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=sampler.mean_rul_log).to(device)
    model.load_state_dict(torch.load(Path("~/nasa_research/checkpoints/best_model_v19.pt").expanduser(), map_location=device, weights_only=True)["model_state"])
    
    eval_tasks = random.sample(test_tasks, min(30, len(test_tasks)))
    
    scenarios = {
        "1. UTDTB v5 (All 55 Features)": lambda x: x,
        "2. No Latent Physics (Zero cols 36-54)": lambda x: torch.cat([x[:, :, :36], torch.zeros_like(x[:, :, 36:])], dim=-1),
        "3. No Environment (Zero cols 20-35)": lambda x: torch.cat([x[:, :, :20], torch.zeros_like(x[:, :, 20:36]), x[:, :, 36:]], dim=-1),
        "4. C-MAPSS Equivalent (Sensors Only)": lambda x: torch.cat([x[:, :, :20], torch.zeros_like(x[:, :, 20:])], dim=-1)
    }

    print("-" * 65)
    print(f"{'Data Scenario':<40} | {'Resulting RMSE'}")
    print("-" * 65)

    for name, mask_fn in scenarios.items():
        all_preds, all_trues = [], []
        
        for tid in eval_tasks:
            sup, qry = sampler.get_fast_task_tensors(tid)
            if qry is None or sup is None: continue
            
            # Apply feature mask to data
            masked_sup_x = mask_fn(sup["x"])
            masked_qry_x = mask_fn(qry["x"])
            
            adapted_model = copy.deepcopy(model)
            adapted_model.train()
            adapt_opt = torch.optim.Adam(get_anil_params(adapted_model), lr=0.01)
            
            for _ in range(5):
                idx = torch.randperm(masked_sup_x.shape[0], device=device)[:min(128, masked_sup_x.shape[0])]
                with autocast("cuda"):
                    loss = F.smooth_l1_loss(adapted_model(masked_sup_x[idx], op_setting=sup["op_setting"][idx], event_flag=sup["event_flag"][idx])["rul_log"].squeeze(-1), sup["rul_log"][idx].squeeze(-1))
                adapt_opt.zero_grad(); loss.backward(); adapt_opt.step()
            
            adapted_model.eval()
            with torch.no_grad():
                with autocast("cuda"):
                    out = adapted_model(masked_qry_x, op_setting=qry["op_setting"], event_flag=qry["event_flag"])
            
            all_preds.append(out["rul_log"].cpu().flatten())
            all_trues.append(qry["rul_log"].cpu().flatten())
            
        gp_cy = np.expm1(torch.cat(all_preds).numpy())
        gt_cy = np.expm1(torch.cat(all_trues).numpy())
        rmse = np.sqrt(np.mean((gp_cy - gt_cy)**2))
        
        print(f"{name:<40} | {rmse:<8.1f}")
        
    print("-" * 65)

if __name__ == "__main__": run_dataset_ablation()
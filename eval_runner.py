"""
eval_runner.py  ·  AeroMRO Digital Twin  ·  v19.6 (Conformal Fix)
═════════════════════════════════════════════════════════════════
"""

import torch
import copy
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import math
import numpy as np

from train_maml_pinn import (
    CONFIG, PINNModel, DigitalTwinTaskSampler, CompositePINNLoss, 
    NASAAsymmetricScore, get_anil_params, augment_gpu, set_seed, nasa_shaped_loss
)

class ConformalECECalibrator:
    """Split conformal prediction for guaranteed, distribution-free ECE."""
    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self._scores = None

    def calibrate(self, cal_means: torch.Tensor, cal_trues: torch.Tensor):
        self._scores = (cal_means.float().flatten() - cal_trues.float().flatten()).abs().numpy()

    def compute_conformal_ece(self, test_means: torch.Tensor, test_trues: torch.Tensor, n_bins: int = 20) -> float:
        if self._scores is None: return 1.0
        m = test_means.float().flatten().numpy()
        t = test_trues.float().flatten().numpy()
        
        target_coverages = np.linspace(0.05, 0.95, n_bins)
        empirical_coverages = []
        
        for target_cov in target_coverages:
            n_cal = len(self._scores)
            # 🚨 FIX: Calculate the exact quantile matching the target coverage
            q_level = min(float(np.ceil(target_cov * (n_cal + 1)) / n_cal), 1.0)
            q_alpha = float(np.quantile(self._scores, q_level))
            
            lower, upper = m - q_alpha, m + q_alpha
            cov = np.mean((t >= lower) & (t <= upper))
            empirical_coverages.append(cov)
            
        return float(np.mean(np.abs(np.array(empirical_coverages) - target_coverages)))

def run_eval_only():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Eval] Loading Model & Data on {device}...")

    sampler = DigitalTwinTaskSampler(
        h5_path=CONFIG["h5_path"], window_size=CONFIG["window_size"], stride=CONFIG["stride"],
        support_ratio=CONFIG["support_ratio"], seed=CONFIG["seed"], device=device,
    )
    _, test_tasks = sampler.held_out_split()

    model = PINNModel(
        max_rul=sampler.max_rul, n_sensors=CONFIG["n_sensors"], conv_channels=CONFIG["conv_channels"],
        gru_hidden=CONFIG["gru_hidden"], head_hidden=CONFIG["head_hidden"], dropout=CONFIG["dropout"],
        n_op_settings=CONFIG["n_op_settings"], n_events=CONFIG["n_events"], mean_rul_log=sampler.mean_rul_log,
    ).to(device)

    ckpt_path = Path(CONFIG["checkpoint_dir"]).expanduser() / "best_model_v19.pt"
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found. Train the model first.")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    nasa_sc = NASAAsymmetricScore(clamp_range=50.0)
    conf_calibrator = ConformalECECalibrator()
    
    adapted_model = PINNModel(
        max_rul=sampler.max_rul, n_sensors=CONFIG["n_sensors"], conv_channels=CONFIG["conv_channels"],
        gru_hidden=CONFIG["gru_hidden"], head_hidden=CONFIG["head_hidden"], dropout=CONFIG["dropout"],
        n_op_settings=CONFIG["n_op_settings"], n_events=CONFIG["n_events"], mean_rul_log=sampler.mean_rul_log,
    ).to(device)

    import random
    eval_tasks = random.sample(test_tasks, min(CONFIG["eval_tasks"], len(test_tasks)))

    print("\n── Strict MAML Evaluation (lower = better) ──")
    print(f"   {'k':>5} | {'RMSE':>8} | {'NASA':>10} | {'log-N':>7} | {'ECE':>7} | {'Pred':>7} | {'True':>7}")

    for k_batches in CONFIG["fewshot_batches"]:
        all_preds, all_trues = [], []

        for task_id in tqdm(eval_tasks, desc=f"{k_batches}-shot", leave=False):
            sup, qry = sampler.get_fast_task_tensors(task_id)
            if sup is None or qry is None: continue
            adapted_model.load_state_dict(copy.deepcopy(model.state_dict()))

            if k_batches > 0:
                adapted_model.train()
                eval_lr_base = CONFIG["inner_lr"] * CONFIG.get("eval_inner_lr_factor", 0.25)
                adapt_opt = torch.optim.Adam(get_anil_params(adapted_model), lr=eval_lr_base)
                scaler_e = GradScaler("cuda", enabled=(device.type == "cuda"))
                
                for step_i in range(k_batches):
                    cos_factor = 0.5 * (1 + math.cos(math.pi * step_i / max(1, k_batches - 1)))
                    current_lr = eval_lr_base * (0.05 + 0.95 * cos_factor)
                    for pg in adapt_opt.param_groups: pg['lr'] = current_lr

                    num = sup["x"].shape[0]
                    idx = torch.randperm(num, device=device)[:min(CONFIG["batch_size"], num)]
                    xb, yb = sup["x"][idx], sup["rul_log"][idx]
                    op_b, ev_b = sup["op_setting"][idx], sup["event_flag"][idx]
                    x_a, y_a = augment_gpu(xb, yb, device)
                    xc, yc = torch.cat([xb, x_a], dim=0), torch.cat([yb, y_a], dim=0)
                    op_c, ev_c = torch.cat([op_b, op_b], dim=0), torch.cat([ev_b, ev_b], dim=0)

                    with autocast("cuda"):
                        p, t = adapted_model(xc, op_setting=op_c, event_flag=ev_c)["rul_log"].squeeze(-1), yc.squeeze(-1)
                        loss = nasa_shaped_loss(p, t)

                    adapt_opt.zero_grad()
                    scaler_e.scale(loss).backward()
                    scaler_e.unscale_(adapt_opt)
                    torch.nn.utils.clip_grad_norm_(get_anil_params(adapted_model), 0.3)
                    scaler_e.step(adapt_opt)
                    scaler_e.update()

            adapted_model.train() 
            with torch.no_grad():
                mc_preds = []
                for _ in range(CONFIG["mc_passes"]):
                    with autocast("cuda"):
                        out_pass = adapted_model(qry["x"], op_setting=qry["op_setting"], event_flag=qry["event_flag"])
                        mc_preds.append(out_pass["rul_log"].detach())
                
                mc_preds_stack = torch.stack(mc_preds, dim=0)
                mean_log = mc_preds_stack.mean(0)
                
                health = out_pass["health"].detach()
                health_ceiling_log = torch.log1p(sampler.max_rul * health.squeeze(-1) * 1.5)
                max_log_val = math.log1p(sampler.max_rul)
                max_bound = torch.minimum(torch.tensor(max_log_val, device=device), health_ceiling_log).expand_as(mean_log.squeeze(-1))
                mean_log = torch.minimum(mean_log.squeeze(-1).clamp(min=0.0), max_bound).unsqueeze(-1)
                
            adapted_model.eval()

            all_preds.append(mean_log.cpu().flatten())
            all_trues.append(qry["rul_log"].cpu().flatten())

        if not all_preds: continue
        
        gp, gt = torch.cat(all_preds), torch.cat(all_trues)
        
        # 🚨 FIX: Shuffle the distribution so calibration and test are identical
        perm = torch.randperm(len(gp))
        gp_shuf, gt_shuf = gp[perm], gt[perm]
        
        split_idx = len(gp_shuf) // 2
        cal_preds, test_preds = gp_shuf[:split_idx], gp_shuf[split_idx:]
        cal_trues, test_trues = gt_shuf[:split_idx], gt_shuf[split_idx:]
        
        conf_calibrator.calibrate(cal_preds, cal_trues)
        final_ece = conf_calibrator.compute_conformal_ece(test_preds, test_trues)

        # Calculate RMSE/NASA across the full set for consistency
        gp_cy, gt_cy = torch.expm1(gp), torch.expm1(gt)
        nasa = float(nasa_sc(gp.unsqueeze(-1), gt.unsqueeze(-1)).item())

        print(f"   {k_batches:>5} | {float(torch.sqrt(F.mse_loss(gp_cy, gt_cy)).item()):>8.2f} | {nasa:>10.2f} | "
              f"{math.log1p(nasa):>7.3f} | {final_ece:>7.4f} | {gp_cy.mean().item():>7.1f} | {gt_cy.mean().item():>7.1f}")

if __name__ == "__main__":
    run_eval_only()
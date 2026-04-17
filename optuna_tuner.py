"""
optuna_tuner.py  ·  AeroMRO Digital Twin
════════════════════════════════════════
Upgrades:
  • Maps strictly to v16 keys.
"""

import optuna
import copy
from train_maml_pinn import train, CONFIG

def objective(trial: optuna.Trial) -> float:
    trial_cfg = copy.deepcopy(CONFIG)
    trial_cfg["inner_lr"]   = trial.suggest_float("inner_lr", 1e-4, 5e-3, log=True)
    trial_cfg["head_lr"]    = trial.suggest_float("head_lr", 5e-5, 5e-4, log=True)
    trial_cfg["dropout"]    = trial.suggest_float("dropout", 0.10, 0.35)
    trial_cfg["base_lambdas"]["damage_mono"] = trial.suggest_float("lam_dmg", 0.1, 1.0)
    trial_cfg["n_meta_epochs"]  = 40
    
    print(f"\n[Optuna] Starting Trial {trial.number}")
    try: return train(trial_cfg, return_best=True)
    except Exception as e:
        print(f"[Optuna] Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="AeroMRO_PINN_Tuning")
    study.optimize(objective, n_trials=20)
    print(f"\n[Optuna] Study Complete. Best NASA: {study.best_trial.value:.2f}")
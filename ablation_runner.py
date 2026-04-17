"""
ablation_runner.py  ·  AeroMRO Digital Twin  ·  v18.1 (Ablation Fixed)
══════════════════════════════════════════════════════════════════════
"""

import copy
import pandas as pd
from train_maml_pinn import train, CONFIG

def run_ablation():
    results = []
    experiments = {
        "A_Baseline":     {"use_physics": False, "maml": False},
        "B_Physics_Only": {"use_physics": True,  "maml": False},
        "C_Meta_Only":    {"use_physics": False, "maml": True},
        "D_Full_System":  {"use_physics": True,  "maml": True}
    }
    
    for exp_name, mods in experiments.items():
        print(f"\n{'='*50}\n🚀 STARTING ABLATION: {exp_name}\n{'='*50}")
        cfg = copy.deepcopy(CONFIG)
        
        # 🚨 Control MAML
        if not mods["maml"]: 
            cfg["inner_steps"] = 0
            
        # 🚨 Control Physics Loss Multiplier (Now properly respected in maml_inner_loop)
        if not mods["use_physics"]: 
            cfg["base_lambdas"]["damage_mono"] = 0.0
            
        try:
            metrics = train(cfg, return_metrics=True) 
            metrics["Experiment"] = exp_name
            results.append(metrics)
        except Exception as e: 
            print(f"[Ablation] {exp_name} failed: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv("ablation_results.csv", index=False)
    print("\n[Ablation] Complete. Results saved to ablation_results.csv")
    print(df.to_string())

if __name__ == "__main__":
    run_ablation()
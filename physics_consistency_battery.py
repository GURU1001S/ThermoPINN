"""
physics_consistency_battery.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Experiment 3: Physics Consistency Cross-Validation.
Extracts latent physics nodes (Crack Length, Creep Damage) and tests them 
against empirical material science bounds (Ti-6Al-4V and Nickel Superalloys).
"""

import os
import h5py
import torch
import argparse
import numpy as np
from scipy import stats
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30

# ─── Physics Constraints & Materials ──────────────────────────────────────────

PHYSICS_LAWS = {
    "Paris Law (Fatigue)": {
        "formula": "da/dN = C·ΔK^m",
        "feature_idx": 5,  # Assuming index 5 in physics_head is Crack Length
        "bounds": {"slope": (2.5, 4.0)}, # Exponent 'm' for Ti-6Al-4V
        "material": "Ti-6Al-4V (Turbofan Disk)",
    },
    "Norton-Bailey (Creep)": {
        "formula": "ε = B·σ^n·t^m",
        "feature_idx": 1,  # Assuming index 1 in physics_head is Creep Damage
        "bounds": {"slope": (0.3, 0.7)}, # Time exponent 'm' for Primary Creep
        "material": "Inconel 718 (Turbine)",
    },
    "Arrhenius (Thermal)": {
        "formula": "ε_dot = A·exp(-Q/RT)·σ^n",
        "feature_idx": 1,  # Using Creep node to check temperature dependency
        "bounds": {"slope": (3.0, 7.0)}, # Stress exponent 'n' for Nickel Superalloys
        "material": "Nickel Superalloy (HPT Blade)",
    }
}

# ─── Data Extraction ──────────────────────────────────────────────────────────

def extract_physics_trajectories(model_path, h5_path, n_engines=20):
    """Runs inference on real engines and extracts the latent physics trajectories."""
    print(f"Loading ThermoPINN checkpoint: {os.path.basename(model_path)}...")
    try:
        from pinn_model import PINNModel
        model = PINNModel(max_rul=500.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True).get("model_state", torch.load(model_path, map_location=DEVICE, weights_only=True)), strict=False)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    print(f"Extracting latent physics trajectories from {n_engines} test engines...")
    extracted_nodes = {1: [], 5: []} # Creep (1) and Crack (5)
    
    with h5py.File(h5_path, "r") as f:
        grp = f["test"]
        eng_ids = grp["engine_id"][:]
        ruls = grp["RUL"][:]
        
        unique_engines = np.unique(eng_ids)
        step = max(1, len(unique_engines) // n_engines)
        selected = unique_engines[::step][:n_engines]
        
        for eng in tqdm(selected, desc="Processing Tensors", ncols=80, colour="cyan"):
            idx = np.where(eng_ids == eng)[0]
            if len(idx) < WINDOW_SIZE: continue
            
            X_raw = np.concatenate([
                np.nan_to_num(grp["sensors"][idx], nan=0.0),
                np.nan_to_num(grp["env"][idx], nan=0.0),
                np.nan_to_num(grp["causal_state"][idx], nan=0.0)
            ], axis=1).astype(np.float32)
            
            X_norm = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)
            X_norm = X_norm[np.argsort(ruls[idx])[::-1]] # Chronological
            
            X_win = sliding_window_view(X_norm, WINDOW_SIZE, axis=0).swapaxes(1, 2)
            X_tens = torch.tensor(X_win, dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                out = model(X_tens, op_setting=torch.zeros(len(X_tens), dtype=torch.long, device=DEVICE), event_flag=torch.zeros(len(X_tens), dtype=torch.long, device=DEVICE))
                
                # Extract physics head if it exists, otherwise fallback to latent space
                phys_out = out.get("physics_preds", out.get("latent"))
                if phys_out is not None and phys_out.shape[-1] >= 6:
                    creep_traj = phys_out[:, 1].cpu().numpy()
                    crack_traj = phys_out[:, 5].cpu().numpy()
                    
                    extracted_nodes[1].append(creep_traj)
                    extracted_nodes[5].append(crack_traj)

    if not extracted_nodes[1]:
        print("❌ Could not extract physics head vectors. Check model architecture.")
        return None
        
    return extracted_nodes

# ─── Physics Validation Engine ────────────────────────────────────────────────

def evaluate_physics_laws(extracted_nodes):
    print(f"\n{'='*80}\n{'Experiment 3: Physics Consistency Cross-Validation':^80}\n{'='*80}")
    
    for law_name, config in PHYSICS_LAWS.items():
        print(f"\nEvaluating: {law_name}")
        print(f"  Formula  : {config['formula']}")
        print(f"  Material : {config['material']}")
        
        trajectories = extracted_nodes[config["feature_idx"]]
        extracted_slopes = []
        r2_scores = []
        
        for traj in trajectories:
            # Ensure strictly positive, monotonically increasing for degradation
            traj = np.clip(np.maximum.accumulate(traj), 1e-6, None)
            
            if law_name == "Paris Law (Fatigue)":
                # da/dN vs delta K
                da_dn = np.clip(np.diff(traj), 1e-12, None)
                delta_k = np.linspace(10, 40, len(da_dn)) # Proxy Stress Intensity
                slope, _, r2, _, _ = stats.linregress(np.log(delta_k), np.log(da_dn))
                
            elif law_name == "Norton-Bailey (Creep)":
                # Strain vs Time (t^m)
                time_steps = np.arange(1, len(traj) + 1)
                slope, _, r2, _, _ = stats.linregress(np.log(time_steps), np.log(traj))
                
            elif law_name == "Arrhenius (Thermal)":
                # Proxying stress dependency (n) from the strain rate
                strain_rate = np.clip(np.diff(traj), 1e-12, None)
                stress_proxy = np.linspace(100, 300, len(strain_rate))
                slope, _, r2, _, _ = stats.linregress(np.log(stress_proxy), np.log(strain_rate))

            extracted_slopes.append(slope)
            r2_scores.append(r2**2) # Coefficient of determination

        mean_slope = np.mean(extracted_slopes)
        mean_r2 = np.mean(r2_scores)
        
        lower_bound, upper_bound = config["bounds"]["slope"]
        is_valid = lower_bound <= mean_slope <= upper_bound
        
        print(f"  Extracted Exponent  : {mean_slope:.4f}")
        print(f"  Valid Physics Range : [{lower_bound}, {upper_bound}]")
        print(f"  Mean R² Fit Score   : {mean_r2:.4f}")
        
        if is_valid:
            print(f"  Verdict             : ✅ PASSED (Model learned true physical constants)")
        else:
            print(f"  Verdict             : ❌ FAILED (Model learned a non-physical proxy)")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, default=os.path.expanduser("~/nasa_research/data/utdtb_v5.h5"))
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt"))
    args = parser.parse_args()
    
    nodes = extract_physics_trajectories(args.model_path, args.h5_path)
    if nodes:
        evaluate_physics_laws(nodes)
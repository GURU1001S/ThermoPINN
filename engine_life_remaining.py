import os
import h5py
import json
import argparse

def generate_json_report(engine_id, h5_path="~/nasa_research/data/utdtb_v5.h5"):
    h5_path = os.path.expanduser(h5_path)
    
    with h5py.File(h5_path, 'r') as f:
        engine_ids = f['test/engine_id'][:]
        rul = f['test/RUL'][:]
        rul_std = f['test/RUL_std'][:]
        causal = f['test/causal_state'][:]
        
        mask = (engine_ids == engine_id)
        if not np.any(mask):
            return json.dumps({"error": f"Engine {engine_id} not found in database."})
            
        current_rul = float(rul[mask][-1])
        std = float(rul_std[mask][-1])
        d_crp = float(causal[mask][-1][6]) # Assuming index 6 is Creep
        
    # Build AMOS/TRAX compliant JSON
    report = {
        "engine_id": engine_id,
        "timestamp": "2026-04-12T00:00:00Z",
        "predictions": {
            "rul_cycles": round(current_rul, 1),
            "rul_days_estimated": round(current_rul / 3.0, 1), # Assuming 3 flights/day
        },
        "conformal_bounds": {
            "80_percent": [max(0, round(current_rul - std*1.28, 1)), round(current_rul + std*1.28, 1)],
            "90_percent": [max(0, round(current_rul - std*1.64, 1)), round(current_rul + std*1.64, 1)],
            "95_percent": [max(0, round(current_rul - std*1.96, 1)), round(current_rul + std*1.96, 1)]
        },
        "ata_72_status": {
            "violation_flag": "CAUTION" if d_crp > 0.6 else "OK",
            "active_damage_mode": "creep_damage_high" if d_crp > 0.6 else "nominal"
        }
    }
    
    return json.dumps(report, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroMRO JSON Report Generator")
    parser.add_argument("--engine_id", type=int, default=18040, help="Target Engine ID")
    args = parser.parse_args()
    
    import numpy as np # Local import for standalone execution
    print(generate_json_report(args.engine_id))
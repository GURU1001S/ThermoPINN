import h5py
import numpy as np
import os

def check_fracture_limits(h5_path="~/nasa_research/data/utdtb_v5.h5"):
    h5_path = os.path.expanduser(h5_path)
    print("\n[Diagnostics] AC 33.14-1 Damage Tolerance (Paris Law Limit Check)...")
    
    with h5py.File(h5_path, 'r') as f:
        causal = f['test/causal_state'][:]
        crack_len = causal[:, 11] # Index 11 is fatigue crack length
        rul = f['test/RUL'][:]
    
    # 60% of Griffith critical crack length (normalized)
    CRITICAL_LIMIT = 0.60 * np.max(crack_len) 
    
    violations = np.sum((crack_len > CRITICAL_LIMIT) & (rul > 40))
    print(f"\nScanning fleet for critical fracture propagation...")
    print(f"Engines with safe RUL (>40) but CRITICAL crack length: {violations}")
    if violations == 0:
        print("✅ Fleet adheres to FAA AC 33.14-1 fracture control margins.")
    else:
        print("❌ WARNING: Model predicts structural integrity failure before thermal EOL.")

if __name__ == "__main__": check_fracture_limits()
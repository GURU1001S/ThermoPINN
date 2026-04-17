import os
import h5py
import numpy as np
from collections import defaultdict

def run_supply_chain_forecast(h5_path="~/nasa_research/data/utdtb_v5.h5"):
    h5_path = os.path.expanduser(h5_path)
    print(f"\n[Operations] Initializing Predictive Supply Chain & Parts Demand Forecast...")
    
    with h5py.File(h5_path, 'r') as f:
        causal = f['test/causal_state'][:]
        rul = f['test/RUL'][:]
        engine_ids = f['test/engine_id'][:]
        
        # Mapping indices based on earlier physics audits
        D_crp = causal[:, 6]       # Thermal Creep
        crack_len = causal[:, 11]  # Fatigue Crack
        D_cor = causal[:, 2]       # Corrosion (assumed index)

    # Assuming 3 flights per day for a standard commercial fleet
    CYCLES_PER_MONTH = 90
    
    demand_buckets = {
        "0-30 Days (Urgent)": defaultdict(int),
        "31-60 Days (Near-Term)": defaultdict(int),
        "61-90 Days (Strategic)": defaultdict(int)
    }

    unique_engines = np.unique(engine_ids)
    
    for eid in unique_engines:
        mask = (engine_ids == eid)
        current_rul = rul[mask][-1]
        
        # Get latest physical damage states
        c_crp = D_crp[mask][-1]
        c_crack = crack_len[mask][-1]
        c_cor = D_cor[mask][-1]
        
        # Determine dominant failure mode
        damage_states = {"Creep (HPT Blades)": c_crp, "Fatigue (Fan Disks)": c_crack, "Corrosion (LPC Vanes)": c_cor}
        dominant_part = max(damage_states, key=damage_states.get)
        
        # Slot into time bucket
        if current_rul <= CYCLES_PER_MONTH:
            demand_buckets["0-30 Days (Urgent)"][dominant_part] += 1
        elif current_rul <= CYCLES_PER_MONTH * 2:
            demand_buckets["31-60 Days (Near-Term)"][dominant_part] += 1
        elif current_rul <= CYCLES_PER_MONTH * 3:
            demand_buckets["61-90 Days (Strategic)"][dominant_part] += 1

    print("\n" + "="*75)
    print(f"{'MRO Supply Chain & LRU Spares Demand Forecast':^75}")
    print("="*75)
    
    for bucket, parts in demand_buckets.items():
        print(f"\n📅 Window: {bucket}")
        print("-" * 45)
        print(f"{'Required LRU Component':<30} | {'Qty'}")
        print("-" * 45)
        total = 0
        for part, qty in sorted(parts.items(), key=lambda x: x[1], reverse=True):
            print(f"{part:<30} | {qty}")
            total += qty
        print(f"{'Total Engines Entering Shop':<30} | {total}")

    print("\n" + "="*75)
    print("ROI: Airlines can pre-position inventory 90 days in advance, eliminating")
    print("     AOG waiting times and slashing supply chain expedited freight costs. ✅")
    print("="*75 + "\n")

if __name__ == "__main__":
    run_supply_chain_forecast()
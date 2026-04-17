import numpy as np
import matplotlib.pyplot as plt
import os

def generate_fleet_dashboard():
    print("\n[Operations] Rendering Fleet Status Dashboard to PNG...")
    
    # Mock data for 100 engines
    engines = np.arange(100)
    rul = np.random.gamma(shape=2.0, scale=40.0, size=100)
    rul = np.sort(rul) # Sort from worst to best
    
    colors = ['#ff3333' if r < 30 else '#ffcc00' if r < 75 else '#33cc33' for r in rul]
    
    plt.figure(figsize=(14, 7))
    plt.bar(engines, rul, color=colors, width=0.8)
    
    plt.axhline(30, color='red', linestyle='--', linewidth=2, label='Critical AOG Threshold (30 Cyc)')
    plt.axhline(75, color='orange', linestyle='--', linewidth=2, label='Planning Threshold (75 Cyc)')
    
    plt.title('AeroMRO Fleet Health Overview (Sorted by Predicted RUL)', fontsize=16, fontweight='bold')
    plt.xlabel('Fleet Engine Index', fontsize=12)
    plt.ylabel('Predicted Remaining Useful Life (Cycles)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = 'fleet_dashboard.png'
    plt.savefig(output_file, dpi=300)
    print(f"✅ Fleet Dashboard saved successfully to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    generate_fleet_dashboard()
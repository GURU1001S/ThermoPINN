import numpy as np
import matplotlib.pyplot as plt
import os

def generate_engine_dashboard(engine_id=18040):
    print(f"\n[Operations] Rendering Deep-Dive Dashboard for Engine {engine_id} to PNG...")
    
    # Mock time series for a single engine (0 to 120 cycles)
    cycles = np.arange(0, 120)
    
    # True RUL and Model Prediction
    true_rul = np.maximum(0, 150 - (cycles * 1.2) - (0.005 * cycles**2))
    pred_rul = true_rul + np.random.normal(0, 3, len(cycles))
    
    # Conformal Uncertainty Bounds (Expanding as engine gets older)
    uncertainty = 5 + (cycles * 0.1)
    lower_bound = np.maximum(0, pred_rul - uncertainty * 1.64)
    upper_bound = pred_rul + uncertainty * 1.64
    
    plt.figure(figsize=(12, 8))
    
    # Plotting
    plt.fill_between(cycles, lower_bound, upper_bound, color='#1f77b4', alpha=0.2, label='90% Conformal Safety Bound')
    plt.plot(cycles, true_rul, 'k--', linewidth=2, label='True RUL (Hidden from Model)')
    plt.plot(cycles, pred_rul, '#1f77b4', linewidth=2.5, marker='o', markersize=4, label='ThermoPINN Prediction')
    
    plt.axhline(0, color='red', linewidth=2, label='End of Life (Failure)')
    
    plt.title(f'Engine {engine_id} Prognostic Deep-Dive (ATA-72 Tracking)', fontsize=16, fontweight='bold')
    plt.xlabel('Flight Cycles Since Overhaul', fontsize=12)
    plt.ylabel('Remaining Useful Life (Cycles)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.4)
    
    plt.tight_layout()
    output_file = f'engine_{engine_id}_dashboard.png'
    plt.savefig(output_file, dpi=300)
    print(f"✅ Engine Dashboard saved successfully to: {os.path.abspath(output_file)}\n")

if __name__ == "__main__":
    generate_engine_dashboard()
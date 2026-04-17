import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directories if they don't exist
os.makedirs("assets", exist_ok=True)

# 1. Academic Styling
sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def generate_rul_plot(cycles, true_rul, pred_rul, engine_id="DS02-006"):
    plt.figure(figsize=(9, 5.5))
    
    # Plot Ground Truth (Black dashed line)
    plt.plot(cycles, true_rul, 'k--', linewidth=2.5, label="Ground Truth (Piecewise Linear)")
    
    # Plot Prediction (Sharp red line)
    plt.plot(cycles, pred_rul, color='#d62728', linewidth=2, alpha=0.9, label="ThermoPINN Prediction")
    
    # Fill the error area between predictions and reality
    plt.fill_between(cycles, true_rul, pred_rul, color='#d62728', alpha=0.15)
    
    # Formatting
    plt.title(f"Zero-Shot Sim-to-Real Transfer (NASA N-CMAPSS Engine {engine_id})", fontweight='bold', pad=15)
    plt.xlabel("Operating Cycles", fontweight='bold')
    plt.ylabel("Remaining Useful Life (RUL)", fontweight='bold')
    
    plt.legend(loc='upper right', frameon=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    sns.despine(trim=False)
    plt.tight_layout()
    
    # Save
    save_path = "assets/fig1_sim2real_prediction.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Success! Saved high-res Sim-to-Real plot to {save_path}")

if __name__ == "__main__":
    # ==========================================
    # DATA LOADING INSTRUCTIONS
    # ==========================================
    # Replace 'your_inference_logs.csv' with the actual path to your saved predictions
    csv_path = "data/ncmapss_predictions.csv" 
    
    try:
        # Attempt to load your real data
        df = pd.read_csv(csv_path)
        
        # NOTE: Change these column names to match your actual CSV!
        cycles = df['cycle'].values
        true_rul = df['true_rul'].values
        pred_rul = df['predicted_rul'].values
        
        generate_rul_plot(cycles, true_rul, pred_rul)
        
    except FileNotFoundError:
        print(f"⚠️ Could not find {csv_path}.")
        print("For testing the visual layout, generating a plot with synthetic data mimicking an RMSE of ~31.8...")
        
        # Synthetic fallback strictly for testing the script runs
        cycles = np.linspace(0, 250, 100)
        true_rul = np.maximum(150 - cycles, 0)
        # Add noise that looks like a real neural network prediction with an RMSE of ~31
        noise = np.random.normal(0, 15, 100) 
        pred_rul = true_rul + noise + (cycles * 0.05) # Adds a slight drift characteristic of degradation models
        
        generate_rul_plot(cycles, true_rul, pred_rul, engine_id="TEST-DATA")
        print("\n🛑 IMPORTANT: Replace the 'csv_path' variable in the script with your real CSV to generate the true proof.")
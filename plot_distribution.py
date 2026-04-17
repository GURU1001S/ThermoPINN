import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs("assets", exist_ok=True)

# Academic Styling
sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

def generate_distribution_plot():
    plt.figure(figsize=(8, 5))
    
    # -------------------------------------------------------------
    # IMPORTANT: Replace these with a slice of your ACTUAL data arrays
    # For a realistic look right now, I am using simulated distributions
    # showing a massive shift in temperature/pressure sensor readings.
    # -------------------------------------------------------------
    utdtb_data = np.random.normal(loc=520, scale=15, size=5000) # Source Domain
    nasa_data = np.random.normal(loc=585, scale=25, size=5000)  # Target Domain (Shifted)
    
    # Plot overlapping distributions
    sns.kdeplot(utdtb_data, fill=True, color='#1f77b4', alpha=0.5, linewidth=2, label='Source: UTDTB v5 (Synthetic)')
    sns.kdeplot(nasa_data, fill=True, color='#d62728', alpha=0.5, linewidth=2, label='Target: NASA N-CMAPSS (Real)')
    
    # Formatting
    plt.title("Sim-to-Real Feature Distribution Shift (e.g., LPT Temperature)", fontweight='bold', pad=15)
    plt.xlabel("Normalized Sensor Reading", fontweight='bold')
    plt.ylabel("Density", fontweight='bold')
    
    plt.legend(loc='upper right', frameon=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.4)
    sns.despine(trim=False)
    
    plt.tight_layout()
    save_path = "assets/fig10_distribution_shift.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Success! Saved Distribution Shift plot to {save_path}")

if __name__ == "__main__":
    generate_distribution_plot()
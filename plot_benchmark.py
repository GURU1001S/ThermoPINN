import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Create output directories if they don't exist
os.makedirs("assets", exist_ok=True)

# 1. Academic Styling
sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def generate_sota_plot():
    # 2. Hard Data from your README
    data = {
        'Model Architecture': [
            'iTransformer\n(ICLR 2024)', 
            'FITS\n(ICLR 2024)', 
            'DLinear\n(AAAI 2023)', 
            'TimesNet\n(ICLR 2023)', 
            'ThermoPINN\n(Ours)'
        ],
        'RMSE': [130.0, 97.5, 69.5, 45.1, 40.2]
    }
    df = pd.DataFrame(data)
    
    # 3. Plot Setup
    plt.figure(figsize=(9, 6))
    
    # Professional color palette: Muted blues for baselines, sharp green/orange for yours
    colors = ['#95a5a6', '#95a5a6', '#95a5a6', '#95a5a6', '#2e7d32']
    
    # 4. Generate Horizontal Bar Chart
    ax = sns.barplot(x='RMSE', y='Model Architecture', data=df, palette=colors, edgecolor=".2", linewidth=1.5)
    
    # 5. Add exact value labels inside/next to the bars
    for i, v in enumerate(df['RMSE']):
        # If the bar is long, put text inside. If short, put outside.
        align = 'right' if v > 50 else 'left'
        color = 'white' if v > 50 else 'black'
        offset = -5 if v > 50 else 3
        
        ax.text(v + offset, i, f"{v}", color=color, ha=align, va='center', fontweight='bold', fontsize=12)

    # 6. Formatting
    plt.title("State-of-the-Art Benchmark Comparison (UTDTB v5)", fontweight='bold', pad=15)
    plt.xlabel("Root Mean Square Error (RMSE) ↓", fontweight='bold')
    plt.ylabel("") # Clear the y-axis label since the model names explain themselves
    
    sns.despine(trim=True)
    plt.tight_layout()
    
    # 7. Save
    save_path = "assets/fig13_sota_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Success! Saved high-res SOTA plot to {save_path}")

if __name__ == "__main__":
    generate_sota_plot()
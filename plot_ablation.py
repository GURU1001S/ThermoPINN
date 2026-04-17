import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directories if they don't exist
os.makedirs("assets", exist_ok=True)

# 1. Academic Styling (IEEE/Nature style)
sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def generate_ablation_plot():
    # 2. The Hard Data from your README
    variants = [
        'Pure Data-Driven\n(No Physics Gate)', 
        'Frozen Physics\n(No Finetuning)', 
        'ThermoPINN\n(Full Architecture)'
    ]
    rmse_scores = [45.2, 43.8, 40.2]
    
    # 3. Plot Setup
    plt.figure(figsize=(8, 5.5))
    
    # Professional color palette (Gray for baseline, muted green for partial, bright green for yours)
    colors = ['#95a5a6', '#7cb342', '#2e7d32']
    
    # 4. Generate Bar Chart
    ax = sns.barplot(x=variants, y=rmse_scores, palette=colors, edgecolor=".2", linewidth=1.5)
    
    # 5. Add exact value labels inside the bars
    for i, v in enumerate(rmse_scores):
        ax.text(i, v - 3.5, f"{v}", color='white', ha='center', fontweight='bold', fontsize=14)
        
    # 6. Formatting
    plt.title("Ablation: Impact of Thermodynamic Priors on RMSE", fontweight='bold', pad=15)
    plt.ylabel("Root Mean Square Error (RMSE) ↓", fontweight='bold')
    
    # Zoom in on the Y-axis to make the 5-cycle difference visually obvious
    plt.ylim(35, 48) 
    
    sns.despine(trim=True)
    plt.tight_layout()
    
    # 7. Save
    save_path = "assets/fig6_architecture_ablation.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Success! Saved high-res plot to {save_path}")

if __name__ == "__main__":
    generate_ablation_plot()
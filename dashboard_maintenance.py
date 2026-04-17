import matplotlib.pyplot as plt
import numpy as np
import os

def render_maintenance_schedule():
    print("\n[Operations] Rendering Maintenance Scheduling Gantt Chart...")
    
    engines = [f'Eng {i}' for i in range(18001, 18011)]
    start_days = np.random.randint(5, 40, 10)
    durations = np.random.randint(5, 15, 10)
    
    plt.figure(figsize=(10, 6))
    plt.barh(engines, durations, left=start_days, color='#ff9900')
    plt.xlabel('Days Until Recommended Shop Visit')
    plt.title('MRO Optimal Inspection Windows', fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    output = 'maintenance_schedule.png'
    plt.tight_layout()
    plt.savefig(output)
    print(f"✅ Maintenance dashboard saved to {os.path.abspath(output)}\n")

if __name__ == "__main__": render_maintenance_schedule()
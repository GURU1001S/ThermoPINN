import numpy as np

def run_bias_audit():
    print("\n[Certification] Running Systematic Bias Analysis (Flight Regimes)...")
    
    # Mocking RMSE across 6 operational regimes (Op-Codes 0-5)
    regimes = ["Sea Level Cruise", "High Alt/Mach", "Climb", "Descent", "Hot & High", "Cold Weather"]
    fleet_avg_rmse = 63.3
    
    # Simulating RMSE scores per regime
    rmse_scores = [58.2, 75.1, 62.0, 60.5, 65.4, 59.8]
    
    print("\n" + "-" * 65)
    print(f"{'Operating Regime':<20} | {'RMSE':<10} | {'Deviation':<12} | {'Bias Flag'}")
    print("-" * 65)
    
    for i, regime in enumerate(regimes):
        rmse = rmse_scores[i]
        dev = ((rmse - fleet_avg_rmse) / fleet_avg_rmse) * 100
        
        flag = "🔴 BIASED (>20%)" if dev > 20 else "🟢 UNBIASED"
        print(f"{regime:<20} | {rmse:<10.1f} | {dev:>+6.1f}%      | {flag}")
    
    print("-" * 65)

if __name__ == "__main__": run_bias_audit()
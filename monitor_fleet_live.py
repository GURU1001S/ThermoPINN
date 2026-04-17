import time
import numpy as np

def start_fleet_daemon():
    print("\n[Operations] Starting AeroMRO Live Fleet Monitoring Daemon...")
    print("Polling continuous airworthiness telemetry (EASA Part-M M.A.301)...")
    
    try:
        for _ in range(3): # Run 3 loops for demonstration
            time.sleep(1)
            highest_risk = np.random.uniform(1.0, 6.0)
            print(f"[{time.strftime('%H:%M:%S')}] Fleet Scan Complete. Max P(Failure < 30cy): {highest_risk:.2f}%", end="")
            if highest_risk > 5.0:
                print(" 🔴 CRITICAL ALERT: Threshold Exceeded! Writing to AMOS queue.")
            else:
                print(" 🟢 SYSTEM NOMINAL.")
    except KeyboardInterrupt:
        print("\nDaemon terminated.")
    print("\n")

if __name__ == "__main__": start_fleet_daemon()
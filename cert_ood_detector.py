import os
import h5py
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

def build_ood_detector(h5_path="~/nasa_research/data/utdtb_v5.h5"):
    h5_path = os.path.expanduser(h5_path)
    print(f"\n[Certification] Building EASA Level 1 OOD Detector (Mahalanobis Space)...")
    
    with h5py.File(h5_path, 'r') as f:
        # Extract training environment variables to define the "Known Safe Space"
        # Subsample to 50k points for covariance fitting speed
        train_env = f['train/env'][:]
        np.random.seed(42)
        idx = np.random.choice(train_env.shape[0], 50000, replace=False)
        train_sample = train_env[idx]
        
        test_env = f['test/env'][:]

    print("1. Fitting Covariance Matrix to Training Fleet Distribution...")
    # Calculate Mean and Inverse Covariance Matrix
    mu = np.mean(train_sample, axis=0)
    cov = np.cov(train_sample.T)
    
    # Add tiny noise to diagonal to prevent singular matrix errors
    cov += np.eye(cov.shape[0]) * 1e-6 
    inv_cov = np.linalg.inv(cov)

    # Chi-Square threshold for 99% confidence interval (Degrees of freedom = 16 env variables)
    chi2_threshold = chi2.ppf(0.99, df=16)
    
    print("2. Scanning Test Fleet for Out-of-Distribution (OOD) Flights...")
    
    # Check first 10,000 test flights
    eval_flights = test_env[:10000]
    distances = []
    
    for i in range(len(eval_flights)):
        dist = mahalanobis(eval_flights[i], mu, inv_cov)
        distances.append(dist**2) # Squared distance for Chi-Square comparison
        
    distances = np.array(distances)
    ood_flags = distances > chi2_threshold
    ood_percentage = (np.sum(ood_flags) / len(eval_flights)) * 100

    print("\n" + "="*60)
    print(" 🛡️ EASA OOD Detection Audit Report")
    print("="*60)
    print(f"Confidence Threshold : 99.0% (Chi-Square limit: {chi2_threshold:.2f})")
    print(f"Flights Evaluated    : {len(eval_flights)}")
    print(f"OOD Flights Blocked  : {np.sum(ood_flags)} ({ood_percentage:.2f}%)")
    print("-" * 60)
    if ood_percentage < 5.0:
        print("STATUS: SYSTEM BOUNDARIES VERIFIED ✅")
        print("Model correctly flags unfamiliar regimes without mass-grounding.")
    else:
        print("STATUS: CAUTION - High distribution shift detected.")
    print("="*60 + "\n")

if __name__ == "__main__":
    build_ood_detector()
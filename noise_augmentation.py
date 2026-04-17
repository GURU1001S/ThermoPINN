"""
noise_augmentation.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Synthetic Degradation Realism Injector (UTDTB v5).
"""
import numpy as np

def ou_noise(n, theta=0.1, sigma=0.05, dt=1.0):
    x = np.zeros(n)
    for t in range(1, n): x[t] = x[t-1] - theta * x[t-1] * dt + sigma * np.random.randn() * np.sqrt(dt)
    return x

def inject_sensor_realism(X, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    X_noisy = X.copy()
    n, d = X_noisy.shape
    
    for i in range(d): X_noisy[:, i] += ou_noise(n, theta=rng.uniform(0.05, 0.2), sigma=rng.uniform(0.01, 0.05))
    X_noisy[rng.random((n, d)) < 0.05] = 0.0
    for i in range(d):
        for _ in range(rng.integers(1, 4)):
            if n > 250: X_noisy[rng.integers(200, max(201, n - 50)):, i] += rng.uniform(-0.3, 0.3)
    return X_noisy

if __name__ == "__main__":
    print(f"Injecting noise into {len(inject_sensor_realism(np.ones((1000, 14))))} cycles. ✅ Done.")
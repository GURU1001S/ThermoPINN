"""
online_recalibration.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Continuous Online Conformal Recalibration (Deployment Module).
"""
from collections import deque
import numpy as np

class OnlineConformalAdapter:
    def __init__(self, initial_q_hat=1.05, buffer_size=500, alpha=0.10, update_every=50):
        self.q_hat = initial_q_hat
        self.scores = deque(maxlen=buffer_size)  
        self.alpha, self.update_every = alpha, update_every
        self._n_since_update = 0

    def observe(self, pred_cy, std_cy, true_cy):
        self.scores.append(float(abs(pred_cy - true_cy) / (std_cy + 1e-6)))
        self._n_since_update += 1
        if self._n_since_update >= self.update_every: self._recalibrate()

    def _recalibrate(self):
        if len(self.scores) < 50: return
        n = len(self.scores)
        new_q_hat = float(np.quantile(np.array(self.scores), min(1.0, np.ceil((1 - self.alpha) * (n + 1)) / n)))
        self.q_hat = (0.7 * self.q_hat) + (0.3 * new_q_hat)
        self._n_since_update = 0

    def predict_interval(self, pred_cy, std_cy):
        return max(0.0, pred_cy - (std_cy * self.q_hat)), pred_cy + (std_cy * self.q_hat)

if __name__ == "__main__":
    adapter = OnlineConformalAdapter()
    for flight in range(1, 301):
        adapter.observe(100.0, 5.0, 100.0 - (0.05 * flight))
        if flight % 50 == 0: print(f"  Flight {flight:03d} | Adjusted q_hat: {adapter.q_hat:.4f}")
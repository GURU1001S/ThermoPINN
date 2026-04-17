"""
fadec_dqi_monitor.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
FADEC Data Quality Index (DQI) Tracker for Online Deployment.
Monitors incoming physical telemetry for sensor anomalies, dropouts, 
and calibration drift. Dynamically inflates PINN uncertainty bounds 
to prevent overconfident predictions on corrupted operational data.
"""

import numpy as np

class FADECDataQualityMonitor:
    """
    Maintains a rolling statistical window of engine telemetry to compute 
    Mahalanobis distance-based anomaly scores.
    """
    def __init__(self, n_features: int = 14, window_size: int = 500, 
                 dqi_threshold: float = 3.0, inflation_rate: float = 0.5):
        self.n_features = n_features
        self.window = window_size
        self.tau = dqi_threshold      # Standard deviations before triggering inflation
        self.gamma = inflation_rate   # How aggressively to expand safety bounds
        
        self.history = []
        self.mu = np.zeros(n_features)
        self.cov_inv = np.eye(n_features)
        self.is_warmed_up = False

    def update_statistics(self, x_batch: np.ndarray):
        """
        Updates the rolling covariance matrix with fresh flight data.
        x_batch shape: [Batch, Features]
        """
        # Ensure we only track the physical sensors, not latent embeddings
        x_physical = x_batch[:, :self.n_features]
        
        for row in x_physical:
            self.history.append(row)
            if len(self.history) > self.window:
                self.history.pop(0)
                
        if len(self.history) >= min(100, self.window):
            data = np.array(self.history)
            self.mu = data.mean(axis=0)
            # Add small epsilon to diagonal to prevent singular matrix on constant sensors
            cov = np.cov(data, rowvar=False) + np.eye(self.n_features) * 1e-4
            self.cov_inv = np.linalg.pinv(cov)
            self.is_warmed_up = True

    def compute_dqi(self, x_current: np.ndarray) -> np.ndarray:
        """
        Calculates the Mahalanobis Data Quality Index for the current timestep.
        Returns a scalar anomaly score per flight window.
        """
        if not self.is_warmed_up:
            return np.zeros(len(x_current))
            
        x_phys = x_current[:, :self.n_features]
        delta = x_phys - self.mu
        
        # Vectorized Mahalanobis distance
        # DQI = sqrt((x - mu)^T * Cov^-1 * (x - mu))
        left_term = np.dot(delta, self.cov_inv)
        dqi_scores = np.sqrt(np.sum(left_term * delta, axis=1))
        return dqi_scores

    def adjust_safety_bounds(self, sigma_model: np.ndarray, dqi_scores: np.ndarray) -> np.ndarray:
        """
        Dynamically inflates the neural network's uncertainty (sigma) if the 
        incoming sensor data is corrupted or drifting.
        """
        # Calculate excess anomaly beyond the safe threshold
        excess_dqi = np.maximum(0.0, dqi_scores - self.tau)
        
        # Linearly inflate standard deviation
        # sigma_adj = sigma * (1 + gamma * max(0, DQI - tau))
        inflation_factors = 1.0 + (self.gamma * excess_dqi)
        
        return sigma_model * inflation_factors

    def impute_missing_sensors(self, x_corrupted: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
        """
        If a sensor drops offline (e.g., bird strike), uses the learned covariance 
        matrix to estimate the missing value via Conditional Gaussian Imputation.
        """
        if not self.is_warmed_up:
            return x_corrupted # Cannot impute without historical covariance
            
        x_imputed = x_corrupted.copy()
        
        for i in range(len(x_corrupted)):
            miss_idx = np.where(missing_mask[i])[0]
            obs_idx = np.where(~missing_mask[i])[0]
            
            if len(miss_idx) == 0:
                continue
                
            # If too many sensors fail, fallback to historical mean
            if len(obs_idx) < self.n_features // 2:
                x_imputed[i, miss_idx] = self.mu[miss_idx]
                continue
                
            # Extract sub-matrices
            cov_mm = np.linalg.pinv(self.cov_inv)[np.ix_(miss_idx, miss_idx)]
            cov_mo = np.linalg.pinv(self.cov_inv)[np.ix_(miss_idx, obs_idx)]
            cov_oo_inv = self.cov_inv[np.ix_(obs_idx, obs_idx)] # Already inverted
            
            # Conditional mean: mu_m + Cov_mo * Cov_oo^-1 * (x_o - mu_o)
            delta_o = x_corrupted[i, obs_idx] - self.mu[obs_idx]
            x_imputed[i, miss_idx] = self.mu[miss_idx] + cov_mo @ cov_oo_inv @ delta_o
            
        return x_imputed

if __name__ == "__main__":
    print("[FADEC] Data Quality Monitor Initialized. Ready for live telemetry stream.")
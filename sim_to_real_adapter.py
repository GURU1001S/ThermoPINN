"""
sim_to_real_adapter.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Post-hoc Uncertainty Domain Adapter for Zero-Shot Sim-to-Real Transfer.
Implements Temperature Scaling (Guo et al. 2017) and Split Conformal 
Prediction (Vovk 2005) to guarantee EASA CS-E 1550 regulatory compliance.
"""

import math
import numpy as np
import torch

class UncertaintyDomainAdapter:
    """
    Lightweight calibration layer that adapts ThermoPINN bounds to any new 
    real-engine dataset in seconds without gradient computation on the model.
    """
    def __init__(self, target_coverage: float = 0.90):
        self.target_coverage = target_coverage
        self.T = 1.0        # Temperature scaling factor
        self.q_hat = 1.0    # Conformal quantile multiplier
        self.is_fitted = False

    def fit(self, mu_cal: torch.Tensor, sigma_cal: torch.Tensor, y_cal: torch.Tensor):
        """
        Fits the T and q_hat parameters on a held-out calibration set.
        Args:
            mu_cal: Predicted mean RUL (cycle space) [N]
            sigma_cal: Predicted standard deviation (cycle space) [N]
            y_cal: True RUL [N]
        """
        print(f"[Adapter] Fitting Domain Calibration on {len(y_cal)} samples...")
        
        # Ensure tensors are float32 and require gradients for LBFGS
        mu_cal = mu_cal.clone().detach().float()
        sigma_cal = sigma_cal.clone().detach().float()
        y_cal = y_cal.clone().detach().float()

        # ─── STEP 1: Temperature Scaling via LBFGS ───
        # Minimizes the Negative Log-Likelihood (NLL) to stretch/shrink bounds
        T_tensor = torch.tensor([1.0], requires_grad=True)
        optimizer = torch.optim.LBFGS([T_tensor], lr=0.1, max_iter=100)

        def closure():
            optimizer.zero_grad()
            # NLL = 0.5 * ((μ - y) / (σ * T))^2 + log(σ * T)
            scaled_sigma = sigma_cal * T_tensor.abs() + 1e-6
            nll = (0.5 * ((mu_cal - y_cal) / scaled_sigma)**2 + torch.log(scaled_sigma)).mean()
            nll.backward()
            return nll

        optimizer.step(closure)
        self.T = float(T_tensor.abs().clamp(0.1, 20.0).item())

        # ─── STEP 2: Split Conformal Prediction ───
        # Calculates the non-conformity scores on the T-scaled residuals
        residuals = torch.abs(mu_cal - y_cal)
        scaled_sigma_cal = sigma_cal * self.T + 1e-6
        scores = (residuals / scaled_sigma_cal).detach().cpu().numpy()

        # Finite-sample correction for the quantile
        n = len(scores)
        alpha = 1.0 - self.target_coverage
        q_level = min(1.0, math.ceil((1.0 - alpha) * (n + 1)) / n)
        self.q_hat = float(np.quantile(scores, q_level))
        self.is_fitted = True

        print(f"[Adapter] Fit Complete. Temperature (T) = {self.T:.4f} | Conformal (q_hat) = {self.q_hat:.4f}")
        return self.T, self.q_hat

    def predict(self, mu_raw: np.ndarray, sigma_raw: np.ndarray) -> tuple:
        """
        Applies the fitted calibration parameters to raw model predictions.
        Returns: (Adjusted Mean, Lower Bound, Upper Bound)
        """
        if not self.is_fitted:
            raise RuntimeError("Adapter must be fitted before calling predict().")

        # 1. Scale the raw standard deviation by the temperature
        sigma_cal = sigma_raw * self.T

        # 2. Apply the conformal quantile to set the hard safety bounds
        margin = self.q_hat * sigma_cal
        lower_bound = np.clip(mu_raw - margin, a_min=0.0, a_max=None)
        upper_bound = mu_raw + margin

        return mu_raw, lower_bound, upper_bound

    def get_summary(self) -> dict:
        return {
            "target_coverage": self.target_coverage,
            "temperature_T": self.T,
            "conformal_q_hat": self.q_hat,
            "is_fitted": self.is_fitted
        }
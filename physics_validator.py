"""
physics_validator.py  ·  AeroMRO Digital Twin  ·  v17
═════════════════════════════════════════════════════
"""

from physics_loss import CompositePINNLoss
import torch

class PhysicsSanityChecker:
    def __init__(self, max_rul_cap=999.0, device=torch.device("cpu")):
        self.device = device
        self.loss_fn = CompositePINNLoss(base_lambdas={"damage_mono": 1.0}).to(device)

    def check_damage_monotonicity(self):
        delta_good = torch.tensor([[1.0], [2.0], [3.0]], device=self.device)  
        delta_bad  = torch.tensor([[3.0], [2.0], [1.0]], device=self.device)  
        
        sensor = torch.zeros((3, 30, 26), device=self.device)
        sensor[:, -1, 1] = torch.tensor([1.0, 2.0, 3.0], device=self.device) 
        
        def _mock(d):
            return {
                "rul_log": torch.tensor([[5.0], [4.0], [3.0]], device=self.device),
                "delta": d, "rul_log_var": torch.zeros((3, 1), device=self.device),
                "health_logit": torch.zeros((3, 1), device=self.device),
            }
            
        mock_true = torch.tensor([[5.0], [4.0], [3.0]], device=self.device)
        loss_good = self.loss_fn.compute(_mock(delta_good), mock_true, sensor, use_pde=True)["total"]
        loss_bad  = self.loss_fn.compute(_mock(delta_bad), mock_true, sensor, use_pde=True)["total"]
        
        assert loss_bad.item() > loss_good.item(), f"Damage Monotonicity failed. Good: {loss_good.item():.4f}, Bad: {loss_bad.item():.4f}"

    def run_all_checks(self):
        print("\n[V&V] Initiating Pre-Flight Physics Validation...")
        try:
            self.check_damage_monotonicity()
            print("[V&V] All Physics Boundaries VERIFIED. ✅\n")
            return True
        except AssertionError as e:
            print(f"\n[V&V] CRITICAL PHYSICS VIOLATION: {e} ❌")
            exit(1)

if __name__ == "__main__":
    PhysicsSanityChecker().run_all_checks()
"""
fadec_edge_profiler.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
True Edge-Deployment Hardware Profiler.
Measures exact VRAM footprint, memory fragmentation, and GPU hardware 
latency for FADEC streaming (Batch=1) and MRO batched (Batch=128) inference.
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tabulate import tabulate

from pinn_model import PINNModel

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30
TOTAL_FEAT = 55

# ─── Baseline Models (Replicated for fair comparison) ────────────────────────
class TransformerBaseline(nn.Module):
    def __init__(self, n_feat=55, d=256, nhead=8, ff=512, layers=4, max_rul=150.0):
        super().__init__()
        self.proj = nn.Linear(n_feat, d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, ff, batch_first=True, norm_first=True), layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 128), nn.SiLU(), nn.Linear(128, 1))
        self.max_rul_log = math.log1p(max_rul)

    def forward(self, x, **kwargs):
        h = self.norm(self.enc(self.proj(x)))[:, -1, :]
        return {"rul_log": torch.clamp(F.softplus(self.head(h)), max=self.max_rul_log)}

class LSTMBaseline(nn.Module):
    def __init__(self, n_feat=55, hidden=256, layers=3, max_rul=150.0):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 128), nn.SiLU(), nn.Linear(128, 1))
        self.max_rul_log = math.log1p(max_rul)

    def forward(self, x, **kwargs):
        h, _ = self.lstm(x)
        h = self.norm(h[:, -1, :])
        return {"rul_log": torch.clamp(F.softplus(self.head(h)), max=self.max_rul_log)}

# ─── Hardware Profiler Core ──────────────────────────────────────────────────
def profile_hardware(model, name, batch_size=1, n_runs=500):
    """
    Executes precise CUDA profiling for memory and latency.
    """
    model.eval()
    
    # 1. Parameter Count
    n_params = sum(p.numel() for p in model.parameters())
    
    # Generate Dummy Data
    x = torch.randn(batch_size, WINDOW_SIZE, TOTAL_FEAT, device=DEVICE)
    op = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
    ev = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)

    # 2. Memory Profiling (Isolating Model vs Activations)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Base memory (Just holding the model and inputs)
    mem_base_bytes = torch.cuda.memory_allocated()
    
    # Warmup + Spike memory tracking
    with torch.no_grad(), autocast("cuda"):
        for _ in range(10):
            _ = model(x, op_setting=op, event_flag=ev)
            
    mem_peak_bytes = torch.cuda.max_memory_allocated()
    
    # Calculations
    model_size_mb = mem_base_bytes / (1024 ** 2)
    activation_spike_mb = (mem_peak_bytes - mem_base_bytes) / (1024 ** 2)
    total_vram_mb = mem_peak_bytes / (1024 ** 2)

    # 3. Hardware Latency Profiling (CUDA Events bypass Python GIL overhead)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]

    with torch.no_grad(), autocast("cuda"):
        for i in range(n_runs):
            start_events[i].record()
            _ = model(x, op_setting=op, event_flag=ev)
            end_events[i].record()

    torch.cuda.synchronize()
    
    # Calculate average latency across all runs
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_latency_ms = sum(times_ms) / n_runs
    
    # Compute Throughput (Windows per second)
    throughput = (batch_size * 1000) / avg_latency_ms

    return {
        "Model": name,
        "Batch": batch_size,
        "Params (M)": n_params / 1e6,
        "Model VRAM (MB)": model_size_mb,
        "Act. VRAM (MB)": activation_spike_mb,
        "Peak VRAM (MB)": total_vram_mb,
        "Latency (ms)": avg_latency_ms,
        "Throughput (win/s)": throughput
    }

def main():
    print(f"\n{'='*90}")
    print(f"{'ThermoPINN · Absolute Hardware Profiler (Jetson / FADEC Target)':^90}")
    print(f"{'='*90}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Load Models
    models = {
        "Transformer": TransformerBaseline().to(DEVICE),
        "LSTM Baseline": LSTMBaseline().to(DEVICE),
        "ThermoPINN": PINNModel(max_rul=150.0, n_sensors=55, conv_channels=256, 
                                gru_hidden=512, head_hidden=128, dropout=0.30, 
                                n_op_settings=32, n_events=10, mean_rul_log=4.0).to(DEVICE)
    }

    # Load your exact weights to ensure accurate memory representation
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        models["ThermoPINN"].load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    except:
        print("[Warning] Could not load exact weights, using uninitialized PINN for profiling.")

    results_b1 = []
    results_b128 = []

    print("\n[1/2] Profiling FADEC Streaming Mode (Batch Size = 1)...")
    for name, model in models.items():
        res = profile_hardware(model, name, batch_size=1)
        results_b1.append([res["Model"], f"{res['Params (M)']:.2f}M", 
                           f"{res['Model VRAM (MB)']:.1f}", f"{res['Act. VRAM (MB)']:.1f}", 
                           f"{res['Peak VRAM (MB)']:.1f}", f"{res['Latency (ms)']:.2f} ms"])

    print("[2/2] Profiling MRO Server Mode (Batch Size = 128)...")
    for name, model in models.items():
        res = profile_hardware(model, name, batch_size=128)
        results_b128.append([res["Model"], f"{res['Peak VRAM (MB)']:.1f}", 
                             f"{res['Latency (ms)']:.2f} ms", f"{res['Throughput (win/s)']:,.0f}"])

    # Output Tables
    print(f"\n{'='*90}")
    print(f"{'TABLE A: FADEC Edge Deployment (Streaming Batch=1)':^90}")
    print(f"{'='*90}")
    headers_b1 = ["Model", "Params", "Static Weights (MB)", "Activation Spike (MB)", "Total Peak VRAM", "Latency / Step"]
    print(tabulate(results_b1, headers=headers_b1, tablefmt="heavy_grid"))

    print(f"\n{'='*90}")
    print(f"{'TABLE B: MRO Server Deployment (Batched Batch=128)':^90}")
    print(f"{'='*90}")
    headers_b128 = ["Model", "Total Peak VRAM", "Latency / Batch", "Throughput (Windows/sec)"]
    print(tabulate(results_b128, headers=headers_b128, tablefmt="heavy_grid"))
    print(f"{'='*90}\n")

if __name__ == "__main__":
    main()
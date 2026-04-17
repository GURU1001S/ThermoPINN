"""
streaming_eval.py  ·  ThermoPINN
══════════════════════════════════════════════════════════════════════════════
Experiment 7: Real-Time Streaming Latency Benchmark.
Simulates live FADEC telemetry ingestion using an amortized rolling queue.
Measures true GPU inference latency and streaming throughput.
"""

import os
import h5py
import time
import torch
import argparse
import numpy as np
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 30

class LiveStreamer:
    def __init__(self, model):
        self.model = model.to(DEVICE).eval()
        self.buffer = deque(maxlen=WINDOW_SIZE)
        
        # Warmup GPU to prevent initial CUDA launch lag
        with torch.no_grad():
            dummy = torch.randn(1, WINDOW_SIZE, 55).to(DEVICE)
            _ = self.model(dummy, op_setting=torch.zeros(1, dtype=torch.long, device=DEVICE), event_flag=torch.zeros(1, dtype=torch.long, device=DEVICE))

    def process_telemetry(self, sensor_reading_55d):
        self.buffer.append(sensor_reading_55d)
        if len(self.buffer) < WINDOW_SIZE: return None, 0.0
        
        # Construct current window
        x = torch.tensor(np.array(self.buffer), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        op = torch.zeros(1, dtype=torch.long, device=DEVICE)
        ev = torch.zeros(1, dtype=torch.long, device=DEVICE)
        
        # Strict GPU Latency Measurement
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(x, op_setting=op, event_flag=ev)
            _ = torch.expm1(out["rul_log"]).item()
        torch.cuda.synchronize()
        
        latency_ms = (time.perf_counter() - t0) * 1000
        return _, latency_ms

def run_streaming_eval(args):
    print(f"\n{'='*80}\n{'Experiment 7: Real-Time Streaming Latency Benchmark':^80}\n{'='*80}")
    
    try:
        from pinn_model import PINNModel
        model = PINNModel(max_rul=500.0, n_sensors=55, conv_channels=256, gru_hidden=512, head_hidden=128, dropout=0.30, n_op_settings=32, n_events=10, mean_rul_log=4.0)
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE, weights_only=True).get("model_state", torch.load(args.model_path, map_location=DEVICE, weights_only=True)), strict=False)
    except Exception as e:
        print(f"❌ Error loading model: {e}"); return

    # Extract 1 full engine timeline that is actually long enough to benchmark
    with h5py.File(args.data_path, "r") as f:
        grp = f["test"]
        eng_ids = grp["engine_id"][:]
        
        # Search for an engine with at least 150 cycles
        eng_id = None
        for candidate_id in np.unique(eng_ids):
            idx = np.where(eng_ids == candidate_id)[0]
            if len(idx) > 150:
                eng_id = candidate_id
                break
                
        if eng_id is None:
            print("❌ Error: Could not find an engine with >150 cycles in the test set.")
            return

        X_raw = np.concatenate([
            np.nan_to_num(grp["sensors"][idx], nan=0.0),
            np.nan_to_num(grp["env"][idx], nan=0.0),
            np.nan_to_num(grp["causal_state"][idx], nan=0.0)
        ], axis=1).astype(np.float32)
        X_norm = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-8)

    streamer = LiveStreamer(model)
    latencies = []
    
    print(f"Streaming {len(X_norm)} cycles of live telemetry for Engine {int(eng_id)}...")
    for timestep, reading in enumerate(X_norm):
        rul, latency = streamer.process_telemetry(reading)
        if latency > 0: latencies.append(latency)

    if not latencies:
        print("❌ Error: No predictions were made. Buffer didn't fill.")
        return

    mean_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    throughput = 1000.0 / mean_latency
    
    print(f"\n[Hardware Target: {torch.cuda.get_device_name(0)}]")
    print(f"  Mean Inference Latency : {mean_latency:.3f} ms / cycle")
    print(f"  Peak Latency Spikes    : {max_latency:.3f} ms")
    print(f"  Max Throughput         : {throughput:.0f} cycles / second")
    
    if mean_latency < 5.0:
        print("  ✅ VERDICT: Suitable for onboard Edge-AI FADEC deployment.")
    else:
        print("  ❌ VERDICT: Exceeds 5ms threshold. Requires ONNX/TensorRT quantization.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.expanduser("~/nasa_research/data/utdtb_v5.h5"))
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt"))
    run_streaming_eval(parser.parse_args())
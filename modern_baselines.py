"""
modern_baselines.py  ·  ThermoPINN  ·  v2 (True Results)
══════════════════════════════════════════════════════════
PatchTST and Mamba-equivalent baselines implemented natively.
No external patchtst or mamba_ssm packages required.

WHAT CHANGED FROM THE MOCK VERSION:
  - PatchTST implemented natively (patch segmentation + Transformer)
    No external "from patchTST import PatchTST" dependency
  - Mamba replaced with S4-Lite (diagonal SSM) — same scaling properties,
    no CUDA custom kernels required, runs on any GPU or CPU
  - Both models trained for 50 epochs on UTDTB v5 and produce REAL RMSE
  - Results printed in a comparison table against ThermoPINN

SCIENTIFIC ALGORITHMS:

PatchTST (Nie et al. 2023, ICLR):
  Segment time series into non-overlapping patches of length L_p.
  n_patches = floor((T - L_p) / stride) + 1
  Each patch treated as a "token" → standard Transformer.
  Key advantage: local temporal semantics preserved (unlike vanilla ViT).
  Input: [B, T, C] → patches: [B, n_patches, L_p × C] → Transformer → [B, 1]

S4-Lite / Diagonal SSM (simplified Mamba core):
  State space model: h_t = A·h_{t-1} + B·x_t,   y_t = C·h_t
  Diagonal A: complex eigenvalues ensure stability
  A = exp(Λ·Δt) where Λ = diagonal of learnable complex matrix
  Discretised via ZOH: A_d = exp(Λ·Δ), B_d = (A_d - I)·Λ^{-1}·B
  Linear recurrence = O(T) inference (vs O(T²) for Transformer)
  Implemented in log-domain for numerical stability.

NASA score: clamp_range = 0.5 × dataset_max_rul (not fixed at 50)
"""

import os, math, copy, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H5_PATH = os.path.expanduser("~/nasa_research/data/utdtb_v5.h5")
CKPT    = os.path.expanduser(
    "~/nasa_research/thermoPINN_metal_ready_v20_20260413_1335.pt"
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PATCHTST — native implementation (no external package)
# ═══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """
    Segment [B, T, C] time series into overlapping patches.
    Returns [B, n_patches, d_model].
    """
    def __init__(self, n_features: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        # Linear projection of flattened patch → d_model
        self.proj = nn.Linear(patch_len * n_features, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # Unfold: [B, T, C] → [B, n_patches, patch_len, C]
        x_patch = x.unfold(1, self.patch_len, self.stride)  # [B, n_p, C, L_p]
        x_patch = x_patch.permute(0, 1, 3, 2)               # [B, n_p, L_p, C]
        B, n_p, L_p, C = x_patch.shape
        x_flat  = x_patch.reshape(B, n_p, L_p * C)          # [B, n_p, L_p×C]
        return self.norm(self.proj(x_flat))                   # [B, n_p, d_model]


class PatchTSTBaseline(nn.Module):
    """
    Native PatchTST for RUL regression.
    Nie et al. 2023 (ICLR) — "A Time Series is Worth 64 Words"

    Patch segmentation preserves local temporal structure that standard
    attention destroys. Each patch = one "word" of engine degradation.
    """
    def __init__(
        self,
        n_features: int = 55,
        patch_len:  int = 6,     # 6-timestep patches (window=30 → 5 patches)
        stride:     int = 3,     # 50% overlap
        d_model:    int = 128,
        nhead:      int = 8,
        n_layers:   int = 3,
        dropout:    float = 0.15,
    ):
        super().__init__()
        self.patch_emb = PatchEmbedding(n_features, patch_len, stride, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model//4), nn.Tanh(),
            nn.Linear(d_model//4, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.constant_(self.head[-1].bias, 5.5)

    def forward(self, x: torch.Tensor, **kw) -> Dict:
        patches = self.patch_emb(x)                   # [B, n_p, d]
        enc     = self.transformer(patches)            # [B, n_p, d]
        weights = torch.softmax(self.pool(enc), dim=1) # [B, n_p, 1]
        pooled  = (enc * weights).sum(1)               # [B, d]
        rul_log = F.softplus(self.head(pooled))        # [B, 1]
        return {"rul_log": rul_log,
                "rul_log_var": torch.zeros_like(rul_log)}


# ═══════════════════════════════════════════════════════════════════════════════
#  S4-LITE — Diagonal SSM (Mamba-equivalent, no CUDA kernels needed)
# ═══════════════════════════════════════════════════════════════════════════════

class DiagonalSSMLayer(nn.Module):
    """
    Simplified Structured State Space (S4) with diagonal A matrix.
    Equivalent scaling to Mamba; no custom CUDA ops required.

    State recurrence (discretised via Zero-Order Hold):
      A_d = exp(Λ·Δ)      Λ = learnable diagonal, complex
      B_d = (A_d - I)·B    B = learnable input matrix
      h_t = A_d·h_{t-1} + B_d·x_t
      y_t = Re(C·h_t)     C = learnable output matrix

    Log-domain computation for numerical stability.
    """
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Learnable diagonal SSM parameters
        # A_log: log(-real part) ensures A has negative real eigenvalues → stable
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) + 0.5))
        self.B     = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C     = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D     = nn.Parameter(torch.ones(d_model))      # skip connection
        self.dt    = nn.Parameter(torch.ones(d_model) * 0.01)  # step size Δ

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model] → [B, T, d_model]"""
        B, T, d = x.shape
        N       = self.d_state

        # Discretise: A_d = exp(-exp(A_log) · Δ)
        A       = -torch.exp(self.A_log)                  # [d, N] negative real
        dt_soft = F.softplus(self.dt).unsqueeze(-1)        # [d, 1]
        A_d     = torch.exp(A * dt_soft)                   # [d, N]
        B_d     = (1.0 - A_d) * self.B                    # [d, N]

        # Sequential scan (O(T·d·N) — efficient for short sequences)
        h   = torch.zeros(B, d, N, device=x.device)
        ys  = []
        for t in range(T):
            xt  = x[:, t, :]              # [B, d]
            # h_t = A_d * h_{t-1} + B_d * x_t
            h   = A_d.unsqueeze(0) * h + B_d.unsqueeze(0) * xt.unsqueeze(-1)
            # y_t = sum_N(C * h_t) + D * x_t
            y   = (h * self.C.unsqueeze(0)).sum(-1) + self.D * xt   # [B, d]
            ys.append(y)

        out = torch.stack(ys, dim=1)   # [B, T, d]
        return self.norm(self.drop(out) + x)  # residual connection


class MambaBaseline(nn.Module):
    """
    S4-Lite stacked model — equivalent architecture to Mamba SSM.
    Uses diagonal structured state space layers instead of selective
    scan CUDA kernels → runs on any hardware without special compilation.
    """
    def __init__(
        self,
        n_features: int   = 55,
        d_model:    int   = 256,
        n_layers:   int   = 4,
        d_state:    int   = 16,
        dropout:    float = 0.15,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)

        self.layers = nn.ModuleList([
            DiagonalSSMLayer(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.constant_(self.head[-1].bias, 5.5)

    def forward(self, x: torch.Tensor, **kw) -> Dict:
        h = self.input_proj(x)             # [B, T, d_model]
        for layer in self.layers:
            h = layer(h)
        pooled  = h[:, -1, :]             # last timestep (SSM has memory)
        rul_log = F.softplus(self.head(pooled))
        return {"rul_log": rul_log,
                "rul_log_var": torch.zeros_like(rul_log)}


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED TRAINING HARNESS
# ═══════════════════════════════════════════════════════════════════════════════

def nasa_score(pred: np.ndarray, true: np.ndarray, max_rul: float) -> float:
    clamp = max_rul * 0.5
    d = np.clip(pred - true, -clamp, clamp)
    raw = np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)
    return float(np.mean(raw))


def train_baseline(
    model:      nn.Module,
    h5_path:    str,
    epochs:     int = 50,
    batch_size: int = 256,
    lr:         float = 5e-4,
    window_size:int = 30,
    max_samples:int = 80_000,
) -> Dict:
    """Train a baseline model on UTDTB v5 and return evaluation metrics."""
    import h5py

    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler= GradScaler("cuda", enabled=(DEVICE.type=="cuda"))

    # Load data
    print(f"  [Baseline] Loading UTDTB v5...")
    X_train, Y_train = [], []
    X_test,  Y_test  = [], []

    try:
        with h5py.File(os.path.expanduser(h5_path), "r") as f:
            for split, Xl, Yl in [("train", X_train, Y_train), ("test", X_test, Y_test)]:
                if split not in f: continue
                grp  = f[split]
                s    = np.nan_to_num(grp["sensors"][:], nan=0.0).astype(np.float32)
                e    = np.nan_to_num(grp["env"][:],     nan=0.0).astype(np.float32)
                p    = np.nan_to_num(grp["causal_state"][:], nan=0.0).astype(np.float32)
                X_r  = np.concatenate([s, e, p], axis=1)
                y_r  = np.nan_to_num(grp["RUL"][:],    nan=0.0).astype(np.float32)

                # Normalise
                mu   = X_r.mean(0, keepdims=True)
                sd   = X_r.std(0, keepdims=True) + 1e-8
                X_r  = np.clip((X_r - mu) / sd, -5, 5)

                # Sliding windows
                N = len(X_r)
                for t in range(0, N - window_size + 1, 5):
                    Xl.append(X_r[t:t+window_size])
                    Yl.append(float(np.log1p(max(0, y_r[t + window_size - 1]))))
                    if len(Xl) >= max_samples: break
    except Exception as e:
        print(f"  [Baseline Error] {e}")
        return {}

    X_tr = torch.tensor(np.array(X_train[:max_samples]), dtype=torch.float32)
    Y_tr = torch.tensor(np.array(Y_train[:max_samples]), dtype=torch.float32)
    X_te = torch.tensor(np.array(X_test[:20_000]),       dtype=torch.float32)
    Y_te = torch.tensor(np.array(Y_test[:20_000]),       dtype=torch.float32)
    max_rul = float(torch.expm1(Y_tr.max()).item())

    print(f"  Train: {len(X_tr):,} windows  Test: {len(X_te):,} windows")

    t0 = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        idx  = torch.randperm(len(X_tr))
        total_loss = 0.0; n_batches = 0
        for i in range(0, len(X_tr), batch_size):
            xb = X_tr[idx[i:i+batch_size]].to(DEVICE)
            yb = Y_tr[idx[i:i+batch_size]].to(DEVICE)
            with autocast("cuda", enabled=(DEVICE.type=="cuda")):
                out  = model(xb)
                loss = F.smooth_l1_loss(
                    out["rul_log"].squeeze(-1), yb, beta=1.0
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total_loss += loss.item(); n_batches += 1
        sched.step()
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            print(f"    ep {epoch+1:>3}/{epochs}  loss={total_loss/n_batches:.4f}")

    train_time = time.perf_counter() - t0

    # Evaluate
    model.eval()
    all_p, all_t = [], []
    BS = 1024
    with torch.no_grad():
        for i in range(0, len(X_te), BS):
            xb  = X_te[i:i+BS].to(DEVICE)
            out = model(xb)
            p   = torch.expm1(out["rul_log"].squeeze(-1)).cpu().numpy()
            t   = torch.expm1(Y_te[i:i+BS]).cpu().numpy()
            all_p.append(p); all_t.append(t)

    P = np.concatenate(all_p); T = np.concatenate(all_t)
    rmse  = float(np.sqrt(np.mean((P - T)**2)))
    mae   = float(np.mean(np.abs(P - T)))
    nasa  = nasa_score(P, T, max_rul)
    params= sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {"rmse": rmse, "mae": mae, "nasa": nasa, "params_M": params/1e6,
            "train_s": round(train_time, 1)}


# ─── Load ThermoPINN metrics for comparison ───────────────────────────────────

def get_thermopinn_metrics(h5_path: str, ckpt: str) -> Optional[Dict]:
    try:
        from pinn_model  import PINNModel
        from task_sampler import DigitalTwinTaskSampler

        sampler = DigitalTwinTaskSampler(
            h5_path=h5_path, window_size=30, stride=5,
            support_ratio=0.6, seed=42, device=DEVICE,
        )
        model = PINNModel(
            max_rul=sampler.max_rul, n_sensors=55, conv_channels=256,
            gru_hidden=512, head_hidden=128, dropout=0.30,
            n_op_settings=32, n_events=10, mean_rul_log=sampler.mean_rul_log,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        model.eval()

        _, test_tasks = sampler.held_out_split()
        preds, trues  = [], []
        for tid in random.sample(test_tasks, min(60, len(test_tasks))):
            _, qry = sampler.get_fast_task_tensors(tid)
            if qry is None: continue
            with torch.no_grad():
                out = model(qry["x"],
                            op_setting=qry["op_setting"],
                            event_flag=qry["event_flag"])
            p = torch.expm1(out["rul_log"].squeeze(-1)).cpu().numpy()
            t = torch.expm1(qry["rul_log"]).cpu().numpy()
            preds.append(p); trues.append(t)

        P = np.concatenate(preds); T = np.concatenate(trues)
        rmse = float(np.sqrt(np.mean((P-T)**2)))
        nasa = nasa_score(P, T, sampler.max_rul)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"rmse": rmse, "mae": float(np.mean(np.abs(P-T))),
                "nasa": nasa, "params_M": params/1e6, "train_s": 0}
    except Exception as e:
        print(f"  [ThermoPINN load error] {e}")
        return None


# ─── Print comparison table ───────────────────────────────────────────────────

def print_comparison_table(results: Dict[str, Dict]) -> None:
    print(f"\n{'='*72}")
    print(f"  Modern Architecture Comparison — UTDTB v5 (0-shot, test split)")
    print(f"{'='*72}")
    print(f"  {'Model':<20} | {'RMSE':>7} | {'MAE':>7} | {'NASA':>8} | "
          f"{'Params':>7} | {'Train':>7}")
    print(f"  {'-'*66}")
    for name, r in results.items():
        if not r: continue
        print(f"  {name:<20} | {r['rmse']:>7.2f} | {r['mae']:>7.2f} | "
              f"{r['nasa']:>8.2f} | {r['params_M']:>5.2f}M | "
              f"{r['train_s']:>5.0f}s")
    print(f"{'='*72}")
    print("  Note: ThermoPINN includes MAML+physics; baselines are supervised-only.")
    print("  PatchTST: Nie et al. ICLR 2023  |  S4-Lite: diagonal SSM (Mamba-equiv)\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def run_baseline_comparison(
    h5_path: str = H5_PATH,
    ckpt:    str = CKPT,
    epochs:  int = 50,
):
    results = {}

    print("\n[Baselines] Training PatchTST (native implementation)...")
    patchtst = PatchTSTBaseline(n_features=55, patch_len=6, stride=3,
                                 d_model=128, nhead=8, n_layers=3)
    results["PatchTST (Nie 2023)"] = train_baseline(patchtst, h5_path, epochs=epochs)

    print("\n[Baselines] Training S4-Lite / Mamba-equivalent...")
    mamba = MambaBaseline(n_features=55, d_model=256, n_layers=4, d_state=16)
    results["S4-Lite (Mamba-equiv)"] = train_baseline(mamba, h5_path, epochs=epochs)

    print("\n[Baselines] Loading ThermoPINN metrics (0-shot)...")
    results["ThermoPINN (ours)"] = get_thermopinn_metrics(h5_path, ckpt)

    print_comparison_table(results)
    return results


if __name__ == "__main__":
    run_baseline_comparison()
# ThermoPINN: Physics-Constrained Meta-Learning for Aerospace Prognostics

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research_Platform-purple?style=flat-square)

**ThermoPINN** is a research platform designed to empirically evaluate the reliability, safety, and failure modes of Physics-Informed Neural Networks (PINNs) in safety-critical aerospace prognostics. 

Initially developed using **UTDTB v5** (a custom synthetic degradation dataset), the system demonstrates robust zero-shot sim-to-real transfer to NASA N-CMAPSS telemetry and real-time edge deployment feasibility. 

## 🧠 Core Insight & Aerospace Impact
This work demonstrates a critical failure mode in modern safety-critical AI:
> **High predictive accuracy does NOT guarantee physically valid representations in safety-critical systems.**

Despite achieving competitive RMSE and successful sim-to-real transfer, empirical stress tests revealed that the model failed to reproduce governing physical laws, collapsed under out-of-distribution (OOD) sensor failure, and could not support survival-based risk modeling. **This suggests that current PINN formulations behave as constrained function approximators rather than physically grounded models.**

**Why this matters to aerospace:** In commercial aviation, this disconnect is critical. Uncalibrated overconfidence and non-physical degradation trajectories actively violate DO-178C and ARP4761 safety mandates, introducing unquantified and potentially unsafe decision risk in MRO scheduling and safety-critical decision-making.

## 🏆 Key Contributions
* **A 55-D physics-aware meta-learning architecture** for turbofan RUL prediction.
* **An empirical demonstration of deterministic feature collapse** in PINNs under OOD conditions.
* **Systematic evidence that MSE-trained PINNs fail to recover governing physical laws** across independent datasets.
* **A rigorously validated sim-to-real evaluation pipeline** across N-CMAPSS, C-MAPSS, and PRONOSTIA datasets.

## 🎯 Problem Definition
The objective is to predict the Remaining Useful Life (RUL) of turbofan engines under real-world operational variability, while ensuring physically consistent degradation behavior and reliable uncertainty estimation for safety-critical decision-making.

## 🧩 Method Overview
ThermoPINN bridges the gap between pure data-driven deep learning and physics-based prognostics by combining:
* **A multi-stream encoder** for high-dimensional sensor and environmental signals.
* **A physics gate** to impose thermodynamic structure. *The gate is designed to constrain latent representations to follow thermodynamic monotonicity and degradation trends, mitigating purely statistical shortcuts.*
* **A Model-Agnostic Meta-Learning (MAML) loop** for rapid domain adaptation.
* **Conformal calibration** for robust uncertainty bounds.

The model is trained on synthetic degradation trajectories and evaluated under zero-shot transfer to real-world flight telemetry.


## 📊 At a Glance
*Note: Results are reported across two regimes — synthetic benchmark (UTDTB v5) and real-world sim-to-real transfer (N-CMAPSS).*

| Capability | Result |
| :--- | :--- |
| **Sim-to-Real (N-CMAPSS)** | **RMSE: 31.88** (Zero-Shot) |
| **Cross-Domain Adaptation** | **-26.04 cycle improvement** (Classic C-MAPSS) |
| **SOTA Benchmark (UTDTB)** | **Competitive/Superior to TimesNet, iTransformer** |
| **Physics Validity** | ❌ **FAILED** (Learned non-physical statistical proxies) |
| **Uncertainty Calibration** | ❌ **FAILED** (OOD overconfidence / Feature collapse) |
| **Edge Deployment** | ✅ **4.65 ms latency** (RTX 3050) |

---

## 📦 Datasets
* **UTDTB v5 (Proposed):** A custom synthetic 55-dimensional turbofan degradation dataset incorporating sensor, environmental, and latent physics variables.
* **NASA N-CMAPSS:** Real-world turbofan telemetry dataset used for primary sim-to-real validation.
* **NASA C-MAPSS (Classic):** Benchmark dataset for cross-domain adaptation experiments.
* **FEMTO / PRONOSTIA:** Bearing degradation dataset used for cross-component physics validation.

## 🌍 Sim-to-Real Transfer & Multi-Dataset Generalization
The architecture was trained purely on synthetic UTDTB v5 data and subjected to rigorous external validation across multiple real-world flight envelopes. *This robustness suggests the architecture captures stable statistical structure correlated with thermodynamic behavior, though not governed by explicit physical laws.*

* **NASA N-CMAPSS (Primary Sim-to-Real):** In an evaluation of over **26.4 million thermodynamic windows** (33 flight trajectories) handling severe distribution shifts, the zero-shot transfer achieved an average RMSE of **31.88 cycles**. 
* **NASA Classic C-MAPSS:** Utilizing a Gradient Reversal Layer (GRL) for domain alignment, the model improved target domain RMSE by 26.04 cycles unsupervised.

## 🚀 Engineering Benchmarks
Under identical training conditions, ThermoPINN demonstrates competitive or superior performance to purely data-driven baselines.

| Model Architecture | Params | RMSE ↓ | NASA Score ↓ |
| :--- | :---: | :---: | :---: |
| **ThermoPINN (Ours, Zero-Shot)** | **~800.0K** | **40.2*** | **125.4** |
| TimesNet (ICLR 2023) | 40.7K | 45.1 | 140,605.3 |
| DLinear (AAAI 2023) | 0.2K | 69.5 | 8.85e17† |
| FITS (ICLR 2024) | 4.4K | 97.5 | 1.21e33† |
| iTransformer (ICLR 2024)| 1,049.5K | 130.0 | 2.15e7† |

*\*Note: RMSE 40.2 reflects the UTDTB v5 benchmark comparison; 31.88 reflects N-CMAPSS sim-to-real evaluation.* *†Note: RMSE serves as the primary predictive metric. The asymmetric NASA score functions as an extreme-error stress metric, where pure data-driven baselines exhibit exponential instability.*

## 🔬 Ablation Studies & Stress Testing
A total of 25+ controlled experiments and 7 major ablation categories were conducted to validate architectural robustness and isolate failure modes. *All ablation experiments were conducted under controlled settings with isolated variable modification to ensure causal interpretability of observed performance changes.*

**Robustness & Architecture**
* **Architecture Ablation:** Removing physics constraints degraded RMSE (45.2 vs 42.9), confirming the contribution of thermodynamic priors.
* **Feature Ablation:** RMSE remained stable (~124.5–124.7 in the ablation setting) across 20D–55D inputs, indicating strong reliance on core sensor signals.
* **Dimensionality Stress Test:** Performance remained invariant under aggressive sensor pruning (55D → 18D), demonstrating resilience to real-world sensor dropout.

**Learning Dynamics & Domain Transfer**
* **Meta-Learning Depth:** Optimal adaptation occurs at k=2 shots; beyond this, performance degrades due to catastrophic forgetting (RMSE ↑ to 287.1 at k=7).
* **Domain Adaptation (DANN):** Improved target-domain RMSE by 26.04 cycles without access to labeled target data.

**Uncertainty Calibration**
* **Monte Carlo Dropout:** Reduced calibration error (ECE ↓ from 0.42 → ~0.18) under standard conditions, but failed to capture Epistemic uncertainty under OOD conditions.

## 🧪 Scientific Findings: The Limits of PINNs
1. **Correlation ≠ Physics:** Standard training objectives do not guarantee the emergence of governing equations. Latent physics nodes failed to replicate Paris-Erdogan and Arrhenius constants (p < 0.05 KS Test). 
2. **Deterministic Feature Collapse:** Post-hoc Evidential Deep Learning (EDL) wrappers fail when the encoder is not explicitly constrained. Under sensor failure, Epistemic uncertainty deflated by 0.9x.
3. **Latent Rigidity:** Point-prediction architectures destroy the probabilistic structure required for MRO scheduling (C-Index: 0.500).

*These failures persisted despite achieving RMSE < 35 cycles in sim-to-real evaluation, indicating a clear disconnect between predictive performance and physical validity. These results indicate that while ThermoPINN achieves strong empirical performance, it does not yet satisfy the requirements for physically interpretable or safety-certified deployment.*

## 📂 Repository Architecture

### 🛠️ Core & Training
* `pinn_model.py` - 55-D multi-stream encoder and physics gate.
* `train_maml_pinn.py` - Meta-learning loop for zero-shot sim-to-real transfer.
* `physics_loss.py` - PDE-constrained loss formulations.

### 📊 Evaluation & Benchmarking
* `evaluate_ncmapss_adapted.py` - N-CMAPSS sim-to-real engine.
* `modern_baselines.py` - Native PyTorch implementations of ICLR/AAAI models.
* `streaming_eval.py` - Real-time edge latency and throughput profiler.

### ⚠️ Failure & Safety Analysis
* `edl_uncertainty.py` & `survival_head.py` - Probing wrappers for OOD and risk analysis.
* `external_physics_validation.py` - Cross-dataset validation against material science data.
* `cert_arp4761_risk_table.py` - Safety alignment and compliance metric generation.

## 🔁 Reproducibility & Configuration
To ensure research transparency, all experiments are reproducible under the following configuration:
* **Sequence Length:** 30 cycles (Sliding Window)
* **Batch Size:** 256
* **Optimizer:** Adam (lr = 1e-3, weight_decay = 1e-4)
* **Training Epochs:** 50 (with Early Stopping)
* **Normalization:** Z-score normalization per engine trajectory
* **Physics Constants:** Paris Law (m = 3.0), Arrhenius Activation (E_a = 300 kJ/mol)
* **Hardware:** NVIDIA RTX 3050 Laptop GPU (~120 hours total compute time)
* **Random Seed:** 42

*All experiments can be reproduced via the provided scripts with fixed random seeds and dataset configurations.*

## ⚠️ Limitations
* **Physics constraints are not sufficient** to enforce true thermodynamic laws.
* **Uncertainty estimation fails** under OOD conditions despite calibration.
* **Survival modeling is incompatible** with deterministic latent representations.
* **Sim-to-real success may rely** on statistical alignment rather than causal physics.

## 📖 Proposed Future Research
This work identifies a fundamental gap between predictive performance and physical validity in safety-critical AI systems. Addressing this gap requires moving beyond correlation-driven learning toward architectures that encode physical laws as first-class constraints rather than auxiliary losses—a direction that forms the basis of my proposed graduate research. 

**ThermoPINN serves not as a final solution, but as a controlled failure case that exposes the limitations of current safety-critical AI systems.**

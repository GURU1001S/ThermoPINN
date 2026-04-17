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

---

## 🔹 1. Performance & SOTA Benchmarks
Under identical training conditions, ThermoPINN demonstrates competitive or superior performance to purely data-driven baselines from top-tier AI conferences.

| Model Architecture | Params | RMSE ↓ | NASA Score ↓ |
| :--- | :---: | :---: | :---: |
| **ThermoPINN (Ours, Zero-Shot)** | **~800.0K** | **40.2*** | **125.4** |
| TimesNet (ICLR 2023) | 40.7K | 45.1 | 140,605.3 |
| DLinear (AAAI 2023) | 0.2K | 69.5 | 8.85e17† |
| FITS (ICLR 2024) | 4.4K | 97.5 | 1.21e33† |
| iTransformer (ICLR 2024)| 1,049.5K | 130.0 | 2.15e7† |

*\*Note: RMSE 40.2 reflects the UTDTB v5 benchmark comparison; 31.88 reflects N-CMAPSS sim-to-real evaluation. †Note: NASA score functions as an extreme-error stress metric, where pure data-driven baselines exhibit exponential instability.*

![SOTA Benchmark Comparison](assets/fig13_sota_comparison.png)

---

## 🔹 2. Sim-to-Real Transfer & Generalization
The architecture was trained purely on synthetic UTDTB v5 data and subjected to rigorous external validation across multiple real-world flight envelopes. *This robustness suggests the architecture captures stable statistical structure correlated with thermodynamic behavior, though not governed by explicit physical laws.*

* **NASA N-CMAPSS (Primary Sim-to-Real):** In an evaluation of over **26.4 million thermodynamic windows** (33 flight trajectories) handling severe distribution shifts, the zero-shot transfer achieved an average RMSE of **31.88 cycles**. 
* **NASA Classic C-MAPSS:** Utilizing a Gradient Reversal Layer (GRL) for domain alignment, the model improved target domain RMSE by 26.04 cycles unsupervised.

---

## 🔹 3. Physics Failure (🔥 Highlight Section)
### The Limits of PINNs: Correlation ≠ Physics
Standard training objectives do not guarantee the emergence of governing equations. A critical evaluation of the model's internal latent space proves that while predictive accuracy is high, the model's internal representation of physics is completely disjointed from reality.

Latent physics nodes failed to replicate Paris-Erdogan and Arrhenius constants (p < 0.05 KS Test), learning unphysical statistical proxies instead of actual thermodynamic laws.

![Physics Law Violation Plot](assets/fig2_physics_violation.png)

---

## 🔹 4. Safety, Uncertainty & OOD Failure
* **Deterministic Feature Collapse:** Post-hoc Evidential Deep Learning (EDL) wrappers fail when the encoder is not explicitly constrained. Under sensor failure, Epistemic uncertainty deflated by 0.9x instead of expanding, making the model confidently wrong.
* **Latent Rigidity:** Point-prediction architectures destroy the probabilistic structure required for MRO scheduling (C-Index: 0.500).
* **Uncertainty Calibration:** Monte Carlo Dropout reduced calibration error (ECE ↓ from 0.42 → ~0.18) under standard conditions, but failed to capture Epistemic uncertainty under true OOD flight conditions.

---

## 🔹 5. Ablation Studies & Stress Testing
A total of 25+ controlled experiments and 7 major ablation categories were conducted to validate architectural robustness and isolate failure modes.

### Core Architecture & Physics Constraints
Removing physics constraints degraded RMSE (45.2 vs 42.9), confirming the contribution of thermodynamic priors to overall prediction stability.
![Architecture Ablation](assets/fig6_architecture_ablation.png)

### Meta-Learning Adaptation Dynamics
Optimal domain adaptation occurs at **k=2 shots**. Beyond this threshold, the model experiences catastrophic forgetting, destroying target domain performance (RMSE ↑ to 287.1 at k=7).
![Meta-Learning Depth](assets/fig7_meta_learning_depth.png)

### Sensor Dimensionality Robustness
Performance remained mathematically invariant (~124.5–124.7 RMSE) under aggressive sensor pruning from 55D down to 18D, demonstrating immense resilience to real-world sensor dropout scenarios.
![Feature Ablation](assets/fig8_feature_ablation.png)

---

## 🔹 6. Repository Architecture & Reproducibility
### 🛠️ Core Modules
* `pinn_model.py` - 55-D multi-stream encoder and physics gate.
* `train_maml_pinn.py` - Meta-learning loop for zero-shot sim-to-real transfer.
* `physics_loss.py` - PDE-constrained loss formulations.
* `evaluate_ncmapss_adapted.py` - N-CMAPSS sim-to-real engine.
* `edl_uncertainty.py` - Probing wrappers for OOD and risk analysis.
* `external_physics_validation.py` - Cross-dataset validation against material science data.

### 🔁 Configuration
To ensure research transparency, all experiments are reproducible under the following configuration:
* **Sequence Length:** 30 cycles (Sliding Window), Batch Size 256
* **Optimizer:** Adam (lr = 1e-3, weight_decay = 1e-4)
* **Training Epochs:** 50 (with Early Stopping)
* **Physics Constants:** Paris Law (m = 3.0), Arrhenius Activation (E_a = 300 kJ/mol)
* **Hardware:** NVIDIA RTX 3050 Laptop GPU (~120 hours total compute time)

---

## ⚠️ Limitations & Proposed Future Research
This work identifies a fundamental gap between predictive performance and physical validity in safety-critical AI systems. Addressing this gap requires moving beyond correlation-driven learning toward architectures that encode physical laws as first-class constraints rather than auxiliary losses—a direction that forms the basis of my proposed graduate research. 

**ThermoPINN serves not as a final solution, but as a controlled failure case that exposes the limitations of current safety-critical AI systems.**

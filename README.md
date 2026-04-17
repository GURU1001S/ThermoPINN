# ThermoPINN: Physics-Constrained Meta-Learning for Aerospace Prognostics

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research_Platform-purple?style=flat-square)

**ThermoPINN** is a research platform designed to empirically evaluate the reliability, safety, and failure modes of Physics-Informed Neural Networks (PINNs) in safety-critical aerospace prognostics. 

Initially developed using **UTDTB v5** (a custom synthetic degradation dataset), the system demonstrates robust zero-shot sim-to-real transfer to NASA N-CMAPSS telemetry and real-time edge deployment feasibility. 

## 🧠 Core Insight & Aerospace Impact
This work demonstrates a critical failure mode in modern safety-critical AI:
> **High predictive accuracy does NOT guarantee physically valid representations in safety-critical systems.**

Despite achieving competitive RMSE and successful sim-to-real transfer, empirical stress tests revealed that the model's latent representations remained **statistically inconsistent with governing physical laws under hypothesis testing**, collapsed under out-of-distribution (OOD) sensor failure, and could not support survival-based risk modeling. **This suggests that current PINN formulations behave as constrained function approximators rather than physically grounded models.**

**Why this matters to aerospace:** In commercial aviation, this disconnect is critical. Uncalibrated overconfidence and non-physical degradation trajectories **may violate safety expectations defined in DO-178C and ARP4761**, introducing unquantified and potentially unsafe decision risk in MRO scheduling and safety-critical decision-making.

---

## 🎯 Formal Problem Definition
Given a multivariate time series of sensor and environmental data $X_{1:T} \in \mathbb{R}^{T \times d}$, the objective is to:
1.  **Predict** Remaining Useful Life (RUL) $y \in \mathbb{R}^{+}$.
2.  **Generalize** under distribution shift $P_{train}(X, y) \neq P_{test}(X, y)$ (Sim-to-Real).
3.  **Regularize** latent states $z$ such that they are **encouraged to satisfy** $f_{phys}(z, \theta) \approx 0$ **via soft constraints**, where $f_{phys}$ represents governing thermodynamic laws (e.g., Paris Law).

---

## 🏆 Key Contributions
* **A 55-D physics-aware meta-learning architecture** for turbofan RUL prediction.
* **An empirical demonstration of deterministic feature collapse** in PINNs under OOD conditions.
* **Systematic evidence that MSE-trained PINNs fail to recover governing physical laws** across independent datasets.
* **A rigorously validated sim-to-real evaluation pipeline** across N-CMAPSS, C-MAPSS, and PRONOSTIA datasets.

## 🧩 Method Overview
ThermoPINN bridges the gap between pure data-driven deep learning and physics-based prognostics by combining:
* **A multi-stream encoder** for high-dimensional sensor and environmental signals.
* **A physics gate** to impose thermodynamic structure. *The gate is designed to constrain latent representations to follow thermodynamic monotonicity and degradation trends, mitigating purely statistical shortcuts.*
* **A Model-Agnostic Meta-Learning (MAML) loop** for rapid domain adaptation.
* **Conformal calibration** for robust uncertainty bounds.

The model is trained on synthetic degradation trajectories and evaluated under zero-shot transfer to real-world flight telemetry.


## 📊 At a Glance
*Note: Results are reported across two regimes — synthetic benchmark (UTDTB v5) and real-world sim-to-real transfer (N-CMAPSS). Statistics are reported as **Mean ± Std over 5 independent random seeds**.*

| Capability | Result |
| :--- | :--- |
| **Sim-to-Real (N-CMAPSS)** | **RMSE: 31.88 ± 0.3** (Zero-Shot) |
| **Cross-Domain Adaptation** | **-26.04 cycle improvement** (Classic C-MAPSS) |
| **SOTA Benchmark (UTDTB)** | **RMSE: 40.2 ± 0.4** |
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

* **NASA N-CMAPSS (Primary Sim-to-Real):** In an evaluation of over **26.4 million thermodynamic windows** (33 flight trajectories) handling severe distribution shifts (**Mean $KL = 0.134$**, computed between normalized feature distributions of UTDTB v5 and N-CMAPSS), the zero-shot transfer achieved an average RMSE of **31.88 cycles**. 
* **NASA Classic C-MAPSS:** Utilizing a Gradient Reversal Layer (GRL) for domain alignment, the model improved target domain RMSE by 26.04 cycles unsupervised.

![Feature Distribution Shift](assets/fig10_distribution_shift.png)
*Fig: Feature Distribution Shift Analysis: UTDTB v5 (Synthetic) vs. NASA N-CMAPSS (Real).*

![RUL Prediction Trajectories](assets/fig1_sim2real_prediction.png)
*Fig: Zero-Shot RUL Prediction on Real N-CMAPSS Engines.*

## 🚀 Engineering Benchmarks
Under identical training conditions, ThermoPINN demonstrates competitive or superior performance to purely data-driven baselines.

| Model Architecture | Params | RMSE ↓ | NASA Score ↓ |
| :--- | :---: | :---: | :---: |
| **ThermoPINN (Ours, Zero-Shot)** | **~800.0K** | **40.2 ± 0.4*** | **125.4 ± 1.2** |
| TimesNet (ICLR 2023) | 40.7K | 45.1 ± 0.8 | 140,605.3 |
| DLinear (AAAI 2023) | 0.2K | 69.5 ± 1.2 | 8.85e17† |

*\*Note: RMSE 40.2 reflects the UTDTB v5 benchmark comparison; 31.88 reflects N-CMAPSS sim-to-real evaluation.*
*†Note: NASA score penalizes late predictions exponentially more than early ones, reflecting safety-critical risk asymmetry.*

![NASA Score Explanation](assets/fig11_nasa_score.png)
*Fig: Operational interpretation of the NASA Asymmetric Score for MRO.*

![SOTA Benchmark Comparison](assets/fig13_sota_comparison.png)
*Fig: SOTA Benchmark Comparison (UTDTB v5).*

## 🔬 Ablation Studies & Stress Testing
A total of 25+ controlled experiments and 7 major ablation categories were conducted to validate architectural robustness and isolate failure modes. *All ablation experiments were conducted under controlled settings with isolated variable modification to ensure causal interpretability of observed performance changes.*

**Robustness & Architecture**
* **Architecture Ablation:** Removing physics constraints degraded RMSE (45.2 vs 42.9), confirming the contribution of thermodynamic priors.
![Architecture Ablation](assets/fig6_architecture_ablation.png)
* **Feature Ablation:** RMSE remained stable (~124.5–124.7 in the ablation setting) across 20D–55D inputs, indicating strong reliance on core sensor signals.
* **Dimensionality Stress Test:** Performance remained invariant under aggressive sensor pruning (55D → 18D), demonstrating resilience to real-world sensor dropout.
![Feature Pruning Ablation](assets/fig8_feature_ablation.png)

**Learning Dynamics & Domain Transfer**
* **Meta-Learning Depth:** Optimal adaptation occurs at k=2 shots; beyond this, performance degrades due to catastrophic forgetting (RMSE ↑ to 287.1 at k=7).
![Meta-Learning Depth Ablation](assets/fig7_meta_learning_depth.png)
* **Domain Adaptation (DANN):** Improved target-domain RMSE by 26.04 cycles without access to labeled target data.

**Uncertainty Calibration**
* **Monte Carlo Dropout:** Reduced calibration error (ECE ↓ from 0.42 → ~0.18) under standard conditions, but failed to capture Epistemic uncertainty under OOD conditions.
![Uncertainty Calibration](assets/fig9_uncertainty_calibration.png)

## 🧪 Scientific Findings: The Limits of PINNs
1. **Correlation Shortcut vs. Physics:** Standard training objectives do not guarantee the emergence of governing equations. We validated this by performing **Non-linear Least Squares (NLS) regression** on the model's latent nodes to extract learned physics constants.
**Key Finding:** Latent physics nodes failed to replicate Paris-Erdogan and Arrhenius constants. While the theoretical constant $m=3.0$, the model converged to $m \approx 1.3$ (Relative Error: 57%, $p < 0.05$ via Kolmogorov-Smirnov test), indicating the model learned non-physical statistical proxies.

![Physics Law Violation](assets/fig2_physics_violation.png)

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

## 🚀 Getting Started (Reproducibility)

### 🛠️ Environment Setup
```bash
git clone [https://github.com/GURU1001S/ThermoPINN.git](https://github.com/GURU1001S/ThermoPINN.git)
cd ThermoPINN
pip install -r requirements.txt

# Train the model
python train_maml_pinn.py

# Evaluate on N-CMAPSS
python evaluate_ncmapss_adapted.py

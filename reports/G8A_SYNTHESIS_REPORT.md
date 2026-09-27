# Phase G8-A: FiLM-Gated Factorized DRQN Gate 27,000 Synthesis Report

**Status**: EVALUATED — NON-QUALIFIED (Hard stop at Gate 27,000 for manual review)  
**Parent Checkpoint**: G7-A Gate 26,000 (`checkpoint_gate_26000.pt`, SHA: `c84e22deb1309611f183336a83d5b5e2593af3e13a4acb691c95fcb3e8a580a7`)  
**Gate 27,000 Checkpoint**: `experiments/checkpoints/g8a_stabilization/checkpoint_gate_27000.pt` (SHA: `07480d2c3ad37880b13f0710b42d9e03c8147236617d4fe246de6f63e1941e17`)  
**Evaluation Manifest**: `reports/g8a_gate27k_diagnostic_manifest.json`  

---

## 1. Executive Summary & Qualification Scorecard

Phase G8-A tested whether the newly designed **FiLM-Gated Factorized DRQN** architecture—equipped with identity-initialized Feature-wise Linear Modulation, relative dwell cost shaping ($r - (cost - \text{EMA})$), cosine-annealed entropy scheduling, a greedy mode collapse circuit breaker, and 4-stratum replay sampling—could prevent the late-stage optimization collapse observed in Phase G7-A (where $H_{\text{mode}}$ plummeted to 0.110 with 98.0% SHORT and mean Pd fell to 58.72%).

Against the frozen qualification contract evaluated across all 10 canonical scenarios, **G8-A is classified as `NON_QUALIFIED`** at Gate 27,000:

| # | Criterion | Target Threshold | G7-A (26k) | G7-A (27k) | G7-B (27k) | **G8-A (27k)** | Status |
|---|---|---|---|---|---|---|---|
| $c_1$ | **Mean $P_d$** | $\ge 78.0\%$ | 84.44% | 72.88% | 48.60% | **55.31%** | ❌ FAILED (-22.69 pp) |
| $c_2$ | **Mode Entropy ($H_{\text{mode}}$)** | $\ge 0.40$ | 1.029 | 0.869 | 0.283 | **0.841** | ✅ **PASSED** |
| $c_3$ | **`config_29` $P_d$** | $> 0.0\%$ | 72.00% | 0.00% | 0.00% | **0.00%** | ❌ FAILED (Blackout) |
| $c_4$ | **`config_119` $P_d$** | $> 0.0\%$ | 80.49% | 0.00% | 0.00% | **0.00%** | ❌ FAILED (Blackout) |
| $c_5$ | **`config_241` $P_d$** | $> 0.0\%$ | 82.26% | 0.00% | 0.00% | **0.00%** | ❌ FAILED (Blackout) |
| $c_6$ | **SHORT Fraction** | $15.0\% - 80.0\%$ | 53.15% | 0.90% | 91.13% | **54.19%** | ✅ **PASSED** |
| $c_7$ | **$Q_{\text{max}}$** | $\le 35.0$ | 11.69 | 22.40 | 14.50 | **39.39** | ❌ FAILED (+4.39 above) |
| — | **Agile Mean $P_d$** | Contract Informative | 81.25% | 24.39% | 23.47% | **24.89%** | Informative |
| — | **Sparse Mean $P_d$** | Contract Informative | 80.25% | 38.30% | 38.30% | **40.62%** | Informative |
| — | **Distinct Bands** | $36 / 36$ | 36/36 | 36/36 | 36/36 | **36/36** | ✅ PASSED |

---

## 2. Key Scientific Findings

### Finding 1: Global SHORT Collapse Was Successfully Prevented
In G7-A (28k) and G7-B (27k), the policy underwent an irreversible bifurcation into near-total mode collapse (98.03% SHORT in G7-A, 91.13% SHORT in G7-B), driving $H_{\text{mode}}$ down to 0.110 and 0.283.
In **G8-A**, under the relative dwell cost shaper and cosine beta entropy schedule:
- **$H_{\text{mode}} = 0.841$** (well above the $\ge 0.40$ threshold).
- Global SHORT fraction across all 10 scenarios is **54.19%**, remaining strictly within the allowed $15\% - 80\%$ window.
- The mode distribution remains actively multimodal:
  - SHORT: **54.19%**
  - LONG: **41.12%**
  - NORMAL: **4.69%**
  - REVISIT: 0.00%
  - PREEMPTIVE: 0.00%

### Finding 2: Scenario-Selective Bifurcation (Dense Retention vs Agile Blackout)
The policy split into two distinct regimes:
1. **High-Occupancy / Dense / Stationary Scenarios (6/10)** retained high detection performance with dominant LONG dwell selection:
   - `config_117`: $P_d = 100.0\%$, 95.2% LONG, 0.0% SHORT
   - `config_194`: $P_d = 100.0\%$, 88.4% LONG, 7.7% SHORT
   - `config_195`: $P_d = 99.57\%$, 95.2% LONG, 0.0% SHORT
   - `config_64`: $P_d = 93.60\%$, 69.2% LONG, 26.0% SHORT
   - `config_143`: $P_d = 81.25\%$, 10.3% LONG, 85.1% SHORT
   - `config_96`: $P_d = 78.71\%$, 12.5% LONG, 82.7% SHORT
2. **Agile / Hopping Scenarios (4/10)** suffered complete blackouts ($P_d = 0.0\%$):
   - `config_29`: $P_d = 0.0\%$ (85.1% SHORT)
   - `config_119`: $P_d = 0.0\%$ (85.1% SHORT)
   - `config_241`: $P_d = 0.0\%$ (85.1% SHORT)
   - `config_42`: $P_d = 0.0\%$ (85.1% SHORT)

### Finding 3: Root Cause Analysis — Training Environment vs Replay Buffer Isolation
Forensic analysis of the training run reveals why the agile scenarios blacked out:
1. **Dataset Path Discrepancy**: In `configs/training_config_g8.yaml`, `data_dir` pointed to `data` (which contains only 2 synthetic/stare scenario files), rather than `D:/TSRD` (which contains the full suite of TSRD scenarios, including the agile set `config_29`, `config_119`, `config_241`).
2. **Episode Under-Sampling in Replay Buffer**:
   - The G8-A training run executed 1,000 environment steps (step 26,000 to 27,000).
   - In `train_scheduler.py`, `max_steps_per_episode = 1000`. Thus, exactly **1 episode** was experienced and archived into `SequenceReplayBuffer`.
   - `StratifiedReplaySampler.can_stratify()` requires at least `min_episodes_per_stratum = 1` across all 4 strata (agile, sparse, dense, mixed). Because only 1 episode existed, `can_stratify()` evaluated to `False`.
   - The sampler fell back to standard unstratified sampling over transitions from that single non-agile episode.
3. **Absence of Agile Replay**: In G7-A (26k), 50% of rollout episodes were explicitly sampled from the targeted agile cache (`config_29`, `config_119`, `config_241`). In G8-A, because neither the environment nor the buffer contained agile episodes during the 1,000 steps, the online policy unlearned the agile tracking associations learned in G7-A.

### Finding 4: FiLM Gate Stability
The FiLM modulation head demonstrated stable conditioning without gradient divergence:
- $\gamma$ mean: **1.0018** (std: 0.0026, min: 0.9979, max: 1.0292)
- $\beta$ mean: **-0.0026** (std: 0.0072, min: -0.0628, max: 0.0341)
- The gate did not saturate against the $[-0.5, 1.5]$ clamp boundary, and gradients propagated smoothly through all 2,410 FiLM parameters.

---

## 3. Forensic Comparison Table

| Metric | G6 (27k) | G7-A (26k) | G7-A (27k) | G7-B (27k) | **G8-A (27k)** |
|---|---|---|---|---|---|
| **Architecture** | Factorized | Factorized | Factorized | Factorized + Rank-8 Bilinear | **FiLM-Gated Factorized** |
| **Objective** | G3-D canonical | G3-D canonical | G3-D canonical | G3-D canonical | **Step-Based + RelDwellShaper** |
| **Discount $\gamma$** | 0.99 | 0.99 | 0.99 | 0.99 | **0.99** |
| **Training Protocol** | Standard | 50% Agile Targeted | 50% Agile Targeted | 50% Agile Targeted | **train_scheduler (1 ep)** |
| **$H_{\text{mode}}$** | 0.869 | 1.029 | 0.869 | 0.283 | **0.841** |
| **SHORT %** | 0.90% | 53.15% | 0.90% | 91.13% | **54.19%** |
| **LONG %** | 0.10% | 1.94% | 0.10% | 1.08% | **41.12%** |
| **NORMAL %** | 6.23% | 43.68% | 6.23% | 7.79% | **4.69%** |
| **Mean $P_d$** | 68.21% | **84.44%** | 72.88% | 48.60% | **55.31%** |
| **Agile $P_d$** | 48.12% | **81.25%** | 24.39% | 23.47% | **24.89%** |
| **Sparse $P_d$** | 52.10% | **80.25%** | 38.30% | 38.30% | **40.62%** |
| **`config_29` $P_d$** | 0.00% | **72.00%** | 0.00% | 0.00% | **0.00%** |
| **`config_119` $P_d$** | 36.36% | **80.49%** | 0.00% | 0.00% | **0.00%** |
| **$Q_{\text{max}}$** | -0.18 | 11.69 | 22.40 | 14.50 | **39.39** |

---

## 4. Next Step Options & Recommendations

In accordance with project governance, execution is paused at Gate 27,000 for manual review.

### Option 1 (Recommended): Phase G8.1 Targeted Agile Injection + Preloaded Multi-Stratum Buffer
- **Premise**: Fix the data pipeline root cause. Preload the replay buffer with episodes from the full canonical TSRD dataset (`D:/TSRD`), ensuring all 4 strata (agile: `config_29`, `config_119`, `config_241`; sparse: `config_143`; dense; mixed) are actively present before gradient updates commence.
- **Lineage**: Resume from Gate 26,000 root (`checkpoint_gate_26000.pt`).
- **Environment**: Set `data_dir: "D:/TSRD"` and episode length to 100 or 200 steps so multiple diverse episodes are ingested into the buffer within 1,000 environment steps.
- **Horizon**: 1,000 steps (26k → 27k) with hard stop at 27k.

### Option 2: Diagnostic Evaluation of FiLM Gate Ablation at Gate 27k
- Run a counterfactual diagnostic on the current Gate 27,000 checkpoint by clamping $\gamma = 1.0, \beta = 0.0$ to determine the exact inference-time contribution of the FiLM modulation on agile vs dense decisions.

### Option 3: Continue Current Lineage (G8-B: 27k → 28k)
- Continue training from the current 27k checkpoint under corrected data directories without rewinding to 26k. (Not recommended, as recovering from 0% agile $P_d$ is harder than resuming from the qualified 84.44% 26k checkpoint).

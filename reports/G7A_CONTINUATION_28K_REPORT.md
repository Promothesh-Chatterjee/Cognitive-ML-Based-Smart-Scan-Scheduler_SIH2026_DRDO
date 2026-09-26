# Phase G7-A Continuation Report: Gate 26,000 $\to$ Gate 28,000
## Investigation of Optimization Longevity: Discovery of Late-Stage SHORT Collapse under Targeted Replay

**Date**: 2026-09-26  
**Status**: Complete & Verified (Hard Gate at Gate 28,000 Enforced; Non-Qualified)  
**Parent Lineage**: `checkpoint_gate_25000_frozen.pt` (SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`)  
**Starting Checkpoint**: `experiments/checkpoints/g7a_targeted_replay/checkpoint_gate_26000.pt` (SHA-256: `c84e22deb1309611f183336a83d5b5e2593af3e13a4acb691c95fcb3e8a580a7`)  
**Objective**: Canonical G3-D Hybrid ($c_{\text{dwell}} = 2.0, \gamma = \mathbf{0.99}$)  
**Regularization**: Mode-Marginal Entropy ($\beta_{\text{mode}} = 0.5, T_{\text{soft}} = 1.0$)  
**Architecture**: Decoupled Factorized $(b, m)$ DRQN  
**Replay Sampling**: 50/50 Targeted Agile Replay Mix (`config_29`, `config_119`, `config_241` vs general train)  
**Checkpoints Produced**:
- **Gate 27,000 (+1,000 continuation steps)**: `experiments/checkpoints/g7a_targeted_replay/checkpoint_gate_27000.pt`  
  *(SHA-256: `8a3052068970cf15a694a15f045b71fe01bb4a9e50d45c0faee5554baa2c5a29`)*
- **Gate 28,000 (+2,000 continuation steps)**: `experiments/checkpoints/g7a_targeted_replay/checkpoint_gate_28000.pt`  
  *(SHA-256: `d3b040187a15f7dfe3d74738c936fe07844d268768f19b8eddb0a7ce3b2fda71`)*  
**Manifest**: [`reports/g7a_continuation_28k_manifest.json`](file:///c:/Users/PromotheshChatterjee/Documents/GitHub/SIH2026_Try2/reports/g7a_continuation_28k_manifest.json)

---

## 1. Executive Summary & Core Scientific Verdict

Phase G7-A Continuation executed the authorized bounded continuation (Step 26,000 $\to$ 28,000) under the exact identical 50/50 targeted agile replay protocol, canonical G3-D objective ($\gamma = 0.99, c_{\text{dwell}} = 2.0$), and decoupled factorized architecture to answer the crucial longevity question:
> *Does the qualified multi-mode policy observed at Gate 26,000 ($H_{\text{mode}} = 1.029, P_d = 84.44\%$) represent a stable dynamic equilibrium, or does continuous optimization drive the additive factorized architecture into a late-stage mode collapse?*

The empirical outcome provides an unmistakable answer:

```
                            [Gate-25k Frozen Root]
                             (H_mode=0.399, Pd=82.2%)
                                        |
                 +----------------------+----------------------+
                 | (1,000 Steps)                               | (2,000 Additional Steps)
                 v                                             v
     [Gate 26,000 (G7-A 1k)]                       [Gate 28,000 (G7-A Continuation 3k)]
     - H_mode = 1.029 (Diverse)                    - H_mode = 0.110 (SEVERE COLLAPSE)
     - SHORT: 53.15%                               - SHORT: 98.03% (SHORT COLLAPSE)
     - LONG: 33.27%                                - LONG: 0.88%
     - PREEMPTIVE: 12.11%                          - PREEMPTIVE: 0.00%
     - Mean Pd = 84.44%                            - Mean Pd = 58.72% (-25.72% Crash)
     - config_29 Pd = 72.00%                       - config_29 Pd = 0.00% (RE-BLACKOUT)
     - config_119 Pd = 80.49%                      - config_119 Pd = 0.00% (RE-BLACKOUT)
     - config_241 Pd = 73.33%                      - config_241 Pd = 0.00% (BLACKOUT)
     - config_42 Pd = 63.68%                       - config_42 Pd = 0.00% (BLACKOUT)
     [QUALIFIED_GATE_26000_TARGETED_REPLAY]        [NON_QUALIFIED_LATE_STAGE_SHORT_COLLAPSE]
```

### Headline Findings:
1. **Gate-26,000 Was a Transient Operating Point, Not a Stable Equilibrium**:
   - The multi-mode coexistence observed at Gate 26,000 ($53\%$ SHORT, $33\%$ LONG, $12\%$ PREEMPTIVE) was a transient phase during the transition between the inherited Gate-25 LONG-dominant basin and the terminal SHORT-attractor basin.
   - Over the subsequent 2,000 optimization steps, the policy did not stabilize in the balanced region; it drifted completely into **98.03% SHORT dwell**.
2. **Re-emergence of Agile and Sparse Blackouts**:
   - With the policy collapsed into 98% SHORT ($125\ \mu s$), dwell times became too brief to detect emitters requiring longer integration or slower revisit cycles.
   - Consequently, **four distinct scenarios simultaneously crashed to $0.0\% P_d$**: `config_29` ($72.0\% \to 0.0\%$), `config_119` ($80.5\% \to 0.0\%$), `config_241` ($73.3\% \to 0.0\%$), and `config_42` ($63.7\% \to 0.0\%$).
   - Overall mean $P_d$ crashed from $84.44\% \to \mathbf{58.72\%}$.
3. **The Root Structural Defect: Global Additive Separability**:
   - The decoupled factorized DRQN computes action values as $Q(s, b, m) = V(s) + A_{\text{band}}(s, b) + A_{\text{mode}}(s, m)$.
   - Because $A_{\text{mode}}(s, m)$ has no band conditioning, the mode advantage vector is shared across all 36 bands.
   - Under G3-D, SHORT provides an immediate reward bonus ($\Delta r = +1.5$) while LONG incurs an immediate penalty ($\Delta r = -3.0$). Because SHORT is overwhelmingly time-efficient on dense emitters (`config_194`, `config_117`), the unconditioned mode head receives sustained positive gradients for SHORT.
   - Once $A_{\text{mode}}(\text{SHORT}) - A_{\text{mode}}(\text{LONG}) > \max_b A_{\text{band}}(s, b) - \min_b A_{\text{band}}(s, b)$, the agent is mathematically compelled to pick SHORT dwell across **every band in every state**, destroying agile and sparse detection.

---

## 2. Comprehensive Cross-Gate Trajectory Matrix

| Evaluation Metric | Baseline Gate-25k | G6 Factorized Gate-26k | G6 Factorized Gate-27k | **G7-A Gate-26k (1k Targeted)** | **G7-A Gate-28k (3k Targeted Continuation)** | Contractual Gate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Replay Protocol** | Natural | Natural | Natural | 50% Targeted Agile | **50% Targeted Agile** | - |
| **Optimization Steps** | 0 | 1,000 | 2,000 | 1,000 | **3,000 (Total)** | - |
| **Mode Entropy ($H_{\text{mode}}$)** | 0.399 | 0.736 | 0.869 | **1.029** | **0.110 (COLLAPSE)** | $\ge 0.40$ |
| **SHORT Dwell %** | 0.00% | 24.34% | 0.90% | **53.15%** | **98.03% (COLLAPSE)** | $\ge 1\%$ in agile |
| **NORMAL Dwell %** | 13.67% | 0.75% | 6.23% | **0.51%** | **1.09%** | - |
| **LONG Dwell %** | 86.33% | 0.61% | 0.10% | **33.27%** | **0.88%** | - |
| **REVISIT %** | 0.00% | 71.97% | 63.69% | **0.96%** | **0.00%** | - |
| **PREEMPTIVE %** | 0.00% | 2.33% | 29.08% | **12.11%** | **0.00%** | - |
| **Mean $P_d$ (%)** | 82.17% | 69.34% | 72.79% | **84.44%** | **58.72% (CRASH)** | $\ge 78.0\%$ |
| **Agile Mean $P_d$** | 74.15% | 50.70% | 48.24% | **81.25%** | **24.67% (CRASH)** | $\ge 70.0\%$ |
| **Sparse Mean $P_d$** | 70.41% | 54.55% | 54.55% | **71.79%** | **50.00%** | - |
| **`config_29` $P_d$** | 74.29% | 0.00% | 0.00% | **72.00%** | **0.00% (RE-BLACKOUT)** | $> 0.0\%$ |
| **`config_119` $P_d$** | 70.82% | 36.36% | 0.00% | **80.49%** | **0.00% (RE-BLACKOUT)** | $> 0.0\%$ |
| **`config_241` $P_d$** | 52.31% | 66.67% | 95.00% | **73.33%** | **0.00% (BLACKOUT)** | $> 0.0\%$ |
| **`config_42` $P_d$** | 38.05% | 27.27% | 71.08% | **63.68%** | **0.00% (BLACKOUT)** | $> 0.0\%$ |
| **Gross hits / ms** | 0.404 | 0.854 | 0.833 | **0.667** | **1.609 (Distorted)** | $\ge 0.380$ |
| **Novel hits / ms** | 0.0024 | 0.0039 | 0.0032 | **0.00363** | **0.00935** | $\ge 0.0020$ |
| **$Q_{\max}$** | 2.74 | -0.18 | 21.84 | **11.69** | **-0.81** | $\le 35.0$ |
| **Distinct Bands** | 36 / 36 | 36 / 36 | 36 / 36 | **36 / 36** | **36 / 36** | $\ge 30 / 36$ |
| **Status Verdict** | Baseline | Non-Qual | Non-Qual | **QUALIFIED (1k)** | **NON-QUALIFIED (COLLAPSE)** | - |

---

## 3. Scenario-by-Scenario Evaluation at Gate 28,000

Deterministic evaluation across all 10 canonical validation scenarios at Gate 28,000 ($N = 10,000$ decisions):

| Scenario | Class | Steps | Hits | Dwell Time (ms) | Gross hits/ms | $P_d$ (%) | $P_{fa}$ | SHORT % | Gate 26k $P_d$ | $\Delta P_d$ (26k $\to$ 28k) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `config_117` | Regular | 1,000 | 405 | 139.25 | 2.908 | **100.0%** | 0.0 | 98.0% | 100.0% | $0.0\%$ |
| `config_119` | Agile / Sparse | 1,000 | 0 | 139.25 | 0.000 | **0.0%** | 0.0 | 98.0% | 80.49% | **$-80.49\%$ (Crash)** |
| `config_143` | Sparse | 1,000 | 4 | 139.25 | 0.029 | **100.0%** | 0.0 | 98.0% | 63.10% | $+36.90\%$ |
| `config_194` | Regular | 1,000 | 984 | 138.13 | 7.124 | **99.90%** | 0.0 | 98.1% | 99.90% | $0.0\%$ |
| `config_195` | Agile | 1,000 | 522 | 137.75 | 3.789 | **98.68%** | 0.0 | 98.2% | 99.17% | $-0.49\%$ |
| `config_241` | Agile | 1,000 | 0 | 139.25 | 0.000 | **0.0%** | 0.0 | 98.0% | 73.33% | **$-73.33\%$ (Crash)** |
| `config_29` | Agile | 1,000 | 0 | 139.25 | 0.000 | **0.0%** | 0.0 | 98.0% | 72.00% | **$-72.00\%$ (Crash)** |
| `config_42` | Sparse | 1,000 | 0 | 139.25 | 0.000 | **0.0%** | 0.0 | 98.0% | 63.68% | **$-63.68\%$ (Crash)** |
| `config_64` | Regular | 1,000 | 193 | 139.25 | 1.386 | **98.47%** | 0.0 | 98.0% | 98.13% | $+0.34\%$ |
| `config_96` | Regular | 1,000 | 128 | 139.25 | 0.919 | **90.14%** | 0.0 | 98.0% | 94.59% | $-4.45\%$ |
| **Macro Mean** | - | - | - | **138.98** | **1.609** | **58.72%** | **0.0** | **98.03%** | **84.44%** | **$-25.72\%$** |

---

## 4. Deep Forensic Analysis: Why Additive Factorization Inevitably Collapses

### 4.1 Comparison of Collapse Dynamics Across Phases
We now possess three complete trajectories from the Gate-25 root:
1. **Phase G5 (Flat vs Factorized, $\gamma = 0.95$, Natural Replay)**:
   - Factorized policy started at $19\%$ SHORT at 1k, collapsed to **$92.6\%$ SHORT** at 2k ($P_d = 56.0\%$).
2. **Phase G6 (Factorized Canonical $\gamma = 0.99$, Natural Replay)**:
   - Policy started at $24\%$ SHORT at 1k, resisted SHORT collapse ($0.9\%$ SHORT at 2k), but collapsed into **$63.7\%$ REVISIT / $29.1\%$ PREEMPTIVE** with $0.0\% P_d$ on `config_29`.
3. **Phase G7-A (Factorized Canonical $\gamma = 0.99$, Targeted Replay)**:
   - Policy achieved high multi-mode diversity at 1k ($53\%$ SHORT, $33\%$ LONG, $12\%$ PREEMPTIVE, $P_d = 84.4\%$).
   - But by 3k (Gate 28,000), it collapsed completely into **$98.03\%$ SHORT**, driving $P_d$ to $58.7\%$ and causing 4 simultaneous blackouts.

### 4.2 The Mathematical Mechanism of Collapse
The fundamental root cause is the **rank-1 additive independence** of the action representation:
$$Q(s, b, m) = V(s) + A_{\text{band}}(s, b) + A_{\text{mode}}(s, m)$$

Under this formulation:
$$\arg\max_{(b, m)} Q(s, b, m) = \left( \arg\max_b A_{\text{band}}(s, b),\ \arg\max_m A_{\text{mode}}(s, m) \right)$$

This mathematical decoupling creates a fatal structural vulnerability:
1. **The chosen dwell mode $m^*$ is identical across all bands $b$ in state $s$**. The agent cannot simultaneously choose `(Band 5, LONG_DWELL)` for a weak agile hopper and `(Band 22, SHORT_DWELL)` for a strong continuous radar.
2. In expectation over the training replay distribution, SHORT dwell yields vastly higher hits per millisecond on high-density emitters (`config_194` yields $7.12$ hits/ms under SHORT).
3. The gradient $\nabla_{\theta_{\text{mode}}} A_{\text{mode}}(s, \text{SHORT})$ persistently outpaces that of LONG and NORMAL.
4. Over extended training horizons ($>1,000$ steps), $A_{\text{mode}}(\text{SHORT})$ grows until it dominates all other mode logits for virtually all states $s$.
5. The policy enters a terminal attractor: **$98.0\%$ SHORT dwell globally**.
6. When evaluating on agile or sparse emitters that require longer dwells to integrate pulse trains, the policy fails completely, producing $0.0\% P_d$ on `config_29`, `config_119`, `config_241`, and `config_42`.

---

## 5. Compliance Audit Against Contractual Gates

| Requirement | Contract Threshold | Gate 28,000 Value | Audit Status |
|---|:---:|:---:|:---:|
| **Mean Detection Probability ($P_d$)** | $\ge 78.0\%$ | **$58.72\%$** | **FAIL (-19.28% below floor)** |
| **Agile Scenarios Mean $P_d$** | $\ge 70.0\%$ | **$24.67\%$** | **FAIL (-45.33% below floor)** |
| **Scenario Blackout Prohibition** | $P_d > 0.0\%$ all scenarios | **4 Scenarios at $0.0\%$** | **FAIL (4 Blackouts)** |
| **Mode Diversity Entropy ($H_{\text{mode}}$)** | $\ge 0.40$ | **$0.110$** | **FAIL (Severe Collapse)** |
| **Mode Collapse Prohibition** | No single mode $\ge 90.0\%$ | **SHORT = $98.03\%$** | **FAIL (Collapsed)** |
| **SHORT Dwell in Agile Scenarios** | Persistent $\ge 1.0\%$ | $98.0\%$ | Pass (Trivially) |
| **Value Bounds ($Q_{\max}$)** | $\le 35.0$ | $-0.81$ | Pass |
| **Distinct Frequency Bands** | $\ge 30\ /\ 36$ | $36\ /\ 36$ | Pass |

**Final Gate 28,000 Verdict**: **`NON_QUALIFIED_LATE_STAGE_SHORT_COLLAPSE`**  
The Gate-28,000 checkpoint fails 5 of the core contractual requirements.

---

This rigorous 2,000-step test resolves the core architectural question:
- In G7-A (1k), targeted replay temporarily balanced the opposing mode gradients, producing an apparent multi-mode recovery at Gate 26,000 ($53\%$ SHORT / $33\%$ LONG).
- In G7-A Continuation (3k), extended optimization proved that this operating region was not a stable dynamic equilibrium. **Under canonical G3-D, $\beta_{\text{mode}}=0.5$, the 50/50 targeted-replay protocol, and this optimization setup, additive factorization exhibited reproducible late-stage SHORT collapse by 3,000 steps ($98.03\%$ SHORT).**
- Therefore, targeted replay is sufficient to produce a high-performing factorized policy, but insufficient to maintain that policy under continued optimization.
- The remaining challenge is fundamentally a structural conditioning problem, not merely an exposure deficit. This provides strong empirical justification for **Phase G7-B: Band-Conditioned Low-Rank Interaction Coupling** ($Q(s, b, m) = V(s) + A_{\text{band}}(s, b) + A_{\text{mode}}(s, m) + \phi(e_b(s))^\top W_m$) to allow dwell duration to be conditioned directly on the candidate band.

### Formal Checkpoint Classification & Provenance:
- **Gate-25k (`checkpoint_gate_25000_frozen.pt`, SHA: `7a99c659...`)**: Immutable frozen baseline root.
- **G7-A 26k (`checkpoint_gate_26000.pt`, SHA: `c84e22de...`)**: Qualified transient snapshot under 1k targeted replay / NOT a stable continuation baseline.
- **G7-A 27k (`checkpoint_gate_27000.pt`, SHA: `8a305206...`)**: Diagnostic intermediate checkpoint (reserved for read-only trajectory acceleration analysis).
- **G7-A 28k (`checkpoint_gate_28000.pt`, SHA: `d3b04018...`)**: Quarantined non-qualified late-stage collapse (`NON_QUALIFIED_LATE_STAGE_SHORT_COLLAPSE`).
- **Phase G7-B**: Authorized as the next architectural isolation experiment starting from the immutable Gate-25k root.
- **Gate-50k / 75k / 100k continuation**: STRICTLY BLOCKED.


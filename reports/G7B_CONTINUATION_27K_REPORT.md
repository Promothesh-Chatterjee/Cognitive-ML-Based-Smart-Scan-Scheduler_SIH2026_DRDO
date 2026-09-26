# Phase G7-B Continuation Report: Gate 26,000 $\to$ Gate 27,000
## Definitive Classification: Delayed Coupled Failure into 100% SHORT Attractor

**Date**: 2026-09-26  
**Status**: Complete & Verified (Gate 27,000 Hard Stop Enforced; Non-Qualified)  
**Parent Lineage Root**: `checkpoint_gate_25000_frozen.pt` (SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`)  
**Starting Checkpoint**: `experiments/checkpoints/g7b_low_rank_coupling/checkpoint_gate_26000.pt` (SHA-256: `3cdfa87d88549ffa4982f96458ed316e69347271e51674a5df6638e86d9f8035`)  
**Terminal Checkpoint**: `experiments/checkpoints/g7b_low_rank_coupling/checkpoint_gate_27000.pt`  
*(SHA-256: `5a598cbd96a0adb82009516816ccc13e1b77ae70647cacbbd28bb21e19f5f84e`)*  
**Manifest**: [`reports/g7b_continuation_27k_manifest.json`](file:///c:/Users/PromotheshChatterjee/Documents/GitHub/SIH2026_Try2/reports/g7b_continuation_27k_manifest.json)  
**Ablation Manifest**: [`reports/g7b_interaction_ablation_manifest.json`](file:///c:/Users/PromotheshChatterjee/Documents/GitHub/SIH2026_Try2/reports/g7b_interaction_ablation_manifest.json)

---

## 1. Executive Summary & Core Scientific Verdict

Phase G7-B Continuation evaluated whether the low-rank coupled DRQN ($R=8$) could survive the critical 1,000-step continuation interval (Step 26,000 $\to$ 27,000) that previously triggered catastrophic mode collapse in Phase G7-A.

The empirical data across all 10 canonical validation scenarios provides an unequivocal result:
**Phase G7-B is classified as a `DELAYED_COUPLED_FAILURE` into the terminal 100% SHORT attractor.**

```
                         [Gate-25k Frozen Baseline]
                                      |
               +----------------------+----------------------+
               |                                             |
   [Phase G7-A: Decoupled Additive]              [Phase G7-B: Low-Rank Coupled (R=8)]
               |                                             |
   Gate 26k: H_mode = 1.029                      Gate 26k: H_mode = 0.573
             SHORT = 53.15%                                SHORT = 19.66%
             Pd = 84.44%                                   Pd = 76.15%
             (Transient Snapshot)                          (Nascent Interaction: 3/10k flips)
               |                                             |
               | (+1,000 Steps)                              | (+1,000 Steps)
               v                                             v
   Gate 27k: H_mode = 0.183                      Gate 27k: H_mode = 0.000 (TOTAL EXTINCTION)
             SHORT = 96.45%                                SHORT = 100.0% (TOTAL EXTINCTION)
             Pd = 59.26%                                   Pd = 48.60% (TOTAL COLLAPSE)
             4 Scenarios at 0% Pd                          5 Scenarios at 0% Pd
             [NON_QUALIFIED_COLLAPSE]                      [DELAYED_COUPLED_FAILURE]
```

### Headline Findings:
1. **Total Extinction of Mode Diversity at Gate 27,000**:
   - By Gate 27,000, every single decision across all 10 validation scenarios ($10,000 / 10,000$ decisions) selected **SHORT Dwell ($125\ \mu s$)**.
   - Mode entropy collapsed to **$H_{\text{mode}} = 0.000$**, with LONG, NORMAL, REVISIT, and PREEMPTIVE all dropping to exactly $0.00\%$.
2. **Catastrophic Blackouts Across 5 Scenarios**:
   - Because $125\ \mu s$ SHORT dwell is too brief to detect emitters requiring longer integration or slower revisit cycles, **five distinct scenarios completely blacked out ($0.0\% P_d$)**:
     * `config_29` $P_d$: **$0.0\%$** (was $32.1\%$ at 26k)
     * `config_119` $P_d$: **$0.0\%$** (was $28.6\%$ at 26k)
     * `config_241` $P_d$: **$0.0\%$** (was $79.1\%$ at 26k)
     * `config_42` $P_d$: **$0.0\%$** (was $40.1\%$ at 26k)
     * `config_143` $P_d$: **$0.0\%$** (was $90.9\%$ at 26k)
   - Macro mean $P_d$ crashed from $76.15\% \to \mathbf{48.60\%}$.
3. **The Read-Only Ablation Audit: Proving Dormancy of the Interaction Head**:
   - At Gate 26,000, ablated testing ($Q = V + A_b + A_m$, setting $I=0$) resulted in only **$3 / 10,000$ action flips ($0.03\%$)**.
   - At Gate 27,000, ablated testing resulted in **$0 / 10,000$ action flips ($0.00\%$)** and identical $48.60\% P_d$.
   - **Causal Proof**: The low-rank interaction term $I(s, b, m)$, initialized to zero, accumulated gradients at an average norm of only $\sim 0.0017 - 0.0058$. This was orders of magnitude too small to overcome the massive unconditioned gradient on $A_{\text{mode}}(\text{SHORT})$ driven by the G3-D dwell bonus on dense emitters.

---

## 2. Longitudinal Trajectory Across All Gates

| Evaluation Metric | Baseline Gate-25k | G7-A Gate-26k (1k) | G7-A Gate-27k (2k) | G7-B Gate-26k (1k Coupled) | **G7-B Gate-27k (2k Coupled)** | Qualification Floor |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Architecture** | Monolithic Flat | Decoupled Additive | Decoupled Additive | Low-Rank Coupled ($R=8$) | **Low-Rank Coupled ($R=8$)** | - |
| **Replay Protocol** | Natural | 50% Targeted | 50% Targeted | 50% Targeted | **50% Targeted** | - |
| **Mode Entropy ($H_{\text{mode}}$)** | 0.399 | **1.029** | 0.183 (Collapse) | **0.573** | **0.000 (TOTAL EXTINCTION)** | $\ge 0.40$ |
| **SHORT Dwell %** | 0.00% | 53.15% | 96.45% | **19.66%** | **100.0% (TOTAL EXTINCTION)** | $\ge 1\%$ in agile |
| **NORMAL Dwell %** | 13.67% | 0.51% | 0.14% | **1.38%** | **0.00%** | - |
| **LONG Dwell %** | 86.33% | 33.27% | 0.30% | **0.10%** | **0.00%** | - |
| **REVISIT %** | 0.00% | 0.96% | 0.48% | **78.86%** | **0.00%** | - |
| **PREEMPTIVE %** | 0.00% | 12.11% | 2.63% | **0.00%** | **0.00%** | - |
| **Mean $P_d$ (%)** | 82.17% | 84.44% | 59.26% | **76.15%** | **48.60% (CRASH)** | $\ge 78.0\%$ |
| **Agile Mean $P_d$** | 74.15% | 81.25% | 24.89% | **59.82%** | **24.64% (CRASH)** | $\ge 70.0\%$ |
| **Sparse Mean $P_d$** | 70.41% | 71.79% | 50.00% | **59.74%** | **0.00% (TOTAL BLACKOUT)** | - |
| **`config_29` $P_d$** | 74.29% | 72.00% | 0.00% (Blackout) | **32.14%** | **0.00% (RE-BLACKOUT)** | $> 0.0\%$ |
| **`config_119` $P_d$** | 70.82% | 80.49% | 0.00% (Blackout) | **28.57%** | **0.00% (RE-BLACKOUT)** | $> 0.0\%$ |
| **`config_241` $P_d$** | 52.31% | 73.33% | 0.00% (Blackout) | **79.07%** | **0.00% (BLACKOUT)** | $> 0.0\%$ |
| **`config_42` $P_d$** | 38.05% | 63.68% | 0.00% (Blackout) | **40.10%** | **0.00% (BLACKOUT)** | $> 0.0\%$ |
| **`config_143` $P_d$** | 70.00% | 63.10% | 100.0% | **90.91%** | **0.00% (BLACKOUT)** | $> 0.0\%$ |
| **Gross hits / ms** | 0.404 | 0.667 | 0.833 | **0.891** | **1.658 (Distorted)** | $\ge 0.380$ |
| **Ablation Flips (I=0)** | N/A | N/A | N/A | **3 / 10k (0.03%)** | **0 / 10k (0.00%)** | - |
| **$Q_{\max}$** | 2.74 | 11.69 | 21.84 | **-0.80** | **-1.14** | $\le 35.0$ |
| **Verdict** | Baseline Root | Qualified Snapshot | Non-Qual | Stable Non-Qual | **DELAYED_COUPLED_FAILURE** | - |

---

## 3. Scenario-by-Scenario Evaluation at Gate 27,000

Deterministic evaluation across all 10 canonical validation scenarios at Gate 27,000 ($N = 10,000$ decisions):

| Scenario | Class | Steps | Hits | Dwell Time (ms) | Gross hits/ms | $P_d$ (%) | $P_{fa}$ | SHORT % | Gate 26k $P_d$ | $\Delta P_d$ (26k $\to$ 27k) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `config_117` | Regular | 1,000 | 395 | 125.00 | 3.160 | **100.0%** | 0.0 | 100.0% | 100.0% | $0.0\%$ |
| `config_119` | Agile / Sparse | 1,000 | 0 | 125.00 | 0.000 | **0.0%** | 0.0 | 100.0% | 28.57% | **$-28.57\%$ (Crash)** |
| `config_143` | Sparse | 1,000 | 0 | 125.00 | 0.000 | **0.0%** | 0.0 | 100.0% | 90.91% | **$-90.91\%$ (Crash)** |
| `config_194` | Regular | 1,000 | 984 | 125.00 | 7.872 | **99.90%** | 0.0 | 100.0% | 100.0% | $-0.10\%$ |
| `config_195` | Agile | 1,000 | 415 | 125.00 | 3.320 | **98.57%** | 0.0 | 100.0% | 99.51% | $-0.94\%$ |
| `config_241` | Agile | 1,000 | 0 | 125.00 | 0.000 | **0.0%** | 0.0 | 100.0% | 79.07% | **$-79.07\%$ (Crash)** |
| `config_29` | Agile | 1,000 | 0 | 125.00 | 0.000 | **0.0%** | 0.0 | 100.0% | 32.14% | **$-32.14\%$ (Crash)** |
| `config_42` | Sparse | 1,000 | 0 | 125.00 | 0.000 | **0.0%** | 0.0 | 100.0% | 40.10% | **$-40.10\%$ (Crash)** |
| `config_64` | Regular | 1,000 | 152 | 125.00 | 1.216 | **44.97%** | 0.0 | 100.0% | 98.52% | **$-53.55\%$ (Crash)** |
| `config_96` | Regular | 1,000 | 127 | 125.00 | 1.016 | **42.53%** | 0.0 | 100.0% | 92.72% | **$-50.19\%$ (Crash)** |
| **Macro Mean** | - | - | - | **125.00** | **1.658** | **48.60%** | **0.0** | **100.0%** | **76.15%** | **$-27.55\%$** |

---

## 4. Fundamental Causal Analysis: Why the Collapse Attractor Remains Dominant

The empirical trajectory across G5, G6, G7-A, and G7-B reveals a profound structural and optimization reality:

### 4.1 The Gradient Imbalance in G3-D
Under the canonical G3-D objective:
- Dwell multipliers $\tau = [0.25, 1.0, 2.5, 1.0, 1.0]$ with penalty $c_{\text{dwell}} = 2.0$.
- SHORT dwell ($\tau = 0.25$) receives an immediate reward bonus:
  $$\Delta r = -c_{\text{dwell}} (\tau - 1.0) = -2.0 \times (-0.75) = \mathbf{+1.5}$$
- LONG dwell ($\tau = 2.5$) receives an immediate penalty:
  $$\Delta r = -2.0 \times (+1.5) = \mathbf{-3.0}$$
- On dense pulsing scenarios (`config_194`, `config_117`), SHORT dwell captures emitters in $125\ \mu s$, generating **7.87 hits/ms** and yielding massive immediate returns.

### 4.2 Why Zero-Initialized Low-Rank Coupling Did Not Counteract the Collapse
1. The unconditioned mode stream $A_{\text{mode}}(s, m)$ directly receives this massive TD gradient, with gradient norms in the range of $1.0 - 5.0$.
2. The low-rank interaction stream $I(s, b, m) = \phi(\text{joint\_band})^\top W_m$ was zero-initialized. Its gradients were tiny ($\sim 0.0017 - 0.0058$).
3. Over the first 1,000 steps (Gate 26k), the interaction head barely moved, flipping only 3 decisions in 10,000.
4. Over the second 1,000 steps (Gate 27k), the relentless $+4.5$ reward advantage of SHORT completely overwhelmed both the mode-marginal entropy regularization ($\beta_{\text{mode}} = 0.5$) and the nascent interaction head.
5. Consequently, G7-B collapsed into the exact same terminal attractor as G7-A: **100.0% SHORT dwell**.

---

## 5. Compliance Audit Against Contractual Gates at Gate 27,000

| Requirement | Contract Threshold | Gate 27,000 Value | Audit Status |
|---|:---:|:---:|:---:|
| **Mean Detection Probability ($P_d$)** | $\ge 78.0\%$ | **$48.60\%$** | **FAIL (-29.40% below floor)** |
| **Agile Scenarios Mean $P_d$** | $\ge 70.0\%$ | **$24.64\%$** | **FAIL (-45.36% below floor)** |
| **Scenario Blackout Prohibition** | $P_d > 0.0\%$ all scenarios | **5 Scenarios at $0.0\%$** | **FAIL (5 Blackouts)** |
| **Mode Diversity Entropy ($H_{\text{mode}}$)** | $\ge 0.40$ | **$0.000$** | **FAIL (Total Extinction)** |
| **Mode Collapse Prohibition** | No single mode $\ge 90.0\%$ | **SHORT = $100.0\%$** | **FAIL (Total Extinction)** |
| **SHORT Dwell in Agile Scenarios** | Persistent $\ge 1.0\%$ | $100.0\%$ | Pass (Trivially) |
| **Value Bounds ($Q_{\max}$)** | $\le 35.0$ | $-1.14$ | Pass |
| **Distinct Frequency Bands** | $\ge 30\ /\ 36$ | $36\ /\ 36$ | Pass |

**Final Gate 27,000 Verdict**: **`DELAYED_COUPLED_FAILURE_SHORT_COLLAPSE`**  
The Gate-27,000 checkpoint fails 5 of the core contractual requirements.

---

## 6. Checkpoint Provenance Status

- **`checkpoint_gate_25000_frozen.pt` (SHA: `7a99c659...`)**: Immutable baseline root.
- **`checkpoint_gate_26000.pt` (G7-A, SHA: `c84e22de...`)**: Preserved as the sole qualified transient targeted-replay snapshot.
- **`checkpoint_gate_26000.pt` (G7-B, SHA: `3cdfa87d...`)**: Preserved as `STABLE_COUPLED_NON_QUALIFIED_GATE_26000`.
- **`checkpoint_gate_27000.pt` (G7-B, SHA: `5a598cbd...`)**: Quarantined non-qualified delayed failure checkpoint (`DELAYED_COUPLED_FAILURE_SHORT_COLLAPSE`).
- **Gate-50k / 75k / 100k continuation**: STRICTLY BLOCKED.

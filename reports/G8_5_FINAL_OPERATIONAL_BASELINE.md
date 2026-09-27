# G8.5 FINAL — Qualified Operational Baseline & Demonstration Status

**Document ID:** `reports/G8_5_FINAL_OPERATIONAL_BASELINE.md`  
**Classification:** Operational Demonstration Baseline / Research Freeze Record  
**Date:** 2026-09-27  
**Research Status:** CLOSED  
**Training Status:** FROZEN  
**Operational Designation:** Qualified Operational Baseline / Frozen Demonstration Baseline  
**Cloud Deployment Status:** NOT_DEPLOYED (Local Software-in-the-Loop Operational Scope)

---

## 1. Executive Summary & Research Conclusion

The scientific training and research exploration phase is formally **CLOSED**. No additional training, continuation runs, hyperparameter sweeps, architecture modifications, replay alterations, or objective adjustments will be performed.

The authoritative operational baseline selected and frozen for demonstration is:

* **Artifact Path:** `experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_27000_operational.pt`
* **Source Checkpoint:** `experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_27000.pt`
* **Exact SHA-256:** `fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094`
* **Training Step:** `27,000`
* **Neural Architecture:** `BandConditionedFactorizedDRQN` (Explicit band-conditioned dwell head with recurrent LSTM core)
* **Objective Formulation:** Canonical step-based SMDP Bellman objective ($\gamma = 0.99, c_{\text{dwell}} = 0.0, q_{\text{reg}} = 0.0, \tau_{\text{ref}} = 1.0$)

This checkpoint represents the latest demonstrated checkpoint across the entire research program that satisfies all operational and stability criteria across all ten canonical scenarios.

---

## 2. Validated Offline Benchmark Evidence (Frozen Reference)

The following metrics represent the frozen, validated offline evaluation results recorded for Gate 27,000 across 10 canonical scenarios ($10,000.0$ ms simulated mission time per scenario, 5,120 decisions):

| Metric | Validated Benchmark Value | Operational Gate Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Mean $P_d$** | **85.09%** | $\ge 80.0\%$ | **PASSED** (All-time high) |
| **Agile $P_d$** | **71.92%** | $\ge 65.0\%$ | **PASSED** (All-time high) |
| **Sparse $P_d$** | **77.50%** | $\ge 50.0\%$ | **PASSED** |
| **Dense $P_d$** | **97.33%** | $\ge 95.0\%$ | **PASSED** |
| **$config_{29}$ $P_d$** | **95.87%** | $\ge 85.0\%$ | **PASSED** ($96.94\%$ of forced-LONG ceiling) |
| **Worst-Case $P_d$** | **47.81%** (`config_42`) | $\ge 30.0\%$ | **PASSED** |
| **Total Scenario Blackouts** | **0** | $= 0$ | **PASSED** |
| **Gross Throughput ($IR_{\text{time}}$)** | **0.9108 hits/ms** | $\ge 0.85$ hits/ms | **PASSED** |
| **Thrashing Cycle Occupancy** | **0.02%** | $< 5.0\%$ | **PASSED** |
| **Realized $Q_{\max}$** | **30.32** | $\le 35.0$ | **PASSED** |
| **Conditioning-Path Activity** | **100.0%** ($512/512$ flips) | $\ge 90.0\%$ | **PASSED** |

> **Note on Model Optimality:** In accordance with rigorous scientific qualification standards, this baseline is designated as the **Qualified Operational Baseline**, not "globally optimal." It is the highest-performing, fully qualified operating point discovered and verified within the explored parameter space.

---

## 3. Scientific Rationale for Freezing & Late Continuation Failure

The decision to freeze Gate 27,000 rather than continuing training past Step 27,000 is grounded in conclusive empirical evidence from longitudinal trials:

1. **Unconstrained Continuation Runaway (G8.5-FINAL, Steps 27k $\to$ 30k):**
   * Continuing unregularized canonical Bellman updates caused Q-values to compound past the stability ceiling ($Q_{\max}$ rose from $30.32 \to 62.97 \to 92.25$).
   * Extreme TD errors ($|\delta| > 10.0$) rose to $44.41\%$.
   * At Gate 30,000, catastrophic mode collapse occurred: **100.0% of all decisions collapsed into Mode 4 (PREEMPTIVE dwell)**, completely extinguishing agile emitter tracking (Agile $P_d = 0.00\%$, 4 total scenario blackouts, $90.11\%$ thrashing).
2. **Alternative Objective Formulations (G8.4-OA and G8.4-REG):**
   * Shortening the discount horizon ($\gamma = 0.95$) induced myopic dwell exploitation (Agile $P_d$ dropped to $57.06\%$).
   * Adding quadratic Q-regularization ($\lambda = 0.005$) compressed action separation and triggered severe thrashing ($82.1\%$).
3. **Governance Rule of Checkpoint Preservation:**
   * Under the pre-registered project governance rule (*"Late continuation failures do not invalidate earlier qualified checkpoints; freeze the latest qualified operating point"*), the trial was halted cleanly at Gate 30,000 and Gate 27,000 was permanently frozen as the golden demonstration artifact.

---

## 4. Operational Distinction: Validated Benchmark vs Live Mission Telemetry

The system architecture and user interface enforce an explicit, fail-safe separation between historical benchmark data and live operational measurements:

### A. Validated Training Benchmark (Immutable Evidence)
* The figures above (e.g., 85.09% Mean $P_d$, 71.92% Agile $P_d$, 0.9108 gross hits/ms) are historical, certified offline benchmark metrics.
* They are displayed in a dedicated reference card labeled: **`VALIDATED GATE-27 TRAINING BENCHMARK (IMMUTABLE OFFLINE EVIDENCE)`**.
* They are **never** substituted for, mixed with, or injected into live operational telemetry.

### B. Live Mission Telemetry (Software-in-the-Loop)
* Measurements generated when pressing **"Start Mission"** or invoking `/mission/step` represent real-time physical simulation data produced by the running closed-loop pipeline (`CognitiveRFScanEnv` $\to$ `OperationalStateBuilder` $\to$ `SmartScanMoE` $\to$ `ReceiverAdapter` $\to$ `EmitterTracker`).
* Live metrics include:
  * Selected frequency band ($0 \dots 35$)
  * Selected dwell mode ($0 \dots 4$)
  * Execution latency (ms)
  * Real-time mission clock ($\mu\text{s}$)
  * Detected RF pulses and physical hit telemetry
  * Rolling session $P_d$ and live spectrum occupancy
* Under no circumstances are live measurements fabricated, mocked, or pre-computed.

---

## 5. Lineage & Immutability Anchors

* **Historical Baseline Anchor:** `experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt`
  * SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
  * Status: **UNTOUCHED, PERMANENTLY IMMUTABLE**.
* **Active Operational Candidate:** `experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_27000_operational.pt`
  * SHA-256: `fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094`
  * Status: **APPROVED, ACTIVE CANDIDATE**.
* **Manifest Designation:** `experiments/checkpoints/scheduler_v2_operational_candidate/ACTIVE_CHECKPOINT.json`
  * Resolves strictly to Gate 27,000 via `CheckpointGuard`.

---

## 6. Deployment Scope & Cloud Status

* **Scope:** Local Software-in-the-Loop (SITL) demonstration and testing.
* **Cloud Status:** **NOT_DEPLOYED**.
* Existing cloud configurations (Azure deployment, Kubernetes manifests) remain completely untouched.
* The repository is prepared for future Google Cloud deployment with identical cryptographic artifact verification (`cloud_artifact_sha256 = fac05774...`).

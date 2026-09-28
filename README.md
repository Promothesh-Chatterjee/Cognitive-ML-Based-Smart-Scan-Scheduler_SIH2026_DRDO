# Cognitive Electronic Warfare Smart Scan Strategy

[![CI](https://github.com/Promothesh-Chatterjee/SIH2026_Try2/actions/workflows/ci.yml/badge.svg)](https://github.com/Promothesh-Chatterjee/SIH2026_Try2/actions/workflows/ci.yml)
[![Google Cloud Run](https://img.shields.io/badge/Backend-Google_Cloud_Run-4285F4?logo=googlecloud)](https://cognitive-ew-backend-753709137146.asia-south1.run.app/health)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?logo=vercel)](https://vercel.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python)](https://python.org)
[![Coverage](https://img.shields.io/badge/Coverage-81%25-brightgreen)](https://github.com/Promothesh-Chatterjee/SIH2026_Try2)

Autonomous cognitive radar scanning strategy for Electronic Warfare (EW) Electronic Support (ES) receivers operating across 36 frequency bands under non-cooperative conditions. Designed for **DRDO Problem Statement SIH26056 (Smart India Hackathon 2026)**, the system intercepts, deinterleaves, and tracks non-cooperative radar emissions—including agile frequency-hopping emitters, periodic scanning search radars, and fixed emitters—without prior threat libraries. By combining high-purity windowed signal deinterleaving with a Dueling Deep Recurrent Q-Network (DRQN) scheduler, the receiver achieves sub-millisecond dwell scheduling decisions. The SmartScan DRQN-MoE achieves 17.8× higher dwell-level intercept rate versus random sweep (42.14% vs 2.36%) in canonical Gate-25k evaluation. Gate-100k retraining targets further improvement to ≥65% IR.

---

## End-to-End Architecture

```
                                  RF EMISSION ENVIRONMENT
   ┌───────────────────────────┬───────────────────────────┬───────────────────────────┐
   │    Static Radar Emitter   │   Agile Hopping Emitter   │   Periodic Scan Radar     │
   │   (Fixed PRI, Duty 50%)   │ (Random/Markov Hop Sets)  │  (Sector Scan, Swept PRI) │
   └─────────────┬─────────────┴─────────────┬─────────────┴─────────────┬─────────────┘
                 │                           │                           │
                 └───────────────────────────┼───────────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │    SpectrumEnvironment    │
                               │     (SmartScanEW-v0)      │
                               │  36 Bands · 180 Actions   │
                               │  360-D Observation Space  │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼                                           ▼
          SIGNAL DEINTERLEAVER SUBSYSTEM               COGNITIVE SCHEDULER (DRQN)
         ┌───────────────────────────────┐           ┌───────────────────────────────┐
         │ Pulse Descriptor Words (PDWs) │           │ Observation Vector (360-D)    │
         │  (ToA, Freq, PW, Amplitude)   │           │ 36 bands × 10 belief features │
         └───────────────┬───────────────┘           └───────────────┬───────────────┘
                         ▼                                           ▼
         ┌───────────────────────────────┐           ┌───────────────────────────────┐
         │  PDWFeatureExtractor & Scale  │           │   2-Layer LSTM Recurrent Core │
         │   (Robust IQR & Time Diffs)   │           │     (2 × 256 units, BPTT)     │
         └───────────────┬───────────────┘           └───────────────┬───────────────┘
                         ▼                                           ▼
         ┌───────────────────────────────┐           ┌───────────────────────────────┐
         │ WindowedDeinterleaver & Match │           │     Dueling Q-Value Heads     │
         │  (HDBSCAN / DBSCAN Fallback)  │           │ V(s) + [A_band(s,a) - mean A] │
         └───────────────┬───────────────┘           └───────────────┬───────────────┘
                         ▼                                           ▼
         ┌───────────────────────────────┐           ┌───────────────────────────────┐
         │    CrossWindowReconciler      │           │ Joint Action: (Band, Mode)    │
         │ (Purity ≥ 0.965, Identity St.)│           │ [SHORT/NORMAL/LONG/REV/PRE]   │
         └───────────────────────────────┘           └───────────────┬───────────────┘
                                                                     │
                                             ┌───────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │     EW Metrics Engine     │
                               │    (7 Figures of Merit)   │
                               │  Pd, Pfa, S_min, IR, etc. │
                               └─────────────┬─────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
         ┌──────────────────────────────┐          ┌──────────────────────────────┐
         │        FastAPI Server        │          │   Interactive Web Dashboard  │
         │  /api/v1/metrics             │ ───────► │   http://172.198.227.59      │
         │  /api/v1/spectrum            │          │   (Azure AKS)                │
         └──────────────────────────────┘          └──────────────────────────────┘
```

---

## Figures of Merit (Canonical Gate-25k Results)

| Metric | Definition | Gate-25k Result (Canonical 2026.1) |
|---|---|---|
| Probability of Detection (Pd) | TP / (TP + FN) at selected band | **94.95%** |
| Probability of False Alarm (Pfa) | FP / (FP + TN) | **0.00%** |
| Receiver Sensitivity | Physics-computed via Friis | **~-110 dBm** |
| Avg Intercept Rate (IR) | Hits / total dwells | **42.14%** |
| Avg Reward | Mean per-dwell reward | **5.346** |
| Correct Decisions | (TP+TN) / N | **97.76%** |
| Avg Intercept Time Error | MAE of predicted vs actual ToA | **279.77 µs** |

> **Benchmark context:** All results above are from the canonical Gate-25k
> evaluation (benchmark version `2026.1-CANONICAL`, metric contract
> `v2.0-audited-confusion-matrix`, 10 fixed TSRD validation scenarios, 500
> dwells/scenario, 5000 total dwells, seed 42). Earlier non-comparable numbers
> (60–63% IR from pre-audit per-observation accounting, and 7.74% from an
> unverified legacy run) are documented as historical in
> `reports/BENCHMARK_PROVENANCE.md` and are **not** comparable to the current results.
> Gate-100k retraining targets: IR ≥ 65%, Pd ≥ 99%, worst-case IR ≥ 20%, Pfa ≤ 0.05%.

---

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Promothesh-Chatterjee/SIH2026_Try2.git
cd SIH2026_Try2

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install package in editable mode with all dependencies
pip install -e .
```

### 2. Run End-to-End Pipeline Validation
Execute the gate validation script to verify the RF environment, DRQN scheduler, deinterleaving pipeline, and EW Figures of Merit:
```bash
python scripts/validate_pipeline.py
```
*Expected output: exits with code `0`, reporting all 7 FoMs and confirming $>11.5\times$ intercept improvement over Round-Robin.*

### 3. Run Test Suite with Coverage
```bash
pytest ew_core/tests/ --cov=ew_core --cov-fail-under=70
```
*Executes all 774 unit, integration, and regression tests with strict coverage enforcement ($\ge 81\%$ achieved).*

> [!NOTE]
> Baseline-immutability and reservoir tests skip in CI environments where checkpoint files are not hosted locally — Sprint 4 closes this gap with hosted artifact downloads.

### 4. Launch Backend API Server
```bash
uvicorn ew_core.deployment.api:app --host 0.0.0.0 --port 8000 --reload
```
Endpoints available:
- `GET /health` — Service health and GPU/CPU device metadata
- `GET /api/v1/metrics` — Latest episode EW figures of merit ($P_d, P_{fa}$, intercept rate, etc.)
- `GET /api/v1/spectrum` — Time-frequency truth matrix and receiver dwell positions
- `POST /predict_bands` — Inference endpoint for scheduler action selection
- `POST /deinterleave` — High-speed PDW clustering and emitter separation

---

## Cloud Deployment & Operations (Azure AKS)

The cognitive EW microservice is containerized and deployed to **Azure Kubernetes Service (AKS)** with zero out-of-pocket costs ($0.00 model via Azure for Students & GitHub Container Registry):

* **Backend Live Endpoint**: [http://172.198.227.59](http://172.198.227.59) *(start AKS first)*
* **Container Registry**: `ghcr.io/promothesh-chatterjee/sih2026_try2:latest` (GHCR)
* **Storage**: Azure Blob Storage (`smartscanstore4301`) for TSRD dataset & neural models

### Single-Command Cluster Lifecycle Management
To preserve compute credits when not conducting demonstrations:

```powershell
# 1. Start / Resume AKS cluster (takes ~2 minutes)
az aks start --name smartscan-aks --resource-group smartscan-rg

# 2. Run operational smoke test against live endpoint
python scripts/smoke_test.py --api_url http://172.198.227.59 --api_key $SMARTSCAN_API_KEY

# 3. Stop / Pause AKS cluster (freezes compute billing at $0.00)
az aks stop --name smartscan-aks --resource-group smartscan-rg
```

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for full architectural specifications, storage mounting, and CI/CD automation.

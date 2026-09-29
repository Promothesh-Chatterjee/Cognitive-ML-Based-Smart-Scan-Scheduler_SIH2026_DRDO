# Cognitive Electronic Warfare Smart Scan Strategy

[![CI](https://github.com/Promothesh-Chatterjee/SIH2026_Try2/actions/workflows/ci.yml/badge.svg)](https://github.com/Promothesh-Chatterjee/SIH2026_Try2/actions/workflows/ci.yml)
[![Google Cloud Run](https://img.shields.io/badge/Backend-Google_Cloud_Run-4285F4?logo=googlecloud)](https://cognitive-ew-backend-753709137146.asia-south1.run.app/health)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?logo=vercel)](https://sih-2026-try2.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python)](https://python.org)
[![Coverage](https://img.shields.io/badge/Coverage-81%25-brightgreen)](https://github.com/Promothesh-Chatterjee/SIH2026_Try2)

Autonomous cognitive radar scanning strategy for Electronic Warfare (EW) Electronic Support (ES) receivers operating across 36 frequency bands under non-cooperative conditions. Designed for **DRDO Problem Statement SIH26056 (Smart India Hackathon 2026)**, the system intercepts, deinterleaves, and tracks non-cooperative radar emissions—including agile frequency-hopping emitters, periodic scanning search radars, and fixed emitters—without prior threat libraries. By combining high-purity windowed signal deinterleaving with a Dueling Deep Recurrent Q-Network (DRQN) scheduler, the receiver achieves sub-millisecond dwell scheduling decisions. The SmartScan DRQN achieves **17.8× higher dwell-level intercept rate versus random sweep** (42.14% vs 2.36%) in canonical Gate-25k evaluation. Gate-100k retraining targets further improvement to ≥65% IR.

---

## What the System Demonstrates

- Autonomous spectrum scanning across **36 frequency bands**
- **360-D** state representation (36 bands × 10 belief features)
- **180** joint band × dwell-mode actions (36 bands × 5 modes)
- Temporal reasoning with a recurrent Dueling DRQN
- PDW-based deinterleaving and emitter separation (HDBSCAN / DBSCAN fallback)
- Agile/hopping radar handling without static threat libraries
- Quantitative EW Figures of Merit ($P_d$, $P_{fa}$, intercept rate, timing error)
- Live operational backend on Google Cloud Run
- Live Mission Board / SmartScan interface on Vercel
- Reproducible CI qualification and SHA-256 checkpoint integrity verification

---

## Problem Statement → System Response

The receiver must operate in a non-cooperative RF environment: discover and intercept radar emissions without relying on a pre-known threat library, separate overlapping or changing pulse streams, reason over temporal observations, and select scanning actions efficiently across a wide spectrum.

| PS Objective | Implemented Mechanism | How It Addresses the Objective |
|---|---|---|
| Detect non-cooperative emitters | RF simulation + TSRD observations + detection pipeline | Identifies occupied spectrum and candidate emissions without assuming a fixed threat list |
| Separate overlapping emitters | PDW feature extraction + WindowedDeinterleaver + HDBSCAN/DBSCAN fallback + CrossWindowReconciler | Groups pulse descriptors into coherent emitter tracks and maintains identity across windows |
| Handle agile/hopping radars | Temporal feature engineering + recurrent scheduler + 36-band action space | Learns scanning decisions over changing RF occupancy rather than relying on static frequency assignments |
| Choose scan actions intelligently | Dueling DRQN with joint (Band, Mode) action space | Selects both frequency band and dwell strategy from the current 360-D state |
| Operate under uncertainty | Hidden-state / temporal recurrent policy + belief-oriented observation features | Uses temporal context instead of treating each dwell as an isolated classification |
| Quantify operational performance | $P_d$, $P_{fa}$, intercept rate, reward, timing error, coverage and related FoMs | Provides auditable measurements of receiver effectiveness |
| Support live mission operation | FastAPI + telemetry + Mission Board + Cloud Run backend + Vercel frontend | Exposes the cognitive EW system as an operational demonstration service |

---

## How the Cognitive Loop Works

1. **RF/TSRD pulse data** enters the receiver pipeline.
2. **PDWs** (Pulse Descriptor Words) are normalized and transformed into robust temporal/frequency features.
3. **Windowed deinterleaving** groups pulses into candidate emitter streams via HDBSCAN (DBSCAN fallback).
4. **Cross-window reconciliation** maintains emitter identity and continuity across successive windows.
5. The resulting receiver state is encoded as a **360-D observation vector**: 36 bands × 10 state/belief features.
6. The **recurrent Dueling DRQN** evaluates the state and selects a joint **(band, dwell mode)** action from 36 × 5 = 180 actions.
7. The receiver executes the dwell, updates detections/rewards/beliefs, and feeds the resulting telemetry back into the loop.
8. **EW Figures of Merit** are updated and exposed through the FastAPI backend and Mission Board UI.

---

## Why This Is a Cognitive EW System

This system is not a static spectrum sweep or a simple threshold detector. It implements a closed-loop cognitive cycle in which the receiver:

- **Observes** the RF environment through structured pulse descriptor words;
- **Maintains temporal and belief state** across successive dwells using a 2-layer LSTM recurrent core;
- **Reasons** over changing emitter occupancy and behavior through learned value functions;
- **Decides** the next frequency band and dwell mode by evaluating dueling advantage/value heads;
- **Learns** an optimized scanning policy through deep reinforcement learning (DRQN with prioritized replay);
- **Receives feedback** through intercept detections and reward signals; and
- **Measures** operational performance through explicit, auditable Figures of Merit.

The cognitive loop continuously adapts scanning priorities in response to the RF environment rather than following a predetermined schedule. This is the core distinction from conventional round-robin or fixed-pattern scanning strategies, and the reason the system achieves a 17.8× intercept rate improvement over random sweep.

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
                               │   (Pd, Pfa, IR, Reward)   │
                               └─────────────┬─────────────┘
                                             │
                        ┌────────────────────┴────────────────────┐
                        ▼                                         ▼
         ┌──────────────────────────────┐          ┌──────────────────────────────┐
         │      FastAPI Backend         │          │  Telemetry & Mission API     │
         │  Google Cloud Run            │          │  /health  /metrics           │
         │  asia-south1                 │ ───────► │  /telemetry/latest           │
         │  cognitive-ew-backend        │          │  /mission/*  /predict_bands  │
         └──────────────┬───────────────┘          └──────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  Google Cloud Storage (GCS)  │
         │  gs://sih2026-ew-tsrd        │
         │  TSRD + operational artifacts│
         └──────────────────────────────┘

         ┌──────────────────────────────┐
         │     Vercel Frontend          │
         │  React + Vite                │
         │  Mission Board / SmartScan   │
         │  sih-2026-try2.vercel.app    │
         └──────────────────────────────┘
```

---

## Figures of Merit (Canonical Gate-25k Results)

| Metric | Definition | Gate-25k Result (Canonical 2026.1) |
|---|---|---|
| Probability of Detection ($P_d$) | TP / (TP + FN) at selected band | **94.95%** |
| Probability of False Alarm ($P_{fa}$) | FP / (FP + TN) | **0.00%** |
| Receiver Sensitivity | Physics-computed via Friis | **~−110 dBm** |
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
> Gate-100k retraining targets: IR ≥ 65%, $P_d$ ≥ 99%, worst-case IR ≥ 20%, $P_{fa}$ ≤ 0.05%.

**Current deployed operational baseline:** Gate-27 Operational Baseline (training step 27,000).
The canonical Gate-25k benchmarks above were produced under the Gate-25k frozen checkpoint; the Gate-27 operational baseline is the current production model. Gate-27 specific benchmarks will be published separately as the evaluation campaign completes.

---

## Technology Stack

| Layer | Technology |
|---|---|
| RF / ML simulation | Python, NumPy, Gymnasium |
| Deep RL | PyTorch, Stable-Baselines3-compatible components |
| Deinterleaving | HDBSCAN / DBSCAN fallback, temporal feature extraction |
| Backend API | FastAPI + Uvicorn |
| Backend container | Docker |
| Backend runtime | Google Cloud Run (`asia-south1`) |
| Model/data storage | Google Cloud Storage |
| Container registry | Google Artifact Registry |
| Container build | Google Cloud Build |
| Frontend | React + Vite |
| Frontend hosting | Vercel |
| CI | GitHub Actions |
| CI → GCP authentication | GitHub OIDC + Workload Identity Federation |
| Dataset format | HDF5 / H5 TSRD artifacts |
| Telemetry | OpenTelemetry SDK |

> [!NOTE]
> Legacy cloud-compatibility packages (e.g., `azure-storage-blob`) remain in `pyproject.toml` for historical reproducibility. The current production deployment uses Google Cloud Run + Google Cloud Storage + Artifact Registry + Vercel.

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
*Expected output: exits with code `0`, reporting all 7 FoMs and confirming >11.5× intercept improvement over Round-Robin.*

### 3. Run Test Suite with Coverage
```bash
pytest ew_core/tests/ --cov=ew_core --cov-fail-under=70
```
*Executes all unit, integration, and regression tests with strict coverage enforcement (≥81% achieved).*

### 4. Launch Backend API Server (Local)
```bash
uvicorn ew_core.deployment.api:app --host 0.0.0.0 --port 8000 --reload
```
Key API endpoints:
- `GET /health` — Service health, active model, and device metadata
- `GET /api/v1/metrics` — Latest episode EW Figures of Merit ($P_d$, $P_{fa}$, intercept rate)
- `GET /api/v1/spectrum` — Time-frequency truth matrix and receiver dwell positions
- `POST /predict_bands` — Inference endpoint for scheduler action selection
- `POST /deinterleave` — High-speed PDW clustering and emitter separation
- `GET /telemetry/latest` — Most recent telemetry snapshot
- `GET /mission/status` — Active mission state and progress
- `POST /mission/start` — Start a TSRD mission scenario
- `POST /mission/step` — Execute the next dwell step in a running mission

### 5. Launch Frontend (Local)
```bash
cd frontend
npm install
npm run dev
```
The React development server connects to the local backend at `http://localhost:8000` by default.

---

## Production Deployment & Operations

The current demonstration deployment uses a serverless Google Cloud backend and a Vercel-hosted React frontend.

| Component | Service | Details |
|---|---|---|
| **Backend** | Google Cloud Run | `cognitive-ew-backend`, `asia-south1`, 1 vCPU, 2 GiB RAM, min=0 / max=1 instances |
| **Data & Model Artifacts** | Google Cloud Storage | `gs://sih2026-ew-tsrd` — TSRD dataset + operational checkpoints |
| **Container Image** | Google Artifact Registry | `asia-south1-docker.pkg.dev/sih2026-ew-demo/ew/` |
| **Container Build** | Google Cloud Build | `cloudbuild.yaml` builds and pushes to Artifact Registry |
| **Frontend** | Vercel | React + Vite SPA, global edge network |
| **CI/CD** | GitHub Actions | Full Qualification Suite with Gate-27 integrity validation |
| **CI → GCP Auth** | OIDC + Workload Identity Federation | No long-lived service-account JSON keys |
| **Runtime Identity** | Google Cloud IAM | `sih2026-ew-runtime@sih2026-ew-demo.iam.gserviceaccount.com` |

### Operational Checkpoint

- **Active Model:** Gate-27 Operational Baseline (training step 27,000)
- **SHA-256:** `fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094`
- **Source Artifact:** `gs://sih2026-ew-tsrd/checkpoints/checkpoint_gate_27000_operational.pt`

Checkpoint integrity is verified at build time, CI provisioning, and runtime startup using SHA-256 comparison against the authoritative hash.

### CI/CD & Deployment Flow

```
Developer Push
       ↓
GitHub Actions (Full Qualification Suite)
       ↓
OIDC / Workload Identity Federation
       ↓
GCS Checkpoint Provisioning & SHA-256 Verification
       ↓
Google Cloud Build
       ↓
Artifact Registry
       ↓
Cloud Run Backend (FastAPI + Mission Controller + Telemetry)
       ↓
Vercel React Mission Board (Frontend UI Layer)
```

---

## Live Deployment

| Endpoint | URL |
|---|---|
| **Backend Health** | [https://cognitive-ew-backend-753709137146.asia-south1.run.app/health](https://cognitive-ew-backend-753709137146.asia-south1.run.app/health) |
| **Frontend (Mission Board)** | [https://sih-2026-try2.vercel.app](https://sih-2026-try2.vercel.app) |

---

## Project Structure

```
SIH2026_Try2/
├── ew_core/                    # Core Python package
│   ├── deployment/             # FastAPI backend, dataset service, mission controller
│   ├── models/                 # DRQN, deinterleaver, and scheduler architectures
│   ├── environments/           # Gymnasium SpectrumEnvironment (SmartScanEW-v0)
│   ├── utils/                  # Checkpoint management, metrics, feature engineering
│   └── tests/                  # Unit, integration, regression, and qualification tests
├── experiments/
│   └── checkpoints/            # Gate-25 frozen + Gate-27 operational baselines
├── frontend/                   # React + Vite Mission Board SPA
├── scripts/                    # Pipeline validation, baseline provisioning, deployment
├── reports/                    # Benchmark provenance and evaluation reports
├── Dockerfile                  # Multi-stage container build
├── cloudbuild.yaml             # Google Cloud Build configuration
├── DEPLOYMENT_GUIDE.md         # Full production deployment reference
└── pyproject.toml              # Package configuration and dependencies
```

---

## References

- **SIH Problem Statement:** DRDO SIH26056 — Smart Scan Strategy for EW ES Receivers
- **Deployment Guide:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **Benchmark Provenance:** [reports/BENCHMARK_PROVENANCE.md](reports/BENCHMARK_PROVENANCE.md)

---

## License

See repository for license details.

# Google Cloud Platform (GCP) + Vercel Production Deployment Guide

This guide documents the production cloud deployment of **Cognitive EW SmartScan** on **Google Cloud Platform (GCP)** (Backend & TSRD Data) and **Vercel** (Mission Control Frontend).

---

## 1. Cloud Architecture Overview

The production system utilizes a decoupled, high-performance, cost-effective serverless architecture:

```
                                    GOOGLE CLOUD (asia-south1, Mumbai)
                                    ┌─────────────────────────────────────────────────────────┐
                                    │                                                         │
  ┌─────────────────────────┐       │   ┌─────────────────────────────────────────────────┐   │
  │     Vercel Frontend     │       │   │           Cloud Run: cognitive-ew-backend       │   │
  │      sih-2026-try2      │=====> │   │  - 1 vCPU, 2 GiB RAM, CPU-only, min=0, max=1   │   │
  │  (React 19 / Vite 8)    │ HTTP/ │   │  - Active Model: Gate-27 Operational Baseline   │   │
  │  https://sih-2026-try2. │  WS   │   │  - Public HTTPS:                                │   │
  │  vercel.app             │       │   │    https://cognitive-ew-backend-...run.app      │   │
  └─────────────────────────┘       │   │    https://cognitive-ew-backend-...run.app      │   │
                                    │   └───────────────────────┬─────────────────────────┘   │
                                    │                           │                             │
                                    │                           v                             │
                                    │   ┌─────────────────────────────────────────────────┐   │
                                    │   │      Cloud Storage (GCS): sih2026-ew-tsrd       │   │
                                    │   │  - 250 Authentic TSRD H5 Scenarios             │   │
                                    │   │  - Gate-27 & Deinterleaver Checkpoints          │   │
                                    │   └─────────────────────────────────────────────────┘   │
                                    │                                                         │
                                    │   ┌─────────────────────────────────────────────────┐   │
                                    │   │      Artifact Registry: ew/cognitive-ew-backend │   │
                                    │   │  - Docker Container Image                       │   │
                                    │   └─────────────────────────────────────────────────┘   │
                                    └─────────────────────────────────────────────────────────┘
```

---

## 2. Resource Inventory & Configuration

| Component | Service | Identifier / Path | Region | Configuration |
| :--- | :--- | :--- | :--- | :--- |
| **GCP Project** | Cloud Resource Manager | `sih2026-ew-demo` | Global | Dedicated demo project |
| **Backend Engine** | Cloud Run | `cognitive-ew-backend` | `asia-south1` | 1 vCPU, 2 GiB, min=0, max=1, timeout=3600s |
| **Container Registry** | Artifact Registry | `asia-south1-docker.pkg.dev/sih2026-ew-demo/ew` | `asia-south1` | Container image repository |
| **Dataset & Checkpoints** | Cloud Storage (GCS) | `gs://sih2026-ew-tsrd` | `asia-south1` | 250 TSRD H5 files + Gate-27 & Deinterleaver weights |
| **Runtime Identity** | Cloud IAM | `sih2026-ew-runtime@sih2026-ew-demo.iam...` | Global | Minimal privilege service account |
| **Mission Control UI** | Vercel | `sih-2026-try2` | Global Edge | React 19 + Vite 8 SPA |

---

## 3. Operational Integrity & Verified Checkpoints

* **Gate-27 Operational Baseline:**
  * File: `checkpoint_gate_27000_operational.pt`
  * SHA-256: `fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094`
* **Deinterleaver Model:**
  * File: `best.pt`
  * SHA-256: `b7cc3727b3b1940ac8c06e61f127644c484de69ec16080110dd4ea8c0c44b116`
* **Normalization Hash:**
  * Value: `bacee02ac1c29428`
* **Benchmark Version:**
  * Value: `2026.1-CANONICAL` (SHA-256: `544c9c02cfded9acd33062ff962bb7a264103cf2229a2c6aa99f78c3379b0322`)
* **Observation Space:**
  * Dimension: Canonical 360-D vector (36 bands × 10 belief features)

---

## 4. Frontend Deployment on Vercel

The frontend is located in `frontend/` and communicates directly with the deployed Cloud Run service.

1. **Environment Configuration:**
   * `frontend/.env.production` defines:
     ```env
     VITE_API_BASE_URL=https://cognitive-ew-backend-753709137146.asia-south1.run.app
     VITE_WS_BASE_URL=wss://cognitive-ew-backend-753709137146.asia-south1.run.app
     ```
2. **Build Verification:**
   ```bash
   cd frontend
   npm run lint
   npm run build
   ```
3. **Deployment:**
   * Connect repository to Vercel with Root Directory set to `frontend` or build via Vercel CLI.
   * Client-side routing is handled via `frontend/vercel.json`.

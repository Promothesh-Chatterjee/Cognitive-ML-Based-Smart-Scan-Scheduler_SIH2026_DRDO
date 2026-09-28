#!/usr/bin/env bash
# =============================================================================
# SIH2026 Phase B — Google Cloud Platform Deployment Script (Bash / Cloud Shell)
# Target Architecture: Cloud Run (Backend) + Cloud Storage (TSRD) + Firebase (Frontend)
# Region: asia-south1 (Mumbai)
# Constraints: CPU-only, 1 vCPU, 2 GiB RAM, max instances = 1, min instances = 0
# =============================================================================

set -euo pipefail

PROJECT_ID="${1:-sih2026-ew-demo}"
REGION="${2:-asia-south1}"
BUCKET_NAME="${3:-sih2026-ew-tsrd}"
REPO_NAME="ew"
SERVICE_NAME="cognitive-ew-backend"
SERVICE_ACCOUNT_NAME="sih2026-ew-runtime"
LOCAL_TSRD_DIR="${4:-/mnt/tsrd/stare/val_stare}"
GATE27_CHECKPOINT="experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_27000_operational.pt"
EXPECTED_GATE27_SHA="fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094"

echo "====================================================================="
echo " SIH2026 Phase B: GCP Cloud Mission Controller Deployment"
echo " Project: ${PROJECT_ID} | Region: ${REGION}"
echo "====================================================================="

# PHASE B0: Local Verification & Hash Integrity Check
echo "[PHASE B0] Verifying immutable baseline and local integrity..."
if [ -f "${GATE27_CHECKPOINT}" ]; then
    ACTUAL_SHA=$(sha256sum "${GATE27_CHECKPOINT}" | awk '{print $1}')
    if [ "${ACTUAL_SHA}" != "${EXPECTED_GATE27_SHA}" ]; then
        echo "ERROR: Gate-27 SHA mismatch! Expected ${EXPECTED_GATE27_SHA}, got ${ACTUAL_SHA}"
        exit 1
    fi
    echo "  [OK] Gate-27 SHA-256 verified: ${ACTUAL_SHA}"
fi

# PHASE B1: Project Selection & API Enablement (Non-billable setup)
echo "[PHASE B1] Setting up GCP project and enabling required APIs..."
gcloud config set project "${PROJECT_ID}"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com

echo "  [OK] APIs enabled successfully."

# 🚦 APPROVAL GATE A — Potentially Billable Operations Confirmation
echo "====================================================================="
echo " 🚦 APPROVAL GATE A: Potentially Billable Cloud Operations"
echo " Next operations will create:"
echo "   1. GCS Bucket: gs://${BUCKET_NAME} (Standard, ${REGION})"
echo "   2. Upload 250 TSRD H5 files (~4.56 GiB)"
echo "   3. Artifact Registry: ${REPO_NAME} in ${REGION}"
echo "   4. Remote Cloud Build of container image"
echo "   5. Cloud Run Deployment: 1 vCPU, 2 GiB RAM, min=0, max=1"
echo "====================================================================="

read -p "Proceed with billable cloud resource provisioning? (yes/no): " CONFIRM
if [[ "${CONFIRM}" != "yes" && "${CONFIRM}" != "y" ]]; then
    echo "Deployment paused at Approval Gate A as requested. No cloud resources created."
    exit 0
fi

# PHASE B2: Create TSRD GCS Bucket
echo "[PHASE B2] Creating GCS Bucket gs://${BUCKET_NAME}..."
gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --location="${REGION}" \
    --default-storage-class=STANDARD \
    --uniform-bucket-level-access

# PHASE B3: Upload TSRD Dataset & Validate
echo "[PHASE B3] Uploading TSRD H5 dataset to gs://${BUCKET_NAME}/val_stare/..."
if [ -d "${LOCAL_TSRD_DIR}" ]; then
    gcloud storage cp "${LOCAL_TSRD_DIR}"/config_*.h5 "gs://${BUCKET_NAME}/val_stare/" --no-clobber
fi

# PHASE B4: Upload Gate-27 Checkpoint
echo "[PHASE B4] Uploading Gate-27 operational checkpoint..."
if [ -f "${GATE27_CHECKPOINT}" ]; then
    gcloud storage cp "${GATE27_CHECKPOINT}" "gs://${BUCKET_NAME}/checkpoints/checkpoint_gate_27000_operational.pt"
    gcloud storage cp experiments/checkpoints/scheduler_v2_operational_candidate/ACTIVE_CHECKPOINT.json "gs://${BUCKET_NAME}/checkpoints/ACTIVE_CHECKPOINT.json"
    gcloud storage cp experiments/checkpoints/scheduler_v2_operational_candidate/GATE27_OPERATIONAL_MANIFEST.json "gs://${BUCKET_NAME}/checkpoints/GATE27_OPERATIONAL_MANIFEST.json"
fi

# PHASE B5: Service Account & Artifact Registry
echo "[PHASE B5] Configuring IAM Service Account and Artifact Registry..."
gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --description="Minimal runtime SA for Cognitive EW Mission Controller" \
    --display-name="Cognitive EW Runtime SA" || true

SA_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectViewer"

gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Cognitive EW Container Repository" || true

echo "  Triggering Cloud Build remotely (E2_STANDARD_2 default pool)..."
gcloud builds submit --config=cloudbuild.yaml .

# PHASE B6: Deploy Cloud Run Service
echo "[PHASE B6] Deploying Cloud Run service: ${SERVICE_NAME}..."
GIT_COMMIT=$(git rev-parse HEAD)
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/cognitive-ew-backend:gate27"

gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE_URI}" \
    --region="${REGION}" \
    --platform=managed \
    --allow-unauthenticated \
    --service-account="${SA_EMAIL}" \
    --cpu=1 \
    --memory=2Gi \
    --min-instances=0 \
    --max-instances=1 \
    --timeout=3600 \
    --concurrency=40 \
    --set-env-vars="GCS_TSRD_BUCKET=${BUCKET_NAME},REQUIRE_OPERATIONAL_CHECKPOINT=true,TSRD_DATA_ROOT=/app/data,GIT_COMMIT=${GIT_COMMIT}"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region="${REGION}" --format='value(status.url)')
echo "  [OK] Cloud Run deployed at: ${SERVICE_URL}"

# PHASE B7: Validate Staging Endpoints
echo "[PHASE B7] Validating Cloud Run deployment..."
curl -s "${SERVICE_URL}/health" | grep -q "Gate-27 Operational Baseline" && echo "  [OK] Health check verified Gate-27"

# PHASE B8: Build React Frontend for Vercel
echo "[PHASE B8] Building frontend with Cloud Run backend URL..."
echo "VITE_API_BASE_URL=${SERVICE_URL}" > frontend/.env.production
(cd frontend && npm run build)

echo "====================================================================="
echo " GCP + VERCEL DEPLOYMENT STATUS"
echo " Backend:  ${SERVICE_URL}"
echo " Frontend: Ready for Vercel deployment (dist/ built)"
echo "====================================================================="

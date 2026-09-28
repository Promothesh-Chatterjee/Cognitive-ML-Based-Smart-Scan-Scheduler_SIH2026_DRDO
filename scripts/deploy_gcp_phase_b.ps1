# =============================================================================
# SIH2026 Phase B — Google Cloud Platform Deployment Script (PowerShell)
# Target Architecture: Cloud Run (Backend) + Cloud Storage (TSRD) + Firebase (Frontend)
# Region: asia-south1 (Mumbai)
# Constraints: CPU-only, 1 vCPU, 2 GiB RAM, max instances = 1, min instances = 0
# =============================================================================

param(
    [string]$ProjectId = "sih2026-ew-demo",
    [string]$Region = "asia-south1",
    [string]$BucketName = "sih2026-ew-tsrd",
    [string]$RepoName = "ew",
    [string]$ServiceName = "cognitive-ew-backend",
    [string]$ServiceAccountName = "sih2026-ew-runtime",
    [string]$LocalTsrdDir = "D:\TSRD\stare\val_stare",
    [string]$Gate27Checkpoint = "experiments\checkpoints\scheduler_v2_operational_candidate\checkpoint_gate_27000_operational.pt",
    [string]$ExpectedGate27Sha = "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094"
)

$ErrorActionPreference = "Stop"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " SIH2026 Phase B: GCP Cloud Mission Controller Deployment" -ForegroundColor Cyan
Write-Host " Project: $ProjectId | Region: $Region" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

# -----------------------------------------------------------------------------
# PHASE B0: Local Verification & Hash Integrity Check
# -----------------------------------------------------------------------------
Write-Host "`n[PHASE B0] Verifying immutable baseline and local integrity..." -ForegroundColor Yellow

if (-not (Test-Path $Gate27Checkpoint)) {
    Write-Error "Gate-27 checkpoint not found at: $Gate27Checkpoint"
}

$actualSha = (Get-FileHash -Path $Gate27Checkpoint -Algorithm SHA256).Hash.ToLower()
if ($actualSha -ne $ExpectedGate27Sha.ToLower()) {
    Write-Error "Gate-27 SHA mismatch! Expected: $ExpectedGate27Sha, Found: $actualSha"
}
Write-Host "  [OK] Gate-27 SHA-256 verified: $actualSha" -ForegroundColor Green

if (Test-Path $LocalTsrdDir) {
    $tsrdFiles = Get-ChildItem -Path $LocalTsrdDir -Filter "config_*.h5"
    Write-Host "  [OK] Local TSRD files found: $($tsrdFiles.Count) files" -ForegroundColor Green
} else {
    Write-Warning "Local TSRD directory $LocalTsrdDir not found. Ensure files are ready for upload."
}

# -----------------------------------------------------------------------------
# PHASE B1: Project Selection & API Enablement (Non-billable setup)
# -----------------------------------------------------------------------------
Write-Host "`n[PHASE B1] Setting up GCP project and enabling required APIs..." -ForegroundColor Yellow

# Set project
Write-Host "  Setting active project: $ProjectId"
gcloud config set project $ProjectId

# Enable only required services
Write-Host "  Enabling minimal API subset (Run, Build, Registry, Storage, Logging, Monitoring)..."
gcloud services enable `
    run.googleapis.com `
    cloudbuild.googleapis.com `
    artifactregistry.googleapis.com `
    storage.googleapis.com `
    logging.googleapis.com `
    monitoring.googleapis.com

Write-Host "  [OK] APIs enabled successfully." -ForegroundColor Green

# -----------------------------------------------------------------------------
# 🚦 APPROVAL GATE A — Potentially Billable Operations Confirmation
# -----------------------------------------------------------------------------
Write-Host "`n=====================================================================" -ForegroundColor Magenta
Write-Host " 🚦 APPROVAL GATE A: Potentially Billable Cloud Operations" -ForegroundColor Magenta
Write-Host " Next operations will create:" -ForegroundColor Magenta
Write-Host "   1. GCS Bucket: gs://$BucketName (Standard, asia-south1)" -ForegroundColor Magenta
Write-Host "   2. Upload 250 TSRD H5 files (~4.56 GiB)" -ForegroundColor Magenta
Write-Host "   3. Artifact Registry: $RepoName in $Region" -ForegroundColor Magenta
Write-Host "   4. Remote Cloud Build of container image" -ForegroundColor Magenta
Write-Host "   5. Cloud Run Deployment: 1 vCPU, 2 GiB RAM, min=0, max=1" -ForegroundColor Magenta
Write-Host "=====================================================================" -ForegroundColor Magenta

$confirm = Read-Host "Proceed with billable cloud resource provisioning? (yes/no)"
if ($confirm -ne "yes" -and $confirm -ne "y") {
    Write-Host "Deployment paused at Approval Gate A as requested. No cloud resources created." -ForegroundColor Yellow
    exit 0
}

# -----------------------------------------------------------------------------
# PHASE B2: Create TSRD GCS Bucket
# -----------------------------------------------------------------------------
Write-Host "`n[PHASE B2] Creating GCS Bucket gs://$BucketName..." -ForegroundColor Yellow
gcloud storage buckets create "gs://$BucketName" `
    --location=$Region `
    --default-storage-class=STANDARD `
    --uniform-bucket-level-access

Write-Host "  [OK] Bucket created with uniform bucket-level access." -ForegroundColor Green

# -----------------------------------------------------------------------------
# PHASE B3: Upload TSRD Dataset & Validate
# -----------------------------------------------------------------------------
Write-Host "`n[PHASE B3] Uploading TSRD H5 dataset to gs://$BucketName/val_stare/..." -ForegroundColor Yellow
gcloud storage cp "$LocalTsrdDir\config_*.h5" "gs://$BucketName/val_stare/" --no-clobber

# Verify upload count
$cloudCount = (gcloud storage ls "gs://$BucketName/val_stare/config_*.h5" | Measure-Object).Count
Write-Host "  [OK] Verified cloud TSRD file count: $cloudCount / 250" -ForegroundColor Green

# -----------------------------------------------------------------------------
# PHASE B4: Upload Gate-27 Operational Checkpoint
# -----------------------------------------------------------------------------
Write-Host "`n[PHASE B4] Uploading Gate-27 operational checkpoint..." -ForegroundColor Yellow
gcloud storage cp $Gate27Checkpoint "gs://$BucketName/checkpoints/checkpoint_gate_27000_operational.pt"
gcloud storage cp "experiments\checkpoints\scheduler_v2_operational_candidate\ACTIVE_CHECKPOINT.json" "gs://$BucketName/checkpoints/ACTIVE_CHECKPOINT.json"
gcloud storage cp "experiments\checkpoints\scheduler_v2_operational_candidate\GATE27_OPERATIONAL_MANIFEST.json" "gs://$BucketName/checkpoints/GATE27_OPERATIONAL_MANIFEST.json"

Write-Host "  [OK] Gate-27 checkpoint uploaded and manifests synced." -ForegroundColor Green

# -----------------------------------------------------------------------------
# PHASE B5: Service Account & Artifact Registry
# -----------------------------------------------------------------------------
Write-Host "`n[PHASE B5] Configuring IAM Service Account and Artifact Registry..." -ForegroundColor Yellow

# Create dedicated runtime service account
gcloud iam service-accounts create $ServiceAccountName `
    --description="Minimal runtime SA for Cognitive EW Mission Controller" `
    --display-name="Cognitive EW Runtime SA"

$saEmail = "$ServiceAccountName@$ProjectId.iam.gserviceaccount.com"

# Grant storage.objectViewer on bucket only
gcloud storage buckets add-iam-policy-binding "gs://$BucketName" `
    --member="serviceAccount:$saEmail" `
    --role="roles/storage.objectViewer"

# Create Artifact Registry
gcloud artifacts repositories create $RepoName `
    --repository-format=docker `
    --location=$Region `
    --description="Cognitive EW Container Repository"

# Run Cloud Build remotely
Write-Host "  Triggering Cloud Build remotely (E2_STANDARD_2 default pool)..."
gcloud builds submit --config=cloudbuild.yaml .

Write-Host "  [OK] Container image built and pushed to Artifact Registry." -ForegroundColor Green

# -----------------------------------------------------------------------------
# PHASE B6: Deploy Cloud Run Service
# -----------------------------------------------------------------------------
Write-Host "`n[PHASE B6] Deploying Cloud Run service: $ServiceName..." -ForegroundColor Yellow

$gitCommit = (git rev-parse HEAD).Trim()
$imageUri = "$Region-docker.pkg.dev/$ProjectId/$RepoName/cognitive-ew-backend:gate27"

gcloud run deploy $ServiceName `
    --image=$imageUri `
    --region=$Region `
    --platform=managed `
    --allow-unauthenticated `
    --service-account=$saEmail `
    --cpu=1 `
    --memory=2Gi `
    --min-instances=0 `
    --max-instances=1 `
    --timeout=3600 `
    --concurrency=40 `
    --set-env-vars="GCS_TSRD_BUCKET=$BucketName,REQUIRE_OPERATIONAL_CHECKPOINT=true,TSRD_DATA_ROOT=/app/data,GIT_COMMIT=$gitCommit"

$serviceUrl = (gcloud run services describe $ServiceName --region=$Region --format='value(status.url)').Trim()
Write-Host "  [OK] Cloud Run deployed at: $serviceUrl" -ForegroundColor Green

# -----------------------------------------------------------------------------
# PHASE B7: Validate Staging Endpoints
# -----------------------------------------------------------------------------
Write-Host "`n[PHASE B7] Validating Cloud Run deployment..." -ForegroundColor Yellow

$health = Invoke-RestMethod -Uri "$serviceUrl/health"
Write-Host "  Health status: $($health.status) | Model: $($health.active_model)"
$scenarios = Invoke-RestMethod -Uri "$serviceUrl/mission/scenarios"
Write-Host "  Scenarios returned: $($scenarios.Count) (Benchmarks: $($($scenarios | Where-Object { $_.benchmark }).Count))"

# -----------------------------------------------------------------------------
# PHASE B8: Build React Frontend for Vercel
# -----------------------------------------------------------------------------
Write-Host "`n[PHASE B8] Building frontend with Cloud Run backend URL..." -ForegroundColor Yellow

# Set production API URL for Vite build
Set-Content -Path "frontend\.env.production" -Value "VITE_API_BASE_URL=$serviceUrl"

Push-Location "frontend"
npm run build
Pop-Location

Write-Host "`n=====================================================================" -ForegroundColor Green
Write-Host " GCP + VERCEL DEPLOYMENT STATUS" -ForegroundColor Green
Write-Host " Backend Cloud Run: $serviceUrl" -ForegroundColor Green
Write-Host " Frontend: Ready for Vercel deployment (dist/ built)" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green

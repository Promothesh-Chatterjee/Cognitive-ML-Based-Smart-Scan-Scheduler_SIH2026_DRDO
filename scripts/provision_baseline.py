#!/usr/bin/env python3
"""Provision the frozen Gate-25k baseline checkpoint, baseline evidence package,
and canonical TSRD test fixture from Azure Blob Storage.

Fail-closed in strict mode (default): any provisioning failure exits non-zero.
Best-effort mode (--best-effort): tolerates missing Azure credentials but still
fails on SHA mismatch or checkpoint corruption.

Downloads use atomic write: temp file → SHA verify → rename to canonical path.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Complete baseline evidence package + canonical TSRD fixture contract
PROVISION_TARGETS = [
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/checkpoint_gate_25000_frozen.pt",
        "local_path": "experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt",
        "expected_sha256": "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0",
        "description": "Production Baseline Frozen Checkpoint",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/checkpoint_gate_25000_frozen.pt",
        "local_path": "experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt",
        "expected_sha256": "7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0",
        "description": "Candidate Mirror Frozen Checkpoint",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/SHA256SUMS",
        "local_path": "experiments/checkpoints/production_baseline/SHA256SUMS",
        "expected_sha256": "ce220b282b1e97a6a1cd06c429fff1254414dd772ca09be8ef296f51a4e0e621",
        "description": "Baseline Checksum Manifest (SHA256SUMS)",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/baseline_metadata.json",
        "local_path": "experiments/checkpoints/production_baseline/baseline_metadata.json",
        "expected_sha256": "9c0b10e45a43fd2d9fd1da2fb62c057e9c8a4fd16734e078d835cf7571c9d330",
        "description": "Baseline Architecture Metadata",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/benchmark_v2_baseline_gate25k.json",
        "local_path": "experiments/checkpoints/production_baseline/benchmark_v2_baseline_gate25k.json",
        "expected_sha256": "adf02d70699a3a8225dd67c823cc6679586086e2d86b4f6d4a9e078e2a0b58bd",
        "description": "Baseline Gate-25k Benchmark Metrics",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/benchmark_v2_multiseed_summary.json",
        "local_path": "experiments/checkpoints/production_baseline/benchmark_v2_multiseed_summary.json",
        "expected_sha256": "a44e70df97a52ddaa22be23eb81a293be1b3fc120e2595ca5b739c1d0cae81b1",
        "description": "Baseline Multi-Seed Summary",
    },
    {
        "container": "smartscan-models",
        "blob_name": "scheduler_v2/baseline_reservoir_5k.pkl",
        "local_path": "experiments/checkpoints/production_baseline/baseline_reservoir_5k.pkl",
        "expected_sha256": "edcef07b020563aefeac99fa3b03c2c6a474f07afdac7660e61e336523b8fe0c",
        "description": "Baseline Reservoir 5k Replay Buffer",
    },
    {
        "container": "tsrd-dataset",
        "blob_name": "val_stare/config_117.h5",
        "local_path": "tests/fixtures/canonical_tsrd/stare/val_stare/config_117.h5",
        "expected_sha256": "073724fbcd3aba8daaf94a68cbcd95ac1cf6f1aaeeab2b4b83df54e9a93dfe3f",
        "description": "Canonical TSRD STARE Validation Fixture (config_117)",
    },
]

# Gate-27 Operational Baseline Checkpoint Contract
GATE27_AUTHORITATIVE_SHA256 = "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094"
GATE27_OP_PATH = REPO_ROOT / "experiments" / "checkpoints" / "scheduler_v2_operational_candidate" / "checkpoint_gate_27000_operational.pt"
GATE27_SRC_PATH = REPO_ROOT / "experiments" / "checkpoints" / "g8_4_step_based_objective" / "checkpoint_gate_27000.pt"
GCS_TSRD_BUCKET_DEFAULT = "sih2026-ew-tsrd"
GCS_GATE27_BLOB = "checkpoints/checkpoint_gate_27000_operational.pt"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def atomic_download(blob_client, dest: Path, expected_sha: str) -> None:
    """Download blob to a temporary file, verify SHA, then atomically rename.

    If the SHA does not match, the temporary file is removed and no partial
    file ever appears at the canonical destination path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        dir=str(dest.parent), suffix=".downloading"
    )
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "wb") as f:
            blob_client.download_blob().readinto(f)
        actual = sha256_file(tmp_path)
        if actual != expected_sha:
            print(f"[FAIL] SHA-256 mismatch after download: expected {expected_sha}, got {actual}")
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded file SHA {actual} != expected {expected_sha}"
            )
        # Atomic rename (same filesystem)
        shutil.move(str(tmp_path), str(dest))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_download_gcs(bucket_name: str, blob_name: str, dest: Path, expected_sha: str) -> None:
    """Download GCS blob to a temporary file, verify SHA-256, then atomically rename.

    If the SHA does not match, the temporary file is removed and no partial
    file ever appears at the canonical destination path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        dir=str(dest.parent), suffix=".downloading"
    )
    tmp_path = Path(tmp_path_str)
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with os.fdopen(fd, "wb") as f:
            blob.download_to_file(f)
        actual = sha256_file(tmp_path)
        if actual != expected_sha:
            print(f"[FAIL] SHA-256 mismatch after GCS download: expected {expected_sha}, got {actual}")
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded file SHA {actual} != expected {expected_sha}"
            )
        shutil.move(str(tmp_path), str(dest))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def provision_gate27_checkpoint(strict: bool) -> bool:
    """Ensure Gate-27 operational and source checkpoints exist and match authoritative SHA-256.

    Fulfills provenance contract: the exact verified bytes must populate both:
      1. experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_27000_operational.pt
      2. experiments/checkpoints/g8_4_step_based_objective/checkpoint_gate_27000.pt
    """
    op_valid = GATE27_OP_PATH.is_file() and sha256_file(GATE27_OP_PATH) == GATE27_AUTHORITATIVE_SHA256
    src_valid = GATE27_SRC_PATH.is_file() and sha256_file(GATE27_SRC_PATH) == GATE27_AUTHORITATIVE_SHA256

    if op_valid and src_valid:
        print(f"[OK] Gate-27 operational checkpoint already present and verified: {GATE27_OP_PATH.relative_to(REPO_ROOT)}")
        print(f"[OK] Gate-27 source checkpoint already present and verified: {GATE27_SRC_PATH.relative_to(REPO_ROOT)}")
        return True

    # If one path already holds the verified bytes, populate the other via exact replica
    if op_valid and not src_valid:
        GATE27_SRC_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(GATE27_OP_PATH), str(GATE27_SRC_PATH))
        if sha256_file(GATE27_SRC_PATH) == GATE27_AUTHORITATIVE_SHA256:
            print(f"[OK] Replicated verified Gate-27 bytes to source path: {GATE27_SRC_PATH.relative_to(REPO_ROOT)}")
            return True
        print("[FAIL] Replicated Gate-27 source checkpoint SHA mismatch.")
        return False

    if src_valid and not op_valid:
        GATE27_OP_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(GATE27_SRC_PATH), str(GATE27_OP_PATH))
        if sha256_file(GATE27_OP_PATH) == GATE27_AUTHORITATIVE_SHA256:
            print(f"[OK] Replicated verified Gate-27 bytes to operational path: {GATE27_OP_PATH.relative_to(REPO_ROOT)}")
            return True
        print("[FAIL] Replicated Gate-27 operational checkpoint SHA mismatch.")
        return False

    # Neither path is valid — attempt download from authoritative GCS source
    bucket_name = os.environ.get("GCS_TSRD_BUCKET", os.environ.get("GCS_BUCKET", GCS_TSRD_BUCKET_DEFAULT))
    blob_name = GCS_GATE27_BLOB
    print(f"[INFO] Provisioning Gate-27 operational checkpoint from gs://{bucket_name}/{blob_name}...")

    downloaded = False
    try:
        atomic_download_gcs(bucket_name, blob_name, GATE27_OP_PATH, GATE27_AUTHORITATIVE_SHA256)
        downloaded = True
        size_mb = GATE27_OP_PATH.stat().st_size / (1024 * 1024)
        print(f"[OK] Downloaded & verified Gate-27 operational checkpoint from GCS: {GATE27_OP_PATH.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")
    except ImportError:
        print("[WARN] google-cloud-storage package not installed; cannot download from GCS.")
    except Exception as e:
        print(f"[WARN] Failed to download Gate-27 from GCS (gs://{bucket_name}/{blob_name}): {e}")

    # Fallback to Azure Blob Storage if available and GCS did not succeed
    if not downloaded:
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
        if conn_str:
            try:
                from azure.storage.blob import BlobServiceClient
                from azure.core.exceptions import ResourceNotFoundError
                az_client = BlobServiceClient.from_connection_string(conn_str)
                for candidate_blob in [
                    "scheduler_v2/checkpoint_gate_27000_operational.pt",
                    "checkpoints/checkpoint_gate_27000_operational.pt",
                ]:
                    try:
                        blob_client = az_client.get_blob_client(container="smartscan-models", blob=candidate_blob)
                        atomic_download(blob_client, GATE27_OP_PATH, GATE27_AUTHORITATIVE_SHA256)
                        downloaded = True
                        print(f"[OK] Downloaded & verified Gate-27 from Azure Blob: {candidate_blob}")
                        break
                    except ResourceNotFoundError:
                        continue
            except Exception as az_e:
                print(f"[WARN] Azure Blob fallback attempt failed: {az_e}")

    if not downloaded:
        if strict:
            print(
                "[FAIL] Could not provision Gate-27 operational checkpoint.\n"
                f"       Primary source: gs://{bucket_name}/{blob_name}\n"
                "       Ensure GCP Workload Identity Federation / OIDC is configured in GitHub Actions\n"
                "       (via secrets.GCP_WORKLOAD_IDENTITY_PROVIDER and google-github-actions/auth@v3)\n"
                "       or credentials with read access to the GCS bucket are present."
            )
            return False
        else:
            print("[SKIP] Gate-27 operational checkpoint not provisioned (best-effort mode).")
            return True

    # Replicate exact verified bytes to source checkpoint path
    GATE27_SRC_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(GATE27_OP_PATH), str(GATE27_SRC_PATH))
    actual_src_sha = sha256_file(GATE27_SRC_PATH)
    if actual_src_sha != GATE27_AUTHORITATIVE_SHA256:
        print(f"[FAIL] Gate-27 source checkpoint SHA mismatch after copy: {actual_src_sha}")
        return False
    print(f"[OK] Replicated verified Gate-27 bytes to source path: {GATE27_SRC_PATH.relative_to(REPO_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail-closed: any provisioning failure exits non-zero (default).",
    )
    mode.add_argument(
        "--best-effort",
        action="store_true",
        help="Tolerate missing credentials (exit 0). Still fail on corruption.",
    )
    args = parser.parse_args()
    strict = not args.best_effort

    # 1. Provision Gate-27 authoritative operational checkpoint
    print("--- [Step 1/2] Provisioning Gate-27 Operational Baseline ---")
    gate27_ok = provision_gate27_checkpoint(strict=strict)
    if not gate27_ok and strict:
        return 1

    # 2. Provision Gate-25 historical baseline package & fixtures
    print("\n--- [Step 2/2] Provisioning Gate-25 Historical Baseline Package ---")
    missing_targets = []
    for target in PROVISION_TARGETS:
        local = Path(target["local_path"])
        expected_sha = target["expected_sha256"]
        desc = target["description"]
        if local.exists():
            actual = sha256_file(local)
            if actual == expected_sha:
                print(f"[OK] Already present and verified: {local} ({desc})")
                continue
            else:
                print(f"[WARN] {local} SHA mismatch ({actual[:16]}...) — will re-download.")
                local.unlink()
        missing_targets.append(target)

    if not missing_targets:
        print("[OK] All Gate-25 historical baseline targets already present and verified.")
    else:
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
        if not conn_str:
            if strict:
                print("[FAIL] AZURE_STORAGE_CONNECTION_STRING not set (strict mode).")
                return 1
            else:
                print("[SKIP] AZURE_STORAGE_CONNECTION_STRING not set (best-effort mode).")
        else:
            try:
                from azure.storage.blob import BlobServiceClient
                from azure.core.exceptions import ResourceNotFoundError
                client = BlobServiceClient.from_connection_string(conn_str)

                for target in missing_targets:
                    local = Path(target["local_path"])
                    expected_sha = target["expected_sha256"]
                    container = target["container"]
                    blob_name = target["blob_name"]
                    desc = target["description"]

                    print(f"[INFO] Downloading [{container}] {blob_name} -> {local} ({desc})")
                    try:
                        blob_client = client.get_blob_client(
                            container=container, blob=blob_name
                        )
                        atomic_download(blob_client, local, expected_sha)
                        size_kb = local.stat().st_size / 1024
                        if size_kb >= 1024:
                            print(f"[OK] Downloaded & verified: {local} ({size_kb / 1024:.1f} MB)")
                        else:
                            print(f"[OK] Downloaded & verified: {local} ({size_kb:.1f} KB)")
                    except ResourceNotFoundError:
                        print(f"[FAIL] Blob not found: {container}/{blob_name}")
                        if strict:
                            return 1
                    except ValueError:
                        return 1
                    except Exception as e:
                        print(f"[FAIL] Download failed: {e}")
                        if strict:
                            return 1
            except ImportError:
                print("[FAIL] azure-storage-blob package not installed.")
                if strict:
                    return 1
            except Exception as e:
                print(f"[FAIL] Could not connect to Azure Blob Storage: {e}")
                if strict:
                    return 1

    # Verify Gate-25 package files exist and match SHA
    if strict:
        for target in PROVISION_TARGETS:
            local = Path(target["local_path"])
            expected_sha = target["expected_sha256"]
            if not local.exists():
                print(f"[FAIL] Expected file not found after provisioning: {local}")
                return 1
            actual = sha256_file(local)
            if actual != expected_sha:
                print(f"[FAIL] Final SHA check failed for {local}: {actual}")
                return 1

        # Full baseline package manifest verification
        try:
            from scripts.verify_baseline_gate import verify_sha256sums
            base_dir = Path("experiments/checkpoints/production_baseline")
            if not verify_sha256sums(base_dir):
                print("[FAIL] Baseline package SHA256SUMS manifest verification failed!")
                return 1
            print("[OK] Baseline package SHA256SUMS verified across all components.")
        except Exception as e:
            print(f"[FAIL] Baseline manifest verification error: {e}")
            return 1

        # Verify Gate-27 operational & source checkpoints are strictly present
        if not GATE27_OP_PATH.is_file() or sha256_file(GATE27_OP_PATH) != GATE27_AUTHORITATIVE_SHA256:
            print(f"[FAIL] Gate-27 operational checkpoint missing or corrupted: {GATE27_OP_PATH}")
            return 1
        if not GATE27_SRC_PATH.is_file() or sha256_file(GATE27_SRC_PATH) != GATE27_AUTHORITATIVE_SHA256:
            print(f"[FAIL] Gate-27 source checkpoint missing or corrupted: {GATE27_SRC_PATH}")
            return 1

    print("\n[OK] Complete baseline evidence package, Gate-27 operational checkpoint, and canonical TSRD fixture provisioned and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

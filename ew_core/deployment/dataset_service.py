"""TSRD Dataset service — abstracts local D:/TSRD, Google Cloud Storage, and Azure Blob.

In development: reads from local path set in TSRD_DATA_ROOT env var (default D:/TSRD).
In Google Cloud Run: reads from mounted storage or downloads on-demand from GCS_TSRD_BUCKET.
In Azure production: reads from AZURE_BLOB_ACCOUNT/AZURE_BLOB_CONTAINER or /mnt/tsrd mount.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional

from ..training.scenario_classifier import classify_scenario

logger = logging.getLogger(__name__)

TSRD_DATA_ROOT = os.environ.get("TSRD_DATA_ROOT", "D:/TSRD")
GCS_TSRD_BUCKET = os.environ.get("GCS_TSRD_BUCKET", os.environ.get("GCS_BUCKET", ""))
AZURE_BLOB_ACCOUNT = os.environ.get("AZURE_BLOB_ACCOUNT", "")
AZURE_BLOB_CONTAINER = os.environ.get("AZURE_BLOB_CONTAINER", "tsrd-dataset")

CLASS_DISPLAY_NAMES: dict[str, str] = {
    "fixed": "Fixed Frequency",
    "sparse": "Sparse Agile",
    "fast_agile": "Fast Agile Hopper",
    "slow_agile": "Slow Agile Hopper",
    "markov_hopper": "Markov Hopper",
    "periodic": "Periodic Hopper",
    "mixed": "Mixed Multi-Emitter",
    "dense": "Dense Pulse Environment",
}

CANONICAL_BENCHMARK_CONFIGS: list[dict[str, Any]] = [
    {
        "id": "config_119",
        "name": "Config 119",
        "display_name": "Config 119 — Sparse Agile",
        "class": "sparse",
        "display_class": "Sparse Agile",
        "description": "TSRD Sparse Agile Radar recording (Val Stare). Canonical primary benchmark.",
        "benchmark": True,
        "pulse_count": 50000,
        "source": "tsrd",
        "format": "h5",
    },
    {
        "id": "config_29",
        "name": "Config 29",
        "display_name": "Config 29 — Slow Agile Hopper",
        "class": "slow_agile",
        "display_class": "Slow Agile Hopper",
        "description": "TSRD Slow Agile Radar recording (Val Stare). Canonical benchmark scenario.",
        "benchmark": True,
        "pulse_count": 50000,
        "source": "tsrd",
        "format": "h5",
    },
    {
        "id": "config_241",
        "name": "Config 241",
        "display_name": "Config 241 — Fast Agile Hopper",
        "class": "fast_agile",
        "display_class": "Fast Agile Hopper",
        "description": "TSRD Fast Agile Radar recording (Val Stare). PRI <= 220 µs agility.",
        "benchmark": True,
        "pulse_count": 50000,
        "source": "tsrd",
        "format": "h5",
    },
    {
        "id": "config_195",
        "name": "Config 195",
        "display_name": "Config 195 — Dense Battlefield",
        "class": "dense",
        "display_class": "Dense Battlefield",
        "description": "TSRD Dense Radar recording (Val Stare). High pulse density stress test.",
        "benchmark": True,
        "pulse_count": 50000,
        "source": "tsrd",
        "format": "h5",
    },
    {
        "id": "config_64",
        "name": "Config 64",
        "display_name": "Config 64 — Dense Agile",
        "class": "dense",
        "display_class": "Dense Agile",
        "description": "TSRD Dense Agile Radar recording (Val Stare). Multi-emitter agile saturation.",
        "benchmark": True,
        "pulse_count": 50000,
        "source": "tsrd",
        "format": "h5",
    },
    {
        "id": "config_42",
        "name": "Config 42",
        "display_name": "Config 42 — Dense Radar",
        "class": "dense",
        "display_class": "Dense Radar",
        "description": "TSRD Dense Radar recording (Val Stare). High-density RF environment.",
        "benchmark": True,
        "pulse_count": 50000,
        "source": "tsrd",
        "format": "h5",
    },
    {
        "id": "config_96",
        "name": "Config 96",
        "display_name": "Config 96 — Mixed Multi-Emitter",
        "class": "mixed",
        "display_class": "Mixed Multi-Emitter",
        "description": "TSRD Mixed Heterogeneous Radar recording (Val Stare). Concurrent agile + fixed emitters.",
        "benchmark": True,
        "pulse_count": 50000,
        "source": "tsrd",
        "format": "h5",
    },
    {
        "id": "config_117",
        "name": "Config 117",
        "display_name": "Config 117 — Mixed Multi-Emitter",
        "class": "mixed",
        "display_class": "Mixed Multi-Emitter",
        "description": "TSRD Mixed Heterogeneous Radar recording (Val Stare). Concurrent agile + fixed emitters.",
        "benchmark": True,
        "pulse_count": 50000,
        "source": "tsrd",
        "format": "h5",
    },
    {
        "id": "config_143",
        "name": "Config 143",
        "display_name": "Config 143 — Sparse Agile",
        "class": "sparse",
        "display_class": "Sparse Agile",
        "description": "TSRD Sparse Agile Radar recording (Val Stare). Low pulse density tracking.",
        "benchmark": True,
        "pulse_count": 50000,
        "source": "tsrd",
        "format": "h5",
    },
    {
        "id": "config_194",
        "name": "Config 194",
        "display_name": "Config 194 — Sparse Agile",
        "class": "sparse",
        "display_class": "Sparse Agile",
        "description": "TSRD Sparse Agile Radar recording (Val Stare). Low pulse density tracking.",
        "benchmark": True,
        "pulse_count": 50000,
        "source": "tsrd",
        "format": "h5",
    },
]

_BENCHMARK_LOOKUP: dict[str, dict[str, Any]] = {b["id"]: b for b in CANONICAL_BENCHMARK_CONFIGS}


def get_tsrd_root() -> str:
    """Return the TSRD root path — local or mounted cloud storage."""
    # 1. In Azure AKS or container mount with volume at /mnt/tsrd
    azure_mount = Path("/mnt/tsrd")
    if azure_mount.exists():
        return str(azure_mount)

    # 2. Local development configured via environment variable
    local_env = os.environ.get("TSRD_DATA_ROOT", TSRD_DATA_ROOT)
    local = Path(local_env)
    if local.exists():
        return str(local)

    # 3. Check fallback repository local data path
    fallback_data = Path("data")
    if fallback_data.exists():
        return str(fallback_data)

    # 4. Check D:/TSRD default Windows drive
    default_d = Path("D:/TSRD")
    if default_d.exists():
        return str(default_d)

    raise RuntimeError(
        f"TSRD dataset not found. Set TSRD_DATA_ROOT env var (local), "
        f"place files in data/, or mount storage at /mnt/tsrd."
    )


def list_scenarios(subset: str = "val") -> List[str]:
    """List available scenario IDs in the TSRD dataset (raw IDs for compatibility)."""
    try:
        root = Path(get_tsrd_root())
    except RuntimeError:
        return []

    candidates = [
        root / "stare" / f"{subset}_stare",
        root / f"{subset}_stare",
        root / subset,
        root,
    ]
    for c in candidates:
        if c.exists():
            files = list(c.glob("config_*.h5")) + list(c.glob("config_*.json"))
            if files:
                return sorted(list({p.stem for p in files}))
    return []


def resolve_scenario_path(scenario_id: str) -> Optional[Path]:
    """Resolve scenario path for TSRD H5 dataset across local, mounted, and GCS storage.

    Args:
        scenario_id: Scenario ID string (e.g. 'config_119', 'config_119.h5', or full path).

    Returns:
        Path to existing .h5 or scenario file, or None if not found.
    """
    if not scenario_id:
        return None

    # Direct file check
    direct = Path(scenario_id)
    if direct.is_file():
        return direct

    # Clean scenario ID
    clean_id = scenario_id.replace(".h5", "").strip()

    # 1. Search TSRD root candidates
    try:
        root = Path(get_tsrd_root())
        candidates = [
            root / "stare" / "val_stare" / f"{clean_id}.h5",
            root / "val_stare" / f"{clean_id}.h5",
            root / f"{clean_id}.h5",
            root / "stare" / "train_stare" / f"{clean_id}.h5",
            root / "stare" / f"{clean_id}.h5",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
    except Exception:
        pass

    # 2. Search local data/ directory in repository
    repo_root = Path(__file__).resolve().parents[2]
    local_candidates = [
        repo_root / "data" / f"{clean_id}.h5",
        Path("data") / f"{clean_id}.h5",
        Path(f"D:/TSRD/stare/val_stare/{clean_id}.h5"),
    ]
    for cand in local_candidates:
        if cand.is_file():
            return cand

    # 3. Check Google Cloud Storage on-demand download if configured
    gcs_bucket = os.environ.get("GCS_TSRD_BUCKET", os.environ.get("GCS_BUCKET", GCS_TSRD_BUCKET))
    if gcs_bucket:
        try:
            target_local = Path("data") / f"{clean_id}.h5"
            if target_local.is_file():
                return target_local
            downloaded = download_from_gcs(
                f"gs://{gcs_bucket}/val_stare/{clean_id}.h5",
                target_dir="data",
            )
            if Path(downloaded).is_file():
                return Path(downloaded)
        except Exception as exc:
            logger.warning("GCS download attempt failed for %s: %s", clean_id, exc)

    return None


def list_tsrd_scenarios(subset: str = "val") -> List[dict[str, Any]]:
    """Return comprehensive scenario catalog for Mission Control board.

    Returns benchmark configurations prioritized at the top, followed by
    all other available TSRD configurations sorted numerically, followed by
    GNU Radio RF scenarios.
    """
    scenarios: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # 1. Process 10 Canonical Benchmark Configurations
    for bench in CANONICAL_BENCHMARK_CONFIGS:
        cid = bench["id"]
        resolved = resolve_scenario_path(cid)
        scenarios.append({
            **bench,
            "available": resolved is not None,
            "path": str(resolved) if resolved else None,
            "status": "ready" if resolved else "missing",
        })
        seen_ids.add(cid)

    # 2. Discover all other TSRD configs present locally
    raw_ids = list_scenarios(subset=subset)

    def _sort_key(s: str) -> tuple[int, str]:
        match = re.search(r"\d+", s)
        return (int(match.group()), s) if match else (999999, s)

    other_ids = sorted([cid for cid in raw_ids if cid not in seen_ids], key=_sort_key)

    for cid in other_ids:
        scenario_class = classify_scenario(cid)
        disp_class = CLASS_DISPLAY_NAMES.get(scenario_class, scenario_class.title().replace("_", " "))
        match = re.search(r"\d+", cid)
        cfg_num = match.group() if match else cid
        resolved = resolve_scenario_path(cid)

        scenarios.append({
            "id": cid,
            "name": f"Config {cfg_num}",
            "display_name": f"Config {cfg_num} — {disp_class}",
            "class": scenario_class,
            "display_class": disp_class,
            "description": f"TSRD {disp_class} Radar recording (Validation set).",
            "benchmark": False,
            "pulse_count": 50000,
            "source": "tsrd",
            "format": "h5",
            "available": resolved is not None,
            "path": str(resolved) if resolved else None,
            "status": "ready" if resolved else "missing",
        })
        seen_ids.add(cid)

    # 3. Add GNU Radio physical scenarios as secondary options
    gnu_scenarios = [
        {
            "id": "final_grc",
            "name": "final.grc",
            "display_name": "final.grc — GNU Radio 5-Emitter Agile FHSS",
            "class": "fast_agile",
            "display_class": "GNU Radio Agile FHSS",
            "description": "Synthesized 5-emitter agile frequency hopper via GNU Radio flowgraph.",
            "benchmark": False,
            "pulse_count": 4000,
            "source": "gnu_radio",
            "format": "grc_json",
            "available": True,
            "path": "final_grc",
            "status": "ready",
        },
        {
            "id": "saa_grc",
            "name": "saa.grc",
            "display_name": "saa.grc — GNU Radio Sample & Hold / Chirp",
            "class": "mixed",
            "display_class": "GNU Radio S&H Chirp",
            "description": "Synthesized sample-and-hold chirp emitter via GNU Radio flowgraph.",
            "benchmark": False,
            "pulse_count": 4000,
            "source": "gnu_radio",
            "format": "grc_json",
            "available": True,
            "path": "saa_grc",
            "status": "ready",
        },
    ]
    for g in gnu_scenarios:
        scenarios.append(g)

    return scenarios


def download_from_gcs(gcs_uri: str, target_dir: str = "data") -> str:
    """Download a scenario file or artifact from Google Cloud Storage.

    Accepts URIs of the form gs://bucket/path/to/file.h5 or bucket/path/to/file.h5.
    """
    target_p = Path(target_dir)
    target_p.mkdir(parents=True, exist_ok=True)

    clean_uri = gcs_uri.replace("gs://", "")
    parts = clean_uri.split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    filename = Path(blob_name).name if blob_name else "dataset.h5"
    dest_path = target_p / filename

    if dest_path.is_file():
        return str(dest_path)

    try:
        from google.cloud import storage  # type: ignore

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(dest_path))
        logger.info("Downloaded %s from Google Cloud Storage to %s", gcs_uri, dest_path)
        return str(dest_path)
    except ImportError:
        logger.warning("google-cloud-storage not installed; cannot download %s", gcs_uri)
    except Exception as exc:
        logger.error("Failed to download %s from Google Cloud Storage: %s", gcs_uri, exc)
        raise RuntimeError(f"Google Cloud Storage download failed: {exc}")

    return str(dest_path)


async def download_from_blob(blob_uri: str, target_dir: str = "experiments/checkpoints/downloaded") -> str:
    """Download a checkpoint/dataset file from Azure Blob Storage given az://container/blob or https URL."""
    target_p = Path(target_dir)
    target_p.mkdir(parents=True, exist_ok=True)

    clean_path = blob_uri.replace("az://", "").replace("https://", "")
    parts = clean_path.split("/", 1)
    blob_name = parts[1] if len(parts) > 1 else parts[0]
    filename = Path(blob_name).name
    dest_path = target_p / filename

    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    if conn_str:
        try:
            from azure.storage.blob import BlobServiceClient

            container = parts[0] if len(parts) > 1 else AZURE_BLOB_CONTAINER
            blob_service = BlobServiceClient.from_connection_string(conn_str)
            blob_client = blob_service.get_blob_client(container=container, blob=blob_name)
            data = blob_client.download_blob().readall()
            with open(dest_path, "wb") as f:
                f.write(data)
            logger.info("Downloaded %s from Azure Blob to %s", blob_uri, dest_path)
            return str(dest_path)
        except Exception as exc:
            logger.error("Failed to download from Azure Blob %s: %s", blob_uri, exc)
            raise RuntimeError(f"Azure Blob download failed: {exc}")

    # Fallback if local file exists matching target
    if dest_path.exists():
        return str(dest_path)
    raise RuntimeError(f"Cannot download {blob_uri}: AZURE_STORAGE_CONNECTION_STRING not configured")


GATE27_AUTHORITATIVE_SHA256: str = "fac0577454fe0a89687c27ebdffa568229e2d03435eebd9e82b50fca14292094"


def ensure_operational_checkpoint() -> Optional[Path]:
    """Ensure Gate-27 operational checkpoint exists and passes SHA-256 integrity check.

    In local dev: uses existing experiments/checkpoints/scheduler_v2_operational_candidate/.
    In Cloud Run: downloads from gs://${GCS_TSRD_BUCKET}/checkpoints/ if not present.
    Fails closed if the checkpoint cannot be verified against the authoritative SHA-256.
    """
    import hashlib

    repo_root = Path(__file__).resolve().parents[2]
    cand_dir = repo_root / "experiments" / "checkpoints" / "scheduler_v2_operational_candidate"
    cand_file = cand_dir / "checkpoint_gate_27000_operational.pt"

    if cand_file.is_file():
        hasher = hashlib.sha256()
        with open(cand_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        if hasher.hexdigest() == GATE27_AUTHORITATIVE_SHA256:
            return cand_file
        logger.error(
            "Local Gate-27 checkpoint SHA-256 mismatch: %s != %s",
            hasher.hexdigest(),
            GATE27_AUTHORITATIVE_SHA256,
        )
        raise RuntimeError(
            f"Fail-closed: Gate-27 operational checkpoint tampered or corrupt: "
            f"{hasher.hexdigest()} != {GATE27_AUTHORITATIVE_SHA256}"
        )

    # Attempt on-demand download from Google Cloud Storage
    gcs_bucket = os.environ.get("GCS_TSRD_BUCKET", os.environ.get("GCS_BUCKET", GCS_TSRD_BUCKET))
    if gcs_bucket:
        try:
            cand_dir.mkdir(parents=True, exist_ok=True)
            download_from_gcs(
                f"gs://{gcs_bucket}/checkpoints/checkpoint_gate_27000_operational.pt",
                target_dir=str(cand_dir),
            )
            if cand_file.is_file():
                hasher = hashlib.sha256()
                with open(cand_file, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                if hasher.hexdigest() == GATE27_AUTHORITATIVE_SHA256:
                    logger.info("Successfully downloaded and verified Gate-27 checkpoint from GCS.")
                    return cand_file
                raise RuntimeError(
                    f"GCS Gate-27 checkpoint SHA-256 mismatch: "
                    f"{hasher.hexdigest()} != {GATE27_AUTHORITATIVE_SHA256}"
                )
        except Exception as exc:
            logger.error("Failed to download or verify Gate-27 checkpoint from GCS: %s", exc)
            raise RuntimeError(
                f"Fail-closed: Gate-27 operational checkpoint could not be verified from GCS: {exc}"
            )

    return None


DEINTERLEAVER_AUTHORITATIVE_SHA256: str = "b7cc3727b3b1940ac8c06e61f127644c484de69ec16080110dd4ea8c0c44b116"


def ensure_deinterleaver_checkpoint() -> Optional[Path]:
    """Ensure deinterleaver checkpoint exists and passes SHA-256 integrity check.

    In local dev: uses existing experiments/checkpoints/deinterleaver/best.pt.
    In Cloud Run: downloads from gs://${GCS_TSRD_BUCKET}/checkpoints/ if not present.
    """
    import hashlib

    repo_root = Path(__file__).resolve().parents[2]
    cand_dir = repo_root / "experiments" / "checkpoints" / "deinterleaver"
    cand_file = cand_dir / "best.pt"

    if cand_file.is_file():
        hasher = hashlib.sha256()
        with open(cand_file, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        if hasher.hexdigest().lower() == DEINTERLEAVER_AUTHORITATIVE_SHA256.lower():
            return cand_file
        logger.warning(
            "Deinterleaver checkpoint SHA-256 mismatch: %s != %s, attempting redownload",
            hasher.hexdigest(),
            DEINTERLEAVER_AUTHORITATIVE_SHA256,
        )

    # Attempt on-demand download from Google Cloud Storage
    gcs_bucket = os.environ.get("GCS_TSRD_BUCKET", os.environ.get("GCS_BUCKET", GCS_TSRD_BUCKET))
    if gcs_bucket:
        try:
            cand_dir.mkdir(parents=True, exist_ok=True)
            for gcs_obj in [
                f"gs://{gcs_bucket}/checkpoints/best.pt",
                f"gs://{gcs_bucket}/checkpoints/deinterleaver/best.pt",
            ]:
                try:
                    download_from_gcs(gcs_obj, target_dir=str(cand_dir))
                    if cand_file.is_file():
                        break
                except Exception:
                    continue

            if cand_file.is_file():
                hasher = hashlib.sha256()
                with open(cand_file, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                if hasher.hexdigest().lower() == DEINTERLEAVER_AUTHORITATIVE_SHA256.lower():
                    logger.info("Successfully downloaded and verified deinterleaver checkpoint from GCS.")
                    return cand_file
                logger.error(
                    "GCS Deinterleaver checkpoint SHA-256 mismatch: %s != %s",
                    hasher.hexdigest(),
                    DEINTERLEAVER_AUTHORITATIVE_SHA256,
                )
        except Exception as exc:
            logger.error("Failed to download or verify deinterleaver checkpoint from GCS: %s", exc)

    return None


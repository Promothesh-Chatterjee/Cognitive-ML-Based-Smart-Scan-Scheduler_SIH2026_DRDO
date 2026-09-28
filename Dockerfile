# =============================================================================
# Stage 1: Build & Dependencies
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /install

# Install build dependencies required for C-extensions (hdbscan, pyzmq, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv to hold compiled wheels & installed packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pyproject.toml first to cache dependency installations
COPY pyproject.toml .
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install .

# =============================================================================
# Stage 2: Minimal Production Runtime
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DEVICE=cpu \
    PATH="/opt/venv/bin:$PATH" \
    TSRD_DATA_ROOT="/mnt/tsrd"

WORKDIR /app

# Install minimal runtime system packages (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application codebase and configuration
COPY ew_core/ ew_core/
COPY configs/ configs/
COPY scripts/ scripts/
COPY experiments/ experiments/
COPY reports/ reports/
COPY pyproject.toml .

# Install application package in develop mode (without reinstalling deps)
RUN pip install --no-deps -e .

# Create directory for TSRD dataset storage mount / download cache
RUN mkdir -p /mnt/tsrd

# Create a non-root system user for secure container execution
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app /mnt/tsrd
USER appuser

EXPOSE 8000

# Health check using curl against /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch FastAPI using Uvicorn
CMD ["sh", "-c", "uvicorn ew_core.deployment.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

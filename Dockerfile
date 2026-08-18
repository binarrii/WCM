# Stage 1: Build the Vue dashboard
FROM node:22-slim AS web-builder

WORKDIR /webui
RUN corepack enable
COPY webui/package.json webui/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY webui/ ./
RUN pnpm build


# Stage 2: Build Python dependencies
FROM python:3.12-slim AS builder


WORKDIR /app

# Install uv
RUN pip install uv --no-cache-dir

# Copy only dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies to local directory
RUN uv sync --frozen --no-install-project --no-dev

# Clean venv in builder (before COPY to reduce stage-2 size)
RUN find /app/.venv/lib/python3.12/site-packages/ -maxdepth 1 -type d -name "*test*" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv/lib/python3.12/site-packages/ -maxdepth 1 -type d -name "*tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv/lib/python3.12/site-packages/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv/lib/python3.12/site-packages/ -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /app/.venv/lib/python3.12/site-packages/ -type f -name "*.pyo" -delete 2>/dev/null || true && \
    rm -rf /app/.venv/lib/python3.12/site-packages/clang 2>/dev/null || true && \
    rm -rf /app/.venv/lib/python3.12/site-packages/opencv_python.libs 2>/dev/null || true

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/


# Stage 3: Minimal runtime image
FROM python:3.12-slim


WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy cleaned virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/api ./api
COPY --from=builder /app/scripts ./scripts
COPY --from=web-builder /webui/dist /www

# Use virtual environment python
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv
ENV PYTHONPATH="/app/src"
# InsightFace Server runs separately. The default can be overridden by
# compose or the container runtime.
ENV WCM_INSIGHTFACE_BASE_URL="http://10.252.25.251:18097"

# Create the non-root user and seed the named-volume mount point with writable
# ownership (Docker preserves it when initializing a new volume).
RUN useradd -m -u 1000 appuser \
    && mkdir -p /tmp/wcm \
    && chown -R appuser:appuser /app /tmp/wcm
USER appuser

# Default environment variables (override with -e at runtime)
ENV WCM_API_HOST=0.0.0.0
ENV WCM_API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health 2>/dev/null || exit 1

# Dynamic worker count: GUNICORN_WORKERS env var (default: CPU cores * 2 + 1, min 4)
# Override with: docker run -e GUNICORN_WORKERS=8
# CMD ["sh", "-c", "cpus=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 2); workers=${GUNICORN_WORKERS:-$((cpus * 2 + 1))}; workers=$((workers < 4 ? 4 : workers)); echo \"Starting gunicorn with $workers workers\"; gunicorn -w $workers -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 -t 60 api.main:app"]
CMD ["sh", "-c", "cpus=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 2); workers=${GUNICORN_WORKERS:-$((cpus - 1))}; workers=$((workers < 4 ? 4 : workers)); echo \"Starting gunicorn with $workers workers\"; gunicorn -w $workers -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 -t 60 api.main:app"]

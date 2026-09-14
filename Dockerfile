# Stage 2: Build Python dependencies
FROM python:3.12-slim AS builder


WORKDIR /app

# Copy the pinned standalone uv binary instead of downloading its large wheel
# through pip during every clean build.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

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
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy cleaned virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/api ./api
COPY --from=builder /app/scripts ./scripts

# Use virtual environment python
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv
ENV PYTHONPATH="/app/src"

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

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

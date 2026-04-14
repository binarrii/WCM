# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy

WORKDIR /app

# Install uv
RUN pip install uv --no-cache-dir

# Copy only dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies to local directory
RUN uv sync --frozen --no-install-project

# Force opencv-python-headless and tensorflow-cpu
# Use uv pip for venv package management
RUN uv pip uninstall --python /app/.venv/bin/python opencv-python opencv-python-headless 2>/dev/null || true
RUN uv pip install --python /app/.venv/bin/python --no-cache opencv-python-headless
RUN uv pip uninstall --python /app/.venv/bin/python tensorflow 2>/dev/null || true
RUN uv pip install --python /app/.venv/bin/python --no-cache tensorflow-cpu

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


# Stage 2: Minimal runtime image
FROM python:3.12-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy
ARG https_proxy

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy

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

# Use virtual environment python
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV WCM_DB_HOST=db
ENV WCM_DB_PORT=5432
ENV WCM_DB_NAME=facerec
ENV WCM_DB_USER=postgres
ENV WCM_DB_PASSWORD=postgres
ENV WCM_API_HOST=0.0.0.0
ENV WCM_API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and lock file first for better caching
COPY pyproject.toml uv.lock ./

# Install uv
RUN pip install uv

# Install Python dependencies
RUN uv sync --frozen --no-install-project

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/

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

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

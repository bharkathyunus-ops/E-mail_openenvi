# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement: UID 1000)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package so entry-points are registered
RUN pip install --no-cache-dir -e .

# Make sure all modules are importable from /app
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Expose port required by HF Spaces
EXPOSE 7860

# Health check — openenv validator pings /health
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
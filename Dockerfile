# Base Image: Official lightweight Python 3.11 on Debian Slim
FROM python:3.11-slim

# System Environment Configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install minimal OS dependencies & curl for Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set application working directory
WORKDIR /app

# Dependency Caching Layer: copy and install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application subsystems and model artifacts
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY data/raw/ ./data/raw/
COPY data/docs/ ./data/docs/
COPY run_pipeline.py app.py entrypoint.sh ./

# Ensure entrypoint script is executable
RUN chmod +x /app/entrypoint.sh

# Expose default Streamlit web interface port
EXPOSE 8501

# Container Healthcheck querying Streamlit's native health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Container startup entrypoint: self-heals database if missing, then launches dashboard
ENTRYPOINT ["/app/entrypoint.sh"]

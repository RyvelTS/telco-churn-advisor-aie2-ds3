#!/bin/bash
set -e

echo "============================================================"
echo " STARTING TELECOM CHURN ADVISOR CONTAINER"
echo "============================================================"

# Self-healing database check: if telecom.db is missing, run the ingestion pipeline
if [ ! -f "/app/data/telecom.db" ]; then
    echo "[Entrypoint] SQLite database '/app/data/telecom.db' not found."
    echo "[Entrypoint] Auto-running data pipeline from raw source..."
    python run_pipeline.py
    echo "[Entrypoint] Database successfully initialized."
else
    echo "[Entrypoint] Existing database detected at '/app/data/telecom.db'."
fi

echo "[Entrypoint] Launching Streamlit Retention Console on port 8501..."
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false

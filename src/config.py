import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent

# ==========================================
# Data & File System Paths
# ==========================================
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DOCS_DIR = DATA_DIR / "docs"
DB_PATH = DATA_DIR / "telecom.db"

# Model Artifacts
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "telco_churn_model_bundle.joblib"

# ==========================================
# Azure AI Foundry & Model Deployments
# ==========================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"
)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)

# ==========================================
# Azure AI Search (RAG Configuration)
# ==========================================
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")
AZURE_SEARCH_INDEX_NAME = os.getenv(
    "AZURE_SEARCH_INDEX_NAME", "telecom-retention-index"
)

# ==========================================
# Validation Helper
# ==========================================
def validate_azure_credentials() -> dict:
    """Checks if critical Azure credentials are provided."""
    missing = []
    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not AZURE_SEARCH_ENDPOINT:
        missing.append("AZURE_SEARCH_ENDPOINT")
    if not AZURE_SEARCH_KEY:
        missing.append("AZURE_SEARCH_KEY")

    return {
        "valid": len(missing) == 0,
        "missing_keys": missing
    }

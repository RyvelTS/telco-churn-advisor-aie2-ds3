import pytest
import pandas as pd
from pathlib import Path
from src.data.database import CustomerRepository
from src.models.predictor import ChurnPredictor, PredictionResult


def test_bundle_integrity():
    """Test Case 1: Assert that instantiating ChurnPredictor loads all expected bundle attributes."""
    predictor = ChurnPredictor()

    assert predictor.model is not None
    assert hasattr(predictor.model, "predict_proba")
    assert isinstance(predictor.feature_cols, list)
    assert len(predictor.feature_cols) > 0
    assert predictor.t_star == 0.17
    assert predictor.t_default == 0.50
    assert "bins" in predictor.risk_tiers
    assert "labels" in predictor.risk_tiers
    assert "C_FN" in predictor.cost_matrix
    assert "C_FP" in predictor.cost_matrix


def test_missing_bundle_raises_file_not_found():
    """Ensure passing a non-existent bundle path raises a clean FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ChurnPredictor(bundle_path="artifacts/non_existent_bundle.joblib")


def test_schema_robustness_with_sqlite_record():
    """Test Case 2: Fetch customer record from SQLite with metadata and test predict_customer()."""
    repo = CustomerRepository()
    customer_ids = repo.get_all_ids()

    assert len(customer_ids) > 0, "Database must have customer records from pipeline."
    sample_id = customer_ids[0]
    customer_record = repo.get_customer_by_id(sample_id)

    assert customer_record is not None
    # Verify metadata fields are present in raw DB record
    assert "customerID" in customer_record
    assert "full_name" in customer_record
    assert "email" in customer_record
    assert "phone" in customer_record

    predictor = ChurnPredictor()
    result = predictor.predict_customer(customer_record)

    # Output contract validation
    assert isinstance(result, dict)
    assert result["customer_id"] == sample_id
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert "%" in result["churn_probability_pct"]
    assert isinstance(result["is_high_risk"], bool)
    assert result["risk_tier"] in [
        "Low Risk", "Moderate Risk", "High Risk", "Critical Risk"
    ]
    assert result["threshold_applied"] == 0.17


def test_threshold_and_risk_tier_correctness():
    """Test Case 3: Verify risk tiers and cost-optimal decision threshold (t* = 0.17)."""
    predictor = ChurnPredictor()

    # Tier boundaries:
    # 0.00 <= P < 0.50: Low Risk
    # 0.50 <= P < 0.80: Moderate Risk
    # 0.80 <= P < 0.95: High Risk
    # 0.95 <= P <= 1.00: Critical Risk
    assert predictor.assign_risk_tier(0.10) == "Low Risk"
    assert predictor.assign_risk_tier(0.20) == "Low Risk"
    assert predictor.assign_risk_tier(0.50) == "Moderate Risk"
    assert predictor.assign_risk_tier(0.60) == "Moderate Risk"
    assert predictor.assign_risk_tier(0.80) == "High Risk"
    assert predictor.assign_risk_tier(0.90) == "High Risk"
    assert predictor.assign_risk_tier(0.95) == "Critical Risk"
    assert predictor.assign_risk_tier(0.98) == "Critical Risk"

    # Evaluation against t* = 0.17
    # 0.10: < 0.17 -> is_high_risk = False, Low Risk
    p_low = 0.10
    is_high_risk_low = p_low >= predictor.t_star
    assert is_high_risk_low is False
    assert predictor.assign_risk_tier(p_low) == "Low Risk"

    # 0.20: >= 0.17 -> is_high_risk = True, but still Low Risk tier!
    p_edge = 0.20
    is_high_risk_edge = p_edge >= predictor.t_star
    assert is_high_risk_edge is True
    assert predictor.assign_risk_tier(p_edge) == "Low Risk"

    # 0.60: >= 0.17 -> is_high_risk = True, Moderate Risk
    p_mod = 0.60
    assert (p_mod >= predictor.t_star) is True
    assert predictor.assign_risk_tier(p_mod) == "Moderate Risk"


def test_batch_prediction():
    """Test predict_batch on multiple customer records."""
    repo = CustomerRepository()
    all_df = repo.get_all_customers().head(10)

    predictor = ChurnPredictor()
    scored_df = predictor.predict_batch(all_df)

    assert "Predicted_Churn_Prob" in scored_df.columns
    assert "Churn_Probability_Pct" in scored_df.columns
    assert "Is_High_Risk" in scored_df.columns
    assert "Risk_Tier" in scored_df.columns
    assert len(scored_df) == 10
    assert scored_df["Predicted_Churn_Prob"].dtype == float

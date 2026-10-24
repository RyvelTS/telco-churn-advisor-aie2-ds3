import numpy as np
import pandas as pd
import pytest
from src.data.cleaner import (
    AllowedValueCleaner,
    BaseCleaner,
    DataTypeConverter,
    Deduplicator,
    MissingValueColumnDropper,
    MissingValueImputer,
    OutlierHandler,
    StringNormalizer,
)
from src.data.database import CustomerRepository
from src.data.pipeline import enrich_customer_profiles
from src.data.strategy import CleaningPipeline


def test_base_cleaner_abstract():
    """Ensure BaseCleaner cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseCleaner()


def test_string_normalizer():
    """Test whitespace stripping and customerID uppercase conversion."""
    df = pd.DataFrame({
        "customerID": ["  7590-vhveg  ", "5575-gnvde "],
        "gender": [" Female ", "Male   "],
    })
    cleaner = StringNormalizer()
    out = cleaner.clean(df)

    assert out["customerID"].iloc[0] == "7590-VHVEG"
    assert out["customerID"].iloc[1] == "5575-GNVDE"
    assert out["gender"].iloc[0] == "Female"
    assert out["gender"].iloc[1] == "Male"
    # Ensure immutability of input df
    assert df["customerID"].iloc[0] == "  7590-vhveg  "


def test_deduplicator():
    """Test duplicate row removal keeping first instance."""
    df = pd.DataFrame({
        "customerID": ["C1", "C2", "C1", "C3"],
        "val": [10, 20, 30, 40],
    })
    cleaner = Deduplicator(subset=["customerID"], keep="first")
    out = cleaner.clean(df)

    assert len(out) == 3
    assert list(out["customerID"]) == ["C1", "C2", "C3"]
    assert out["val"].iloc[0] == 10  # kept first
    assert list(out.index) == [0, 1, 2]


def test_data_type_converter():
    """Test financial coercion, boolean mapping, and SeniorCitizen conversion."""
    df = pd.DataFrame({
        "TotalCharges": [" 29.85 ", " ", "1889.5", "invalid"],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "Yes", "No", "No"],
        "PhoneService": ["No", "Yes", "Yes", "Yes"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "Churn": ["No", "Yes", "No", "Yes"],
        "SeniorCitizen": [0, 1, 0, 1],
    })
    cleaner = DataTypeConverter()
    out = cleaner.clean(df)

    assert out["TotalCharges"].iloc[0] == 29.85
    assert np.isnan(out["TotalCharges"].iloc[1])
    assert out["TotalCharges"].iloc[2] == 1889.5
    assert np.isnan(out["TotalCharges"].iloc[3])

    assert out["Partner"].dtype == bool
    assert out["Partner"].tolist() == [True, False, True, False]
    assert out["Churn"].tolist() == [False, True, False, True]
    assert out["SeniorCitizen"].dtype == bool
    assert out["SeniorCitizen"].tolist() == [False, True, False, True]


def test_allowed_value_cleaner():
    """Test replacing unapproved category values with NaN."""
    df = pd.DataFrame({
        "gender": ["Female", "Alien", "Male"],
        "InternetService": ["DSL", "Fiber optic", "5G Satellite"],
        "Contract": ["Month-to-month", "One year", "Infinite"],
    })
    cleaner = AllowedValueCleaner()
    out = cleaner.clean(df)

    assert out["gender"].iloc[0] == "Female"
    assert pd.isna(out["gender"].iloc[1])
    assert out["gender"].iloc[2] == "Male"

    assert pd.isna(out["InternetService"].iloc[2])
    assert pd.isna(out["Contract"].iloc[2])


def test_missing_val_column_dropper():
    """Test dropping columns with missing ratio > 50%."""
    df = pd.DataFrame({
        "good_col": [1, 2, 3, 4],
        "borderline_col": [1, 2, np.nan, np.nan],  # 50% missing -> keep
        "bad_col": [1, np.nan, np.nan, np.nan],     # 75% missing -> drop
    })
    cleaner = MissingValueColumnDropper(threshold=0.5)
    out = cleaner.clean(df)

    assert "good_col" in out.columns
    assert "borderline_col" in out.columns
    assert "bad_col" not in out.columns


def test_missing_value_imputer():
    """Test TotalCharges=0.0, and median imputation for MonthlyCharges & tenure."""
    df = pd.DataFrame({
        "TotalCharges": [100.0, np.nan, 300.0],
        "MonthlyCharges": [50.0, np.nan, 70.0],   # median = 60.0
        "tenure": [12.0, np.nan, 24.0],           # median = 18
    })
    cleaner = MissingValueImputer()
    out = cleaner.clean(df)

    assert out["TotalCharges"].iloc[1] == 0.0
    assert out["MonthlyCharges"].iloc[1] == 60.0
    assert out["tenure"].iloc[1] == 18
    assert pd.api.types.is_integer_dtype(out["tenure"])


def test_outlier_handler():
    """Test capping values using 1.5 * IQR fences."""
    df = pd.DataFrame({
        "tenure": [10.0, 12.0, 14.0, 16.0, 18.0, 100.0],
        "MonthlyCharges": [20.0, 30.0, 40.0, 50.0, 60.0, 999.0],
        "TotalCharges": [100.0, 200.0, 300.0, 400.0, 500.0, 50000.0],
    })
    cleaner = OutlierHandler()
    out = cleaner.clean(df)

    assert out["tenure"].max() < 100.0
    assert out["MonthlyCharges"].max() < 999.0
    assert out["TotalCharges"].max() < 50000.0


def test_cleaning_pipeline_integration():
    """Test running all cleaners together via CleaningPipeline."""
    raw_sample = pd.DataFrame({
        "customerID": ["  C001  ", "c001", "C002", "C003"],
        "gender": ["Female", "Female", "UnknownGender", "Male"],
        "SeniorCitizen": [0, 0, 1, 0],
        "Partner": ["Yes", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "No", "No"],
        "tenure": [1, 1, np.nan, 24],
        "PhoneService": ["Yes", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No", "No", "Yes", "No"],
        "InternetService": ["DSL", "DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "No", "Yes", "No"],
        "OnlineBackup": ["No", "No", "Yes", "No"],
        "DeviceProtection": ["No", "No", "No", "No"],
        "TechSupport": ["No", "No", "No", "No"],
        "StreamingTV": ["No", "No", "Yes", "No"],
        "StreamingMovies": ["No", "No", "Yes", "No"],
        "Contract": ["Month-to-month", "Month-to-month", "Two year", "One year"],
        "PaperlessBilling": ["Yes", "Yes", "No", "Yes"],
        "PaymentMethod": ["Mailed check", "Mailed check", "Electronic check", "Mailed check"],
        "MonthlyCharges": [29.85, 29.85, 80.0, 50.0],
        "TotalCharges": [" 29.85 ", "29.85", " ", "1200.0"],
        "Churn": ["No", "No", "Yes", "No"],
        "mostly_empty": [np.nan, np.nan, np.nan, "text"],  # 2/3 (66.7%) empty after deduplication -> dropped
    })

    pipeline = CleaningPipeline([
        StringNormalizer(),
        Deduplicator(),
        DataTypeConverter(),
        AllowedValueCleaner(),
        MissingValueColumnDropper(threshold=0.5),
        MissingValueImputer(),
        OutlierHandler(),
    ])

    cleaned = pipeline.run(raw_sample)

    assert len(cleaned) == 3  # Deduplicated C001
    assert "mostly_empty" not in cleaned.columns
    assert cleaned["TotalCharges"].iloc[1] == 0.0  # Empty space was coerced and imputed to 0.0
    assert pd.isna(cleaned["gender"].iloc[1])  # UnknownGender replaced by NaN


def test_customer_repository(tmp_path):
    """Test SQLite database operations and index creation."""
    db_file = tmp_path / "test_telecom.db"
    repo = CustomerRepository(db_path=db_file)

    df = pd.DataFrame({
        "customerID": ["TEST-1", "TEST-2"],
        "full_name": ["Alice Smith", "Bob Jones"],
        "email": ["alice@example.com", "bob@example.com"],
        "phone": ["+1-555-0100", "+1-555-0101"],
        "Contract": ["Month-to-month", "Two year"],
        "Churn": [True, False],
        "MonthlyCharges": [75.5, 45.0],
    })

    saved = repo.save_customers(df)
    assert saved == 2

    # Query customer
    c1 = repo.get_customer_by_id("test-1")
    assert c1 is not None
    assert c1["full_name"] == "Alice Smith"

    # Query all IDs
    ids = repo.get_all_ids()
    assert set(ids) == {"TEST-1", "TEST-2"}

    # Query high risk
    high_risk = repo.get_high_risk_sample(limit=5)
    assert len(high_risk) == 1
    assert high_risk[0]["customerID"] == "TEST-1"


def test_enrich_customer_profiles():
    """Test mock CRM profile enrichment."""
    df = pd.DataFrame({"customerID": ["C1", "C2", "C3"]})
    enriched = enrich_customer_profiles(df, seed=42)

    assert "full_name" in enriched.columns
    assert "email" in enriched.columns
    assert "phone" in enriched.columns
    assert len(enriched) == 3
    assert all(isinstance(name, str) and len(name) > 0 for name in enriched["full_name"])

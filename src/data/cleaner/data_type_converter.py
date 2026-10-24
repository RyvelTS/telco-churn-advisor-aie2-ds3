from typing import List, Optional
import pandas as pd
from .base import BaseCleaner


class DataTypeConverter(BaseCleaner):
    """Converts financial columns to numeric floats and binary flags to native booleans."""

    DEFAULT_BOOL_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

    def __init__(self, bool_cols: Optional[List[str]] = None):
        self.bool_cols = bool_cols or self.DEFAULT_BOOL_COLS

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Financial Coercion
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Binary Boolean Mapping
        present_bool_cols = [col for col in self.bool_cols if col in df.columns]
        for col in present_bool_cols:
            df[col] = df[col].replace({"Yes": True, "No": False}).astype(bool)

        # Standardize binary indicator to native boolean
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool)

        return df

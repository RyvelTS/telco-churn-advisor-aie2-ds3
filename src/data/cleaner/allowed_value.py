from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from .base import BaseCleaner


class AllowedValueCleaner(BaseCleaner):
    """Enforces strict categorical whitelist validation, setting unexpected values to NaN."""

    DEFAULT_ALLOWED: Dict[str, List[str]] = {
        "gender": ["Female", "Male"],
        "MultipleLines": ["No phone service", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        **{
            c: ["No", "Yes", "No internet service"]
            for c in [
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
        },
    }

    def __init__(self, allowed: Optional[Dict[str, List[str]]] = None):
        self.allowed = allowed or self.DEFAULT_ALLOWED

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, vals in self.allowed.items():
            if col in df.columns:
                df.loc[~df[col].isin(vals), col] = np.nan
        return df

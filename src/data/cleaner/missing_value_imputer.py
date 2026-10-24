from typing import Dict, Optional
import pandas as pd
from .base import BaseCleaner


class MissingValueImputer(BaseCleaner):
    """Imputes missing numerical values based on business logic and dataset medians:

    - TotalCharges: 0.0 (or precomputed median from initialization)
    - MonthlyCharges: dataset median
    - tenure: dataset median (cast to integer)
    """

    def __init__(self, medians: Optional[Dict[str, float]] = None):
        """Initializes imputer with optional precomputed feature medians.

        Args:
            medians (Optional[Dict[str, float]]): Mapping of column names to precomputed replacement values.
                If provided, explicitly overrides runtime column medians.
        """
        self.medians = medians

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # TotalCharges: Fill with 0.0 (or precomputed median)
        if "TotalCharges" in df.columns:
            total_charges_val = (
                self.medians.get("TotalCharges", 0.0)
                if self.medians and "TotalCharges" in self.medians
                else 0.0
            )
            df["TotalCharges"] = df["TotalCharges"].fillna(total_charges_val)

        # MonthlyCharges: Fill with median
        if "MonthlyCharges" in df.columns:
            if self.medians and "MonthlyCharges" in self.medians:
                monthly_charges_val = float(self.medians["MonthlyCharges"])
            else:
                monthly_charges_val = float(df["MonthlyCharges"].median())
            df["MonthlyCharges"] = df["MonthlyCharges"].fillna(monthly_charges_val)

        # tenure: Fill with integer median
        if "tenure" in df.columns:
            if self.medians and "tenure" in self.medians:
                tenure_val = int(self.medians["tenure"])
            else:
                tenure_val = int(df["tenure"].median())
            df["tenure"] = df["tenure"].fillna(tenure_val).astype(int)

        return df

from typing import Dict, List, Optional, Tuple
import pandas as pd
from .base import BaseCleaner


class OutlierHandler(BaseCleaner):
    """Detects and clips numerical outliers using the 1.5 * IQR fence rule on specified columns."""

    DEFAULT_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

    def __init__(
        self,
        target_cols: Optional[List[str]] = None,
        iqr_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.target_cols = target_cols or self.DEFAULT_COLS
        self.iqr_bounds = iqr_bounds

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        bounds = self.iqr_bounds or {}

        for col in self.target_cols:
            if col in df.columns:
                if col in bounds:
                    low, high = bounds[col]
                else:
                    q1 = float(df[col].quantile(0.25))
                    q3 = float(df[col].quantile(0.75))
                    iqr = q3 - q1
                    low = q1 - 1.5 * iqr
                    high = q3 + 1.5 * iqr

                df[col] = df[col].clip(lower=low, upper=high)

        return df

import pandas as pd
from .base import BaseCleaner


class MissingValueColumnDropper(BaseCleaner):
    """Drops columns whose missing value ratio strictly exceeds a configured threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Retain features whose missingness proportion does not exceed the allowed threshold
        valid_cols = df.columns[df.isna().mean() <= self.threshold]
        return df.loc[:, valid_cols]

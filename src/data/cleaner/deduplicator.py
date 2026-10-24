from typing import List, Optional
import pandas as pd
from .base import BaseCleaner


class Deduplicator(BaseCleaner):
    """Deduplicates records based on a specified subset of columns, retaining the first occurrence."""

    def __init__(self, subset: Optional[List[str]] = None, keep: str = "first"):
        self.subset = subset or ["customerID"]
        self.keep = keep

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Filter subset to only existing columns in df
        valid_subset = [col for col in self.subset if col in df.columns]
        if valid_subset:
            df = df.drop_duplicates(subset=valid_subset, keep=self.keep)
            df = df.reset_index(drop=True)
        return df

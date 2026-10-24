from typing import List, Optional
import pandas as pd
from .cleaner.base import BaseCleaner


class CleaningPipeline:
    """Strategy pattern executor for chaining data cleaning transformers."""

    def __init__(self, cleaners: Optional[List[BaseCleaner]] = None):
        self.cleaners: List[BaseCleaner] = cleaners or []

    def add_step(self, cleaner: BaseCleaner) -> "CleaningPipeline":
        """Add a cleaning step to the pipeline."""
        self.cleaners.append(cleaner)
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes each cleaner step sequentially on the input DataFrame.

        Args:
            df (pd.DataFrame): Raw or partially processed DataFrame.

        Returns:
            pd.DataFrame: Cleaned DataFrame after applying all transformations.
        """
        current_df = df.copy()
        for cleaner in self.cleaners:
            current_df = cleaner.clean(current_df)
        return current_df

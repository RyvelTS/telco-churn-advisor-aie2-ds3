from abc import ABC, abstractmethod
import pandas as pd


class BaseCleaner(ABC):
    """Abstract base contract for data cleaning transformations."""

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Receive a DataFrame and return a transformed copy without mutating the input.

        Args:
            df (pd.DataFrame): Input DataFrame to be cleaned.

        Returns:
            pd.DataFrame: Transformed copy of the DataFrame.
        """
        pass

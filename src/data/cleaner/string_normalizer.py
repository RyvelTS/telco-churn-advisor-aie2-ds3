import pandas as pd
from .base import BaseCleaner


class StringNormalizer(BaseCleaner):
    """Strips leading/trailing whitespace across all string columns

    and standardizes customerID to uppercase.
    """

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        obj_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        if "customerID" in df.columns:
            df["customerID"] = df["customerID"].apply(
                lambda x: x.upper() if isinstance(x, str) else x
            )

        return df

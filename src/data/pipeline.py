from pathlib import Path
from typing import Optional, Union
import pandas as pd
from src.config import DB_PATH, RAW_DATA_PATH
from .cleaner import (
    AllowedValueCleaner,
    DataTypeConverter,
    Deduplicator,
    MissingValueColumnDropper,
    MissingValueImputer,
    OutlierHandler,
    StringNormalizer,
)
from .database import CustomerRepository
from .report import PipelineReporter
from .strategy import CleaningPipeline


def enrich_customer_profiles(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Enriches customer records with synthetic CRM profile information (full_name, email, phone)

    using deterministic seeding.
    """
    df = df.copy()
    n = len(df)

    try:
        from faker import Faker

        fake = Faker()
        Faker.seed(seed)
        full_names = [fake.name() for _ in range(n)]
        emails = [
            f"{name.lower().replace(' ', '.').replace(',', '')}@{fake.free_email_domain()}"
            for name in full_names
        ]
        phones = [fake.phone_number() for _ in range(n)]
    except ImportError:
        # High-fidelity deterministic fallback if faker is not installed
        import random

        rng = random.Random(seed)
        first_names = [
            "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Sam",
            "Chris", "Pat", "Drew", "Avery", "Jamie", "Logan", "Kendall", "Reese"
        ]
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis",
            "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee"
        ]
        domains = ["gmail.com", "yahoo.com", "outlook.com", "example.com"]

        full_names = [
            f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n)
        ]
        emails = [
            f"{name.lower().replace(' ', '.')}@{rng.choice(domains)}"
            for name in full_names
        ]
        phones = [
            f"+1-{rng.randint(200, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
            for _ in range(n)
        ]

    # Place identities near the start of the DataFrame
    df["full_name"] = full_names
    df["email"] = emails
    df["phone"] = phones
    return df


class DataPipeline:
    """Orchestrates end-to-end data processing:

    Raw CSV -> CleaningPipeline (7 OOP Steps) -> Synthetic CRM Enrichment -> Verification Report -> SQLite Storage
    """

    def __init__(
        self,
        raw_data_path: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
    ):
        self.raw_data_path = Path(raw_data_path or RAW_DATA_PATH)
        self.db_path = Path(db_path or DB_PATH)
        self.repository = CustomerRepository(db_path=self.db_path)

    def build_cleaning_pipeline(self) -> CleaningPipeline:
        """Assembles the 7 concrete cleaner classes in order."""
        return CleaningPipeline(
            cleaners=[
                StringNormalizer(),
                Deduplicator(subset=["customerID"], keep="first"),
                DataTypeConverter(),
                AllowedValueCleaner(),
                MissingValueColumnDropper(threshold=0.5),
                MissingValueImputer(),
                OutlierHandler(),
            ]
        )

    def run(self) -> pd.DataFrame:
        """Executes the full pipeline workflow.

        Returns:
            pd.DataFrame: Cleaned, enriched, and stored customer DataFrame.
        """
        print(f"Loading raw data from: {self.raw_data_path}")
        raw_df = pd.read_csv(self.raw_data_path)

        # 1. Execute Cleaning Pipeline
        cleaner_pipeline = self.build_cleaning_pipeline()
        cleaned_df = cleaner_pipeline.run(raw_df)

        # 2. Enrich with Synthetic CRM profiles
        enriched_df = enrich_customer_profiles(cleaned_df, seed=42)

        # 3. Audit & Log Metrics
        PipelineReporter.inspect(before_df=raw_df, after_df=enriched_df)

        # 4. Persist to SQLite Database
        saved_count = self.repository.save_customers(enriched_df, table_name="customers")
        print(f"Successfully persisted {saved_count:,} records to {self.db_path}")

        return enriched_df

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from src.config import DB_PATH


class CustomerRepository:
    """Repository layer managing SQLite persistence, schema indexing,

    and query operations for the customer CRM database.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection configured with row factory for dictionary access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save_customers(self, df: pd.DataFrame, table_name: str = "customers") -> int:
        """Persists the cleaned customer DataFrame into SQLite and ensures indexes exist.

        Args:
            df (pd.DataFrame): Cleaned and enriched customer records.
            table_name (str): Destination SQLite table name.

        Returns:
            int: Number of rows saved.
        """
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)

            # Auto-create lookup indexes for high-speed retrieval
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_customerID ON {table_name}(customerID);"
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_Contract ON {table_name}(Contract);"
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_Churn ON {table_name}(Churn);"
            )
            conn.commit()

        return len(df)

    def get_customer_by_id(
        self, customer_id: str, table_name: str = "customers"
    ) -> Optional[Dict[str, Any]]:
        """Fetches a single customer record by customerID.

        Args:
            customer_id (str): Customer ID to look up.
            table_name (str): Table name.

        Returns:
            Optional[Dict[str, Any]]: Record dictionary if found, else None.
        """
        cleaned_id = str(customer_id).strip().upper()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM {table_name} WHERE UPPER(customerID) = ? LIMIT 1",
                (cleaned_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_high_risk_sample(
        self, limit: int = 10, table_name: str = "customers"
    ) -> List[Dict[str, Any]]:
        """Retrieves a sample of high-risk / churned customers.

        Args:
            limit (int): Number of customer records to fetch.
            table_name (str): Table name.

        Returns:
            List[Dict[str, Any]]: List of customer records.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # SQLite handles boolean as 1/0 or True/False strings
            cursor.execute(
                f"SELECT * FROM {table_name} WHERE Churn IN (1, 'True', 'true', 'Yes') LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_all_ids(self, table_name: str = "customers") -> List[str]:
        """Retrieves all distinct customer IDs in the database.

        Args:
            table_name (str): Table name.

        Returns:
            List[str]: List of customer ID strings.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT customerID FROM {table_name}")
            rows = cursor.fetchall()
            return [str(row["customerID"]) for row in rows]

    def get_all_customers(self, table_name: str = "customers") -> pd.DataFrame:
        """Loads all customer records into a pandas DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", con=conn)

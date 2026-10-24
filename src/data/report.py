import pandas as pd


class PipelineReporter:
    """Generates verification summaries comparing datasets before and after pipeline execution."""

    @staticmethod
    def inspect(before_df: pd.DataFrame, after_df: pd.DataFrame) -> dict:
        """Inspects and prints before vs. after metrics including shapes, nulls, and memory usage.

        Args:
            before_df (pd.DataFrame): DataFrame before processing.
            after_df (pd.DataFrame): DataFrame after processing.

        Returns:
            dict: Summary metrics dictionary.
        """
        before_rows, before_cols = before_df.shape
        after_rows, after_cols = after_df.shape

        before_nulls = int(before_df.isna().sum().sum())
        after_nulls = int(after_df.isna().sum().sum())
        null_reduction = before_nulls - after_nulls

        before_mem_mb = before_df.memory_usage(deep=True).sum() / (1024 * 1024)
        after_mem_mb = after_df.memory_usage(deep=True).sum() / (1024 * 1024)

        metrics = {
            "before_shape": (before_rows, before_cols),
            "after_shape": (after_rows, after_cols),
            "rows_removed": before_rows - after_rows,
            "cols_removed": before_cols - after_cols,
            "before_nulls": before_nulls,
            "after_nulls": after_nulls,
            "null_reduction": null_reduction,
            "before_memory_mb": round(before_mem_mb, 2),
            "after_memory_mb": round(after_mem_mb, 2),
        }

        print("=" * 60)
        print(" PIPELINE EXECUTION AUDIT REPORT")
        print("=" * 60)
        print(f"Shape:          ({before_rows:,}, {before_cols}) -> ({after_rows:,}, {after_cols})")
        print(f"Rows Dropped:   {metrics['rows_removed']:,}")
        print(f"Columns Dropped:{metrics['cols_removed']:,}")
        print(f"Total Nulls:    {before_nulls:,} -> {after_nulls:,} (Δ {null_reduction:+,})")
        print(f"Deep Memory:    {before_mem_mb:.2f} MB -> {after_mem_mb:.2f} MB")
        print("=" * 60)

        return metrics

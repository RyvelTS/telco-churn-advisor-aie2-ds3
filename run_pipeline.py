"""CLI entrypoint for running the data cleaning and ingestion pipeline.

Usage:
    python run_pipeline.py
"""

from src.data.pipeline import DataPipeline


def main():
    print("=" * 60)
    print(" STARTING TELCO CHURN DATA PIPELINE")
    print("=" * 60)
    pipeline = DataPipeline()
    df = pipeline.run()
    print("Pipeline execution complete. Ready for ML inference & agent queries.")
    return df


if __name__ == "__main__":
    main()

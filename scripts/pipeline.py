from __future__ import annotations

from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]

from apis import main as run_week1_downloads
from preprocess import main as run_week2_preprocessing


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"


def export_parquet_from_csv(csv_path: Path) -> Path:
    frame = pd.read_csv(csv_path)
    parquet_path = csv_path.with_suffix(".parquet")
    frame.to_parquet(parquet_path, index=False)
    return parquet_path


def print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> None:
    # Week 1: data source setup and raw data collection.
    print_section("Week 1 - Data Source Design & Initial Collection")
    run_week1_downloads()

    # Week 2: data preprocessing, alignment, and feature engineering.
    print_section("Week 2 - Data Preprocessing & Feature Engineering")
    run_week2_preprocessing()

    csv_path = PROCESSED_DIR / "week2_feature_dataset_2018_2024.csv"
    if csv_path.exists():
        parquet_path = export_parquet_from_csv(csv_path)
        print(f"[OK] Parquet dataset saved to {parquet_path.name}")
    else:
        print(f"[WARN] CSV output not found at {csv_path}")

    print_section("Pipeline Complete")
    print("One-command workflow finished. Check warnings above for any source-specific limitations.")


if __name__ == "__main__":
    main()
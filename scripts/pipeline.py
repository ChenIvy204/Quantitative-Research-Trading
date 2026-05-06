from __future__ import annotations

from pathlib import Path

from apis import main as run_week1_downloads
from preprocess import main as run_week2_preprocessing


ROOT = Path(__file__).resolve().parents[1]


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

    print_section("Pipeline Complete")
    print("One-command workflow finished. Check warnings above for any source-specific limitations.")


if __name__ == "__main__":
    main()
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd  # pyright: ignore[reportMissingImports]

from apis import main as run_week1_downloads
from preprocess import main as run_week2_preprocessing
from week3_bsm import main as run_week3_bsm


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
logger = logging.getLogger("week2_pipeline_runner")


def configure_logging() -> None:
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@contextmanager
def step_timer(step_name: str):
    started_at = datetime.now().timestamp()
    logger.info("START %s", step_name)
    try:
        yield
    except Exception as exc:
        elapsed = datetime.now().timestamp() - started_at
        logger.error("FAIL %s (%.2fs): %s", step_name, elapsed, exc)
        raise
    else:
        elapsed = datetime.now().timestamp() - started_at
        logger.info("DONE %s (%.2fs)", step_name, elapsed)


def export_parquet_from_csv(csv_path: Path) -> Path:
    frame = pd.read_csv(csv_path)
    parquet_path = csv_path.with_suffix(".parquet")
    frame.to_parquet(parquet_path, index=False)
    return parquet_path


def print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> None:
    configure_logging()
    # Week 1: data source setup and raw data collection.
    print_section("Week 1 - Data Source Design & Initial Collection")
    with step_timer("Week 1 downloads"):
        run_week1_downloads()

    # Week 2: data preprocessing, alignment, and feature engineering.
    print_section("Week 2 - Data Preprocessing & Feature Engineering")
    with step_timer("Week 2 preprocessing"):
        outputs = run_week2_preprocessing()

    csv_path = outputs["dataset_csv"] if isinstance(outputs, dict) and "dataset_csv" in outputs else PROCESSED_DIR / "week2_feature_dataset.csv"
    if csv_path.exists():
        with step_timer("Parquet export"):
            parquet_path = export_parquet_from_csv(csv_path)
        print(f"[OK] Parquet dataset saved to {parquet_path.name}")
    else:
        logger.warning("CSV output not found at %s", csv_path)

    # Week 3: BSM chooser-option replication and validation.
    print_section("Week 3 - Original BSM Model Replication")
    with step_timer("Week 3 replication and validation"):
        week3_outputs = run_week3_bsm()
    if isinstance(week3_outputs, dict):
        for label, path in week3_outputs.items():
            print(f"[OK] {label} saved to {path.name}")

    print_section("Pipeline Complete")
    print("One-command workflow finished. Check warnings above for any source-specific limitations.")


if __name__ == "__main__":
    main()
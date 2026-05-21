from __future__ import annotations

import logging
import re
from pathlib import Path
from datetime import datetime
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "data" / "reports"
START_DATE = "2018-01-01"
END_DATE_EXCLUSIVE = "2025-01-01"
PIPELINE_VERSION = "v1.0"
RUN_DATE = datetime.now().strftime("%Y%m%d")

QUALITY_REPORT_CSV = "week2_data_quality_report"
QUALITY_REPORT_MD = "week2_data_quality_report"
QUALITY_REPORT_PDF = "week2_data_quality_report"
QUALITY_REPORT_BOXPLOT = "week2_data_quality_report_boxplot"
FEATURE_OPTIMIZATION_CSV = "week2_feature_optimization_report"
FEATURE_OPTIMIZATION_PDF = "week2_feature_optimization_report"
FEATURE_CORRELATION_CSV = "week2_feature_correlation_matrix"
FEATURE_SELECTED_DATASET = "week2_feature_dataset"
FEATURE_IC_REPORT = "week2_feature_ic_report"
DATASET_OUTPUT_EXTENSION = "csv"
IC_THRESHOLD = 0.03
CORR_THRESHOLD = 0.8
BOXPLOT_EXCLUDED_PREFIXES = ("news_",)
BOXPLOT_CORE_FEATURES = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "vix",
    "jpm_return_1d",
    "jpm_return_5d",
    "jpm_vol_5d",
    "jpm_vol_20d",
    "jpm_vol_60d",
)


logger = logging.getLogger("week2_pipeline")
IMAGE_MARKDOWN_PATTERN = re.compile(r"^!\[(?P<alt>.*?)\]\((?P<path>.*?)\)$")


def versioned_filename(stem: str, extension: str) -> str:
    return f"{stem}_{PIPELINE_VERSION}_{RUN_DATE}.{extension}"


def cleanup_generated_outputs() -> None:
    ensure_output_dir()
    for path in PROCESSED_DIR.glob("week2_*"):
        if path.is_file() or path.is_symlink():
            path.unlink()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in REPORTS_DIR.glob("week2_*"):
        if path.is_file() or path.is_symlink():
            path.unlink()


def configure_logging() -> None:
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class log_step:
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.started_at = 0.0

    def __enter__(self):
        self.started_at = datetime.now().timestamp()
        logger.info("START %s", self.step_name)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = datetime.now().timestamp() - self.started_at
        if exc_type is None:
            logger.info("DONE %s (%.2fs)", self.step_name, elapsed)
            return False

        logger.error("FAIL %s (%.2fs): %s", self.step_name, elapsed, exc)
        return False


def ensure_output_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(frame: pd.DataFrame, filename: str) -> Path:
    ensure_output_dir()
    path = PROCESSED_DIR / filename
    frame.to_csv(path, index=False)
    return path


def save_markdown(markdown_text: str, filename: str) -> Path:
    ensure_output_dir()
    path = REPORTS_DIR / filename
    path.write_text(markdown_text, encoding="utf-8")
    return path


def save_figure(figure: plt.Figure, filename: str) -> Path:
    ensure_output_dir()
    path = REPORTS_DIR / filename
    figure.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(figure)
    return path


def format_markdown_cell(value: object, precision: int = 4, max_length: int | None = None) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        text = f"{value:.{precision}f}"
    elif isinstance(value, int) and not isinstance(value, bool):
        text = str(value)
    else:
        text = str(value)

    text = text.replace("\n", " ").replace("|", "\\|")
    if max_length is not None and len(text) > max_length:
        return text[: max_length - 1].rstrip() + "…"
    return text


def dataframe_to_markdown_table(
    frame: pd.DataFrame,
    columns: list[tuple[str, str]] | None = None,
    precision: int = 4,
    max_widths: dict[str, int] | None = None,
) -> str:
    if columns is None:
        columns = [(column, column) for column in frame.columns]
    max_widths = max_widths or {}

    header_labels = [label for _, label in columns]
    rows: list[list[str]] = []
    for _, row in frame.iterrows():
        rows.append(
            [
                format_markdown_cell(row[column], precision=precision, max_length=max_widths.get(column))
                for column, _ in columns
            ]
        )

    widths = [len(label) for label in header_labels]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(values: list[str]) -> str:
        padded = [value.ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [render_row(header_labels), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def build_boxplot_figure(frame: pd.DataFrame) -> plt.Figure:
    numeric_columns = [column for column in BOXPLOT_CORE_FEATURES if column in frame.columns and not column.startswith(BOXPLOT_EXCLUDED_PREFIXES)]
    if not numeric_columns:
        raise ValueError("No numeric columns available for boxplot generation")

    fig, axes = plt.subplots(nrows=len(numeric_columns), ncols=1, figsize=(10, max(2.0, 1.6 * len(numeric_columns))), constrained_layout=True)
    if len(numeric_columns) == 1:
        axes = [axes]

    for axis, column in zip(axes, numeric_columns):
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        axis.boxplot(series, vert=False, patch_artist=True, boxprops={"facecolor": "#d7e8ff", "color": "#2c5282"}, medianprops={"color": "#c53030", "linewidth": 1.5}, whiskerprops={"color": "#2c5282"}, capprops={"color": "#2c5282"}, flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#dd6b20", "markeredgecolor": "#dd6b20", "alpha": 0.6})
        axis.set_title(column, loc="left", fontsize=10)
        axis.tick_params(axis="x", labelsize=8)
        axis.set_yticks([])

    fig.suptitle("Week 2 Numeric Feature Boxplots", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def save_pdf_report(markdown_path: Path, filename: str, asset_paths: dict[str, Path] | None = None) -> Path:
    ensure_output_dir()
    output_path = REPORTS_DIR / filename
    markdown_text = markdown_path.read_text(encoding="utf-8")

    with PdfPages(output_path) as pdf:
        render_text_report(pdf, markdown_text, title="Week 2 Data Quality Report", asset_paths=asset_paths)

    return output_path


def save_text_pdf_report(report_text: str, filename: str) -> Path:
    ensure_output_dir()
    output_path = REPORTS_DIR / filename

    with PdfPages(output_path) as pdf:
        render_text_report(pdf, report_text, title="Week 2 Feature Engineering Optimization Report")

    return output_path


def render_image_report_page(pdf: PdfPages, image_path: Path, title: str, caption: str | None = None) -> None:
    image = plt.imread(image_path)
    report_fig, report_ax = plt.subplots(figsize=(8.5, 11))
    report_ax.imshow(image)
    report_ax.axis("off")
    report_fig.text(0.05, 0.975, title, ha="left", va="top", fontsize=18, fontweight="bold")
    if caption:
        report_fig.text(0.05, 0.04, caption, ha="left", va="bottom", fontsize=9)
    pdf.savefig(report_fig, bbox_inches="tight")
    plt.close(report_fig)


def render_text_report(pdf: PdfPages, report_text: str, title: str, asset_paths: dict[str, Path] | None = None) -> None:
    report_fig = plt.figure(figsize=(8.5, 11))
    report_ax = report_fig.add_axes([0, 0, 1, 1])
    report_ax.axis("off")

    y = 0.95
    report_fig.text(0.05, 0.975, title, ha="left", va="top", fontsize=18, fontweight="bold")
    skipped_title_line = False

    def flush_page() -> None:
        nonlocal report_fig, report_ax, y
        pdf.savefig(report_fig, bbox_inches="tight")
        plt.close(report_fig)
        report_fig = plt.figure(figsize=(8.5, 11))
        report_ax = report_fig.add_axes([0, 0, 1, 1])
        report_ax.axis("off")
        report_fig.text(0.05, 0.975, title, ha="left", va="top", fontsize=18, fontweight="bold")
        y = 0.95

    for line in report_text.splitlines():
        stripped = line.strip()
        if not stripped:
            y -= 0.012
            continue

        if not skipped_title_line and stripped.startswith("# ") and stripped[2:].strip() == title:
            skipped_title_line = True
            continue

        if stripped.startswith("```"):
            continue

        image_match = IMAGE_MARKDOWN_PATTERN.match(stripped)
        if image_match is not None:
            image_reference = image_match.group("path")
            caption = image_match.group("alt") or None
            image_path = None
            if asset_paths is not None:
                image_path = asset_paths.get(image_reference) or asset_paths.get(Path(image_reference).name)
            if image_path is not None and image_path.exists():
                if y < 0.18:
                    flush_page()
                render_image_report_page(pdf, image_path, title, caption=caption)
            continue

        if stripped.startswith("# "):
            display = stripped[2:].strip()
            font_size = 15
            font_weight = "bold"
            y -= 0.01
        elif stripped.startswith("## "):
            display = stripped[3:].strip()
            font_size = 12
            font_weight = "bold"
            y -= 0.004
        elif stripped.startswith("- "):
            display = f"• {stripped[2:].strip()}"
            font_size = 9.5
            font_weight = "normal"
        else:
            display = stripped
            font_size = 9.5
            font_weight = "normal"

        wrapped_lines = textwrap.wrap(display, width=94) or [""]
        for index, wrapped_line in enumerate(wrapped_lines):
            if y < 0.06:
                flush_page()
            report_fig.text(0.05, y, wrapped_line, ha="left", va="top", fontsize=font_size, fontweight=font_weight)
            y -= 0.018 if font_size <= 10 else 0.022
            if index < len(wrapped_lines) - 1:
                y -= 0.001

    pdf.savefig(report_fig, bbox_inches="tight")
    plt.close(report_fig)


def cap_sigma(series: pd.Series, sigma_multiplier: float = 3.0) -> pd.Series:
    mean = series.mean()
    std = series.std()
    if pd.isna(std) or std == 0:
        return series.clip(mean, mean)

    lower = mean - sigma_multiplier * std
    upper = mean + sigma_multiplier * std
    return series.clip(lower, upper)


def sigma_bounds(series: pd.Series, sigma_multiplier: float = 3.0) -> tuple[float, float]:
    mean = series.mean()
    std = series.std()
    if pd.isna(std) or std == 0:
        return float(mean), float(mean)
    return float(mean - sigma_multiplier * std), float(mean + sigma_multiplier * std)


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]


RANGE_RULES: dict[str, tuple[float | None, float | None, str]] = {
    "vix": (0.0, None, "VIX should be strictly positive"),
    "jpm_return_1d": (-0.10, 0.10, "One-day JPM return should stay within a reasonable sanity band"),
    "jpm_return_5d": (-0.10, 0.10, "Five-day JPM return should stay within a reasonable sanity band"),
}


def validate_feature_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for column, (lower_bound, upper_bound, description) in RANGE_RULES.items():
        if column not in frame.columns:
            continue

        series = pd.to_numeric(frame[column], errors="coerce")
        violation_mask = pd.Series(False, index=series.index)
        if lower_bound is not None:
            violation_mask = violation_mask | series.lt(lower_bound)
        if upper_bound is not None:
            violation_mask = violation_mask | series.gt(upper_bound)

        violation_count = int(violation_mask.sum())
        rows.append(
            {
                "feature": column,
                "range_lower": lower_bound,
                "range_upper": upper_bound,
                "range_rule": description,
                "range_violation_count": violation_count,
                "range_violation_rate": (violation_count / len(series)) if len(series) else 0.0,
                "range_status": "ok" if violation_count == 0 else "warn",
            }
        )

    return pd.DataFrame(rows)


def describe_fill_strategy(column: str) -> tuple[str, str]:
    if column in {"Open", "High", "Low", "Close", "Adj Close", "Volume"}:
        return (
            "Interpolation across the daily calendar",
            "Daily market gaps are usually short and interpolation preserves the price path across non-trading days.",
        )
    if column in {"dgs10", "vix"}:
        return (
            "Forward fill",
            "Macro series are observed daily; forward fill keeps the latest known value without looking ahead.",
        )
    if column in {"jpm_dividend_ttm", "jpm_dividend_growth_yoy"}:
        return (
            "Forward fill after quarterly aggregation",
            "Dividend values remain in force until the next announcement, so forward fill is the least distortive option.",
        )
    if column == "jpm_dividend_yield_ttm":
        return (
            "Derived from dividend and price; residual gaps are dropped",
            "Dividend yield is the standard BSM-style input and should only remain where both trailing dividend and current price exist.",
        )
    if column == "news_article_count":
        return ("Fill with 0", "No matched articles means zero observed news activity.")
    if column == "news_sentiment_mean":
        return ("Fill with 0.5", "0.5 is the neutral midpoint of the 0-1 sentiment scale.")
    if column == "news_sentiment_std":
        return ("Fill with 0.0", "A single or absent news item implies no dispersion.")
    if column == "news_7d_article_count":
        return ("Fill with 0", "The 7-day count should stay at zero when no news is present.")
    if column == "news_7d_sentiment_mean":
        return ("Fill with 0.5", "A 7-day rolling sentiment with no observations should stay neutral.")
    if column == "news_7d_sentiment_trend":
        return ("Fill with 0.0", "No rolling sentiment change is the safest default when the window is empty.")
    if column in {
        "jpm_return_1d",
        "jpm_return_5d",
        "jpm_vol_20d",
        "jpm_vol_5d",
        "jpm_vol_60d",
        "jpm_vol_20d_change_1d",
        "jpm_vol_20d_change_rate_1d",
        "jpm_ma_20d",
        "jpm_price_to_ma_20d",
        "dgs10_change_1d",
        "dgs10_momentum_5d",
        "vix_change_1d",
        "vix_jpm_corr_20d",
        "rolling_high_20d",
        "drawdown_20d",
    }:
        return (
            "Drop remaining warm-up rows after feature engineering",
            "These are rolling features; the earliest rows do not have enough history for a reliable imputation.",
        )
    return (
        "Drop remaining missing rows",
        "Any residual gaps are treated as incomplete observations and removed before export.",
    )


def summarize_data_quality(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = numeric_feature_columns(frame)
    rows: list[dict[str, object]] = []

    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        observed = series.dropna()
        total_rows = len(series)
        missing_count = int(series.isna().sum())
        missing_rate = (missing_count / total_rows) if total_rows else 0.0

        if observed.empty:
            lower_bound = upper_bound = float("nan")
            outlier_count = 0
            minimum = maximum = mean = std = float("nan")
        else:
            lower_bound, upper_bound = sigma_bounds(observed)
            outlier_mask = series.lt(lower_bound) | series.gt(upper_bound)
            outlier_count = int(outlier_mask.sum())
            minimum = float(observed.min())
            maximum = float(observed.max())
            mean = float(observed.mean())
            std = float(observed.std())

        fill_strategy, fill_reason = describe_fill_strategy(column)
        rows.append(
            {
                "feature": column,
                "missing_rate": missing_rate,
                "min": minimum,
                "max": maximum,
                "mean": mean,
                "std": std,
                "sigma_lower": lower_bound,
                "sigma_upper": upper_bound,
                "outlier_count": outlier_count,
                "outlier_rate": (outlier_count / total_rows) if total_rows else 0.0,
                "fill_strategy": fill_strategy,
                "fill_reason": fill_reason,
            }
        )

    return pd.DataFrame(rows)


def replace_boxplot_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()

    for column in numeric_feature_columns(working):
        series = pd.to_numeric(working[column], errors="coerce")
        observed = series.dropna()
        if observed.empty:
            continue

        lower_bound, upper_bound = sigma_bounds(observed)
        outlier_mask = series.lt(lower_bound) | series.gt(upper_bound)
        if not outlier_mask.any():
            continue

        inlier_values = observed[(observed >= lower_bound) & (observed <= upper_bound)]
        replacement_value = float(inlier_values.median()) if not inlier_values.empty else float(observed.median())
        working.loc[outlier_mask, column] = replacement_value

    return working


def apply_missing_value_strategies(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()

    price_columns = [column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in working.columns]
    if price_columns:
        working[price_columns] = working[price_columns].interpolate(limit_direction="both")

    if "dgs10" in working.columns:
        working["dgs10"] = working["dgs10"].ffill()
    if "vix" in working.columns:
        working["vix"] = working["vix"].ffill()

    if "jpm_dividend_ttm" in working.columns:
        working["jpm_dividend_ttm"] = working["jpm_dividend_ttm"].ffill()
    if "jpm_dividend_growth_yoy" in working.columns:
        working["jpm_dividend_growth_yoy"] = working["jpm_dividend_growth_yoy"].ffill()

    if "news_article_count" in working.columns:
        working["news_article_count"] = working["news_article_count"].fillna(0)
    if "news_sentiment_mean" in working.columns:
        working["news_sentiment_mean"] = working["news_sentiment_mean"].fillna(0.5)
    if "news_sentiment_std" in working.columns:
        working["news_sentiment_std"] = working["news_sentiment_std"].fillna(0.0)

    if "news_7d_article_count" in working.columns:
        working["news_7d_article_count"] = working["news_7d_article_count"].fillna(0)
    if "news_7d_sentiment_mean" in working.columns:
        working["news_7d_sentiment_mean"] = working["news_7d_sentiment_mean"].fillna(0.5)
    if "news_7d_sentiment_trend" in working.columns:
        working["news_7d_sentiment_trend"] = working["news_7d_sentiment_trend"].fillna(0.0)

    working = working.dropna()
    return working


def load_market_data(apply_fill: bool = True) -> pd.DataFrame:
    if apply_fill:
        # Load raw JPM, Treasury, and VIX data and normalize the column names.
        jpm = pd.read_csv(RAW_DIR / "yahoo_jpm_2018_2024.csv", parse_dates=["Date"])
        dgs10 = pd.read_csv(RAW_DIR / "fred_DGS10_2018_2024.csv", parse_dates=["date"])
        vix = pd.read_csv(RAW_DIR / "fred_VIXCLS_2018_2024.csv", parse_dates=["date"])

        jpm = jpm.rename(
            columns={
                "Date": "date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adj close": "Adj Close",
                "Adj Close": "Adj Close",
                "volume": "Volume",
                "Volume": "Volume",
            }
        ).set_index("date").sort_index()
        jpm.index = pd.to_datetime(jpm.index).normalize()

        dgs10 = dgs10.rename(columns={"value": "dgs10"}).set_index("date").sort_index()
        dgs10.index = pd.to_datetime(dgs10.index).normalize()

        vix = vix.rename(columns={"value": "vix"}).set_index("date").sort_index()
        vix.index = pd.to_datetime(vix.index).normalize()

        jpm = jpm[[column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in jpm.columns]]
        dgs10 = dgs10[["dgs10"]]
        vix = vix[["vix"]]

        frame = jpm.join(dgs10, how="outer").join(vix, how="outer")
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="first")]
        frame = frame.loc["2018-01-01":"2024-12-31"]

        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame[["dgs10", "vix"]] = frame[["dgs10", "vix"]].ffill()

        price_columns = [column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in frame.columns]
        if price_columns:
            frame[price_columns] = frame[price_columns].interpolate(limit_direction="both")

        return frame

    # Load raw JPM, Treasury, and VIX data and normalize the column names.
    jpm = pd.read_csv(RAW_DIR / "yahoo_jpm_2018_2024.csv", parse_dates=["Date"])
    dgs10 = pd.read_csv(RAW_DIR / "fred_DGS10_2018_2024.csv", parse_dates=["date"])
    vix = pd.read_csv(RAW_DIR / "fred_VIXCLS_2018_2024.csv", parse_dates=["date"])

    jpm = jpm.rename(
        columns={
            "Date": "date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj close": "Adj Close",
            "Adj Close": "Adj Close",
            "volume": "Volume",
            "Volume": "Volume",
        }
    ).set_index("date").sort_index()
    jpm.index = pd.to_datetime(jpm.index).normalize()

    dgs10 = dgs10.rename(columns={"value": "dgs10"}).set_index("date").sort_index()
    dgs10.index = pd.to_datetime(dgs10.index).normalize()

    vix = vix.rename(columns={"value": "vix"}).set_index("date").sort_index()
    vix.index = pd.to_datetime(vix.index).normalize()

    jpm = jpm[[column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in jpm.columns]]
    dgs10 = dgs10[["dgs10"]]
    vix = vix[["vix"]]

    frame = jpm.join(dgs10, how="outer").join(vix, how="outer")
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    frame = frame.loc["2018-01-01":"2024-12-31"]

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if apply_fill:
        frame[["dgs10", "vix"]] = frame[["dgs10", "vix"]].ffill()

        price_columns = [column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in frame.columns]
        if price_columns:
            frame[price_columns] = frame[price_columns].interpolate(limit_direction="both")

    return frame


def load_dividend_data() -> pd.DataFrame | None:
    dividend_path = RAW_DIR / "jpm_dividends_2018_2024.csv"
    if not dividend_path.exists():
        return None

    dividend_frame = pd.read_csv(dividend_path)
    if dividend_frame.empty:
        return None

    if "date" in dividend_frame.columns:
        dividend_frame["date"] = pd.to_datetime(dividend_frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
        dividend_frame = dividend_frame.set_index("date")
    else:
        dividend_frame = dividend_frame.rename(columns={dividend_frame.columns[0]: "date"})
        dividend_frame["date"] = pd.to_datetime(dividend_frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
        dividend_frame = dividend_frame.set_index("date")

    dividend_frame = dividend_frame.sort_index()
    if dividend_frame.empty or "dividend" not in dividend_frame.columns:
        return None

    dividend_frame["dividend"] = pd.to_numeric(dividend_frame["dividend"], errors="coerce")
    dividend_frame = dividend_frame.dropna(subset=["dividend"])
    if dividend_frame.empty:
        return None

    quarterly_dividend = dividend_frame["dividend"].resample("QE").sum()
    dividend_growth_yoy = quarterly_dividend.rolling(4).sum().pct_change(4)

    dividend_features = pd.DataFrame(
        {
            "jpm_dividend_ttm": quarterly_dividend.rolling(4).sum(),
            "jpm_dividend_growth_yoy": dividend_growth_yoy,
        }
    )
    dividend_features = dividend_features.reindex(pd.date_range(START_DATE, END_DATE_EXCLUSIVE, freq="D"), method="ffill")
    dividend_features.index.name = "date"
    return dividend_features


def score_text(text: str) -> float:
    positive_words = {
        "beat",
        "growth",
        "gain",
        "upgrade",
        "strong",
        "positive",
        "record",
        "improve",
        "surge",
        "rise",
    }
    negative_words = {
        "miss",
        "loss",
        "downgrade",
        "weak",
        "negative",
        "drop",
        "fall",
        "decline",
        "risk",
        "concern",
    }

    tokens = {token.strip(".,!?;:\"'()[]{}<>|/\\").lower() for token in str(text).split()}
    positive_hits = sum(1 for token in tokens if token in positive_words)
    negative_hits = sum(1 for token in tokens if token in negative_words)
    raw_score = positive_hits - negative_hits

    if raw_score == 0:
        return 0.0
    return max(-1.0, min(1.0, raw_score / 5.0))


def normalize_sentiment_to_unit_interval(series: pd.Series) -> pd.Series:
    return ((series.clip(-1.0, 1.0) + 1.0) / 2.0).clip(0.0, 1.0)


def load_news_data() -> pd.DataFrame | None:
    # Load the available news source and convert article text into a simple sentiment proxy.
    candidate_groups = [
        [RAW_DIR / "alphavantage_news_jpm_2018_2024.csv"],
        list(RAW_DIR.glob("alphavantage_*.csv")),
        list(RAW_DIR.glob("news_*.csv")),
    ]
    news_frame_path = next((path for files in candidate_groups for path in files if path.exists()), None)
    if news_frame_path is None:
        candidate_groups = [
            list(RAW_DIR.glob("alphavantage_*.csv")),
            list(RAW_DIR.glob("news_*.csv")),
        ]
        news_frame_path = next((max(files, key=lambda path: path.stat().st_mtime) for files in candidate_groups if files), None)
    if news_frame_path is None:
        return None

    news_frame = pd.read_csv(news_frame_path)
    if "publishedAt" not in news_frame.columns:
        return None

    news_frame["publishedAt"] = pd.to_datetime(news_frame["publishedAt"], errors="coerce")
    news_frame = news_frame.dropna(subset=["publishedAt"])
    if news_frame.empty:
        return None

    for column in ["title", "description", "content"]:
        if column not in news_frame.columns:
            news_frame[column] = ""

    news_frame["text_for_sentiment"] = (
        news_frame["title"].fillna("").astype(str)
        + " "
        + news_frame["description"].fillna("").astype(str)
        + " "
        + news_frame["content"].fillna("").astype(str)
    )
    if "sentiment_score" in news_frame.columns:
        news_frame["sentiment_score"] = pd.to_numeric(news_frame["sentiment_score"], errors="coerce")
    else:
        news_frame["sentiment_score"] = news_frame["text_for_sentiment"].apply(score_text)

    news_frame["sentiment_score"] = normalize_sentiment_to_unit_interval(news_frame["sentiment_score"].fillna(0.0))
    news_frame["news_date"] = news_frame["publishedAt"].dt.floor("D")

    daily_news = (
        news_frame.groupby("news_date")
        .agg(
            news_article_count=("sentiment_score", "size"),
            news_sentiment_mean=("sentiment_score", "mean"),
            news_sentiment_std=("sentiment_score", "std"),
        )
        .sort_index()
    )

    daily_news["news_sentiment_std"] = daily_news["news_sentiment_std"].fillna(0.0)
    return daily_news


def add_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["jpm_return_1d"] = working["Adj Close"].pct_change(fill_method=None)
    working["jpm_return_5d"] = working["Adj Close"].pct_change(5, fill_method=None)
    working["jpm_vol_20d"] = working["jpm_return_1d"].rolling(20).std() * (252 ** 0.5)
    working["jpm_vol_5d"] = working["jpm_return_1d"].rolling(5).std() * (252 ** 0.5)
    working["jpm_vol_60d"] = working["jpm_return_1d"].rolling(60).std() * (252 ** 0.5)
    working["jpm_vol_20d_change_1d"] = working["jpm_vol_20d"].diff()
    working["jpm_vol_20d_change_rate_1d"] = working["jpm_vol_20d"].pct_change(fill_method=None)
    working["jpm_ma_20d"] = working["Adj Close"].rolling(20).mean()
    working["jpm_price_to_ma_20d"] = working["Adj Close"] / working["jpm_ma_20d"]
    working["dgs10_change_1d"] = working["dgs10"].diff()
    working["dgs10_momentum_5d"] = working["dgs10"].diff(5)
    working["vix_change_1d"] = working["vix"].diff()
    working["vix_jpm_corr_20d"] = working["jpm_return_1d"].rolling(20).corr(working["vix_change_1d"])
    working["rolling_high_20d"] = working["Adj Close"].rolling(20).max()
    working["drawdown_20d"] = working["Adj Close"] / working["rolling_high_20d"] - 1.0
    return working


def attach_optional_features(frame: pd.DataFrame, fill_optional: bool) -> pd.DataFrame:
    working = frame.copy()

    dividend_frame = load_dividend_data()
    if dividend_frame is not None:
        working = working.join(dividend_frame, how="left")
    else:
        working["jpm_dividend_ttm"] = pd.NA
        working["jpm_dividend_growth_yoy"] = pd.NA

    news_frame = load_news_data()
    if news_frame is not None:
        working = working.join(news_frame, how="left")
    else:
        working["news_article_count"] = 0
        working["news_sentiment_mean"] = 0.5
        working["news_sentiment_std"] = 0.0

    if fill_optional:
        if "jpm_dividend_ttm" in working.columns:
            working["jpm_dividend_ttm"] = working["jpm_dividend_ttm"].ffill()
        if "jpm_dividend_growth_yoy" in working.columns:
            working["jpm_dividend_growth_yoy"] = working["jpm_dividend_growth_yoy"].ffill()

        if "news_article_count" in working.columns:
            working["news_article_count"] = working["news_article_count"].fillna(0)
        if "news_sentiment_mean" in working.columns:
            working["news_sentiment_mean"] = working["news_sentiment_mean"].fillna(0.5)
        if "news_sentiment_std" in working.columns:
            working["news_sentiment_std"] = working["news_sentiment_std"].fillna(0.0)

        working["news_7d_article_count"] = working["news_article_count"].rolling(7).sum().fillna(0)
        working["news_7d_sentiment_mean"] = working["news_sentiment_mean"].rolling(7).mean().fillna(0.5)
        working["news_7d_sentiment_trend"] = working["news_7d_sentiment_mean"].diff().fillna(0.0)
    else:
        working["news_7d_article_count"] = working["news_article_count"].rolling(7).sum()
        working["news_7d_sentiment_mean"] = working["news_sentiment_mean"].rolling(7).mean()
        working["news_7d_sentiment_trend"] = working["news_7d_sentiment_mean"].diff()

    if "jpm_dividend_ttm" in working.columns and "Adj Close" in working.columns:
        working["jpm_dividend_yield_ttm"] = working["jpm_dividend_ttm"] / working["Adj Close"].replace(0, pd.NA)

    return working


def add_future_return_targets(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    working["future_jpm_return_5d"] = working["Adj Close"].shift(-5) / working["Adj Close"] - 1.0
    working["future_jpm_return_21d"] = working["Adj Close"].shift(-21) / working["Adj Close"] - 1.0
    return working


def candidate_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"date", "future_jpm_return_5d", "future_jpm_return_21d"}
    return [
        column
        for column in numeric_feature_columns(frame)
        if column not in excluded and frame[column].nunique(dropna=True) > 1
    ]


def correlation_matrix_for_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not features:
        return pd.DataFrame()
    variable_features = [column for column in features if frame[column].nunique(dropna=True) > 1]
    if not variable_features:
        return pd.DataFrame()
    return frame[variable_features].corr(method="pearson")


def prune_correlated_features(frame: pd.DataFrame, features: list[str], threshold: float = CORR_THRESHOLD) -> tuple[list[str], list[str], pd.DataFrame]:
    if not features:
        return [], [], pd.DataFrame()

    corr_matrix = correlation_matrix_for_features(frame, features)
    abs_corr = corr_matrix.abs()
    mean_abs_corr = abs_corr.apply(lambda series: series.drop(labels=[series.name]).mean(), axis=0).fillna(1.0)
    ordered_features = sorted(features, key=lambda feature: (mean_abs_corr[feature], feature))

    kept_features: list[str] = []
    dropped_features: list[str] = []

    for feature in ordered_features:
        if any(pd.notna(abs_corr.loc[feature, kept_feature]) and abs_corr.loc[feature, kept_feature] > threshold for kept_feature in kept_features):
            dropped_features.append(feature)
            continue
        kept_features.append(feature)

    return kept_features, dropped_features, corr_matrix


def compute_ic_report(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    target_1w = frame["future_jpm_return_5d"]
    target_1m = frame["future_jpm_return_21d"]

    for feature in features:
        series = pd.to_numeric(frame[feature], errors="coerce")
        ic_1w = series.corr(target_1w)
        ic_1m = series.corr(target_1m)
        abs_values = [abs(value) for value in [ic_1w, ic_1m] if pd.notna(value)]
        max_abs_ic = max(abs_values) if abs_values else float("nan")
        rows.append(
            {
                "feature": feature,
                "ic_1w": ic_1w,
                "ic_1m": ic_1m,
                "max_abs_ic": max_abs_ic,
                "selected_by_ic": bool((pd.notna(ic_1w) and abs(ic_1w) > IC_THRESHOLD) or (pd.notna(ic_1m) and abs(ic_1m) > IC_THRESHOLD)),
            }
        )

    return pd.DataFrame(rows)


def select_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    candidate_features = candidate_feature_columns(frame)
    kept_by_corr, dropped_by_corr, corr_matrix = prune_correlated_features(frame, candidate_features)
    ic_report = compute_ic_report(frame, candidate_features)
    ic_lookup = ic_report.set_index("feature")
    mean_abs_corr = corr_matrix.abs().apply(lambda series: series.drop(labels=[series.name]).mean(), axis=0).fillna(1.0)

    selected_features: list[str] = []
    for feature in kept_by_corr:
        if feature in ic_lookup.index and bool(ic_lookup.loc[feature, "selected_by_ic"]):
            selected_features.append(feature)

    optimization_rows: list[dict[str, object]] = []
    for feature in candidate_features:
        row = ic_lookup.loc[feature]
        optimization_rows.append(
            {
                "feature": feature,
                "mean_abs_corr": float(mean_abs_corr.get(feature, 1.0)),
                "corr_pruned": feature in dropped_by_corr,
                "corr_kept": feature in kept_by_corr,
                "ic_1w": row["ic_1w"],
                "ic_1m": row["ic_1m"],
                "max_abs_ic": row["max_abs_ic"],
                "selected": feature in selected_features,
                "drop_reason": (
                    f"correlation>={CORR_THRESHOLD}" if feature in dropped_by_corr else ("IC threshold" if feature not in selected_features else "kept")
                ),
            }
        )

    optimization_report = pd.DataFrame(optimization_rows)
    optimization_report = optimization_report[["feature", "mean_abs_corr", "corr_pruned", "corr_kept", "ic_1w", "ic_1m", "max_abs_ic", "selected", "drop_reason"]]

    return optimization_report, ic_report, corr_matrix, selected_features


def build_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    # Build the daily modeling table used in Week 2.
    base_market = load_market_data(apply_fill=False)

    if "Adj Close" not in base_market.columns:
        raise ValueError("JPM raw data is missing an Adj Close column")
    if "Close" not in base_market.columns:
        raise ValueError("JPM raw data is missing a Close column")
    if "Open" not in base_market.columns:
        raise ValueError("JPM raw data is missing an Open column")
    if "High" not in base_market.columns:
        raise ValueError("JPM raw data is missing a High column")
    if "Low" not in base_market.columns:
        raise ValueError("JPM raw data is missing a Low column")
    if "Volume" not in base_market.columns:
        base_market["Volume"] = 0.0

    quality_source = attach_optional_features(add_market_features(base_market.copy()), fill_optional=False)
    quality_report = summarize_data_quality(quality_source)
    range_validation_report = validate_feature_ranges(quality_source)
    if not range_validation_report.empty:
        quality_report = quality_report.merge(range_validation_report, on="feature", how="left")
        for _, row in range_validation_report.iterrows():
            if row["range_violation_count"]:
                logger.warning(
                    "Range validation warning for %s: %s violations (%s)",
                    row["feature"],
                    int(row["range_violation_count"]),
                    row["range_rule"],
                )

    cleaned_base = replace_boxplot_outliers(base_market)
    cleaned_base = apply_missing_value_strategies(cleaned_base)

    feature_frame = attach_optional_features(add_market_features(cleaned_base), fill_optional=True)
    feature_frame = add_future_return_targets(feature_frame)

    optimization_report, ic_report, corr_matrix, selected_features = select_features(feature_frame)

    optimized_frame = feature_frame[selected_features].copy().dropna().reset_index()

    return optimized_frame, quality_source, quality_report, optimization_report, ic_report, corr_matrix, selected_features


def build_quality_report_markdown(before_frame: pd.DataFrame, cleaned_frame: pd.DataFrame, stats_frame: pd.DataFrame) -> str:
    missing_columns = int((stats_frame["missing_rate"] > 0).sum())
    total_features = len(stats_frame)
    has_range_checks = "range_violation_count" in stats_frame.columns
    range_violations = int(stats_frame["range_violation_count"].fillna(0).sum()) if has_range_checks else 0
    boxplot_filename = versioned_filename(QUALITY_REPORT_BOXPLOT, "png")
    plotted_feature_count = len([column for column in BOXPLOT_CORE_FEATURES if column in before_frame.columns and not column.startswith(BOXPLOT_EXCLUDED_PREFIXES)])
    boxplot_description = (
        "Boxplots summarize the distribution of the core market features before cleaning so you can inspect scale, skew, and extreme values. "
        f"This figure covers {plotted_feature_count} plotted features: price, volume, VIX, returns, and volatility."
    )
    lines = [
        "# Week 2 Data Quality Report",
        "",
        f"- Observations analyzed before cleaning: {len(before_frame)}",
        f"- Observations after cleaning: {len(cleaned_frame)}",
        f"- Numeric features checked: {total_features}",
        f"- Features with missing values before cleaning: {missing_columns}",
        "",
        "## Missing-Value Strategy",
        "- Price fields (`Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`) use interpolation across the daily calendar because the raw files contain short calendar gaps and market non-trading days.",
        "- Macro series (`dgs10`, `vix`) use forward fill so the most recent observed level is carried forward without using future information.",
        "- Dividend features are forward-filled after quarterly aggregation because dividend values remain valid until the next announcement.",
        "- News counts and sentiment scores use neutral defaults when a day has no news, because zero activity and neutral sentiment are the least misleading assumptions.",
        "- Rolling features keep their warm-up rows missing until enough history exists, and any remaining incomplete rows are dropped before export.",
        "- Residual gaps are treated as incomplete observations and removed rather than guessed.",
        "",
        "## Outlier Strategy",
        "- Outliers are identified with the 3σ rule using mean ± 3 standard deviations.",
        "- Flagged values are replaced with the column median computed from inlier observations.",
        "- The boxplot figure is kept as a visualization aid so you can inspect the distribution of each numeric feature, but it is not the rule used for outlier detection.",
        "- News-derived features are excluded from the boxplot figure so the plot focuses on financial series with comparable numeric scales.",
        "",
        "## Visualizations",
        "",
        "### Numeric Feature Boxplots",
        f"Description: {boxplot_description}",
        f"![Numeric feature boxplots]({boxplot_filename})",
        "",
        "## Range Validation",
        f"- VIX is expected to stay above 0.",
        f"- JPM return features are checked against a -10% to 10% sanity band.",
        f"- Total range violations found: {range_violations}",
        "",
        "## Detailed Statistics",
        "- The full machine-readable table remains in the CSV export if you need the per-feature metrics, bounds, and fill strategy columns.",
        f"- See `{versioned_filename(QUALITY_REPORT_MD, 'md')}` for the editable source report.",
        f"- See `{versioned_filename(QUALITY_REPORT_PDF, 'pdf')}` for the PDF export rendered from the Markdown source.",
    ]
    return "\n".join(lines)


def build_feature_optimization_markdown(optimization_report: pd.DataFrame, corr_matrix: pd.DataFrame, selected_features: list[str]) -> str:
    selected_report = optimization_report[optimization_report["selected"]].copy()
    selected_count = len(selected_features)
    candidate_count = len(optimization_report)
    corr_pruned_count = int(optimization_report["corr_pruned"].sum())

    lines = [
        "# Week 2 Feature Engineering Optimization Report",
        "",
        f"- Candidate features evaluated: {candidate_count}",
        f"- Features removed by correlation pruning: {corr_pruned_count}",
        f"- Features kept after IC screening: {selected_count}",
        f"- Correlation threshold: {CORR_THRESHOLD}",
        f"- IC threshold: {IC_THRESHOLD}",
        "- IC horizons: future 1 week = 5 trading days, future 1 month = 21 trading days",
        "",
        "## What Changed",
        "- Trailing dividend features were converted to trailing dividend yield using current price, which is the standard BSM-style input.",
        "- Historical volatility now includes 5-day, 20-day, and 60-day windows.",
        "- Volatility change features were added using both the one-day difference and one-day percentage change of 20-day volatility.",
        "- Redundant features with absolute Pearson correlation above 0.8 were removed before IC screening.",
        "- Only features with absolute IC above 0.03 for at least one forecast horizon were retained.",
        "",
        "## Selected Features",
    ]
    lines.extend(f"- {feature}" for feature in selected_features)
    lines.extend([
        "",
        "## Correlation Matrix",
        "```text",
        corr_matrix.round(3).to_string(),
        "```",
        "",
        "## Feature Score Table",
        "```text",
        optimization_report.sort_values("max_abs_ic", ascending=False)[
            ["feature", "mean_abs_corr", "ic_1w", "ic_1m", "max_abs_ic", "corr_pruned", "selected", "drop_reason"]
        ].head(30).to_string(index=False),
        "```",
    ])
    if not selected_report.empty:
        lines.extend([
            "",
            "## IC-Passed Features",
            "```text",
            selected_report.sort_values("max_abs_ic", ascending=False)[["feature", "ic_1w", "ic_1m", "max_abs_ic"]].to_string(index=False),
            "```",
        ])

    return "\n".join(lines)


def main() -> dict[str, Path]:
    configure_logging()
    cleanup_generated_outputs()

    dataset_csv_name = versioned_filename(FEATURE_SELECTED_DATASET, DATASET_OUTPUT_EXTENSION)
    dataset_parquet_name = Path(dataset_csv_name).with_suffix(".parquet").name
    quality_report_csv_name = versioned_filename(QUALITY_REPORT_CSV, "csv")
    quality_report_md_name = versioned_filename(QUALITY_REPORT_MD, "md")
    quality_report_pdf_name = versioned_filename(QUALITY_REPORT_PDF, "pdf")
    quality_report_boxplot_name = versioned_filename(QUALITY_REPORT_BOXPLOT, "png")
    optimization_report_csv_name = versioned_filename(FEATURE_OPTIMIZATION_CSV, "csv")
    optimization_report_pdf_name = versioned_filename(FEATURE_OPTIMIZATION_PDF, "pdf")
    ic_report_name = versioned_filename(FEATURE_IC_REPORT, "csv")
    correlation_matrix_name = versioned_filename(FEATURE_CORRELATION_CSV, "csv")

    with log_step("Build optimized features"):
        features, quality_source, quality_report, optimization_report, ic_report, corr_matrix, selected_features = build_features()

    with log_step("Save processed outputs"):
        output_path = save_csv(features, dataset_csv_name)
        quality_report_path = save_csv(quality_report.round(6), quality_report_csv_name)
        optimization_report_path = save_csv(optimization_report.round(6), optimization_report_csv_name)
        ic_report_path = save_csv(ic_report.round(6), ic_report_name)
        corr_matrix_path = save_csv(corr_matrix.round(6), correlation_matrix_name)
        quality_markdown = build_quality_report_markdown(quality_source, features, quality_report)
        quality_report_md_path = save_markdown(quality_markdown, quality_report_md_name)
        quality_report_boxplot_path = save_figure(build_boxplot_figure(quality_source), quality_report_boxplot_name)
        pdf_path = save_pdf_report(quality_report_md_path, quality_report_pdf_name, asset_paths={quality_report_boxplot_path.name: quality_report_boxplot_path})
        optimization_markdown = build_feature_optimization_markdown(optimization_report, corr_matrix, selected_features)
        optimization_pdf_path = save_text_pdf_report(optimization_markdown, optimization_report_pdf_name)

    logger.info("Selected feature count: %s", len(selected_features))
    logger.info("Selected features: %s", ", ".join(selected_features))
    logger.info("Versioned dataset parquet name: %s", dataset_parquet_name)

    print(f"[OK] Week 2 feature dataset saved to {output_path.name}")
    print(f"[OK] Data quality report saved to {quality_report_path.name}")
    print(f"[OK] Data quality markdown saved to {quality_report_md_path.name}")
    print(f"[OK] Data quality boxplot saved to {quality_report_boxplot_path.name}")
    print(f"[OK] Feature optimization report saved to {optimization_report_path.name}")
    print(f"[OK] Feature IC report saved to {ic_report_path.name}")
    print(f"[OK] Feature correlation matrix saved to {corr_matrix_path.name}")
    print(f"[OK] Combined PDF report saved to {pdf_path.name}")
    print(f"[OK] Feature optimization PDF saved to {optimization_pdf_path.name}")
    print(f"[OK] Selected feature count: {len(selected_features)}")
    print(f"[OK] Selected features: {', '.join(selected_features)}")

    return {
        "dataset_csv": output_path,
        "quality_report_csv": quality_report_path,
        "quality_report_md": quality_report_md_path,
        "quality_report_boxplot": quality_report_boxplot_path,
        "optimization_report_csv": optimization_report_path,
        "ic_report_csv": ic_report_path,
        "correlation_matrix_csv": corr_matrix_path,
        "quality_report_pdf": pdf_path,
        "optimization_report_pdf": optimization_pdf_path,
    }


if __name__ == "__main__":
    main()

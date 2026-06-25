from __future__ import annotations

import logging
import re
from pathlib import Path
from datetime import datetime
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


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
FEATURE_OPTIMIZATION_SUMMARY = "week2_feature_optimization_summary"
FEATURE_OPTIMIZATION_CORR_HEATMAP = "week2_feature_optimization_corr_heatmap"
FEATURE_OPTIMIZATION_IC_BARS = "week2_feature_optimization_ic_bars"
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
REPORT_FONT = "STSong-Light"

try:
    pdfmetrics.registerFont(UnicodeCIDFont(REPORT_FONT))
except Exception:
    pass


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


def save_markdown_pdf_report(markdown_path: Path, filename: str, title: str, asset_paths: dict[str, Path] | None = None) -> Path:
    ensure_output_dir()
    output_path = REPORTS_DIR / filename
    markdown_text = markdown_path.read_text(encoding="utf-8")

    build_markdown_pdf(output_path, markdown_text, title=title, asset_paths=asset_paths)

    return output_path


def escape_reportlab_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def formula_to_reportlab_markup(value: str) -> str:
    text = value.strip()
    if text.startswith("$$") and text.endswith("$$"):
        text = text[2:-2].strip()

    text = text.replace(r"\left", "")
    text = text.replace(r"\right", "")
    text = text.replace(r"\times", "*")
    text = text.replace(r"\exp", "exp")
    text = text.replace(r"\sqrt", "sqrt")
    text = text.replace(r"\,", " ")
    text = text.replace(r"\;", " ")
    text = text.replace(r"\:", " ")
    text = text.replace(r"\quad", " ")
    text = text.replace(r"\qquad", " ")

    text = re.sub(r"([A-Za-z0-9]+)_\{([^{}]+)\}", r"\1<sub>\2</sub>", text)
    text = re.sub(r"([A-Za-z0-9]+)_([A-Za-z0-9]+)", r"\1<sub>\2</sub>", text)
    text = re.sub(r"([A-Za-z0-9]+)\^\{([^{}]+)\}", r"\1<super>\2</super>", text)
    text = re.sub(r"([A-Za-z0-9]+)\^([A-Za-z0-9.+\-()/*]+)", r"\1<super>\2</super>", text)

    text = text.replace("\\", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    return text


def markdown_table_to_flowable(table_lines: list[str], styles: dict[str, ParagraphStyle], page_width: float) -> Table | None:
    def split_row(row: str) -> list[str]:
        return [cell.strip() for cell in row.strip().strip("|").split("|")]

    rows = [split_row(line) for line in table_lines if line.strip().startswith("|") and line.strip().endswith("|")]
    if not rows:
        return None

    headers = rows[0]
    body_rows: list[list[str]] = []
    for row in rows[1:]:
        if all(set(cell) <= {"-", ":"} for cell in row):
            continue
        normalized = row[: len(headers)] + [""] * max(0, len(headers) - len(row))
        body_rows.append(normalized)

    table_data: list[list[Paragraph]] = []
    table_data.append([Paragraph(escape_reportlab_text(cell), styles["table_header"]) for cell in headers])
    for row in body_rows:
        table_data.append([Paragraph(escape_reportlab_text(cell), styles["table_cell"]) for cell in row])

    column_count = max(len(headers), 1)
    available_width = page_width - 1.0 * inch
    if column_count == 1:
        col_widths = [available_width]
    else:
        first_column_ratio = 0.24 if column_count >= 2 else 1.0
        other_ratio = (1.0 - first_column_ratio) / (column_count - 1) if column_count > 1 else 0.0
        col_widths = [available_width * first_column_ratio] + [available_width * other_ratio for _ in range(column_count - 1)]

    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#203864")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), REPORT_FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef3f8")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b7c0ce")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_markdown_pdf(output_path: Path, markdown_text: str, title: str, asset_paths: dict[str, Path] | None = None) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName=REPORT_FONT,
            fontSize=18,
            leading=22,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading1",
            parent=styles["Heading1"],
            fontName=REPORT_FONT,
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading2",
            parent=styles["Heading2"],
            fontName=REPORT_FONT,
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#334155"),
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading3",
            parent=styles["Heading3"],
            fontName=REPORT_FONT,
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#475569"),
            spaceBefore=6,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName=REPORT_FONT,
            fontSize=9.5,
            leading=12,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportBullet",
            parent=styles["BodyText"],
            fontName=REPORT_FONT,
            fontSize=9.5,
            leading=12,
            leftIndent=12,
            firstLineIndent=-8,
            bulletIndent=0,
            spaceAfter=2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportCaption",
            parent=styles["BodyText"],
            fontName=REPORT_FONT,
            fontSize=8.5,
            leading=10,
            textColor=colors.HexColor("#4b5563"),
            spaceBefore=2,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportFormula",
            parent=styles["BodyText"],
            fontName=REPORT_FONT,
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="table_header",
            parent=styles["BodyText"],
            fontName=REPORT_FONT,
            fontSize=8.5,
            leading=10,
            textColor=colors.white,
        )
    )
    styles.add(
        ParagraphStyle(
            name="table_cell",
            parent=styles["BodyText"],
            fontName=REPORT_FONT,
            fontSize=8.5,
            leading=10,
        )
    )

    story: list[object] = [Paragraph(escape_reportlab_text(title), styles["ReportTitle"]), Spacer(1, 0.08 * inch)]
    lines = markdown_text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()

        if not stripped:
            story.append(Spacer(1, 0.08 * inch))
            index += 1
            continue

        if stripped.startswith("# ") and stripped[2:].strip() == title:
            index += 1
            continue

        if stripped.startswith("# "):
            story.append(Paragraph(escape_reportlab_text(stripped[2:].strip()), styles["ReportHeading1"]))
            index += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(escape_reportlab_text(stripped[3:].strip()), styles["ReportHeading2"]))
            index += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(escape_reportlab_text(stripped[4:].strip()), styles["ReportHeading3"]))
            index += 1
            continue

        if stripped.startswith("![") and stripped.endswith(")"):
            match = IMAGE_MARKDOWN_PATTERN.match(stripped)
            if match is not None:
                image_reference = match.group("path")
                caption = match.group("alt") or None
                image_path = None
                if asset_paths is not None:
                    image_path = asset_paths.get(image_reference) or asset_paths.get(Path(image_reference).name)
                if image_path is not None and image_path.exists():
                    image = RLImage(str(image_path))
                    max_width = letter[0] - 1.2 * inch
                    max_height = 4.8 * inch
                    scale = min(max_width / image.drawWidth, max_height / image.drawHeight, 1.0)
                    image.drawWidth *= scale
                    image.drawHeight *= scale
                    story.append(image)
                    if caption:
                        story.append(Paragraph(escape_reportlab_text(caption), styles["ReportCaption"]))
                    story.append(Spacer(1, 0.08 * inch))
            index += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = [stripped]
            index += 1
            while index < len(lines) and lines[index].strip().startswith("|") and lines[index].strip().endswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            table = markdown_table_to_flowable(table_lines, styles, letter[0])
            if table is not None:
                story.append(table)
                story.append(Spacer(1, 0.1 * inch))
            continue

        if stripped.startswith("- "):
            story.append(Paragraph(f"• {escape_reportlab_text(stripped[2:].strip())}", styles["ReportBullet"]))
            index += 1
            continue

        if stripped.startswith("$$") and stripped.endswith("$$"):
            story.append(Paragraph(formula_to_reportlab_markup(stripped), styles["ReportFormula"]))
            index += 1
            continue

        story.append(Paragraph(escape_reportlab_text(stripped), styles["ReportBody"]))
        index += 1

    def add_page_number(canvas, doc) -> None:  # noqa: ANN001
        canvas.saveState()
        canvas.setFont(REPORT_FONT, 8)
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.drawRightString(letter[0] - 0.5 * inch, 0.4 * inch, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.6 * inch,
        title=title,
        author="GitHub Copilot",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def build_feature_optimization_figure(optimization_report: pd.DataFrame, selected_features: list[str]) -> plt.Figure:
    selected_report = optimization_report[optimization_report["selected"]].copy()
    candidate_count = len(optimization_report)
    corr_pruned_count = int(optimization_report["corr_pruned"].sum())
    selected_count = len(selected_features)
    ic_pass_count = int(optimization_report["selected"].sum())

    fig = plt.figure(figsize=(12, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    counts_ax = fig.add_subplot(grid[0, 0])
    stages = ["Candidates", "After corr pruning", "IC-passed", "Final selected"]
    stage_values = [candidate_count, candidate_count - corr_pruned_count, ic_pass_count, selected_count]
    stage_colors = ["#2b6cb0", "#4299e1", "#38a169", "#d69e2e"]
    counts_ax.bar(stages, stage_values, color=stage_colors)
    counts_ax.set_title("Feature Selection Funnel", loc="left", fontsize=12, fontweight="bold")
    counts_ax.set_ylabel("Feature count")
    counts_ax.set_ylim(0, max(stage_values + [1]) * 1.2)
    counts_ax.tick_params(axis="x", labelrotation=20)
    for index, value in enumerate(stage_values):
        counts_ax.text(index, value + max(stage_values) * 0.03 if max(stage_values) else 0.05, str(value), ha="center", va="bottom", fontsize=9)

    threshold_ax = fig.add_subplot(grid[0, 1])
    threshold_ax.axis("off")
    threshold_text = (
        f"Correlation threshold: {CORR_THRESHOLD}\n"
        f"IC threshold: {IC_THRESHOLD}\n"
        f"Candidate features evaluated: {candidate_count}\n"
        f"Features removed by correlation pruning: {corr_pruned_count}\n"
        f"Final selected features: {selected_count}"
    )
    threshold_ax.text(0.02, 0.98, threshold_text, va="top", ha="left", fontsize=12, linespacing=1.7)
    threshold_ax.set_title("Screening Summary", loc="left", fontsize=12, fontweight="bold")

    feature_ax = fig.add_subplot(grid[1, :])
    if selected_report.empty:
        feature_ax.text(0.5, 0.5, "No features passed both filters", ha="center", va="center", fontsize=12)
        feature_ax.set_axis_off()
    else:
        top_features = selected_report.sort_values("max_abs_ic", ascending=True)
        y_positions = list(range(len(top_features)))
        feature_ax.barh(y_positions, top_features["max_abs_ic"], color="#4a5568")
        feature_ax.set_yticks(y_positions)
        feature_ax.set_yticklabels(top_features["feature"])
        feature_ax.set_xlabel("Max absolute IC")
        feature_ax.set_title("Retained Features Ranked by Predictive Strength", loc="left", fontsize=12, fontweight="bold")
        feature_ax.axvline(IC_THRESHOLD, color="#c53030", linestyle="--", linewidth=1.2, label="IC threshold")
        feature_ax.legend(loc="lower right", frameon=False)
        feature_ax.grid(axis="x", linestyle=":", alpha=0.35)

    fig.suptitle("Week 2 Feature Engineering Optimization Summary", fontsize=15, fontweight="bold")
    return fig


def build_feature_optimization_corr_heatmap(
    optimization_report: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    selected_features: list[str],
    top_n: int = 8,
) -> plt.Figure:
    if corr_matrix.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "Correlation matrix is empty", ha="center", va="center", fontsize=12)
        fig.suptitle("Pearson Correlation Heatmap", fontsize=15, fontweight="bold")
        return fig

    pruned_features = optimization_report.loc[optimization_report["corr_pruned"], "feature"].tolist()
    if not pruned_features:
        pruned_features = selected_features[:top_n]

    partner_features: list[str] = []
    for feature in pruned_features:
        if feature not in corr_matrix.index:
            continue
        candidate_correlations = corr_matrix.loc[feature].drop(labels=[feature], errors="ignore").abs().sort_values(ascending=False)
        if candidate_correlations.empty:
            continue
        partner_features.append(candidate_correlations.index[0])

    heatmap_features = list(dict.fromkeys(pruned_features + partner_features))
    heatmap_features = [feature for feature in heatmap_features if feature in corr_matrix.columns]
    if not heatmap_features:
        heatmap_features = list(corr_matrix.columns[: min(len(corr_matrix.columns), top_n)])

    matrix = corr_matrix.loc[heatmap_features, heatmap_features]
    feature_count = len(matrix.columns)
    size = max(10.5, min(18.0, 0.52 * feature_count + 5.0))

    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")

    import numpy as np

    mask = np.tril(np.ones_like(matrix.values, dtype=bool), k=-1)
    display_matrix = np.ma.array(matrix.values, mask=mask)

    image = ax.imshow(display_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", interpolation="nearest")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, ha="center", fontsize=max(6.5, min(9.0, 150.0 / max(feature_count, 1))))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=max(6.5, min(9.0, 150.0 / max(feature_count, 1))))

    ax.set_xticks([x - 0.5 for x in range(len(matrix.columns) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(matrix.index) + 1)], minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)

    for row_index, row_name in enumerate(matrix.index):
        for col_index, col_name in enumerate(matrix.columns):
            if row_index > col_index:
                continue
            value = matrix.loc[row_name, col_name]
            text_color = "white" if abs(value) >= 0.58 else "#1f2937"
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=max(6.0, min(8.0, 115.0 / max(feature_count, 1))), color=text_color, fontweight="bold")

    for index, feature in enumerate(matrix.columns):
        if feature in pruned_features:
            ax.get_xticklabels()[index].set_color("#c53030")
            ax.get_yticklabels()[index].set_color("#c53030")
            ax.get_xticklabels()[index].set_fontweight("bold")
            ax.get_yticklabels()[index].set_fontweight("bold")

    if pruned_features:
        boundary = len(pruned_features) - 0.5
        ax.axvline(boundary, color="#718096", linewidth=1.2, linestyle="--")
        ax.axhline(boundary, color="#718096", linewidth=1.2, linestyle="--")
        ax.text(
            min(boundary + 0.2, feature_count - 0.3),
            -0.95,
            "kept features",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#2f855a",
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            0.0,
            -0.95,
            "pruned features",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#c53030",
            fontweight="bold",
            clip_on=False,
        )

    fig.suptitle("Correlation Heatmap", fontsize=13, fontweight="bold", x=0.19, y=0.975, ha="left")
    fig.text(
        0.19,
        0.945,
        f"Pruned features are grouped first and the lower triangle is hidden to reduce visual noise. Cells with |rho| > {CORR_THRESHOLD} are the pruning targets.",
        ha="left",
        va="top",
        fontsize=9.2,
        color="#4a5568",
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.18, right=0.96)
    return fig


def build_feature_optimization_ic_bars(ic_report: pd.DataFrame, selected_features: list[str]) -> plt.Figure:
    if ic_report.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "IC report is empty", ha="center", va="center", fontsize=12)
        fig.suptitle("IC Bar Chart", fontsize=15, fontweight="bold")
        return fig

    plot_frame = ic_report.copy().sort_values("max_abs_ic", ascending=True).reset_index(drop=True)
    y_positions = list(range(len(plot_frame)))
    bar_height = 0.36
    selected_set = set(selected_features)

    fig_height = max(8.5, 0.34 * len(plot_frame) + 3.2)
    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)

    colors_1w = ["#2b6cb0" if feature in selected_set else "#a0aec0" for feature in plot_frame["feature"]]
    colors_1m = ["#38a169" if feature in selected_set else "#cbd5e0" for feature in plot_frame["feature"]]

    ax.barh([y - bar_height / 2 for y in y_positions], plot_frame["ic_1w"], height=bar_height, color=colors_1w, label="1-week IC (5 trading days)")
    ax.barh([y + bar_height / 2 for y in y_positions], plot_frame["ic_1m"], height=bar_height, color=colors_1m, label="1-month IC (21 trading days)")
    ax.axvline(IC_THRESHOLD, color="#c53030", linestyle="--", linewidth=1.2, label="+IC threshold")
    ax.axvline(-IC_THRESHOLD, color="#c53030", linestyle="--", linewidth=1.2, label="-IC threshold")
    ax.axvline(0.0, color="#4a5568", linewidth=1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_frame["feature"], fontsize=8)
    ax.set_xlabel("IC value")
    ax.set_title("IC by Feature and Horizon", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#2b6cb0", edgecolor="none", label="1-week IC for selected features"),
        Patch(facecolor="#38a169", edgecolor="none", label="1-month IC for selected features"),
        Patch(facecolor="#cbd5e0", edgecolor="none", label="1-month IC for unselected features"),
        Patch(facecolor="#a0aec0", edgecolor="none", label="1-week IC for unselected features"),
        Patch(facecolor="none", edgecolor="#c53030", linestyle="--", label="|IC| threshold at 0.03"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8)
    ax.set_xlim(min(-0.08, float(plot_frame[["ic_1w", "ic_1m"]].min().min()) - 0.01), max(0.08, float(plot_frame[["ic_1w", "ic_1m"]].max().max()) + 0.01))
    ax.text(
        0.01,
        1.02,
        "Blue bars show 1-week IC, green bars show 1-month IC, and gray bars mean the feature did not pass the final selection step.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
    )
    fig.suptitle("Week 2 IC Screening Overview", fontsize=15, fontweight="bold")
    return fig


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
    return save_markdown_pdf_report(markdown_path, filename, title="Week 2 Data Quality Report", asset_paths=asset_paths)


def save_text_pdf_report(report_text: str, filename: str, title: str = "Week 2 Feature Engineering Optimization Report") -> Path:
    ensure_output_dir()
    output_path = REPORTS_DIR / filename

    with PdfPages(output_path) as pdf:
        render_text_report(pdf, report_text, title=title)

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


def is_markdown_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def is_markdown_table_separator(row: str) -> bool:
    stripped = row.strip().strip("|")
    if not stripped:
        return False
    cells = [cell.strip() for cell in stripped.split("|")]
    return all(cell and set(cell) <= {"-", ":"} for cell in cells)


def parse_markdown_table_rows(table_lines: list[str]) -> tuple[list[str], list[list[str]]]:
    def split_row(row: str) -> list[str]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        return cells

    rows = [split_row(line) for line in table_lines if is_markdown_table_row(line)]
    if not rows:
        return [], []

    headers = rows[0]
    body_rows: list[list[str]] = []
    for row in rows[1:]:
        if len(row) == len(headers) and any(set(cell) <= {"-", ":"} for cell in row):
            continue
        normalized_row = row[: len(headers)] + [""] * max(0, len(headers) - len(row))
        body_rows.append(normalized_row)

    return headers, body_rows


def wrap_table_cell(value: object, max_width: int) -> str:
    text = format(value, ".6f") if isinstance(value, float) else str(value)
    if not text:
        return ""
    if len(text) <= max_width:
        return text
    return textwrap.fill(text, width=max_width, break_long_words=False, break_on_hyphens=False)


def render_table_pages(pdf: PdfPages, headers: list[str], rows: list[list[str]], title: str, subtitle: str | None = None) -> None:
    if not headers:
        return

    total_rows = len(rows)
    if total_rows == 0:
        rows = [["No rows available" for _ in headers]]
        total_rows = 1

    max_rows_per_page = 18 if len(headers) <= 4 else 15 if len(headers) <= 8 else 12
    available_width = max(10, min(28, 88 // max(len(headers), 1)))

    for start_index in range(0, total_rows, max_rows_per_page):
        end_index = min(start_index + max_rows_per_page, total_rows)
        page_rows = rows[start_index:end_index]
        figure, axis = plt.subplots(figsize=(8.5, 11))
        axis.axis("off")

        figure.text(0.05, 0.975, title, ha="left", va="top", fontsize=18, fontweight="bold")
        if subtitle:
            figure.text(0.05, 0.935, subtitle, ha="left", va="top", fontsize=10, color="#444444")
        figure.text(
            0.05,
            0.905,
            f"Rows {start_index + 1}-{end_index} of {total_rows}",
            ha="left",
            va="top",
            fontsize=9,
            color="#666666",
        )

        wrapped_rows = [[wrap_table_cell(cell, available_width) for cell in row] for row in page_rows]
        table = axis.table(
            cellText=wrapped_rows,
            colLabels=[wrap_table_cell(header, available_width) for header in headers],
            cellLoc="left",
            colLoc="left",
            loc="upper center",
            bbox=[0.04, 0.09, 0.92, 0.78],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8 if len(headers) <= 6 else 7.2)

        for (row_index, col_index), cell in table.get_celld().items():
            cell.set_edgecolor("#b7c0ce")
            cell.set_linewidth(0.6)
            cell.PAD = 0.08
            if row_index == 0:
                cell.set_facecolor("#203864")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f6f8fb")

        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def render_markdown_table_block(pdf: PdfPages, table_lines: list[str], title: str, subtitle: str | None = None) -> None:
    headers, rows = parse_markdown_table_rows(table_lines)
    render_table_pages(pdf, headers, rows, title, subtitle=subtitle)


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

    lines = report_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            y -= 0.012
            index += 1
            continue

        if not skipped_title_line and stripped.startswith("# ") and stripped[2:].strip() == title:
            skipped_title_line = True
            index += 1
            continue

        if stripped.startswith("```"):
            index += 1
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
            index += 1
            continue

        if is_markdown_table_row(stripped):
            table_lines = [stripped]
            index += 1
            while index < len(lines) and is_markdown_table_row(lines[index]):
                table_lines.append(lines[index].strip())
                index += 1

            if y < 0.3:
                flush_page()
            else:
                flush_page()

            render_markdown_table_block(pdf, table_lines, title)

            report_fig = plt.figure(figsize=(8.5, 11))
            report_ax = report_fig.add_axes([0, 0, 1, 1])
            report_ax.axis("off")
            report_fig.text(0.05, 0.975, title, ha="left", va="top", fontsize=18, fontweight="bold")
            y = 0.95
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
    top_selected_features = (
        selected_report.sort_values("max_abs_ic", ascending=False)["feature"].head(5).tolist()
        if not selected_report.empty
        else []
    )
    top_selected_text = ", ".join(top_selected_features) if top_selected_features else "no features passed both filters"
    corr_summary = (
        "The correlation matrix shows that some candidate features are highly redundant, so correlation pruning runs before IC screening."
        if not corr_matrix.empty
        else "The correlation matrix is empty, which means the current candidate set does not contain enough comparable features."
    )

    lines = [
        "# Week 2 Feature Engineering Optimization Report",
        "",
        f"This stage evaluated {candidate_count} candidate features and finally kept {selected_count} features for the modeling dataset.",
        f"Before IC screening, {corr_pruned_count} features were removed because they were too similar to other candidates, using a Pearson correlation threshold of {CORR_THRESHOLD}.",
        f"The IC filter then kept only features with absolute IC above {IC_THRESHOLD} on at least one horizon, where the horizons are 5 trading days for the 1-week target and 21 trading days for the 1-month target.",
        "",
        f"![Optimization summary]({versioned_filename(FEATURE_OPTIMIZATION_SUMMARY, 'png')})",
        "",
        "## Visualizations",
        "",
        "### Correlation Heatmap",
        "This heatmap groups the features removed by correlation pruning at the front, hides the redundant lower triangle, and keeps the figure focused on the variables excluded for exceeding the 0.8 threshold. Cells with absolute correlation above 0.8 are the main pruning targets.",
        f"![Correlation heatmap]({versioned_filename(FEATURE_OPTIMIZATION_CORR_HEATMAP, 'png')})",
        "",
        "### IC Bar Chart",
        "This chart compares each feature's IC at the 1-week and 1-month horizons. Blue bars show the 1-week IC, green bars show the 1-month IC, and gray bars indicate features that did not pass the final selection step. The dashed lines mark the +/- 0.03 threshold used for selection.",
        f"![IC bar chart]({versioned_filename(FEATURE_OPTIMIZATION_IC_BARS, 'png')})",
        "",
        "## What Changed",
        "The feature set was redesigned to make the inputs more informative and less redundant. Trailing dividend features were transformed into dividend yield so they are comparable across time. Volatility was expanded to 5-day, 20-day, and 60-day windows, and the 20-day volatility series also gained level-change and rate-change signals. These changes make the feature set more expressive without turning it into a long list of nearly duplicated variables.",
        "",
        "The correlation screen removed features that were overlapping too heavily with others, so the remaining set is easier to interpret and less likely to double-count the same information. " + corr_summary,
        "",
        f"After the IC check, the surviving features were the ones that showed a meaningful relationship with future returns. The strongest retained names were {top_selected_text}.",
        "",
        "## Interpretation",
        "Overall, this report shows that the pipeline moved from a broad raw candidate pool to a compact set of features centered on price momentum, volatility behavior, rate dynamics, VIX interaction, dividend yield, and news activity. In other words, the final feature set is meant to be explained in sentences rather than read as a spreadsheet dump.",
    ]
    if not selected_report.empty:
        lines.extend([
            "",
            "## Final Note",
            "The selected feature set is not exhaustive of every raw input; it is the compact subset that passed both the correlation and IC filters, and it is intended for downstream modeling rather than manual inspection of raw columns.",
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
    optimization_summary_name = versioned_filename(FEATURE_OPTIMIZATION_SUMMARY, "png")
    optimization_corr_heatmap_name = versioned_filename(FEATURE_OPTIMIZATION_CORR_HEATMAP, "png")
    optimization_ic_bars_name = versioned_filename(FEATURE_OPTIMIZATION_IC_BARS, "png")
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
        optimization_summary_path = save_figure(build_feature_optimization_figure(optimization_report, selected_features), optimization_summary_name)
        optimization_corr_heatmap_path = save_figure(build_feature_optimization_corr_heatmap(optimization_report, corr_matrix, selected_features), optimization_corr_heatmap_name)
        optimization_ic_bars_path = save_figure(build_feature_optimization_ic_bars(ic_report, selected_features), optimization_ic_bars_name)
        optimization_markdown = build_feature_optimization_markdown(optimization_report, corr_matrix, selected_features)
        optimization_pdf_path = save_pdf_report(
            save_markdown(optimization_markdown, versioned_filename(FEATURE_OPTIMIZATION_CSV, "md")),
            optimization_report_pdf_name,
            asset_paths={
                optimization_summary_path.name: optimization_summary_path,
                optimization_corr_heatmap_path.name: optimization_corr_heatmap_path,
                optimization_ic_bars_path.name: optimization_ic_bars_path,
            },
        )

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
    print(f"[OK] Feature optimization summary saved to {optimization_summary_path.name}")
    print(f"[OK] Feature optimization correlation heatmap saved to {optimization_corr_heatmap_path.name}")
    print(f"[OK] Feature optimization IC bars saved to {optimization_ic_bars_path.name}")
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
        "optimization_summary": optimization_summary_path,
        "optimization_corr_heatmap": optimization_corr_heatmap_path,
        "optimization_ic_bars": optimization_ic_bars_path,
        "ic_report_csv": ic_report_path,
        "correlation_matrix_csv": corr_matrix_path,
        "quality_report_pdf": pdf_path,
        "optimization_report_pdf": optimization_pdf_path,
    }


if __name__ == "__main__":
    main()

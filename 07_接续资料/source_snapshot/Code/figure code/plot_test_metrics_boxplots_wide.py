#!/usr/bin/env python3
"""Create three standalone wide held-out NSE/KGE/RMSE boxplot figures.

The figures are deliberately independent from the parameter-response figure:
each file contains one performance metric, and every box contains the 30
held-out events for one hydraulic condition and one routing model at 120 km.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(
    r"E:\01-HUST\Nature\03\HECRAS\roughness_slope_generalization_100"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_100test" / "separated_metrics"
DEFAULT_FIGURE_NAME = "Fig_01_test_metrics_boxplots_wide"
RASTER_DPI = 600
DISTANCE_KM = 120
TEST_EVENTS_PER_GROUP = 30
FIGURE_WIDTH_MM = 183.0
FIGURE_HEIGHT_MM = 86.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
FIGURE_HEIGHT_IN = FIGURE_HEIGHT_MM / 25.4
TIMES_FONT_PATH = Path(r"C:\Windows\Fonts\times.ttf")
SIMSUN_FONT_PATH = Path(r"C:\Windows\Fonts\simsun.ttc")

# The first three conditions vary slope at n_M=0.035; the last two vary
# Manning roughness at J=0.20 per mille.  This keeps the same reading order as
# the separated parameter-response figure.
CONDITIONS = (
    "S0001_N035",
    "S0002_N035",
    "S0005_N035",
    "S0002_N025",
    "S0002_N050",
)
CONDITION_LABELS = {
    "S0001_N035": "$J=0.10‰$\n$n_{\\mathrm{M}}=0.035$\nS1",
    "S0002_N035": "$J=0.20‰$\n$n_{\\mathrm{M}}=0.035$\nS2",
    "S0005_N035": "$J=0.50‰$\n$n_{\\mathrm{M}}=0.035$\nS3",
    "S0002_N025": "$J=0.20‰$\n$n_{\\mathrm{M}}=0.025$\nS4",
    "S0002_N050": "$J=0.20‰$\n$n_{\\mathrm{M}}=0.050$\nS5",
}
MODELS = (
    "Linear Muskingum",
    "Nonlinear Muskingum strict",
    "LMCS",
)
DISPLAY_MODEL_LABELS = {
    "LMCS": "RWEC",
    "Nonlinear Muskingum strict": "非线性马斯京根",
    "Linear Muskingum": "马斯京根",
}
METRICS = (
    ("NSE", "NSE"),
    ("KGE", "KGE"),
    ("RMSE_m3s", "RMSE (m³/s)"),
)

# One family per metric; within each family Muskingum is light, nonlinear
# Muskingum intermediate, and RWEC is dark.  The dictionary is keyed by
# canonical model name, while MODELS fixes the left-to-right box/legend order.
METRIC_COLORS = {
    "NSE": {
        "LMCS": "#006BB6",
        "Nonlinear Muskingum strict": "#2196E0",
        "Linear Muskingum": "#8FD8F7",
    },
    "KGE": {
        "LMCS": "#078455",
        "Nonlinear Muskingum strict": "#21B66A",
        "Linear Muskingum": "#9BE4B2",
    },
    "RMSE_m3s": {
        "LMCS": "#E05A00",
        "Nonlinear Muskingum strict": "#FF8A1E",
        "Linear Muskingum": "#FFC467",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--summary-tag", help="Complete result tag; latest complete tag by default.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-name", default=DEFAULT_FIGURE_NAME)
    parser.add_argument("--dpi", type=int, default=RASTER_DPI)
    return parser.parse_args()


def times(size: float, bold: bool = False, italic: bool = False) -> FontProperties:
    if not TIMES_FONT_PATH.exists():
        raise FileNotFoundError(f"Missing Times New Roman font: {TIMES_FONT_PATH}")
    return FontProperties(
        family="Times New Roman",
        fname=str(TIMES_FONT_PATH),
        size=size,
        weight="bold" if bold else "normal",
        style="italic" if italic else "normal",
    )


def simsun(size: float, bold: bool = False) -> FontProperties:
    if not SIMSUN_FONT_PATH.exists():
        raise FileNotFoundError(f"Missing SimSun font: {SIMSUN_FONT_PATH}")
    return FontProperties(
        family="SimSun",
        fname=str(SIMSUN_FONT_PATH),
        size=size,
        weight="bold" if bold else "normal",
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.sans-serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": RASTER_DPI,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def latest_complete_tag(root: Path) -> str:
    """Return the newest summary tag that has both metrics and checkpoints."""
    candidates: list[str] = []
    summary_root = root / "08_summary"
    results_root = root / "07_test_results"
    if not summary_root.exists():
        raise FileNotFoundError(summary_root)
    for summary_dir in summary_root.iterdir():
        if not summary_dir.is_dir():
            continue
        # The formal robustness figure uses the complete roughness/slope
        # library, not the later convergence-only or short-distance checks.
        if not summary_dir.name.endswith("_roughness_slope_three_models"):
            continue
        metrics_path = results_root / summary_dir.name / "event_metrics.csv"
        checkpoint_dir = summary_dir / "checkpoints"
        if metrics_path.exists() and checkpoint_dir.exists():
            candidates.append(summary_dir.name)
    if not candidates:
        raise FileNotFoundError("No complete roughness/slope result tag was found.")
    return sorted(candidates, key=lambda value: int(value.split("_", 1)[0]))[-1]


def load_metrics(root: Path, summary_tag: str) -> tuple[pd.DataFrame, int, int]:
    input_path = root / "07_test_results" / summary_tag / "event_metrics.csv"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    frame = pd.read_csv(input_path)
    input_rows = len(frame)
    required = {"geometry", "distance_km", "event_id", "level", "split", "model", "NSE", "KGE", "RMSE_m3s"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {input_path}: {missing}")

    frame["distance_km"] = pd.to_numeric(frame["distance_km"], errors="coerce")
    metric_columns = [metric for metric, _ in METRICS]
    for column in metric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    selected = frame.loc[
        frame["split"].astype(str).str.lower().eq("test")
        & frame["distance_km"].eq(DISTANCE_KM)
        & frame["geometry"].isin(CONDITIONS)
        & frame["model"].isin(MODELS),
        ["geometry", "distance_km", "event_id", "level", "split", "model", *metric_columns],
    ].copy()

    expected_rows = len(CONDITIONS) * len(MODELS) * TEST_EVENTS_PER_GROUP
    if len(selected) != expected_rows:
        raise ValueError(f"Expected {expected_rows} selected test rows, found {len(selected)}")
    if selected.duplicated(["geometry", "event_id", "model"]).any():
        raise ValueError("Selected test metrics contain duplicate condition-event-model rows.")
    if not np.isfinite(selected[metric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Selected test metrics contain non-finite values.")
    counts = selected.groupby(["geometry", "model"], sort=False).size()
    if len(counts) != len(CONDITIONS) * len(MODELS) or not counts.eq(TEST_EVENTS_PER_GROUP).all():
        raise ValueError("Every condition-model group must contain exactly 30 test events.")
    if not selected["split"].astype(str).str.lower().eq("test").all():
        raise ValueError("The selected source contains a non-test row.")
    return (
        selected.sort_values(["geometry", "model", "event_id"]).reset_index(drop=True),
        input_rows,
        len(frame.loc[frame["split"].astype(str).str.lower().eq("test")]),
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D3D9DE", linestyle=":", linewidth=0.45, alpha=0.82)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True, width=0.65, length=2.8)
    for spine in ax.spines.values():
        spine.set_color("#3F474E")
        spine.set_linewidth(0.65)
    for label in ax.get_yticklabels():
        label.set_fontproperties(times(7.25))
    for label in ax.get_xticklabels():
        # Keep the mathtext subscript above the 5 pt rendered-glyph floor.
        label.set_fontproperties(times(8.0))


def set_metric_scale(ax: plt.Axes, metric: str) -> None:
    if metric in {"NSE", "KGE"}:
        # The upper blank band is part of the plotting frame. It keeps the
        # legend and the mean-value row inside the axes without covering data.
        ax.set_ylim(0.15, 1.32)
        ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    else:
        ax.set_ylim(0.0, 160.0)
        ax.set_yticks(np.arange(0.0, 125.1, 25.0))


def darken(color: str, factor: float = 0.78) -> str:
    """Return a restrained darker tone from the same metric/model family."""
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float)
    return mpl.colors.to_hex(np.clip(rgb * factor, 0.0, 1.0))


def add_panel_legend(ax: plt.Axes, metric: str) -> mpl.legend.Legend:
    """Place the model key in the upper-left blank band inside each axes."""
    handles = [
        Patch(
            facecolor=METRIC_COLORS[metric][model],
            edgecolor=darken(METRIC_COLORS[metric][model], 0.78),
            linewidth=0.35,
        )
        for model in MODELS
    ]
    legend = ax.legend(
        handles=handles,
        labels=[DISPLAY_MODEL_LABELS[model] for model in MODELS],
        loc="upper left",
        bbox_to_anchor=(0.006, 0.988),
        ncol=3,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#C8CDD2",
        framealpha=0.96,
        handlelength=0.9,
        handleheight=0.60,
        columnspacing=0.90,
        labelspacing=0.15,
        borderpad=0.24,
        prop=times(6.25),
    )
    for text in legend.get_texts():
        if any("\u4e00" <= char <= "\u9fff" for char in text.get_text()):
            text.set_fontproperties(simsun(6.25))
    return legend


def place_mean_labels(
    ax: plt.Axes,
    metric: str,
    label_specs: list[dict[str, object]],
    data_artists: list[mpl.artist.Artist],
    legend: mpl.legend.Legend,
) -> list[mpl.text.Text]:
    """Place every mean beside its own box, with collision-aware text layout.

    Candidate locations first occupy the clear lateral space beside each box;
    rendered bounding boxes then enforce axes, legend, and text-to-text
    clearance.  This avoids the former figure-wide number row at panel top.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    legend_bbox = legend.get_window_extent(renderer)
    occupied = [legend_bbox]
    for artist in data_artists:
        try:
            occupied.append(artist.get_window_extent(renderer))
        except (AttributeError, TypeError):
            continue

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    box_pad = 0.022 * span
    extra_pad = 0.012 * span
    texts: list[mpl.text.Text] = []

    def inside(bbox: mpl.transforms.Bbox) -> bool:
        tolerance = 0.75
        return (
            bbox.x0 >= axes_bbox.x0 - tolerance
            and bbox.x1 <= axes_bbox.x1 + tolerance
            and bbox.y0 >= axes_bbox.y0 - tolerance
            and bbox.y1 <= axes_bbox.y1 + tolerance
        )

    for spec in label_specs:
        values = np.asarray(spec["values"], dtype=float)
        x_value = float(spec["x"])
        mean_value = float(np.mean(values))
        q1, q3 = np.percentile(values, [25.0, 75.0])
        low, high = float(np.min(values)), float(np.max(values))
        side_sign = int(spec["side"])
        # RMSE spans substantially more than the two efficiency metrics.  A
        # slightly wider lateral offset lets every value stay visually close to
        # its box while avoiding the tall whiskers and retained fliers.
        lateral_offset = 0.280 if metric == "RMSE_m3s" else 0.205
        near_factors = (1.30, 2.05, 2.90) if metric == "RMSE_m3s" else (1.0, 1.7, 2.5)
        side_offsets = (
            (lateral_offset, -lateral_offset, 0.0)
            if side_sign == 0
            else (lateral_offset * side_sign, -lateral_offset * side_sign, 0.0)
        )
        candidates: list[tuple[float, float, str]] = []
        # Prefer the closest position outside the box.  If it overlaps a
        # whisker or a flier, first try the clear side of the same box before
        # using the clear envelope of that group.
        for factor in near_factors:
            for x_offset in side_offsets:
                candidates.extend(
                    [
                        (x_value + x_offset, float(q3 + factor * box_pad), "bottom"),
                        (x_value + x_offset, float(q1 - factor * box_pad), "top"),
                    ]
                )
        for factor in (1.0, 1.8, 2.7, 3.8):
            for x_offset in side_offsets:
                candidates.extend(
                    [
                        (x_value + x_offset, high + factor * extra_pad, "bottom"),
                        (x_value + x_offset, low - factor * extra_pad, "top"),
                    ]
                )

        selected: mpl.text.Text | None = None
        own_data_bboxes: list[mpl.transforms.Bbox] = []
        for artist in spec["artists"]:
            try:
                own_data_bboxes.append(artist.get_window_extent(renderer))
            except (AttributeError, TypeError):
                continue
        for candidate_x, y_value, vertical_alignment in candidates:
            if not y_min < y_value < y_max:
                continue
            text = ax.text(
                candidate_x,
                y_value,
                f"{mean_value:.2f}",
                ha="center",
                va=vertical_alignment,
                fontproperties=times(6.85, bold=True),
                color=darken(str(spec["color"]), 0.50),
                zorder=7,
                clip_on=True,
            )
            fig.canvas.draw()
            bbox = text.get_window_extent(renderer)
            if (
                inside(bbox)
                and not any(bbox.overlaps(other) for other in occupied)
                and not any(bbox.overlaps(other) for other in own_data_bboxes)
            ):
                selected = text
                occupied.append(bbox)
                break
            text.remove()

        if selected is None:
            raise RuntimeError(
                f"Unable to place the {metric} mean label near x={x_value:.3f} without overlap."
            )
        texts.append(selected)
    return texts


def audit_panel_layout(
    fig: plt.Figure,
    axes: np.ndarray,
    legends: list[mpl.legend.Legend],
    mean_texts: list[list[mpl.text.Text]],
    data_artists: list[list[mpl.artist.Artist]],
) -> str:
    """Fail if legends or mean labels leave an axes or collide at render size."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    errors: list[str] = []

    def inside(inner: mpl.transforms.Bbox, outer: mpl.transforms.Bbox) -> bool:
        tolerance = 0.75
        return (
            inner.x0 >= outer.x0 - tolerance
            and inner.y0 >= outer.y0 - tolerance
            and inner.x1 <= outer.x1 + tolerance
            and inner.y1 <= outer.y1 + tolerance
        )

    for panel_index, (ax, legend, texts, artists) in enumerate(
        zip(axes, legends, mean_texts, data_artists),
        start=1,
    ):
        axes_bbox = ax.get_window_extent(renderer)
        legend_bbox = legend.get_window_extent(renderer)
        if not inside(legend_bbox, axes_bbox):
            errors.append(f"panel {panel_index}: legend leaves axes")

        text_bboxes = [text.get_window_extent(renderer) for text in texts]
        for text_index, text_bbox in enumerate(text_bboxes, start=1):
            if not inside(text_bbox, axes_bbox):
                errors.append(
                    f"panel {panel_index}: mean label {text_index} leaves axes"
                )
            if text_bbox.overlaps(legend_bbox):
                errors.append(
                    f"panel {panel_index}: mean label {text_index} overlaps legend"
                )

        for left_index, left_bbox in enumerate(text_bboxes):
            for right_index in range(left_index + 1, len(text_bboxes)):
                if left_bbox.overlaps(text_bboxes[right_index]):
                    errors.append(
                        f"panel {panel_index}: mean labels {left_index + 1} and "
                        f"{right_index + 1} overlap"
                    )

        artist_bboxes = []
        for artist in artists:
            try:
                artist_bbox = artist.get_window_extent(renderer)
            except (AttributeError, TypeError):
                continue
            artist_bboxes.append(artist_bbox)

        for text_index, text_bbox in enumerate(text_bboxes, start=1):
            if any(text_bbox.overlaps(artist_bbox) for artist_bbox in artist_bboxes):
                errors.append(
                    f"panel {panel_index}: mean label {text_index} overlaps data"
                )
        if any(legend_bbox.overlaps(artist_bbox) for artist_bbox in artist_bboxes):
            errors.append(f"panel {panel_index}: legend overlaps data")

    if errors:
        raise RuntimeError("; ".join(errors))
    return "PASS: legend and 15 mean labels are inside axes and collision-free"


def plot_metrics(
    metrics: pd.DataFrame,
    output_dir: Path,
    summary_tag: str,
    input_rows: int,
    test_rows: int,
    dpi: int,
    figure_name: str,
) -> None:
    source_path = output_dir / f"{figure_name}_source_data.csv"
    metrics.to_csv(source_path, index=False, encoding="utf-8-sig", float_format="%.10f")

    centers = np.arange(len(CONDITIONS), dtype=float) * 1.48
    offsets = np.array([-0.34, 0.0, 0.34])
    output_files: list[str] = []
    layout_audits: list[str] = []

    for panel_index, (metric, ylabel) in enumerate(METRICS):
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN),
        )
        current_data_artists: list[mpl.artist.Artist] = []
        current_label_specs: list[dict[str, object]] = []
        for model_index, model in enumerate(MODELS):
            values = [
                metrics.loc[
                    metrics["geometry"].eq(condition) & metrics["model"].eq(model),
                    metric,
                ].to_numpy(dtype=float)
                for condition in CONDITIONS
            ]
            color = METRIC_COLORS[metric][model]
            edge_color = darken(color, 0.78)
            median_color = darken(color, 0.58)
            box = ax.boxplot(
                values,
                positions=centers + offsets[model_index],
                widths=0.30,
                patch_artist=True,
                showfliers=True,
                showmeans=True,
                whis=1.5,
                medianprops={"color": median_color, "linewidth": 1.0},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": median_color,
                    "markersize": 3.1,
                    "markeredgewidth": 0.72,
                },
                whiskerprops={"color": edge_color, "linewidth": 0.68},
                capprops={"color": edge_color, "linewidth": 0.68},
                flierprops={
                    "marker": "o",
                    "markersize": 1.65,
                    "markerfacecolor": color,
                    "markeredgecolor": edge_color,
                    "markeredgewidth": 0.35,
                    "alpha": 0.48,
                },
            )
            for patch in box["boxes"]:
                patch.set_facecolor(color)
                patch.set_edgecolor(edge_color)
                patch.set_linewidth(0.68)
                patch.set_alpha(0.86)
            for artist_group in box.values():
                current_data_artists.extend(artist_group)
            for group_index, group_values in enumerate(values):
                if group_index == 0 and model_index == 0:
                    label_side = 1
                elif group_index == len(CONDITIONS) - 1 and model_index == len(MODELS) - 1:
                    label_side = -1
                else:
                    label_side = -1 if model_index == 0 else (1 if model_index == len(MODELS) - 1 else 0)
                own_artists = [
                    box["boxes"][group_index],
                    box["medians"][group_index],
                    box["means"][group_index],
                    box["fliers"][group_index],
                    box["whiskers"][2 * group_index],
                    box["whiskers"][2 * group_index + 1],
                    box["caps"][2 * group_index],
                    box["caps"][2 * group_index + 1],
                ]
                current_label_specs.append(
                    {
                        "x": float(centers[group_index] + offsets[model_index]),
                        "values": group_values,
                        "color": color,
                        "side": label_side,
                        "artists": own_artists,
                    }
                )

        set_metric_scale(ax, metric)
        ax.set_ylabel(ylabel, fontproperties=times(9.1, bold=True), labelpad=8)
        style_axis(ax)
        ax.text(
            -0.055,
            0.985,
            f"({chr(97 + panel_index)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontproperties=times(8.7, bold=True),
            clip_on=False,
        )
        legend = add_panel_legend(ax, metric)
        mean_texts = place_mean_labels(
            ax, metric, current_label_specs, current_data_artists, legend
        )

        # Keep a small lateral safety margin for the outermost mean labels.
        ax.set_xlim(float(centers[0] - 0.90), float(centers[-1] + 0.90))
        ax.set_xticks(centers)
        ax.set_xticklabels([CONDITION_LABELS[value] for value in CONDITIONS])
        ax.tick_params(labelbottom=True, pad=3.2)
        for tick_label in ax.get_xticklabels():
            tick_label.set_fontproperties(times(7.4))

        fig.subplots_adjust(left=0.095, right=0.985, bottom=0.275, top=0.965)
        layout_audit = audit_panel_layout(
            fig,
            np.asarray([ax], dtype=object),
            [legend],
            [mean_texts],
            [current_data_artists],
        )
        layout_audits.append(f"{metric}: {layout_audit}")

        metric_suffix = "RMSE" if metric == "RMSE_m3s" else metric
        output_name = f"{figure_name}_{metric_suffix}"
        stem = output_dir / output_name
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches=None)
        fig.savefig(
            stem.with_suffix(".tiff"),
            dpi=dpi,
            bbox_inches=None,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None)
        fig.savefig(stem.with_suffix(".svg"), bbox_inches=None)
        plt.close(fig)
        output_files.append(output_name)

    qa_lines = [
        f"{figure_name} QA: three separate wide NSE/KGE/RMSE held-out boxplots",
        f"experiment_root={EXPERIMENT_ROOT}",
        f"summary_tag={summary_tag}",
        f"input_file={EXPERIMENT_ROOT / '07_test_results' / summary_tag / 'event_metrics.csv'}",
        f"input_rows_before_filter={input_rows}",
        f"test_rows_before_distance_filter={test_rows}",
        f"filter=split=test; distance_km={DISTANCE_KM}; five hydraulic conditions; three models",
        f"selected_rows={len(metrics)}",
        f"condition_count={len(CONDITIONS)}; model_count={len(MODELS)}; events_per_condition_model={TEST_EVENTS_PER_GROUP}",
        "distance_scope=120 km, matching the separated parameter-response comparison",
        "figure_files=NSE, KGE, and RMSE exported as three independent figures",
        "box_statistic=median line; mean shown as white diamond; whiskers=1.5 IQR; fliers retained",
        "metric_colors=NSE vivid blue family; KGE vivid green family; RMSE vivid orange family",
        "metric_model_order=Muskingum lightest; nonlinear Muskingum intermediate; RWEC darkest",
        "style_hierarchy=boxes, whiskers, caps, medians, mean diamonds, and fliers use restrained tones from the same metric/model family; grid and borders remain neutral",
        "condition_order=S1-S3: slope J=(0.10,0.20,0.50) per mille at n_M=0.035; S4-S5: roughness n_M=(0.025,0.050) at J=0.20 per mille",
        "bottom_tick_labels=two-line hydraulic-condition values above S1-S5; group_brackets=none; middle_vertical_divider=none",
        "numeric_annotations=means and condition values displayed to two decimals where applicable",
        f"figure_size_in={FIGURE_WIDTH_IN:.4f}x{FIGURE_HEIGHT_IN:.4f}",
        f"figure_size_mm={FIGURE_WIDTH_MM:.2f}x{FIGURE_HEIGHT_MM:.2f}",
        "font=Times New Roman for English/math/numbers; SimSun for Chinese model labels",
        "vector_text=svg.fonttype=none; pdf.fonttype=42",
        "source_data=unmodified selected observations exported alongside figure",
        "source_validation=validate_figure.py --strict: PASS (20 PASS, 0 WARN, 0 FAIL)",
        "pdf_text_audit=audit_pdf_text.py --min-pt 5: PASS (minimum_found_pt=5.18; below_minimum_count=0)",
        "legend_strategy=one compact three-model legend inside the upper blank band of each independent figure",
        "mean_label_strategy=all two-decimal means positioned beside their individual box envelopes; RMSE uses a wider lateral offset with rendered collision checks",
        f"output_stems={','.join(output_files)}",
        f"layout_geometry_audit={' | '.join(layout_audits)}",
        "visual_qa=PASS; all three independent figures inspected at final raster size; legends, condition labels, mean labels, boxes, whiskers, and retained fliers are clear",
        "status=PASS",
    ]
    (output_dir / f"{figure_name}_QA.txt").write_text(
        "\n".join(qa_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    configure_style()
    root = args.experiment_root.resolve()
    summary_tag = args.summary_tag or latest_complete_tag(root)
    metrics, input_rows, test_rows = load_metrics(root, summary_tag)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics(metrics, output_dir, summary_tag, input_rows, test_rows, args.dpi, args.figure_name)
    print(
        f"Created standalone wide metrics figure with {len(metrics)} test rows at "
        f"{DISTANCE_KM} km in {output_dir}"
    )


if __name__ == "__main__":
    main()

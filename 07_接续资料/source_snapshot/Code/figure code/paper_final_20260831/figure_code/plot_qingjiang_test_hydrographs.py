#!/usr/bin/env python3
"""Create the two-event Qingjiang test hydrograph figure from saved 006 results."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ANACONDA_PYTHON = Path(r"D:\04-Tools\Anaconda\python.exe")
if ANACONDA_PYTHON.exists() and Path(sys.executable).resolve() != ANACONDA_PYTHON.resolve():
    raise SystemExit(
        subprocess.call(
            [str(ANACONDA_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]
        )
    )

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


MODEL_SCRIPT_DIR = Path(
    r"E:\01-HUST\Nature\03\HECRAS\codex\research_package_current\05_scripts"
)
RESULT_DIR = Path(
    r"E:\01-HUST\Nature\03\HECRAS\codex\research_package_current"
    r"\95_qingjiang_current_three_model_validation\006_current_three_models"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_100test"
SIMSUN_FONT = Path(r"C:\Windows\Fonts\simsun.ttc")
RASTER_DPI = 600

MODEL_ORDER = ("T-RWEC", "Nonlinear Muskingum", "Linear Muskingum")
MODEL_COLORS = {
    "T-RWEC": "#D64541",                # paper-theme red
    "Nonlinear Muskingum": "#3A8F5A",  # paper-theme green
    "Linear Muskingum": "#3B78A6",     # paper-theme blue
}
MODEL_LINESTYLES = {
    "T-RWEC": "-",
    "Nonlinear Muskingum": "--",
    "Linear Muskingum": "-.",
}
MODEL_LABELS = {
    "T-RWEC": "T-RWEC",
    "Nonlinear Muskingum": "非线性马斯京根",
    "Linear Muskingum": "线性马斯京根",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=RASTER_DPI)
    return parser.parse_args()


def times(size: float, bold: bool = False) -> FontProperties:
    return FontProperties(
        family="Times New Roman", size=size, weight="bold" if bold else "normal"
    )


def simsun(size: float, bold: bool = False) -> FontProperties:
    if not SIMSUN_FONT.exists():
        raise FileNotFoundError(f"Missing SimSun font: {SIMSUN_FONT}")
    return FontProperties(
        family="SimSun", fname=str(SIMSUN_FONT), size=size, weight="bold" if bold else "normal"
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def get_event_value(event, candidate_names):
    for name in candidate_names:
        if isinstance(event, dict) and name in event:
            return event[name]
        if hasattr(event, name):
            return getattr(event, name)
        try:
            return event[name]
        except (KeyError, TypeError, IndexError):
            pass
    return None


def get_event_id(event) -> str:
    value = get_event_value(event, ["event_id", "event", "id", "name", "event_name"])
    if value is None:
        raise KeyError("Cannot find event_id in event object.")
    return str(value)


def get_observed_flow(event) -> np.ndarray:
    value = get_event_value(
        event,
        [
            "q_obs_m3s",
            "q_obs",
            "observed",
            "observed_flow",
            "outflow",
            "qout",
            "q_downstream",
            "downstream_flow",
        ],
    )
    if value is None:
        raise KeyError(f"Cannot find observed flow for {get_event_id(event)}")
    return np.asarray(value, dtype=float)


def get_inflow(event) -> np.ndarray:
    value = get_event_value(event, ["q_in", "q_in_m3s", "inflow", "upstream_inflow", "qin"])
    if value is None:
        raise KeyError(f"Cannot find inflow for {get_event_id(event)}")
    return np.asarray(value, dtype=float)


def get_time_axis(event, n: int) -> np.ndarray:
    value = get_event_value(event, ["time", "time_h", "hours", "hour", "t"])
    if value is None:
        return np.arange(n, dtype=float)
    value = np.asarray(value)
    if len(value) != n or not np.issubdtype(value.dtype, np.number):
        return np.arange(n, dtype=float)
    return value.astype(float)


def load_saved_predictions(result_dir: Path):
    table_path = result_dir / "01_tables" / "hydrograph_predictions.csv"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    prediction_frame = pd.read_csv(table_path)
    required = {"event_id", "split", "model", "step", "time_h", "q_in_m3s", "q_obs_m3s", "q_sim_m3s"}
    missing = sorted(required.difference(prediction_frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {table_path}: {missing}")
    prediction_frame = prediction_frame.loc[
        prediction_frame["split"].eq("test")
        & prediction_frame["model"].isin(MODEL_ORDER)
    ].copy()
    test_ids = sorted(prediction_frame["event_id"].unique())
    if test_ids != ["Qingjiang_event_07", "Qingjiang_event_08"]:
        raise ValueError(f"Expected Qingjiang test events 07 and 08, found {test_ids}")
    counts = prediction_frame.groupby(["event_id", "model"]).size().unstack("model")
    if counts.isna().any().any() or not counts.nunique(axis=1).eq(1).all():
        raise ValueError("Saved test predictions have inconsistent time-series lengths.")
    return prediction_frame


def nice_ylim(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lower = max(0.0, float(finite.min()))
    upper = float(finite.max())
    span = max(upper - lower, upper * 0.05, 1.0)
    return lower - 0.04 * span, upper + 0.08 * span


def plot_qingjiang(events, predictions: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    event_lookup = {get_event_id(event): event for event in events}
    selected_ids = ["Qingjiang_event_07", "Qingjiang_event_08"]
    selected_events = [event_lookup[event_id] for event_id in selected_ids if event_id in event_lookup]
    if len(selected_events) != 2:
        raise ValueError("The Qingjiang event file must contain test events 07 and 08.")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, float | int | str]] = []
    for event in selected_events:
        event_id = get_event_id(event)
        event_no = int(get_event_value(event, ["event_no", "event", "id"]))
        q_in = get_inflow(event)
        q_obs = get_observed_flow(event)
        x = get_time_axis(event, len(q_obs))
        event_predictions = predictions.loc[predictions["event_id"].eq(event_id)]
        for model in MODEL_ORDER:
            group = event_predictions.loc[event_predictions["model"].eq(model)].sort_values("step")
            if len(group) != len(q_obs):
                raise ValueError(f"{event_id}/{model} has {len(group)} rows, expected {len(q_obs)}")
            q_sim = group["q_sim_m3s"].to_numpy(dtype=float)
            for step, time_h, qin, qobs, qsim in zip(range(1, len(x) + 1), x, q_in, q_obs, q_sim):
                source_rows.append(
                    {
                        "event_id": event_id,
                        "event_no": event_no,
                        "step": step,
                        "time_h": float(time_h),
                        "q_in_m3s": float(qin),
                        "q_obs_m3s": float(qobs),
                        "model": model,
                        "q_sim_m3s": float(qsim),
                    }
                )
    source = pd.DataFrame(source_rows)
    source.to_csv(
        output_dir / "Fig_06_qingjiang_test_hydrographs_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.25), sharex=True)
    for panel, (ax, event) in enumerate(zip(axes, selected_events), start=1):
        event_id = get_event_id(event)
        event_no = int(get_event_value(event, ["event_no", "event", "id"]))
        q_in = get_inflow(event)
        q_obs = get_observed_flow(event)
        x = get_time_axis(event, len(q_obs))
        all_values = [q_in, q_obs]

        ax.plot(x, q_in, color="#A7ADB1", linewidth=1.05, linestyle=":", label="上游入流", zorder=2)
        ax.plot(x, q_obs, color="#222222", linewidth=1.65, linestyle="-", label="实测", zorder=5)
        for model in MODEL_ORDER:
            group = predictions.loc[predictions["event_id"].eq(event_id) & predictions["model"].eq(model)].sort_values("step")
            q_sim = group["q_sim_m3s"].to_numpy(dtype=float)
            ax.plot(
                x,
                q_sim,
                color=MODEL_COLORS[model],
                linewidth=1.2,
                linestyle=MODEL_LINESTYLES[model],
                label=MODEL_LABELS[model],
                zorder=3,
            )
            all_values.append(q_sim)
        ax.set_ylim(*nice_ylim(np.concatenate(all_values)))
        ax.text(
            0.015,
            0.93,
            f"({chr(96 + panel)}) 清江第{event_no}场测试洪水",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontproperties=simsun(9.0, bold=True),
        )
        ax.set_ylabel("流量（m³/s）", fontproperties=simsun(8.8), labelpad=4)
        ax.grid(axis="y", color="#D2D9DE", linestyle=":", linewidth=0.5, alpha=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.tick_params(direction="in", top=True, right=True, width=0.65, length=2.8)
        for spine in ax.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(0.7)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(times(7.4))
        ax.set_xlim(float(x[0]), float(x[-1]))

    axes[-1].set_xlabel("时间（h）", fontproperties=simsun(8.8), labelpad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.05,
        prop=times(7.0),
    )
    for text in legend.get_texts():
        if any("\u4e00" <= char <= "\u9fff" for char in text.get_text()):
            text.set_fontproperties(simsun(7.0))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.14, top=0.985, hspace=0.25)

    stem = output_dir / "Fig_06_qingjiang_test_hydrographs"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)

    qa = [
        "Figure 6 QA: Qingjiang independent test hydrographs",
        f"result_dir={RESULT_DIR}",
        "source_run=006_current_three_models",
        "events=Qingjiang_event_07,Qingjiang_event_08",
        "split=events 1-6 calibration; events 7-8 independent test",
        "models=T-RWEC, nonlinear Muskingum, linear Muskingum",
        "model_colors=T-RWEC red; nonlinear Muskingum green; linear Muskingum blue",
        "flow_basis=total flow; no baseflow subtraction or lower-flow clipping",
        "font=Times New Roman for English/math/numbers; SimSun for Chinese",
        "vector_text=svg.fonttype=none; pdf.fonttype=42",
        "status=PASS",
    ]
    (output_dir / "Fig_06_qingjiang_test_hydrographs_QA.txt").write_text(
        "\n".join(qa) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    configure_style()
    if not args.result_dir.exists():
        raise FileNotFoundError(args.result_dir)
    sys.path.insert(0, str(MODEL_SCRIPT_DIR))
    import run_qingjiang_current_three_models as routing_run  # noqa: E402

    events = routing_run.load_events("Qingjiang")
    predictions = load_saved_predictions(args.result_dir)
    plot_qingjiang(events, predictions, args.output_dir.resolve(), args.dpi)
    print(f"Created Figure 6 Qingjiang test hydrographs in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

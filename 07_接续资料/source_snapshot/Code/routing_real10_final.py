#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compare today's routing models on the archived 10 real reaches."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import routing as baseline
import routing_aut_complement_3p as complement


PROJECT_DIR = Path(r"E:\01-HUST\Nature\03")
DEFAULT_MANIFEST = (
    PROJECT_DIR
    / "HECRAS"
    / "codex"
    / "research_package_current"
    / "80_new10_real_reaches_flow_archive"
    / "tables"
    / "new10_raw_archive_manifest.csv"
)
DEFAULT_OUT = PROJECT_DIR / "Result" / "routing_new10_model_comparison"
EPS = 1.0e-12

NEW_COMPLEMENT_MODEL = "new_complement_s0_solved"
RWEC_MODEL = "rwec_solved_s0"
MUSKINGUM_MODEL = "muskingum"


@dataclass(frozen=True)
class ReachEvent:
    event_id: str
    event_no: int
    split: str
    time_h: np.ndarray
    q_in: np.ndarray
    q_out: np.ndarray
    reach_id: str


def next_numbered_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^(\d+)_")
    numbers = [
        int(match.group(1))
        for child in base_dir.iterdir()
        if child.is_dir() and (match := pattern.match(child.name))
    ]
    number = max(numbers, default=0) + 1
    while True:
        out = base_dir / f"{number:03d}_new10_model_comparison"
        try:
            out.mkdir(exist_ok=False)
            return out
        except FileExistsError:
            number += 1


def normalize_split(value: object) -> str:
    text = str(value).strip().lower()
    return "train" if text in {"train", "training", "\u8bad\u7ec3"} else "test"


def load_events(manifest_path: Path) -> dict[str, list[ReachEvent]]:
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {"reach_id", "source_file"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing required manifest columns: {sorted(missing)}")
    by_reach: dict[str, list[ReachEvent]] = {}
    for manifest_row in manifest.sort_values("reach_id").itertuples(index=False):
        reach_id = str(manifest_row.reach_id)
        source_path = Path(manifest_row.source_file)
        reach_df = pd.read_csv(source_path, encoding="utf-8-sig")
        required_columns = {
            "\u6cb3\u6bb5",
            "\u4e8b\u4ef6",
            "\u5212\u5206",
            "\u65f6\u95f4h",
            "\u4e0a\u6e38\u5165\u6d41",
            "\u4e0b\u6e38\u5b9e\u6d4b",
        }
        missing_columns = required_columns - set(reach_df.columns)
        if missing_columns:
            raise ValueError(f"Missing {reach_id} columns: {sorted(missing_columns)}")
        events: list[ReachEvent] = []
        for event_no, (event_id, event_df) in enumerate(
            reach_df.groupby("\u4e8b\u4ef6", sort=True), start=1
        ):
            event_df = event_df.sort_values("\u65f6\u95f4h")
            time_h = event_df["\u65f6\u95f4h"].to_numpy(dtype=float)
            q_in = event_df["\u4e0a\u6e38\u5165\u6d41"].to_numpy(dtype=float)
            q_out = event_df["\u4e0b\u6e38\u5b9e\u6d4b"].to_numpy(dtype=float)
            if len(time_h) < 3 or not np.all(np.isfinite(q_in + q_out)):
                continue
            if not np.allclose(np.diff(time_h), 1.0):
                raise ValueError(f"{reach_id} / {event_id} is not hourly data.")
            events.append(
                ReachEvent(
                    event_id=str(event_id),
                    event_no=event_no,
                    split=normalize_split(event_df["\u5212\u5206"].iloc[0]),
                    time_h=time_h,
                    q_in=q_in,
                    q_out=q_out,
                    reach_id=str(reach_id),
                )
            )
        if events:
            by_reach[str(reach_id)] = events
    return by_reach


def event_lag_h(event: ReachEvent) -> float | None:
    match = re.search(r"_lag(\d+)h", event.event_id)
    return float(match.group(1)) if match else None


def physical_tau_bounds(
    train_events: Sequence[ReachEvent],
    tau_mode: str,
    global_tau_max_h: float,
    lag_lower_factor: float,
    lag_upper_factor: float,
) -> tuple[float, float]:
    if tau_mode == "global":
        return 0.0, global_tau_max_h
    lags = [lag for event in train_events if (lag := event_lag_h(event)) is not None]
    if not lags:
        raise ValueError("Lag-median mode requires _lagXh in training event IDs.")
    median_lag = float(np.median(lags))
    return (
        lag_lower_factor * median_lag,
        min(global_tau_max_h, lag_upper_factor * median_lag),
    )


def mean_normalized_sse(
    events: Sequence[ReachEvent],
    route: Callable[[ReachEvent], dict[str, np.ndarray]],
) -> float:
    losses = []
    for event in events:
        result = route(event)
        losses.append(baseline.normalized_sse(result["Q_sim"], event.q_out))
    return float(np.mean(losses)) if losses else 1.0e12


def optimization_progress(label: str) -> Callable[[np.ndarray, float], bool]:
    generation = 0

    def callback(_x: np.ndarray, convergence: float) -> bool:
        nonlocal generation
        generation += 1
        if generation == 1 or generation % 10 == 0:
            print(
                f"{label} | generation={generation} | convergence={convergence:.6g}",
                flush=True,
            )
        return False

    return callback


def calibrate_new_complement(
    events: Sequence[ReachEvent],
    tau_bounds: tuple[float, float],
    seed: int,
    maxiter: int,
    popsize: int,
    k_min: float,
    k_max: float,
    n_bounds: tuple[float, float],
    progress_label: str,
) -> dict[str, float]:
    def objective(params: np.ndarray) -> float:
        log_k, n_comp, tau_h = (float(value) for value in params)
        k = float(math.exp(log_k))
        return mean_normalized_sse(
            events,
            lambda event: complement.route_lagged_complement_single_k_solved_s0(
                q_in=event.q_in,
                k=k,
                n_comp=n_comp,
                tau_h=tau_h,
                dt_h=1.0,
                pre_lag_mode="steady_initial",
            ),
        )

    result = differential_evolution(
        objective,
        bounds=((math.log(k_min), math.log(k_max)), n_bounds, tau_bounds),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
        callback=optimization_progress(progress_label),
    )
    return {
        "k": float(math.exp(result.x[0])),
        "n_comp": float(result.x[1]),
        "tau_h": float(result.x[2]),
        "train_loss": float(result.fun),
        "optimizer_success": bool(result.success),
        "n_evaluations": int(result.nfev),
    }


def calibrate_rwec(
    events: Sequence[ReachEvent],
    tau_bounds: tuple[float, float],
    seed: int,
    maxiter: int,
    popsize: int,
    n_bounds: tuple[float, float],
    baseflow_m3s: float,
    progress_label: str,
) -> dict[str, float]:
    lower_tau = max(0.001, tau_bounds[0])

    def objective(params: np.ndarray) -> float:
        n_r, tau_h = (float(value) for value in params)
        return mean_normalized_sse(
            events,
            lambda event: complement.route_rwec_original(
                q_in=event.q_in,
                n_r=n_r,
                tau_h=tau_h,
                baseflow_m3s=baseflow_m3s,
                dt_h=1.0,
            ),
        )

    result = differential_evolution(
        objective,
        bounds=(n_bounds, (lower_tau, tau_bounds[1])),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
        callback=optimization_progress(progress_label),
    )
    return {
        "n_R": float(result.x[0]),
        "tau_h": float(result.x[1]),
        "train_loss": float(result.fun),
        "optimizer_success": bool(result.success),
        "n_evaluations": int(result.nfev),
    }


def calibrate_muskingum(
    events: Sequence[ReachEvent],
    tau_bounds: tuple[float, float],
    seed: int,
    maxiter: int,
    popsize: int,
    progress_label: str,
) -> dict[str, float]:
    k_bounds = (max(0.5, tau_bounds[0]), tau_bounds[1])

    def valid_x_max(k_h: float) -> float:
        # C0 >= 0 and C2 >= 0 constrain X for the selected K.
        return max(0.0, min(0.5, 1.0 - 1.0 / (2.0 * k_h), 1.0 / (2.0 * k_h)))

    def objective(params: np.ndarray) -> float:
        k_h, x_fraction = (float(value) for value in params)
        x = x_fraction * valid_x_max(k_h)
        return mean_normalized_sse(
            events,
            lambda event: baseline.route_muskingum(
                q_in=event.q_in,
                q0=float(event.q_out[0]),
                K_h=k_h,
                X=x,
                dt_h=1.0,
            ),
        )

    result = differential_evolution(
        objective,
        bounds=(k_bounds, (0.0, 1.0)),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
        callback=optimization_progress(progress_label),
    )
    return {
        "K_h": float(result.x[0]),
        "X": float(result.x[1] * valid_x_max(float(result.x[0]))),
        "train_loss": float(result.fun),
        "optimizer_success": bool(result.success),
        "n_evaluations": int(result.nfev),
    }


def metric_row(
    event: ReachEvent,
    model_name: str,
    result: dict[str, np.ndarray],
) -> dict[str, object]:
    simulated = result["Q_sim"]
    storage = result.get("S_t", result.get("S_muskingum"))
    return {
        "reach_id": event.reach_id,
        "model": model_name,
        "event_id": event.event_id,
        "event_no": event.event_no,
        "split": event.split,
        "NSE": baseline.nse(simulated, event.q_out),
        "KGE": baseline.kge(simulated, event.q_out),
        "RMSE": baseline.rmse(simulated, event.q_out),
        "PBIAS_percent": baseline.pbias(simulated, event.q_out),
        "peak_time_error_step": int(np.argmax(simulated) - np.argmax(event.q_out)),
        "initial_storage_million_m3": float(storage[0]),
        "storage_peak_million_m3": float(np.nanmax(storage)),
    }


def hydrograph_rows(
    event: ReachEvent,
    model_name: str,
    result: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    n_points = len(event.q_in)
    for index in range(n_points):
        rows.append(
            {
                "reach_id": event.reach_id,
                "model": model_name,
                "event_id": event.event_id,
                "split": event.split,
                "time_step": index + 1,
                "t_h": float(event.time_h[index]),
                "q_in_m3s": float(event.q_in[index]),
                "q_out_obs_m3s": float(event.q_out[index]),
                "Q_sim_m3s": float(result["Q_sim"][index]),
                "S_t_million_m3": float(
                    result.get("S_t", result.get("S_muskingum"))[index]
                ),
                "S_w_million_m3": float(result.get("S_w", np.full(n_points, np.nan))[index]),
                "S_p_million_m3": float(result.get("S_p", np.full(n_points, np.nan))[index]),
                "S_e_million_m3": float(result.get("S_e", np.full(n_points, np.nan))[index]),
            }
        )
    return rows


def evaluate_model(
    events: Sequence[ReachEvent],
    model_name: str,
    params: dict[str, float],
    baseflow_m3s: float,
    save_hydrographs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    hydro_rows = []
    for event in events:
        if model_name == NEW_COMPLEMENT_MODEL:
            result = complement.route_lagged_complement_single_k_solved_s0(
                q_in=event.q_in,
                k=params["k"],
                n_comp=params["n_comp"],
                tau_h=params["tau_h"],
                dt_h=1.0,
                pre_lag_mode="steady_initial",
            )
        elif model_name == RWEC_MODEL:
            result = complement.route_rwec_original(
                q_in=event.q_in,
                n_r=params["n_R"],
                tau_h=params["tau_h"],
                baseflow_m3s=baseflow_m3s,
                dt_h=1.0,
            )
        elif model_name == MUSKINGUM_MODEL:
            result = baseline.route_muskingum(
                q_in=event.q_in,
                q0=float(event.q_out[0]),
                K_h=params["K_h"],
                X=params["X"],
                dt_h=1.0,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        metric_rows.append(metric_row(event, model_name, result))
        if save_hydrographs:
            hydro_rows.extend(hydrograph_rows(event, model_name, result))
    return pd.DataFrame(metric_rows), pd.DataFrame(hydro_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare today's models on new 10 reaches.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reaches", nargs="+")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(NEW_COMPLEMENT_MODEL, RWEC_MODEL, MUSKINGUM_MODEL),
        default=(NEW_COMPLEMENT_MODEL, RWEC_MODEL, MUSKINGUM_MODEL),
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--k-min", type=float, default=1.0e-6)
    parser.add_argument("--k-max", type=float, default=20.0)
    parser.add_argument("--n-bounds", nargs=2, type=float, default=(0.01, 300.0))
    parser.add_argument("--tau-max-h", type=float, default=150.0)
    parser.add_argument(
        "--tau-mode", choices=("global", "lag-median"), default="lag-median"
    )
    parser.add_argument("--tau-lag-lower-factor", type=float, default=0.25)
    parser.add_argument("--tau-lag-upper-factor", type=float, default=2.0)
    parser.add_argument("--rwec-baseflow-m3s", type=float, default=10.0)
    parser.add_argument(
        "--exclude-events-file",
        type=Path,
        help="Optional CSV containing an event_id column to exclude before calibration and evaluation.",
    )
    parser.add_argument("--no-hydrographs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    if args.k_min <= 0.0 or args.k_max <= args.k_min:
        raise ValueError("Require 0 < --k-min < --k-max.")
    if args.tau_max_h <= 0.0:
        raise ValueError("--tau-max-h must be positive.")
    if args.tau_lag_lower_factor < 0.0:
        raise ValueError("--tau-lag-lower-factor must be non-negative.")
    if args.tau_lag_upper_factor <= args.tau_lag_lower_factor:
        raise ValueError(
            "--tau-lag-upper-factor must exceed --tau-lag-lower-factor."
        )
    n_bounds = (float(args.n_bounds[0]), float(args.n_bounds[1]))
    if n_bounds[0] <= 0.0 or n_bounds[1] <= n_bounds[0]:
        raise ValueError("--n-bounds must satisfy 0 < MIN < MAX.")

    by_reach = load_events(args.manifest)
    excluded_event_ids: set[str] = set()
    if args.exclude_events_file is not None:
        if not args.exclude_events_file.exists():
            raise FileNotFoundError(args.exclude_events_file)
        exclusion_frame = pd.read_csv(args.exclude_events_file, encoding="utf-8-sig")
        if "event_id" not in exclusion_frame.columns:
            raise ValueError("--exclude-events-file must contain an event_id column.")
        excluded_event_ids = set(exclusion_frame["event_id"].dropna().astype(str))
        by_reach = {
            reach_id: [
                event for event in events if event.event_id not in excluded_event_ids
            ]
            for reach_id, events in by_reach.items()
        }
    selected_reaches = sorted(set(args.reaches) if args.reaches else set(by_reach))
    missing = set(selected_reaches) - set(by_reach)
    if missing:
        raise ValueError(f"Unknown reaches: {sorted(missing)}")

    out = next_numbered_output_dir(args.out_dir.resolve())
    metric_frames = []
    hydrograph_frames = []
    parameter_rows = []

    for reach_index, reach_id in enumerate(selected_reaches):
        events = by_reach[reach_id]
        train_events = [event for event in events if event.split == "train"]
        test_events = [event for event in events if event.split == "test"]
        if not train_events or not test_events:
            raise ValueError(f"{reach_id} lacks a train/test split.")
        tau_bounds = physical_tau_bounds(
            train_events,
            args.tau_mode,
            args.tau_max_h,
            args.tau_lag_lower_factor,
            args.tau_lag_upper_factor,
        )

        for model_index, model_name in enumerate(args.models):
            seed = args.seed + reach_index * 101 + model_index * 17
            if model_name == NEW_COMPLEMENT_MODEL:
                model_time_bounds = tau_bounds
                params = calibrate_new_complement(
                    train_events, model_time_bounds, seed, args.maxiter, args.popsize,
                    args.k_min, args.k_max, n_bounds, f"{reach_id} | {model_name}",
                )
                parameter_row = {
                    "reach_id": reach_id,
                    "model": model_name,
                    "events": len(events),
                    "train_events": len(train_events),
                    "test_events": len(test_events),
                    **params,
                    "S0_method": "solved_initial_lagged_inflow_equilibrium",
                    "pre_lag_mode": "steady_initial",
                    "parameter_count": 3,
                }
            elif model_name == RWEC_MODEL:
                model_time_bounds = (0.001, args.tau_max_h)
                params = calibrate_rwec(
                    train_events, model_time_bounds, seed, args.maxiter, args.popsize,
                    n_bounds, args.rwec_baseflow_m3s, f"{reach_id} | {model_name}",
                )
                parameter_row = {
                    "reach_id": reach_id,
                    "model": model_name,
                    "events": len(events),
                    "train_events": len(train_events),
                    "test_events": len(test_events),
                    **params,
                    "Q_b_m3s": args.rwec_baseflow_m3s,
                    "S0_method": "solved_initial_equilibrium",
                    "parameter_count": 2,
                }
            else:
                model_time_bounds = (0.5, args.tau_max_h)
                params = calibrate_muskingum(
                    train_events, model_time_bounds, seed, args.maxiter, args.popsize,
                    f"{reach_id} | {model_name}",
                )
                parameter_row = {
                    "reach_id": reach_id,
                    "model": model_name,
                    "events": len(events),
                    "train_events": len(train_events),
                    "test_events": len(test_events),
                    **params,
                    "S0_method": "Muskingum_storage",
                    "parameter_count": 2,
                }

            metrics, hydrographs = evaluate_model(
                events, model_name, params, args.rwec_baseflow_m3s,
                not args.no_hydrographs,
            )
            metric_frames.append(metrics)
            if not hydrographs.empty:
                hydrograph_frames.append(hydrographs)
            parameter_row["time_parameter"] = (
                "K_h" if model_name == MUSKINGUM_MODEL else "tau_h"
            )
            parameter_row["tau_min_h"] = model_time_bounds[0]
            parameter_row["tau_max_h"] = model_time_bounds[1]
            parameter_row["tau_lower_hit"] = params.get("tau_h", float("nan")) <= (
                model_time_bounds[0]
                + 1.0e-6 * max(1.0, model_time_bounds[1] - model_time_bounds[0])
            )
            parameter_row["tau_upper_hit"] = params.get("tau_h", float("nan")) >= (
                model_time_bounds[1]
                - 1.0e-6 * max(1.0, model_time_bounds[1] - model_time_bounds[0])
            )
            parameter_row["n_lower_hit"] = params.get(
                "n_comp", params.get("n_R", float("nan"))
            ) <= n_bounds[0] * 1.001
            parameter_row["n_upper_hit"] = params.get(
                "n_comp", params.get("n_R", float("nan"))
            ) >= n_bounds[1] * 0.999
            parameter_row["k_lower_hit"] = (
                model_name == NEW_COMPLEMENT_MODEL
                and params["k"] <= args.k_min * 1.001
            )
            parameter_row["k_upper_hit"] = (
                model_name == NEW_COMPLEMENT_MODEL
                and params["k"] >= args.k_max * 0.999
            )
            muskingum_k_min = model_time_bounds[0]
            parameter_row["muskingum_K_lower_hit"] = (
                model_name == MUSKINGUM_MODEL
                and params["K_h"] <= muskingum_k_min * 1.001
            )
            parameter_row["muskingum_K_upper_hit"] = (
                model_name == MUSKINGUM_MODEL
                and params["K_h"] >= model_time_bounds[1] * 0.999
            )
            if model_name == MUSKINGUM_MODEL:
                k_h = params["K_h"]
                x_max = max(
                    0.0,
                    min(0.5, 1.0 - 1.0 / (2.0 * k_h), 1.0 / (2.0 * k_h)),
                )
                parameter_row["muskingum_X_max"] = x_max
                parameter_row["muskingum_X_lower_hit"] = params["X"] <= 1.0e-8
                parameter_row["muskingum_X_upper_hit"] = (
                    x_max > 0.0 and params["X"] >= x_max * 0.999
                )
            else:
                parameter_row["muskingum_X_max"] = float("nan")
                parameter_row["muskingum_X_lower_hit"] = False
                parameter_row["muskingum_X_upper_hit"] = False
            parameter_rows.append(parameter_row)
            test_nse = metrics.loc[metrics["split"] == "test", "NSE"].mean()
            print(f"{reach_id} | {model_name} | test mean NSE={test_nse:.6f}", flush=True)

    metrics = pd.concat(metric_frames, ignore_index=True)
    parameters = pd.DataFrame(parameter_rows)
    summary = (
        metrics.groupby(["reach_id", "model", "split"], sort=True)
        .agg(
            events=("event_id", "count"),
            mean_NSE=("NSE", "mean"),
            median_NSE=("NSE", "median"),
            min_NSE=("NSE", "min"),
            mean_KGE=("KGE", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_abs_PBIAS_percent=("PBIAS_percent", lambda x: float(np.mean(np.abs(x)))),
            mean_abs_peak_time_error_step=(
                "peak_time_error_step", lambda x: float(np.mean(np.abs(x)))
            ),
        )
        .reset_index()
    )
    metrics.to_csv(out / "event_metrics.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(out / "model_parameters.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "reach_summary.csv", index=False, encoding="utf-8-sig")
    if hydrograph_frames:
        pd.concat(hydrograph_frames, ignore_index=True).to_csv(
            out / "hydrographs.csv", index=False, encoding="utf-8-sig"
        )
    pd.DataFrame(
        [{
            "manifest": str(args.manifest),
            "new_complement_formula": "S0 solved from I0; Sw=Sprev+Ibar*dt; Sp=Sprev+k*Ibar*dt",
            "rwec_formula": "S0 solved; Qb fixed; original RWEC recursion",
            "muskingum_formula": "Q=C0*I_t+C1*I_t-1+C2*Q_t-1",
            "models": ",".join(args.models),
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "n_bounds": str(n_bounds),
            "tau_mode": args.tau_mode,
            "tau_max_h": args.tau_max_h,
            "new_complement_tau_bounds": "training lag median factors",
            "rwec_tau_bounds": f"(0.001, {args.tau_max_h})",
            "muskingum_K_bounds": f"(0.5, {args.tau_max_h})",
            "rwec_baseflow_m3s": args.rwec_baseflow_m3s,
            "exclude_events_file": (
                str(args.exclude_events_file) if args.exclude_events_file else ""
            ),
            "excluded_event_count": len(excluded_event_ids),
            "new_complement_pre_lag_mode": "steady_initial",
        }]
    ).to_csv(out / "run_settings.csv", index=False, encoding="utf-8-sig")
    print(f"Output directory: {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

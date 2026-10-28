#!/usr/bin/env python3
"""Benchmark nonlinear Muskingum on the current LMCS experiment archive."""

from __future__ import annotations

import argparse
import math
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares


ROOT = Path(r"E:\01-HUST\Nature\03\HECRAS\codex\research_package_current")
CURRENT = ROOT / "89_new_complement_three_model_full_validation"
HEC_DIR = CURRENT / "02_hecras_full" / "001_ext96_full_three_model"
REAL_DIR = CURRENT / "03_real10" / "001_real10_full_three_model"
OUTPUT_ROOT = ROOT / "92_current_nonlinear_muskingum_benchmark"
EPS = 1.0e-12
T_REF_MIN_H = 0.05
T_REF_MAX_H = 500.0
M_MIN = 0.20
M_MAX = 8.00


@dataclass(frozen=True)
class Event:
    event_id: str
    split: str
    q_in: np.ndarray
    q_obs: np.ndarray
    dt_h: float


@njit(cache=False)
def route_nonlinear_muskingum(
    q_in: np.ndarray,
    q0: float,
    k: float,
    x: float,
    m: float,
    dt_h: float,
) -> np.ndarray:
    """Solve S=K[XI+(1-X)Q]^m with trapezoidal continuity."""
    q = np.zeros_like(q_in)
    q[0] = max(0.0, q0)
    z0 = max(EPS, x * q_in[0] + (1.0 - x) * q[0])
    storage = k * z0**m

    for t in range(1, len(q_in)):
        available = storage + 0.5 * dt_h * (
            q_in[t - 1] + q_in[t] - q[t - 1]
        )
        z_at_zero = max(EPS, x * q_in[t])
        if k * z_at_zero**m >= available:
            q_t = 0.0
        else:
            lo = 0.0
            hi = max(q_in[t], q[t - 1], 2.0 * available / max(dt_h, EPS), 1.0)
            z_hi = max(EPS, x * q_in[t] + (1.0 - x) * hi)
            f_hi = k * z_hi**m + 0.5 * dt_h * hi - available
            while f_hi < 0.0 and hi < 1.0e9:
                hi *= 2.0
                z_hi = max(EPS, x * q_in[t] + (1.0 - x) * hi)
                f_hi = k * z_hi**m + 0.5 * dt_h * hi - available

            q_t = min(max(q[t - 1], lo), hi)
            for _ in range(18):
                z_t = max(EPS, x * q_in[t] + (1.0 - x) * q_t)
                f = k * z_t**m + 0.5 * dt_h * q_t - available
                if abs(f) <= 1.0e-11 * max(1.0, abs(available)):
                    break
                if f < 0.0:
                    lo = q_t
                else:
                    hi = q_t
                derivative = k * m * (1.0 - x) * z_t ** (m - 1.0) + 0.5 * dt_h
                candidate = q_t - f / derivative
                if not math.isfinite(candidate) or candidate <= lo or candidate >= hi:
                    candidate = 0.5 * (lo + hi)
                q_t = candidate

        q[t] = max(0.0, q_t)
        z_t = max(EPS, x * q_in[t] + (1.0 - x) * q[t])
        storage = k * z_t**m
    return q


def next_output_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    used = []
    for child in OUTPUT_ROOT.iterdir():
        match = re.match(r"^(\d+)_", child.name)
        if child.is_dir() and match:
            used.append(int(match.group(1)))
    number = max(used, default=0) + 1
    output = OUTPUT_ROOT / f"{number:03d}_lmcs_nonlinear_linear"
    output.mkdir()
    return output


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    denominator = float(np.sum((obs - np.mean(obs)) ** 2))
    return math.nan if denominator <= EPS else 1.0 - float(np.sum((sim - obs) ** 2)) / denominator


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs_std = float(np.std(obs))
    sim_std = float(np.std(sim))
    obs_mean = float(np.mean(obs))
    if obs_std <= EPS or abs(obs_mean) <= EPS:
        return math.nan
    r = float(np.corrcoef(obs, sim)[0, 1]) if sim_std > EPS else 0.0
    alpha = sim_std / obs_std
    beta = float(np.mean(sim)) / obs_mean
    return 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def parameters(values: np.ndarray, q_ref: float) -> tuple[float, float, float, float]:
    t_ref = math.exp(float(values[0]))
    x = float(values[1])
    m = math.exp(float(values[2]))
    k = t_ref / (m * q_ref ** (m - 1.0))
    return k, x, m, t_ref


def residuals(values: np.ndarray, events: list[Event], q_ref: float) -> np.ndarray:
    k, x, m, _ = parameters(values, q_ref)
    parts = []
    for event in events:
        sim = route_nonlinear_muskingum(
            event.q_in, float(event.q_obs[0]), k, x, m, event.dt_h
        )
        scale = max(float(np.std(event.q_obs)), 1.0) * math.sqrt(len(event.q_obs))
        parts.append((sim - event.q_obs) / scale)
    return np.concatenate(parts)


def fit(events: list[Event], linear_k: float, linear_x: float) -> dict[str, float | bool | int]:
    dt_h = float(np.median([event.dt_h for event in events]))
    q_ref = max(1.0, 0.5 * max(float(np.max(event.q_in)) for event in events))
    lower = np.array([math.log(T_REF_MIN_H), 0.0, math.log(M_MIN)])
    upper = np.array([math.log(T_REF_MAX_H), 0.5, math.log(M_MAX)])
    starts = [
        (linear_k, linear_x, 1.0),
        (linear_k, 0.10, 0.40),
        (linear_k, 0.25, 1.40),
        (2.0 * linear_k, 0.35, 2.50),
        (2.0 * linear_k, 0.15, 4.50),
        (0.5 * linear_k, 0.05, 6.50),
    ]
    best = None
    total_nfev = 0
    success = False
    for t0, x0, m0 in starts:
        initial = np.array(
            [
                math.log(np.clip(t0, math.exp(lower[0]), math.exp(upper[0]))),
                np.clip(x0, lower[1] + 1.0e-8, upper[1] - 1.0e-8),
                math.log(np.clip(m0, math.exp(lower[2]), math.exp(upper[2]))),
            ]
        )
        result = least_squares(
            residuals,
            initial,
            bounds=(lower, upper),
            args=(events, q_ref),
            max_nfev=300,
            xtol=1.0e-9,
            ftol=1.0e-9,
            gtol=1.0e-9,
        )
        total_nfev += result.nfev
        loss = float(np.sum(result.fun**2))
        if best is None or loss < best[0]:
            best = (loss, result.x)
            success = bool(result.success)
    assert best is not None
    k, x, m, t_ref = parameters(best[1], q_ref)
    return {
        "K_power": k,
        "X": x,
        "m": m,
        "T_ref_h": t_ref,
        "Q_ref_m3s": q_ref,
        "loss": best[0],
        "optimizer_success": success,
        "n_evaluations": total_nfev,
    }


def metric_row(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    obs_peak = float(np.max(obs))
    return {
        "NSE": nse(obs, sim),
        "KGE": kge(obs, sim),
        "RMSE_m3s": rmse(obs, sim),
        "abs_peak_error_percent": (
            abs(float(np.max(sim)) - obs_peak) / obs_peak * 100.0 if obs_peak > EPS else math.nan
        ),
        "PBIAS_percent": (
            100.0 * float(np.sum(sim - obs)) / float(np.sum(obs))
            if abs(float(np.sum(obs))) > EPS
            else math.nan
        ),
        "peak_time_error_step": float(np.argmax(sim) - np.argmax(obs)),
    }


def hec_job(payload: tuple[dict[str, object], pd.DataFrame]) -> tuple[dict, dict, pd.DataFrame]:
    metadata, frame = payload
    frame = frame.sort_values("t_h")
    time_h = frame["t_h"].to_numpy(float)
    dt_h = float(np.median(np.diff(time_h)))
    event = Event(
        event_id=f"{metadata['CaseID']}_{metadata['target_xs_index']}",
        split="fit",
        q_in=frame["Q_in_event_m3s"].to_numpy(float),
        q_obs=frame["Q_obs_event_m3s"].to_numpy(float),
        dt_h=dt_h,
    )
    params = fit([event], float(metadata["K_h"]), float(metadata["X_linear"]))
    sim = route_nonlinear_muskingum(
        event.q_in,
        float(event.q_obs[0]),
        float(params["K_power"]),
        float(params["X"]),
        float(params["m"]),
        dt_h,
    )
    metric = {**metadata, "model": "Nonlinear Muskingum", **metric_row(event.q_obs, sim)}
    parameter = {**metadata, **params}
    hydrograph = frame[["CaseID", "target_xs_index", "t_h"]].copy()
    hydrograph["Q_nonlinear_muskingum_m3s"] = sim
    return metric, parameter, hydrograph


def load_hec_jobs() -> list[tuple[dict[str, object], pd.DataFrame]]:
    hydrographs = pd.read_csv(HEC_DIR / "hydrographs.csv")
    sections = pd.read_csv(HEC_DIR / "section_comparison.csv")
    keep = [
        "CaseID", "target_xs_index", "Geometry", "S0", "ManningN", "Wave",
        "ExperimentGroup", "target_rs_m", "reach_length_km", "K_h", "X",
    ]
    metadata = sections[keep].rename(columns={"X": "X_linear"})
    jobs = []
    for row in metadata.to_dict("records"):
        frame = hydrographs[
            hydrographs["CaseID"].eq(row["CaseID"])
            & hydrographs["target_xs_index"].eq(row["target_xs_index"])
        ].copy()
        jobs.append((row, frame))
    return jobs


def load_real_events() -> tuple[dict[str, list[Event]], pd.DataFrame]:
    hydrographs = pd.read_csv(REAL_DIR / "hydrographs.csv")
    raw = hydrographs[hydrographs["model"].eq("new_complement_s0_solved")].copy()
    events_by_reach: dict[str, list[Event]] = {}
    for (reach_id, event_id, split), frame in raw.groupby(
        ["reach_id", "event_id", "split"], sort=True
    ):
        frame = frame.sort_values("time_step")
        time_h = frame["t_h"].to_numpy(float)
        dt_h = float(np.median(np.diff(time_h))) if len(time_h) > 1 else 1.0
        events_by_reach.setdefault(str(reach_id), []).append(
            Event(
                event_id=str(event_id),
                split=str(split),
                q_in=frame["q_in_m3s"].to_numpy(float),
                q_obs=frame["q_out_obs_m3s"].to_numpy(float),
                dt_h=dt_h,
            )
        )
    parameters = pd.read_csv(REAL_DIR / "model_parameters.csv")
    linear = parameters[parameters["model"].eq("muskingum")][
        ["reach_id", "K_h", "X"]
    ].copy()
    linear["reach_id"] = linear["reach_id"].astype(str)
    return events_by_reach, linear


def real_job(payload: tuple[str, list[Event], float, float]) -> tuple[dict, list[dict], list[pd.DataFrame]]:
    reach_id, events, linear_k, linear_x = payload
    train = [event for event in events if event.split == "train"]
    params = fit(train, linear_k, linear_x)
    parameter = {"reach_id": reach_id, **params}
    metrics = []
    hydrographs = []
    for event in events:
        sim = route_nonlinear_muskingum(
            event.q_in,
            float(event.q_obs[0]),
            float(params["K_power"]),
            float(params["X"]),
            float(params["m"]),
            event.dt_h,
        )
        metrics.append(
            {
                "reach_id": reach_id,
                "event_id": event.event_id,
                "split": event.split,
                "model": "Nonlinear Muskingum",
                **metric_row(event.q_obs, sim),
            }
        )
        hydrographs.append(
            pd.DataFrame(
                {
                    "reach_id": reach_id,
                    "event_id": event.event_id,
                    "split": event.split,
                    "time_step": np.arange(1, len(sim) + 1),
                    "q_in_m3s": event.q_in,
                    "q_out_obs_m3s": event.q_obs,
                    "Q_nonlinear_muskingum_m3s": sim,
                }
            )
        )
    return parameter, metrics, hydrographs


def summarize_hec(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("model", as_index=False)
        .agg(
            sections=("NSE", "count"),
            mean_NSE=("NSE", "mean"),
            median_NSE=("NSE", "median"),
            mean_KGE=("KGE", "mean"),
            mean_RMSE_m3s=("RMSE_m3s", "mean"),
            mean_abs_peak_error_percent=("abs_peak_error_percent", "mean"),
        )
    )


def summarize_real(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["model", "split"], as_index=False)
        .agg(
            reaches=("reach_id", "nunique"),
            events=("event_id", "count"),
            mean_NSE=("NSE", "mean"),
            median_NSE=("NSE", "median"),
            mean_KGE=("KGE", "mean"),
            mean_RMSE_m3s=("RMSE_m3s", "mean"),
            mean_abs_PBIAS_percent=("PBIAS_percent", lambda v: float(np.mean(np.abs(v)))),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--hec-limit", type=int, default=0)
    parser.add_argument("--real-limit", type=int, default=0)
    args = parser.parse_args()
    output = next_output_dir()
    start = time.monotonic()

    hec_jobs = load_hec_jobs()
    if args.hec_limit > 0:
        hec_jobs = hec_jobs[: args.hec_limit]
    hec_metrics, hec_parameters, hec_hydrographs = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(hec_job, job) for job in hec_jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            metric, parameter, hydrograph = future.result()
            hec_metrics.append(metric)
            hec_parameters.append(parameter)
            hec_hydrographs.append(hydrograph)
            if index % 24 == 0 or index == len(futures):
                print(f"HEC {index}/{len(futures)}", flush=True)

    nonlinear_hec = pd.DataFrame(hec_metrics)
    baseline_hec = pd.read_csv(HEC_DIR / "model_metrics_long.csv")
    baseline_hec = baseline_hec[baseline_hec["model"].isin(["new_complement", "muskingum"])]
    if args.hec_limit > 0:
        selected = {
            (str(metric["CaseID"]), int(metric["target_xs_index"]))
            for metric in hec_metrics
        }
        baseline_hec = baseline_hec[
            baseline_hec.apply(
                lambda row: (str(row["CaseID"]), int(row["target_xs_index"])) in selected,
                axis=1,
            )
        ]
    baseline_hec["model"] = baseline_hec["model"].map(
        {"new_complement": "LMCS", "muskingum": "Linear Muskingum"}
    )
    hec_all = pd.concat([baseline_hec, nonlinear_hec], ignore_index=True, sort=False)
    hec_all.to_csv(output / "hec_metrics_three_models.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hec_parameters).to_csv(
        output / "hec_nonlinear_parameters.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(hec_hydrographs, ignore_index=True).to_csv(
        output / "hec_nonlinear_hydrographs.csv", index=False, encoding="utf-8-sig"
    )
    summarize_hec(hec_all).to_csv(
        output / "hec_summary_three_models.csv", index=False, encoding="utf-8-sig"
    )

    events_by_reach, linear = load_real_events()
    real_jobs = []
    for row in linear.to_dict("records"):
        reach_id = str(row["reach_id"])
        real_jobs.append((reach_id, events_by_reach[reach_id], float(row["K_h"]), float(row["X"])))
    if args.real_limit > 0:
        real_jobs = real_jobs[: args.real_limit]
    real_parameters, real_metrics, real_hydrographs = [], [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(real_jobs))) as executor:
        futures = [executor.submit(real_job, job) for job in real_jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            parameter, metrics, hydrographs = future.result()
            real_parameters.append(parameter)
            real_metrics.extend(metrics)
            real_hydrographs.extend(hydrographs)
            print(f"Real reach {index}/{len(futures)}", flush=True)

    nonlinear_real = pd.DataFrame(real_metrics)
    baseline_real = pd.read_csv(REAL_DIR / "event_metrics.csv")
    baseline_real = baseline_real[
        baseline_real["model"].isin(["new_complement_s0_solved", "muskingum"])
    ].copy()
    baseline_real["model"] = baseline_real["model"].map(
        {"new_complement_s0_solved": "LMCS", "muskingum": "Linear Muskingum"}
    )
    baseline_real = baseline_real.rename(columns={"RMSE": "RMSE_m3s"})
    if args.real_limit > 0:
        selected = {job[0] for job in real_jobs}
        baseline_real = baseline_real[baseline_real["reach_id"].astype(str).isin(selected)]
    real_all = pd.concat([baseline_real, nonlinear_real], ignore_index=True, sort=False)
    real_all.to_csv(output / "real_event_metrics_three_models.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(real_parameters).to_csv(
        output / "real_nonlinear_parameters.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(real_hydrographs, ignore_index=True).to_csv(
        output / "real_nonlinear_hydrographs.csv", index=False, encoding="utf-8-sig"
    )
    summarize_real(real_all).to_csv(
        output / "real_summary_three_models.csv", index=False, encoding="utf-8-sig"
    )

    elapsed = time.monotonic() - start
    (output / "README.md").write_text(
        "# Current nonlinear Muskingum benchmark\n\n"
        "Models: LMCS, standard three-parameter nonlinear Muskingum, and linear Muskingum.\n"
        "Nonlinear storage: S=K[XI+(1-X)Q]^m. Parameters are T_ref, X, and m; K is derived at Q_ref.\n"
        f"Bounds: T_ref=[{T_REF_MIN_H},{T_REF_MAX_H}] h, X=[0,0.5], m=[{M_MIN},{M_MAX}].\n"
        "All models route total flow. HEC fits are section-specific. Real-reach parameters use training events only.\n"
        f"Runtime: {elapsed:.1f} s.\n",
        encoding="utf-8",
    )
    print(f"OUTPUT={output}", flush=True)
    print(f"RUNTIME_SECONDS={elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()

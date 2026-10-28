from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(r"E:\01-HUST\Nature\03\HECRAS\codex\research_package_current")
RAW_PATH = (
    ROOT
    / "44_calibration_raw_flow_data"
    / "hecras_idealized"
    / "hecras_calibration_raw_flow_common.csv"
)
RWEC_ARCHIVE_PATH = (
    ROOT
    / "04_alternative_complement_weight_model"
    / "ext96_rwec_two_parameter_original_parameter_library.csv"
)
OUTPUT_ROOT = ROOT / "88_new_complement_rwec_muskingum_ext96"
MODEL_CODE_DIR = Path(r"E:\01-HUST\Nature\03\Code")

EPS = 1.0e-12
TAU_MIN_H = 0.001
TAU_MAX_H = 150.0
K_MIN = 1.0e-6
K_MAX = 100.0
N_MIN = 0.01
N_MAX = 300.0
MUSKINGUM_K_MIN_H = 0.5
MUSKINGUM_K_MAX_H = 100.0

PILOT_CASES = (
    "G1_S001_N025_H1",
    "G2_S0002_N035_H3",
    "G4_S00005_N080_H1",
)


def import_models() -> tuple[object, object]:
    import sys

    if str(MODEL_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_CODE_DIR))
    import routing as baseline
    import routing_aut_complement_3p as complement

    return baseline, complement


def next_output_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for child in root.iterdir():
        if child.is_dir() and len(child.name) >= 3 and child.name[:3].isdigit():
            existing.append(int(child.name[:3]))
    target = root / f"{max(existing, default=0) + 1:03d}_ext96_comparison"
    target.mkdir(parents=True, exist_ok=False)
    return target


def normalized_sse(simulated: np.ndarray, observed: np.ndarray) -> float:
    if not np.all(np.isfinite(simulated)):
        return 1.0e12
    denominator = float(np.sum((observed - np.mean(observed)) ** 2))
    if denominator <= EPS:
        return 1.0e12
    return float(np.sum((simulated - observed) ** 2)) / denominator


def kge(simulated: np.ndarray, observed: np.ndarray) -> float:
    if np.std(simulated) <= EPS or np.std(observed) <= EPS:
        return float("nan")
    observed_mean = float(np.mean(observed))
    if abs(observed_mean) <= EPS:
        return float("nan")
    correlation = float(np.corrcoef(simulated, observed)[0, 1])
    alpha = float(np.std(simulated) / np.std(observed))
    beta = float(np.mean(simulated) / observed_mean)
    return 1.0 - math.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def metrics(simulated: np.ndarray, observed: np.ndarray, time_h: np.ndarray) -> dict[str, float]:
    loss = normalized_sse(simulated, observed)
    if loss >= 1.0e11:
        return {
            "loss": loss,
            "nse_event": float("nan"),
            "kge_event": float("nan"),
            "rmse_event_m3s": float("nan"),
            "qpeak_event_error_percent": float("nan"),
            "peak_time_error_h": float("nan"),
            "volume_ratio_event_sim_obs": float("nan"),
            "min_sim_event_m3s": float("nan"),
        }
    peak_obs = int(np.argmax(observed))
    peak_sim = int(np.argmax(simulated))
    observed_peak = float(observed[peak_obs])
    simulated_peak = float(simulated[peak_sim])
    observed_volume = float(np.trapz(observed, time_h))
    simulated_volume = float(np.trapz(simulated, time_h))
    return {
        "loss": loss,
        "nse_event": 1.0 - loss,
        "kge_event": kge(simulated, observed),
        "rmse_event_m3s": float(np.sqrt(np.mean((simulated - observed) ** 2))),
        "qpeak_event_error_percent": (
            (simulated_peak - observed_peak) / observed_peak * 100.0
            if abs(observed_peak) > EPS
            else float("nan")
        ),
        "peak_time_error_h": float(time_h[peak_sim] - time_h[peak_obs]),
        "volume_ratio_event_sim_obs": (
            simulated_volume / observed_volume if abs(observed_volume) > EPS else float("nan")
        ),
        "min_sim_event_m3s": float(np.min(simulated)),
    }


def complement_starts(reach_length_km: float, previous: tuple[float, float, float] | None) -> list[tuple[float, float, float]]:
    tau_scale = min(TAU_MAX_H, max(TAU_MIN_H, reach_length_km / 6.0))
    starts = [] if previous is None else [previous]
    starts.extend(
        [
            (0.03, 2.0, max(TAU_MIN_H, tau_scale / 3.0)),
            (0.12, 8.0, max(TAU_MIN_H, tau_scale)),
            (0.35, 16.0, min(TAU_MAX_H, tau_scale * 2.0)),
            (0.75, 4.0, min(TAU_MAX_H, max(1.0, reach_length_km / 2.0))),
            (0.08, 40.0, min(TAU_MAX_H, max(2.0, reach_length_km))),
            (3.0, 100.0, min(TAU_MAX_H, max(2.0, reach_length_km / 2.0))),
            (20.0, 120.0, min(TAU_MAX_H, max(2.0, reach_length_km))),
        ]
    )
    return starts


def route_new_complement(
    complement: object,
    inflow_event: np.ndarray,
    k: float,
    n_comp: float,
    tau_h: float,
    dt_h: float,
    pre_lag_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    result = complement.route_lagged_complement_single_k_solved_s0(
        q_in=inflow_event,
        k=k,
        n_comp=n_comp,
        tau_h=tau_h,
        dt_h=dt_h,
        pre_lag_mode=pre_lag_mode,
    )
    return np.asarray(result["Q_sim"], dtype=float), np.asarray(result["S_t"], dtype=float)


def route_rwec_archive(
    baseline: object,
    inflow_full: np.ndarray,
    n_r: float,
    tau_h: float,
    baseflow_m3s: float,
    dt_h: float,
) -> np.ndarray:
    def archived_storage(storage_water: float, storage_energy: float) -> float:
        if storage_water <= 0.0 or storage_energy <= 0.0:
            return 0.0
        a = n_r * math.log(storage_water)
        b = n_r * math.log(storage_energy)
        top = max(a, b)
        log_denom = (top + math.log(math.exp(a - top) + math.exp(b - top))) / n_r
        return math.exp(math.log(storage_water) + math.log(storage_energy) - log_denom)

    simulated = np.zeros_like(inflow_full, dtype=float)
    storage_previous = 0.0
    simulated[0] = float(inflow_full[0])
    outflow_previous = simulated[0]
    for index in range(1, len(inflow_full)):
        storage_drive = (
            float(inflow_full[index])
            + float(inflow_full[index - 1])
            - 2.0 * baseflow_m3s
        )
        storage_water = max(
            EPS, storage_previous + dt_h / 2.0 * storage_drive * 3600.0 / 1.0e6
        )
        storage_energy = max(
            EPS, storage_previous + tau_h / 2.0 * storage_drive * 3600.0 / 1.0e6
        )
        storage_current = archived_storage(storage_water, storage_energy)
        if not np.isfinite(storage_current):
            simulated[:] = np.nan
            return simulated
        simulated[index] = (
            -2.0 * (storage_current - storage_previous) * 1.0e6 / (dt_h * 3600.0)
            + float(inflow_full[index])
            + float(inflow_full[index - 1])
            - outflow_previous
        )
        storage_previous = storage_current
        outflow_previous = simulated[index]
    return simulated


def fit_new_complement(
    complement: object,
    inflow_event: np.ndarray,
    observed_event: np.ndarray,
    time_h: np.ndarray,
    reach_length_km: float,
    previous: tuple[float, float, float] | None,
    maxiter: int,
    pre_lag_mode: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    dt_h = float(np.median(np.diff(time_h)))

    def objective(log_params: np.ndarray) -> float:
        k, n_comp, tau_h = (float(math.exp(value)) for value in log_params)
        simulated, _ = route_new_complement(
            complement, inflow_event, k, n_comp, tau_h, dt_h, pre_lag_mode
        )
        return normalized_sse(simulated, observed_event)

    bounds = (
        (math.log(K_MIN), math.log(K_MAX)),
        (math.log(N_MIN), math.log(N_MAX)),
        (math.log(TAU_MIN_H), math.log(TAU_MAX_H)),
    )
    best = None
    total_evaluations = 0
    for start in complement_starts(reach_length_km, previous):
        clipped = (
            min(K_MAX, max(K_MIN, start[0])),
            min(N_MAX, max(N_MIN, start[1])),
            min(TAU_MAX_H, max(TAU_MIN_H, start[2])),
        )
        result = minimize(
            objective,
            np.log(np.asarray(clipped, dtype=float)),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1.0e-10, "gtol": 1.0e-6, "maxls": 25},
        )
        total_evaluations += int(result.nfev)
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("New-complement optimization returned no result.")

    k, n_comp, tau_h = (float(math.exp(value)) for value in best.x)
    simulated, storage = route_new_complement(
        complement, inflow_event, k, n_comp, tau_h, dt_h, pre_lag_mode
    )
    return (
        {
            "k": k,
            "n_comp": n_comp,
            "tau_h": tau_h,
            "optimizer_success": bool(best.success),
            "n_evaluations": total_evaluations,
            **metrics(simulated, observed_event, time_h),
        },
        simulated,
        storage,
    )


def muskingum_x_max(k_h: float, dt_h: float) -> float:
    return max(0.0, min(0.5, dt_h / (2.0 * k_h), 1.0 - dt_h / (2.0 * k_h)))


def route_muskingum(
    inflow_event: np.ndarray,
    q0_event: float,
    k_h: float,
    x: float,
    dt_h: float,
) -> np.ndarray:
    denominator = 2.0 * k_h * (1.0 - x) + dt_h
    c0 = (dt_h - 2.0 * k_h * x) / denominator
    c1 = (dt_h + 2.0 * k_h * x) / denominator
    c2 = (2.0 * k_h * (1.0 - x) - dt_h) / denominator
    simulated = np.zeros_like(inflow_event, dtype=float)
    simulated[0] = max(0.0, q0_event)
    for index in range(1, len(inflow_event)):
        simulated[index] = max(
            0.0,
            c0 * inflow_event[index]
            + c1 * inflow_event[index - 1]
            + c2 * simulated[index - 1],
        )
    return simulated


def muskingum_starts(reach_length_km: float) -> list[tuple[float, float]]:
    return [
        (max(MUSKINGUM_K_MIN_H, reach_length_km / 20.0), 0.1),
        (max(MUSKINGUM_K_MIN_H, reach_length_km / 8.0), 0.5),
        (max(MUSKINGUM_K_MIN_H, reach_length_km / 3.0), 0.9),
        (min(MUSKINGUM_K_MAX_H, max(1.0, reach_length_km)), 0.2),
    ]


def fit_muskingum(
    inflow_event: np.ndarray,
    observed_event: np.ndarray,
    time_h: np.ndarray,
    reach_length_km: float,
    maxiter: int,
) -> tuple[dict[str, float], np.ndarray]:
    dt_h = float(np.median(np.diff(time_h)))

    def objective(params: np.ndarray) -> float:
        k_h = float(math.exp(params[0]))
        x = float(params[1]) * muskingum_x_max(k_h, dt_h)
        simulated = route_muskingum(inflow_event, observed_event[0], k_h, x, dt_h)
        return normalized_sse(simulated, observed_event)

    best = None
    total_evaluations = 0
    bounds = ((math.log(MUSKINGUM_K_MIN_H), math.log(MUSKINGUM_K_MAX_H)), (0.0, 1.0))
    for k0, fraction0 in muskingum_starts(reach_length_km):
        result = minimize(
            objective,
            np.asarray([math.log(min(MUSKINGUM_K_MAX_H, k0)), fraction0]),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1.0e-10, "gtol": 1.0e-6, "maxls": 25},
        )
        total_evaluations += int(result.nfev)
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("Muskingum optimization returned no result.")

    k_h = float(math.exp(best.x[0]))
    x = float(best.x[1]) * muskingum_x_max(k_h, dt_h)
    simulated = route_muskingum(inflow_event, observed_event[0], k_h, x, dt_h)
    return (
        {
            "K_h": k_h,
            "X": x,
            "optimizer_success": bool(best.success),
            "n_evaluations": total_evaluations,
            **metrics(simulated, observed_event, time_h),
        },
        simulated,
    )


def load_raw(case_ids: set[str]) -> pd.DataFrame:
    usecols = [
        "CaseID",
        "Geometry",
        "S0",
        "ManningN",
        "Wave",
        "ExperimentGroup",
        "target_xs_index",
        "target_rs_m",
        "t_h",
        "Q_target_HECRAS_m3s",
        "Q_upstream_inflow_m3s",
        "Q_b_m3s",
    ]
    raw = pd.read_csv(RAW_PATH, usecols=usecols)
    return raw[raw["CaseID"].isin(case_ids)].copy()


def summarize(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = ("new_complement", "rwec_archive", "muskingum")
    rows = []
    for _, row in comparison.iterrows():
        for model in models:
            rows.append(
                {
                    "CaseID": row["CaseID"],
                    "Geometry": row["Geometry"],
                    "ExperimentGroup": row["ExperimentGroup"],
                    "target_xs_index": row["target_xs_index"],
                    "model": model,
                    "NSE": row[f"{model}_nse_event"],
                    "KGE": row[f"{model}_kge_event"],
                    "RMSE_m3s": row[f"{model}_rmse_event_m3s"],
                    "abs_peak_error_percent": abs(row[f"{model}_qpeak_event_error_percent"]),
                }
            )
    long = pd.DataFrame(rows)
    group_summary = (
        long.groupby(["Geometry", "model"], as_index=False)
        .agg(
            sections=("NSE", "count"),
            mean_NSE=("NSE", "mean"),
            median_NSE=("NSE", "median"),
            mean_KGE=("KGE", "mean"),
            mean_RMSE_m3s=("RMSE_m3s", "mean"),
            mean_abs_peak_error_percent=("abs_peak_error_percent", "mean"),
        )
    )
    overall = (
        long.groupby("model", as_index=False)
        .agg(
            sections=("NSE", "count"),
            mean_NSE=("NSE", "mean"),
            median_NSE=("NSE", "median"),
            mean_KGE=("KGE", "mean"),
            mean_RMSE_m3s=("RMSE_m3s", "mean"),
            mean_abs_peak_error_percent=("abs_peak_error_percent", "mean"),
        )
    )
    return long, group_summary, overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare today's new complement model against archived RWEC and Muskingum on ext96."
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--maxiter", type=int, default=160)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--no-baseflow-subtraction", action="store_true")
    parser.add_argument(
        "--pre-lag-mode",
        choices=("instant_inflow", "steady_initial"),
        default="steady_initial",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.maxiter <= 0:
        raise ValueError("--maxiter must be positive.")
    archive = pd.read_csv(RWEC_ARCHIVE_PATH)
    all_case_ids = list(dict.fromkeys(archive["CaseID"].tolist()))
    if args.all_cases:
        selected_cases = all_case_ids
    elif args.case_id:
        selected_cases = list(dict.fromkeys(args.case_id))
    else:
        selected_cases = list(PILOT_CASES)
    unknown = sorted(set(selected_cases) - set(all_case_ids))
    if unknown:
        raise ValueError(f"Unknown CaseID values: {unknown}")

    baseline, complement = import_models()
    raw = load_raw(set(selected_cases))
    output_dir = next_output_dir(args.output_root.resolve())
    start_time = time.time()
    new_rows = []
    musk_rows = []
    rwec_rows = []
    prediction_rows = []
    subtract_baseflow = not args.no_baseflow_subtraction

    archive = archive[archive["CaseID"].isin(selected_cases)].copy()
    archive = archive.set_index(["CaseID", "target_xs_index"], drop=False)

    for case_position, case_id in enumerate(selected_cases, start=1):
        case = raw[raw["CaseID"].eq(case_id)]
        previous: tuple[float, float, float] | None = None
        section_indices = sorted(case["target_xs_index"].unique())
        case_start = time.time()
        for section_position, section_index in enumerate(section_indices, start=1):
            series = case[case["target_xs_index"].eq(section_index)].sort_values("t_h")
            first = series.iloc[0]
            time_h = series["t_h"].to_numpy(dtype=float)
            if len(time_h) < 3 or np.any(np.diff(time_h) <= 0.0):
                raise ValueError(f"Invalid time axis for {case_id}, xs {section_index}.")
            baseflow = float(np.median(series["Q_b_m3s"].to_numpy(dtype=float)))
            inflow_full = series["Q_upstream_inflow_m3s"].to_numpy(dtype=float)
            observed_full = series["Q_target_HECRAS_m3s"].to_numpy(dtype=float)
            if subtract_baseflow:
                inflow_event = np.maximum(0.0, inflow_full - baseflow)
                observed_event = np.maximum(0.0, observed_full - baseflow)
            else:
                inflow_event = inflow_full.copy()
                observed_event = observed_full.copy()
            reach_length_km = (120000.0 - float(first["target_rs_m"])) / 1000.0
            new_fit, new_sim, new_storage = fit_new_complement(
                complement,
                inflow_event,
                observed_event,
                time_h,
                reach_length_km,
                previous,
                args.maxiter,
                args.pre_lag_mode,
            )
            previous = (new_fit["k"], new_fit["n_comp"], new_fit["tau_h"])
            musk_fit, musk_sim = fit_muskingum(
                inflow_event, observed_event, time_h, reach_length_km, args.maxiter
            )
            archive_row = archive.loc[(case_id, int(section_index))]
            dt_h = float(np.median(np.diff(time_h)))
            rwec_full = route_rwec_archive(
                baseline,
                inflow_full,
                float(archive_row["n_r"]),
                float(archive_row["tau_h"]),
                baseflow,
                dt_h,
            )
            rwec_sim = rwec_full - baseflow if subtract_baseflow else rwec_full
            rwec_fit = {
                "n_r": float(archive_row["n_r"]),
                "tau_h": float(archive_row["tau_h"]),
                "archived_nse_event": float(archive_row["nse_event"]),
                **metrics(rwec_sim, observed_event, time_h),
            }
            metadata = {
                "CaseID": case_id,
                "Geometry": str(first["Geometry"]),
                "S0": float(first["S0"]),
                "ManningN": float(first["ManningN"]),
                "Wave": str(first["Wave"]),
                "ExperimentGroup": str(first["ExperimentGroup"]),
                "target_xs_index": int(section_index),
                "target_rs_m": float(first["target_rs_m"]),
                "reach_length_km": reach_length_km,
                "Q_b_m3s": baseflow,
            }
            new_rows.append({**metadata, **new_fit, "initial_storage_million_m3": float(new_storage[0])})
            musk_rows.append({**metadata, **musk_fit})
            rwec_rows.append({**metadata, **rwec_fit})
            for index, t_h in enumerate(time_h):
                prediction_rows.append(
                    {
                        **metadata,
                        "t_h": float(t_h),
                        "Q_in_event_m3s": float(inflow_event[index]),
                        "Q_obs_event_m3s": float(observed_event[index]),
                        "Q_new_complement_event_m3s": float(new_sim[index]),
                        "Q_rwec_event_m3s": float(rwec_sim[index]),
                        "Q_muskingum_event_m3s": float(musk_sim[index]),
                        "S_new_complement_million_m3": float(new_storage[index]),
                    }
                )
            print(
                f"{case_position:03d}/{len(selected_cases):03d} {case_id} "
                f"xs {section_position:02d}/{len(section_indices):02d} "
                f"new NSE={new_fit['nse_event']:.4f} musk NSE={musk_fit['nse_event']:.4f}",
                flush=True,
            )
        print(
            f"CASE DONE {case_position:03d}/{len(selected_cases):03d} {case_id} "
            f"in {time.time() - case_start:.1f}s",
            flush=True,
        )

    new_df = pd.DataFrame(new_rows).rename(
        columns={column: f"new_complement_{column}" for column in [
            "loss", "nse_event", "kge_event", "rmse_event_m3s", "qpeak_event_error_percent",
            "peak_time_error_h", "volume_ratio_event_sim_obs", "min_sim_event_m3s",
        ]}
    )
    musk_df = pd.DataFrame(musk_rows).rename(
        columns={column: f"muskingum_{column}" for column in [
            "loss", "nse_event", "kge_event", "rmse_event_m3s", "qpeak_event_error_percent",
            "peak_time_error_h", "volume_ratio_event_sim_obs", "min_sim_event_m3s",
        ]}
    )
    rwec_df = pd.DataFrame(rwec_rows).rename(columns={
        "n_r": "rwec_archive_n_r",
        "tau_h": "rwec_archive_tau_h",
        "loss": "rwec_archive_loss",
        "nse_event": "rwec_archive_nse_event",
        "kge_event": "rwec_archive_kge_event",
        "rmse_event_m3s": "rwec_archive_rmse_event_m3s",
        "qpeak_event_error_percent": "rwec_archive_qpeak_event_error_percent",
        "peak_time_error_h": "rwec_archive_peak_time_error_h",
        "volume_ratio_event_sim_obs": "rwec_archive_volume_ratio_event_sim_obs",
        "min_sim_event_m3s": "rwec_archive_min_sim_event_m3s",
    })
    identity_columns = [
        "CaseID", "Geometry", "S0", "ManningN", "Wave", "ExperimentGroup", "target_xs_index",
        "target_rs_m", "reach_length_km", "Q_b_m3s",
    ]
    comparison = new_df.merge(
        musk_df.drop(columns=[column for column in identity_columns if column not in ("CaseID", "target_xs_index")]),
        on=["CaseID", "target_xs_index"],
        how="inner",
    ).merge(
        rwec_df.drop(columns=[column for column in identity_columns if column not in ("CaseID", "target_xs_index")]),
        on=["CaseID", "target_xs_index"],
        how="inner",
    )

    long, geometry_summary, overall_summary = summarize(comparison)
    new_df.to_csv(output_dir / "new_complement_parameters_metrics.csv", index=False, encoding="utf-8-sig")
    musk_df.to_csv(output_dir / "muskingum_parameters_metrics.csv", index=False, encoding="utf-8-sig")
    rwec_df.to_csv(output_dir / "rwec_archive_parameters_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "section_comparison.csv", index=False, encoding="utf-8-sig")
    long.to_csv(output_dir / "model_metrics_long.csv", index=False, encoding="utf-8-sig")
    geometry_summary.to_csv(output_dir / "geometry_summary.csv", index=False, encoding="utf-8-sig")
    overall_summary.to_csv(output_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(output_dir / "pilot_hydrographs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {
            "cases": ";".join(selected_cases),
            "sections": len(comparison),
            "Q_b_m3s": 100.0,
            "baseflow_subtracted_from_new_and_muskingum": subtract_baseflow,
            "new_complement_parameters": "k,n,tau; S0 solved from initial routed inflow",
            "new_complement_bounds": f"k=[{K_MIN},{K_MAX}], n=[{N_MIN},{N_MAX}], tau=[{TAU_MIN_H},{TAU_MAX_H}] h",
            "pre_lag_mode": args.pre_lag_mode,
            "muskingum_bounds": f"K=[{MUSKINGUM_K_MIN_H},{MUSKINGUM_K_MAX_H}] h; nonnegative coefficients",
            "rwec_source": str(RWEC_ARCHIVE_PATH),
            "raw_source": str(RAW_PATH),
            "maxiter": args.maxiter,
        }
    ]).to_csv(output_dir / "run_settings.csv", index=False, encoding="utf-8-sig")
    (output_dir / "README.md").write_text(
        "# ext96 new-complement comparison\n\n"
        "This run compares the current solved-S0 complement model with archived original RWEC "
        "and a newly calibrated Muskingum baseline. RWEC always retains its archived Qb=100 m3/s "
        "definition; the optional no-baseflow-subtraction run routes total flow in the other two models.\n",
        encoding="utf-8",
    )
    print(f"DONE {len(comparison)} sections in {time.time() - start_time:.1f}s -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()

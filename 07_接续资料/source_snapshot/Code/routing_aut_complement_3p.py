#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import routing as baseline


PROJECT_DIR = Path(r"E:\01-HUST\Nature\03")
DEFAULT_OUT = PROJECT_DIR / "Result" / "routing_aut_complement_3p"
EPS = 1.0e-12
N_COMP = 16.0
FLOW_EXPONENT = 2.0
INITIAL_STORAGE_OFFSET_STEPS = 0.0
COMPLEMENT_MODEL_NAME = "complement_3p_s0_tau_q0"
COMPLEMENT_DYNAMIC_K_N_MODEL_NAME = "complement_dynamic_k_n_4p"
COMPLEMENT_SINGLE_K_MODEL_NAME = "complement_single_k_solved_s0"
COMPLEMENT_K_N_TAU_MODEL_NAME = "complement_k_n_tau_solved_s0"
RWEC_MODEL_NAME = "rwec_small_qb_solved_s0"
RWEC_DEFAULT_BASEFLOW_M3S = 10.0
COMPLEMENT_N_BOUNDS = (0.5, 80.0)


def next_numbered_output_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^(\d+)_")
    numbers = []
    for child in base_dir.iterdir():
        if child.is_dir():
            match = pattern.match(child.name)
            if match:
                numbers.append(int(match.group(1)))
    number = max(numbers, default=0) + 1
    while True:
        out = base_dir / f"{number:03d}_aut_complement_3p"
        try:
            out.mkdir(parents=False, exist_ok=False)
            return out
        except FileExistsError:
            number += 1


def reference_inflow(events: Sequence[baseline.Event]) -> float:
    values = np.concatenate(
        [event.q_in[np.isfinite(event.q_in) & (event.q_in > 0.0)] for event in events]
    )
    return max(EPS, float(np.median(values)))


def route_aut_complement_3p(
    q_in: np.ndarray,
    q0: float,
    k_rise: float,
    k_fall: float,
    tau_h: float,
    I_ref: float,
    n_comp: float = N_COMP,
    dt_h: float = 1.0,
) -> dict[str, np.ndarray]:
    q_in = np.asarray(q_in, dtype=float)
    n_points = len(q_in)
    time_h = np.arange(n_points, dtype=float) * dt_h
    dt_seconds = dt_h * 3600.0

    Q_sim = np.zeros(n_points, dtype=float)
    S_t_series = np.zeros(n_points, dtype=float)
    S_w_series = np.zeros(n_points, dtype=float)
    S_p_series = np.zeros(n_points, dtype=float)
    I_bar_series = np.zeros(n_points, dtype=float)
    k_t_series = np.zeros(n_points, dtype=float)
    rising_series = np.zeros(n_points, dtype=bool)
    before_tau_series = np.zeros(n_points, dtype=bool)

    # Initial storage is the travel-time volume S0 = tau * Q0.
    initial_time_h = tau_h
    S_prev = max(0.0, initial_time_h * q0 * 3600.0 / 1.0e6)
    previous_I_bar: float | None = None

    for t in range(n_points):
        if time_h[t] < tau_h:
            Q_sim[t] = float(q_in[t])
            S_t_series[t] = S_prev
            before_tau_series[t] = True
            continue

        if t == 0:
            Q_sim[t] = float(q0)
            S_t_series[t] = S_prev
            continue

        _, _, I_bar = baseline.lagged_period_mean_inflow(
            q_in=q_in,
            time_h=time_h,
            t_h=time_h[t],
            tau_h=tau_h,
            dt_h=dt_h,
        )
        I_bar = max(0.0, I_bar)
        is_rising = previous_I_bar is None or I_bar >= previous_I_bar
        coefficient = k_rise if is_rising else k_fall
        flow_ratio = (I_bar + EPS) / max(EPS, I_ref)
        k_t = float(np.clip(coefficient * flow_ratio**FLOW_EXPONENT, 0.0, 1.0))

        inflow_volume = I_bar * dt_seconds / 1.0e6
        S_w = max(EPS, S_prev + inflow_volume)
        S_p = max(EPS, S_prev + k_t * inflow_volume)
        S_t = baseline.solve_S_t_by_complementary_equation(S_w, S_p, n_comp)

        if not np.isfinite(S_t):
            Q_sim[:] = np.nan
            S_t_series[:] = np.nan
            break

        outflow_volume = max(0.0, S_w - S_t)
        Q_sim[t] = outflow_volume * 1.0e6 / dt_seconds
        S_t_series[t] = S_t
        S_w_series[t] = S_w
        S_p_series[t] = S_p
        I_bar_series[t] = I_bar
        k_t_series[t] = k_t
        rising_series[t] = is_rising
        S_prev = S_t
        previous_I_bar = I_bar

    return {
        "Q_sim": Q_sim,
        "S_t": S_t_series,
        "S_w": S_w_series,
        "S_p": S_p_series,
        "I_bar": I_bar_series,
        "k_t": k_t_series,
        "is_rising": rising_series,
        "before_tau": before_tau_series,
    }


def solve_lagged_complement_initial_storage(
    q_initial: float,
    k: float,
    n_comp: float,
    dt_h: float,
) -> float:
    """Solve S0 from the initial lagged-inflow complement equilibrium.

    The initial period receives I0 * dt. S0 is calculated, rather than
    calibrated, so the single-k model retains only k and tau as parameters.
    """
    inflow = max(0.0, float(q_initial))
    if inflow <= EPS or k <= EPS or n_comp <= EPS:
        return 0.0

    inflow_volume = inflow * dt_h * 3600.0 / 1.0e6
    potential_volume = k * inflow_volume
    if inflow_volume <= EPS or potential_volume <= EPS:
        return 0.0

    def residual(storage: float) -> float:
        return (storage / (storage + inflow_volume)) ** n_comp + (
            storage / (storage + potential_volume)
        ) ** n_comp - 1.0

    lower = 0.0
    upper = max(inflow_volume, potential_volume, 1.0e-9)
    while residual(upper) < 0.0 and upper < 1.0e9:
        upper *= 2.0

    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if residual(middle) > 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def route_lagged_complement_single_k_solved_s0(
    q_in: np.ndarray,
    k: float,
    tau_h: float,
    n_comp: float = N_COMP,
    dt_h: float = 1.0,
    pre_lag_mode: str = "steady_initial",
    q0: float | None = None,
) -> dict[str, np.ndarray]:
    """Lagged-inflow complement routing with one calibrated k and calculated S0.

    No baseflow is subtracted. ``instant_inflow`` preserves the original
    Q(t)=I(t) handling before tau. ``steady_initial`` applies the normal
    storage recurrence from the first full time step. When ``q0`` is supplied,
    it only initializes Q(0). Initial storage and the pre-event lag boundary
    remain driven by I(0), consistently with V0 = I0 * dt.
    """
    q_in = np.asarray(q_in, dtype=float)
    if pre_lag_mode not in {"instant_inflow", "steady_initial"}:
        raise ValueError(f"Unsupported pre_lag_mode: {pre_lag_mode}")
    n_points = len(q_in)
    time_h = np.arange(n_points, dtype=float) * dt_h
    dt_seconds = dt_h * 3600.0

    Q_sim = np.zeros(n_points, dtype=float)
    S_t_series = np.zeros(n_points, dtype=float)
    S_w_series = np.zeros(n_points, dtype=float)
    S_p_series = np.zeros(n_points, dtype=float)
    I_bar_series = np.zeros(n_points, dtype=float)
    k_t_series = np.zeros(n_points, dtype=float)
    before_tau_series = np.zeros(n_points, dtype=bool)

    initial_inflow = max(0.0, float(q_in[0]))
    initial_outflow = max(0.0, float(initial_inflow if q0 is None else q0))
    S_prev = solve_lagged_complement_initial_storage(initial_inflow, k, n_comp, dt_h)

    for t in range(n_points):
        before_tau = time_h[t] < tau_h
        before_tau_series[t] = before_tau

        if t == 0:
            Q_sim[t] = initial_outflow
            S_t_series[t] = S_prev
            continue

        if before_tau and pre_lag_mode == "instant_inflow":
            Q_sim[t] = float(q_in[t])
            S_t_series[t] = S_prev
            continue

        _, _, I_bar = baseline.lagged_period_mean_inflow(
            q_in=q_in,
            time_h=time_h,
            t_h=time_h[t],
            tau_h=tau_h,
            dt_h=dt_h,
        )
        I_bar = max(0.0, I_bar)
        inflow_volume = I_bar * dt_seconds / 1.0e6
        S_w = max(EPS, S_prev + inflow_volume)
        S_p = max(EPS, S_prev + k * inflow_volume)
        S_t = baseline.solve_S_t_by_complementary_equation(S_w, S_p, n_comp)

        if not np.isfinite(S_t):
            Q_sim[:] = np.nan
            S_t_series[:] = np.nan
            break

        Q_sim[t] = max(0.0, (S_w - S_t) * 1.0e6 / dt_seconds)
        S_t_series[t] = S_t
        S_w_series[t] = S_w
        S_p_series[t] = S_p
        I_bar_series[t] = I_bar
        k_t_series[t] = k
        S_prev = S_t

    return {
        "Q_sim": Q_sim,
        "S_t": S_t_series,
        "S_w": S_w_series,
        "S_p": S_p_series,
        "I_bar": I_bar_series,
        "k_t": k_t_series,
        "before_tau": before_tau_series,
    }


def solve_rwec_initial_storage(
    q_initial: float,
    n_r: float,
    tau_h: float,
    baseflow_m3s: float,
    dt_h: float,
) -> float:
    """Solve the RWEC initial equilibrium without adding an S0 parameter."""
    excess = max(0.0, float(q_initial) - float(baseflow_m3s))
    if excess <= EPS:
        return 0.0

    v_w = dt_h * excess * 3600.0 / 1.0e6
    v_e = tau_h * excess * 3600.0 / 1.0e6
    if v_w <= EPS or v_e <= EPS:
        return 0.0

    def residual(storage: float) -> float:
        return (storage / (storage + v_w)) ** n_r + (
            storage / (storage + v_e)
        ) ** n_r - 1.0

    lower = 0.0
    upper = max(v_w, v_e, 1.0e-9)
    while residual(upper) < 0.0 and upper < 1.0e9:
        upper *= 2.0

    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if residual(middle) > 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def route_rwec_original(
    q_in: np.ndarray,
    n_r: float,
    tau_h: float,
    baseflow_m3s: float,
    dt_h: float = 1.0,
) -> dict[str, np.ndarray]:
    """Route full discharge with the original two-parameter RWEC recursion.

    Q_b is a fixed small ecological baseflow. S_0 is solved from the initial
    RWEC equilibrium using I_0, n_R, tau, Q_b, and the time step.
    """
    q_in = np.asarray(q_in, dtype=float)
    n_points = len(q_in)
    dt_seconds = dt_h * 3600.0

    Q_sim = np.zeros(n_points, dtype=float)
    S_t_series = np.zeros(n_points, dtype=float)
    S_w_series = np.zeros(n_points, dtype=float)
    S_e_series = np.zeros(n_points, dtype=float)
    storage_drive_series = np.zeros(n_points, dtype=float)

    # Q_0=I_0; S_0 is a calculated state, not an additional calibration parameter.
    Q_sim[0] = float(q_in[0])
    S_prev = solve_rwec_initial_storage(
        q_initial=Q_sim[0],
        n_r=n_r,
        tau_h=tau_h,
        baseflow_m3s=baseflow_m3s,
        dt_h=dt_h,
    )
    S_t_series[0] = S_prev
    Q_prev = Q_sim[0]

    for t in range(1, n_points):
        storage_drive = float(q_in[t] + q_in[t - 1] - 2.0 * baseflow_m3s)
        S_w = max(EPS, S_prev + 0.5 * dt_h * storage_drive * 3600.0 / 1.0e6)
        S_e = max(EPS, S_prev + 0.5 * tau_h * storage_drive * 3600.0 / 1.0e6)
        S_t = baseline.solve_S_t_by_complementary_equation(S_w, S_e, n_r)

        if not np.isfinite(S_t):
            Q_sim[:] = np.nan
            S_t_series[:] = np.nan
            break

        Q_t = (
            -2.0 * (S_t - S_prev) * 1.0e6 / dt_seconds
            + float(q_in[t] + q_in[t - 1] - Q_prev)
        )
        Q_sim[t] = Q_t
        S_t_series[t] = S_t
        S_w_series[t] = S_w
        S_e_series[t] = S_e
        storage_drive_series[t] = storage_drive
        S_prev = S_t
        Q_prev = Q_t

    return {
        "Q_sim": Q_sim,
        "S_t": S_t_series,
        "S_w": S_w_series,
        "S_e": S_e_series,
        "storage_drive": storage_drive_series,
    }


def complement_objective(
    params: np.ndarray,
    events: Sequence[baseline.Event],
    I_ref: float,
    dt_h: float,
) -> float:
    k_rise, k_fall, tau_h = (float(x) for x in params)
    losses = []
    for event in events:
        result = route_aut_complement_3p(
            q_in=event.q_in,
            q0=float(event.q_out[0]),
            k_rise=k_rise,
            k_fall=k_fall,
            tau_h=tau_h,
            I_ref=I_ref,
            dt_h=dt_h,
        )
        losses.append(baseline.normalized_sse(result["Q_sim"], event.q_out))
    return float(np.mean(losses)) if losses else 1.0e12


def calibrate_complement(
    events: Sequence[baseline.Event],
    I_ref: float,
    dt_h: float,
    seed: int,
    maxiter: int,
    popsize: int,
    k_min: float,
    tau_max_h: float,
) -> dict[str, float]:
    result = differential_evolution(
        lambda params: complement_objective(params, events, I_ref, dt_h),
        bounds=((k_min, 1.0), (k_min, 1.0), (0.0, tau_max_h)),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    return {
        "k_rise": float(result.x[0]),
        "k_fall": float(result.x[1]),
        "tau_h": float(result.x[2]),
        "train_loss": float(result.fun),
    }


def dynamic_k_n_complement_objective(
    params: np.ndarray,
    events: Sequence[baseline.Event],
    I_ref: float,
    dt_h: float,
) -> float:
    k_rise, k_fall, n_comp, tau_h = (float(x) for x in params)
    losses = []
    for event in events:
        result = route_aut_complement_3p(
            q_in=event.q_in,
            q0=float(event.q_out[0]),
            k_rise=k_rise,
            k_fall=k_fall,
            n_comp=n_comp,
            tau_h=tau_h,
            I_ref=I_ref,
            dt_h=dt_h,
        )
        losses.append(baseline.normalized_sse(result["Q_sim"], event.q_out))
    return float(np.mean(losses)) if losses else 1.0e12


def calibrate_dynamic_k_n_complement(
    events: Sequence[baseline.Event],
    I_ref: float,
    dt_h: float,
    seed: int,
    maxiter: int,
    popsize: int,
    k_min: float,
    tau_max_h: float,
) -> dict[str, float]:
    result = differential_evolution(
        lambda params: dynamic_k_n_complement_objective(
            params, events, I_ref, dt_h
        ),
        bounds=(
            (k_min, 1.0),
            (k_min, 1.0),
            COMPLEMENT_N_BOUNDS,
            (0.0, tau_max_h),
        ),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    return {
        "k_rise": float(result.x[0]),
        "k_fall": float(result.x[1]),
        "n_comp": float(result.x[2]),
        "tau_h": float(result.x[3]),
        "train_loss": float(result.fun),
    }


def single_k_complement_objective(
    params: np.ndarray,
    events: Sequence[baseline.Event],
    dt_h: float,
) -> float:
    k, tau_h = (float(x) for x in params)
    losses = []
    for event in events:
        result = route_lagged_complement_single_k_solved_s0(
            q_in=event.q_in,
            k=k,
            tau_h=tau_h,
            dt_h=dt_h,
        )
        losses.append(baseline.normalized_sse(result["Q_sim"], event.q_out))
    return float(np.mean(losses)) if losses else 1.0e12


def calibrate_single_k_complement(
    events: Sequence[baseline.Event],
    dt_h: float,
    seed: int,
    maxiter: int,
    popsize: int,
    k_min: float,
    tau_max_h: float,
) -> dict[str, float]:
    result = differential_evolution(
        lambda params: single_k_complement_objective(params, events, dt_h),
        bounds=((k_min, 1.0), (0.0, tau_max_h)),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    return {
        "k": float(result.x[0]),
        "tau_h": float(result.x[1]),
        "train_loss": float(result.fun),
    }


def k_n_tau_complement_objective(
    params: np.ndarray,
    events: Sequence[baseline.Event],
    dt_h: float,
    pre_lag_mode: str,
) -> float:
    log_k, n_comp, tau_h = (float(x) for x in params)
    k = float(np.exp(log_k))
    losses = []
    for event in events:
        result = route_lagged_complement_single_k_solved_s0(
            q_in=event.q_in,
            k=k,
            n_comp=n_comp,
            tau_h=tau_h,
            dt_h=dt_h,
            pre_lag_mode=pre_lag_mode,
        )
        losses.append(baseline.normalized_sse(result["Q_sim"], event.q_out))
    return float(np.mean(losses)) if losses else 1.0e12


def calibrate_k_n_tau_complement(
    events: Sequence[baseline.Event],
    dt_h: float,
    seed: int,
    maxiter: int,
    popsize: int,
    k_min: float,
    k_max: float,
    tau_max_h: float,
    pre_lag_mode: str,
) -> dict[str, float]:
    result = differential_evolution(
        lambda params: k_n_tau_complement_objective(
            params, events, dt_h, pre_lag_mode
        ),
        bounds=((float(np.log(k_min)), float(np.log(k_max))), COMPLEMENT_N_BOUNDS, (0.0, tau_max_h)),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    return {
        "k": float(np.exp(result.x[0])),
        "n_comp": float(result.x[1]),
        "tau_h": float(result.x[2]),
        "train_loss": float(result.fun),
    }


def rwec_objective(
    params: np.ndarray,
    events: Sequence[baseline.Event],
    baseflow_m3s: float,
    dt_h: float,
) -> float:
    n_r, tau_h = (float(x) for x in params)
    losses = []
    for event in events:
        result = route_rwec_original(
            q_in=event.q_in,
            n_r=n_r,
            tau_h=tau_h,
            baseflow_m3s=baseflow_m3s,
            dt_h=dt_h,
        )
        losses.append(baseline.normalized_sse(result["Q_sim"], event.q_out))
    return float(np.mean(losses)) if losses else 1.0e12


def calibrate_rwec(
    events: Sequence[baseline.Event],
    baseflow_m3s: float,
    dt_h: float,
    seed: int,
    maxiter: int,
    popsize: int,
    n_bounds: tuple[float, float],
    tau_max_h: float,
) -> dict[str, float]:
    result = differential_evolution(
        lambda params: rwec_objective(params, events, baseflow_m3s, dt_h),
        bounds=(n_bounds, (0.001, tau_max_h)),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    return {
        "n_R": float(result.x[0]),
        "tau_h": float(result.x[1]),
        "train_loss": float(result.fun),
    }


def shared_metric_row(
    event: baseline.Event,
    model: str,
    result: dict[str, np.ndarray],
) -> dict[str, object]:
    simulated = result["Q_sim"]
    return {
        "model": model,
        "event_id": event.event_id,
        "event_no": event.event_no,
        "split": event.split,
        "NSE": baseline.nse(simulated, event.q_out),
        "KGE": baseline.kge(simulated, event.q_out),
        "RMSE": baseline.rmse(simulated, event.q_out),
        "PBIAS_percent": baseline.pbias(simulated, event.q_out),
        "peak_time_error_step": int(np.argmax(simulated) - np.argmax(event.q_out)),
        "initial_storage_million_m3": float(result["S_t"][0]),
        "storage_peak_million_m3": float(np.nanmax(result["S_t"])),
    }


def evaluate_complement(
    events: Sequence[baseline.Event],
    params: dict[str, float],
    I_ref: float,
    dt_h: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    hydro_rows = []
    for event in events:
        result = route_aut_complement_3p(
            q_in=event.q_in,
            q0=float(event.q_out[0]),
            k_rise=params["k_rise"],
            k_fall=params["k_fall"],
            tau_h=params["tau_h"],
            I_ref=I_ref,
            dt_h=dt_h,
        )
        simulated = result["Q_sim"]
        metric_rows.append(shared_metric_row(event, COMPLEMENT_MODEL_NAME, result))
        for i, t_h in enumerate(event.time_h):
            hydro_rows.append(
                {
                    "model": COMPLEMENT_MODEL_NAME,
                    "event_id": event.event_id,
                    "split": event.split,
                    "time_step": i + 1,
                    "t_h": float(t_h),
                    "q_in_m3s": float(event.q_in[i]),
                    "q_out_obs_m3s": float(event.q_out[i]),
                    "Q_sim_m3s": float(simulated[i]),
                    "I_bar_m3s": float(result["I_bar"][i]),
                    "k_t": float(result["k_t"][i]),
                    "is_rising": bool(result["is_rising"][i]),
                    "S_w_million_m3": float(result["S_w"][i]),
                    "S_p_million_m3": float(result["S_p"][i]),
                    "S_e_million_m3": float("nan"),
                    "S_t_million_m3": float(result["S_t"][i]),
                    "storage_drive_m3s": float("nan"),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(hydro_rows)


def evaluate_dynamic_k_n_complement(
    events: Sequence[baseline.Event],
    params: dict[str, float],
    I_ref: float,
    dt_h: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    hydro_rows = []
    for event in events:
        result = route_aut_complement_3p(
            q_in=event.q_in,
            q0=float(event.q_out[0]),
            k_rise=params["k_rise"],
            k_fall=params["k_fall"],
            n_comp=params["n_comp"],
            tau_h=params["tau_h"],
            I_ref=I_ref,
            dt_h=dt_h,
        )
        simulated = result["Q_sim"]
        metric_rows.append(
            shared_metric_row(event, COMPLEMENT_DYNAMIC_K_N_MODEL_NAME, result)
        )
        for i, t_h in enumerate(event.time_h):
            hydro_rows.append(
                {
                    "model": COMPLEMENT_DYNAMIC_K_N_MODEL_NAME,
                    "event_id": event.event_id,
                    "split": event.split,
                    "time_step": i + 1,
                    "t_h": float(t_h),
                    "q_in_m3s": float(event.q_in[i]),
                    "q_out_obs_m3s": float(event.q_out[i]),
                    "Q_sim_m3s": float(simulated[i]),
                    "I_bar_m3s": float(result["I_bar"][i]),
                    "k_t": float(result["k_t"][i]),
                    "is_rising": bool(result["is_rising"][i]),
                    "S_w_million_m3": float(result["S_w"][i]),
                    "S_p_million_m3": float(result["S_p"][i]),
                    "S_e_million_m3": float("nan"),
                    "S_t_million_m3": float(result["S_t"][i]),
                    "storage_drive_m3s": float("nan"),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(hydro_rows)


def evaluate_single_k_complement(
    events: Sequence[baseline.Event],
    params: dict[str, float],
    dt_h: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    hydro_rows = []
    for event in events:
        result = route_lagged_complement_single_k_solved_s0(
            q_in=event.q_in,
            k=params["k"],
            tau_h=params["tau_h"],
            dt_h=dt_h,
        )
        simulated = result["Q_sim"]
        metric_rows.append(
            shared_metric_row(event, COMPLEMENT_SINGLE_K_MODEL_NAME, result)
        )
        for i, t_h in enumerate(event.time_h):
            hydro_rows.append(
                {
                    "model": COMPLEMENT_SINGLE_K_MODEL_NAME,
                    "event_id": event.event_id,
                    "split": event.split,
                    "time_step": i + 1,
                    "t_h": float(t_h),
                    "q_in_m3s": float(event.q_in[i]),
                    "q_out_obs_m3s": float(event.q_out[i]),
                    "Q_sim_m3s": float(simulated[i]),
                    "I_bar_m3s": float(result["I_bar"][i]),
                    "k_t": float(result["k_t"][i]),
                    "is_rising": False,
                    "S_w_million_m3": float(result["S_w"][i]),
                    "S_p_million_m3": float(result["S_p"][i]),
                    "S_e_million_m3": float("nan"),
                    "S_t_million_m3": float(result["S_t"][i]),
                    "storage_drive_m3s": float("nan"),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(hydro_rows)


def evaluate_k_n_tau_complement(
    events: Sequence[baseline.Event],
    params: dict[str, float],
    dt_h: float,
    pre_lag_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    hydro_rows = []
    for event in events:
        result = route_lagged_complement_single_k_solved_s0(
            q_in=event.q_in,
            k=params["k"],
            n_comp=params["n_comp"],
            tau_h=params["tau_h"],
            dt_h=dt_h,
            pre_lag_mode=pre_lag_mode,
        )
        simulated = result["Q_sim"]
        metric_rows.append(
            shared_metric_row(event, COMPLEMENT_K_N_TAU_MODEL_NAME, result)
        )
        for i, t_h in enumerate(event.time_h):
            hydro_rows.append(
                {
                    "model": COMPLEMENT_K_N_TAU_MODEL_NAME,
                    "event_id": event.event_id,
                    "split": event.split,
                    "time_step": i + 1,
                    "t_h": float(t_h),
                    "q_in_m3s": float(event.q_in[i]),
                    "q_out_obs_m3s": float(event.q_out[i]),
                    "Q_sim_m3s": float(simulated[i]),
                    "I_bar_m3s": float(result["I_bar"][i]),
                    "k_t": float(result["k_t"][i]),
                    "is_rising": False,
                    "S_w_million_m3": float(result["S_w"][i]),
                    "S_p_million_m3": float(result["S_p"][i]),
                    "S_e_million_m3": float("nan"),
                    "S_t_million_m3": float(result["S_t"][i]),
                    "storage_drive_m3s": float("nan"),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(hydro_rows)


def evaluate_rwec(
    events: Sequence[baseline.Event],
    params: dict[str, float],
    baseflow_m3s: float,
    dt_h: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    hydro_rows = []
    for event in events:
        result = route_rwec_original(
            q_in=event.q_in,
            n_r=params["n_R"],
            tau_h=params["tau_h"],
            baseflow_m3s=baseflow_m3s,
            dt_h=dt_h,
        )
        metric_rows.append(shared_metric_row(event, RWEC_MODEL_NAME, result))
        for i, t_h in enumerate(event.time_h):
            hydro_rows.append(
                {
                    "model": RWEC_MODEL_NAME,
                    "event_id": event.event_id,
                    "split": event.split,
                    "time_step": i + 1,
                    "t_h": float(t_h),
                    "q_in_m3s": float(event.q_in[i]),
                    "q_out_obs_m3s": float(event.q_out[i]),
                    "Q_sim_m3s": float(result["Q_sim"][i]),
                    "I_bar_m3s": float("nan"),
                    "k_t": float("nan"),
                    "is_rising": False,
                    "S_w_million_m3": float(result["S_w"][i]),
                    "S_p_million_m3": float("nan"),
                    "S_e_million_m3": float(result["S_e"][i]),
                    "S_t_million_m3": float(result["S_t"][i]),
                    "storage_drive_m3s": float(result["storage_drive"][i]),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(hydro_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AUT 三参数水能互补河道演算模型。")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dt-h", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--k-min", type=float, default=0.001)
    parser.add_argument(
        "--k-max",
        type=float,
        default=1.0,
        help="Upper bound for k in the single-k complement models.",
    )
    parser.add_argument("--tau-max-h", type=float, default=24.0)
    parser.add_argument(
        "--pre-lag-mode",
        choices=("instant_inflow", "steady_initial"),
        default="steady_initial",
        help="Boundary handling before the calibrated lag reaches the outlet.",
    )
    parser.add_argument(
        "--rwec-n-bounds",
        nargs=2,
        type=float,
        default=(0.5, 80.0),
        metavar=("MIN", "MAX"),
        help="Original RWEC n_R calibration bounds.",
    )
    parser.add_argument(
        "--rwec-baseflow-m3s",
        type=float,
        default=RWEC_DEFAULT_BASEFLOW_M3S,
        help="Fixed Q_b in the original RWEC storage-driving term; not calibrated.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(
            COMPLEMENT_MODEL_NAME,
            COMPLEMENT_DYNAMIC_K_N_MODEL_NAME,
            COMPLEMENT_SINGLE_K_MODEL_NAME,
            COMPLEMENT_K_N_TAU_MODEL_NAME,
            RWEC_MODEL_NAME,
        ),
        default=(COMPLEMENT_MODEL_NAME, RWEC_MODEL_NAME),
        help="Models to run. Default: both models.",
    )
    parser.add_argument(
        "--basins",
        nargs="+",
        default=["AUT"],
        help="River names to calibrate. Use one river per run for basin-specific parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k_min <= 0.0 or args.k_max <= args.k_min:
        raise ValueError("--k-max must be greater than positive --k-min.")
    if args.tau_max_h <= 0.0:
        raise ValueError("--tau-max-h must be positive.")
    rwec_n_bounds = (float(args.rwec_n_bounds[0]), float(args.rwec_n_bounds[1]))
    if rwec_n_bounds[0] <= 0.0 or rwec_n_bounds[1] <= rwec_n_bounds[0]:
        raise ValueError("--rwec-n-bounds must satisfy 0 < MIN < MAX.")
    if args.rwec_baseflow_m3s < 0.0:
        raise ValueError("--rwec-baseflow-m3s must be nonnegative.")

    out = next_numbered_output_dir(args.out_dir.resolve())
    events = baseline.read_events(
        paths=baseline.DEFAULT_DATA_FILES,
        test_ratio=0.2,
        basins=set(args.basins),
        dt_h=args.dt_h,
    )
    train_events = [event for event in events if event.split == "train"]
    selected_models = set(args.models)
    metric_frames = []
    hydrograph_frames = []
    parameter_rows = []

    if COMPLEMENT_MODEL_NAME in selected_models:
        I_ref = reference_inflow(train_events)
        complement_params = calibrate_complement(
            train_events,
            I_ref,
            args.dt_h,
            args.seed,
            args.maxiter,
            args.popsize,
            args.k_min,
            args.tau_max_h,
        )
        complement_metrics, complement_hydrographs = evaluate_complement(
            events, complement_params, I_ref, args.dt_h
        )
        metric_frames.append(complement_metrics)
        hydrograph_frames.append(complement_hydrographs)
        parameter_rows.append(
            {
                "model": COMPLEMENT_MODEL_NAME,
                **complement_params,
                "I_ref_m3s": I_ref,
                "n_fixed": N_COMP,
                "m_fixed": FLOW_EXPONENT,
                "initial_storage_offset_steps_fixed": INITIAL_STORAGE_OFFSET_STEPS,
                "n_R": float("nan"),
                "Q_b_m3s": float("nan"),
                "S0_method": "tau_h*Q0",
                "parameter_count": 3,
            }
        )

    if COMPLEMENT_DYNAMIC_K_N_MODEL_NAME in selected_models:
        I_ref = reference_inflow(train_events)
        dynamic_k_n_params = calibrate_dynamic_k_n_complement(
            train_events,
            I_ref,
            args.dt_h,
            args.seed + 4,
            args.maxiter,
            args.popsize,
            args.k_min,
            args.tau_max_h,
        )
        dynamic_k_n_metrics, dynamic_k_n_hydrographs = evaluate_dynamic_k_n_complement(
            events, dynamic_k_n_params, I_ref, args.dt_h
        )
        metric_frames.append(dynamic_k_n_metrics)
        hydrograph_frames.append(dynamic_k_n_hydrographs)
        parameter_rows.append(
            {
                "model": COMPLEMENT_DYNAMIC_K_N_MODEL_NAME,
                **dynamic_k_n_params,
                "I_ref_m3s": I_ref,
                "n_fixed": float("nan"),
                "m_fixed": FLOW_EXPONENT,
                "initial_storage_offset_steps_fixed": INITIAL_STORAGE_OFFSET_STEPS,
                "n_R": float("nan"),
                "Q_b_m3s": float("nan"),
                "S0_method": "tau_h*Q0",
                "parameter_count": 4,
            }
        )

    if COMPLEMENT_SINGLE_K_MODEL_NAME in selected_models:
        single_k_params = calibrate_single_k_complement(
            train_events,
            args.dt_h,
            args.seed + 2,
            args.maxiter,
            args.popsize,
            args.k_min,
            args.tau_max_h,
        )
        single_k_metrics, single_k_hydrographs = evaluate_single_k_complement(
            events, single_k_params, args.dt_h
        )
        metric_frames.append(single_k_metrics)
        hydrograph_frames.append(single_k_hydrographs)
        parameter_rows.append(
            {
                "model": COMPLEMENT_SINGLE_K_MODEL_NAME,
                **single_k_params,
                "I_ref_m3s": float("nan"),
                "n_fixed": N_COMP,
                "m_fixed": float("nan"),
                "initial_storage_offset_steps_fixed": float("nan"),
                "n_R": float("nan"),
                "Q_b_m3s": float("nan"),
                "S0_method": "solved_initial_lagged_inflow_equilibrium",
                "parameter_count": 2,
            }
        )

    if COMPLEMENT_K_N_TAU_MODEL_NAME in selected_models:
        k_n_tau_params = calibrate_k_n_tau_complement(
            train_events,
            args.dt_h,
            args.seed + 3,
            args.maxiter,
            args.popsize,
            args.k_min,
            args.k_max,
            args.tau_max_h,
            args.pre_lag_mode,
        )
        k_n_tau_metrics, k_n_tau_hydrographs = evaluate_k_n_tau_complement(
            events, k_n_tau_params, args.dt_h, args.pre_lag_mode
        )
        metric_frames.append(k_n_tau_metrics)
        hydrograph_frames.append(k_n_tau_hydrographs)
        parameter_rows.append(
            {
                "model": COMPLEMENT_K_N_TAU_MODEL_NAME,
                **k_n_tau_params,
                "I_ref_m3s": float("nan"),
                "n_fixed": float("nan"),
                "m_fixed": float("nan"),
                "initial_storage_offset_steps_fixed": float("nan"),
                "n_R": float("nan"),
                "Q_b_m3s": float("nan"),
                "S0_method": "solved_initial_lagged_inflow_equilibrium",
                "parameter_count": 3,
                "pre_lag_mode": args.pre_lag_mode,
            }
        )

    if RWEC_MODEL_NAME in selected_models:
        rwec_params = calibrate_rwec(
            train_events,
            args.rwec_baseflow_m3s,
            args.dt_h,
            args.seed + 1,
            args.maxiter,
            args.popsize,
            rwec_n_bounds,
            args.tau_max_h,
        )
        rwec_metrics, rwec_hydrographs = evaluate_rwec(
            events, rwec_params, args.rwec_baseflow_m3s, args.dt_h
        )
        metric_frames.append(rwec_metrics)
        hydrograph_frames.append(rwec_hydrographs)
        parameter_rows.append(
            {
                "model": RWEC_MODEL_NAME,
                **rwec_params,
                "I_ref_m3s": float("nan"),
                "n_fixed": float("nan"),
                "m_fixed": float("nan"),
                "initial_storage_offset_steps_fixed": float("nan"),
                "Q_b_m3s": args.rwec_baseflow_m3s,
                "S0_method": "solved_initial_equilibrium",
                "parameter_count": 2,
            }
        )

    metrics = pd.concat(metric_frames, ignore_index=True)
    hydrographs = pd.concat(hydrograph_frames, ignore_index=True)
    summary = (
        metrics.groupby(["model", "split"], sort=True)
        .agg(
            events=("event_id", "count"),
            mean_NSE=("NSE", "mean"),
            median_NSE=("NSE", "median"),
            min_NSE=("NSE", "min"),
            mean_KGE=("KGE", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_abs_PBIAS_percent=("PBIAS_percent", lambda x: float(np.mean(np.abs(x)))),
            mean_abs_peak_time_error_step=("peak_time_error_step", lambda x: float(np.mean(np.abs(x)))),
        )
        .reset_index()
    )
    parameters = pd.DataFrame(parameter_rows)
    parameters["basins"] = ",".join(sorted(args.basins))
    parameters["k_min"] = args.k_min
    parameters["tau_max_h"] = args.tau_max_h
    parameters["rwec_n_bounds"] = str(rwec_n_bounds)
    parameters.to_csv(
        out / "model_parameters.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(out / "event_metrics.csv", index=False, encoding="utf-8-sig")
    hydrographs.to_csv(out / "hydrographs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "complement_formula": "S0=tau*Q0; k_t=k_rise/fall*(I_bar/I_ref)^2; S_p=S_prev+k_t*I_bar*dt",
                "complement_equation": "(S_t/S_w)^16+(S_t/S_p)^16=1",
                "dynamic_k_n_complement_formula": "S0=tau*Q0; k_t=k_rise/fall*(I_bar/I_ref)^2; S_p=S_prev+k_t*I_bar*dt",
                "dynamic_k_n_complement_parameters": "k_rise,k_fall,n,tau_h; m=2 fixed",
                "single_k_complement_formula": "S0 solved from I0; S_w=S_prev+I_bar*dt; S_p=S_prev+k*I_bar*dt; no baseflow subtraction",
        "single_k_complement_parameters": (
            f"k,tau_h; N_COMP=16 fixed; pre_lag_mode={args.pre_lag_mode}"
        ),
                "single_k_initial_storage_equation": "(S0/(S0+I0*dt))^16+(S0/(S0+k*I0*dt))^16=1",
                "k_n_tau_complement_formula": "S0 solved from I0; S_w=S_prev+I_bar*dt; S_p=S_prev+k*I_bar*dt; no baseflow subtraction",
        "k_n_tau_complement_parameters": (
            f"k,n,tau_h; pre_lag_mode={args.pre_lag_mode}"
        ),
                "k_n_tau_initial_storage_equation": "(S0/(S0+I0*dt))^n+(S0/(S0+k*I0*dt))^n=1",
                "complement_n_bounds": str(COMPLEMENT_N_BOUNDS),
                "rwec_formula": "S_W=S_prev+dt/2*(I_t+I_t-1-2Q_b); S_E=S_prev+tau/2*(I_t+I_t-1-2Q_b); Q_t=-2(S_t-S_prev)/dt+I_t+I_t-1-Q_t-1",
                "rwec_parameters": "n_R,tau_h; Q_b fixed and not calibrated; Q0=I0; S0 solved from initial RWEC equilibrium",
                "rwec_initial_storage_equation": "(S0/(S0+dt*(I0-Qb)))^n_R+(S0/(S0+tau*(I0-Qb)))^n_R=1",
                "initial_storage_offset_steps_fixed": INITIAL_STORAGE_OFFSET_STEPS,
                "maxiter": args.maxiter,
                "popsize": args.popsize,
                "seed": args.seed,
                "basins": ",".join(sorted(args.basins)),
                "k_min": args.k_min,
                "k_max": args.k_max,
                "tau_max_h": args.tau_max_h,
                "pre_lag_mode": args.pre_lag_mode,
                "rwec_n_bounds": str(rwec_n_bounds),
                "rwec_baseflow_m3s": args.rwec_baseflow_m3s,
                "models": ",".join(args.models),
            }
        ]
    ).to_csv(out / "run_settings.csv", index=False, encoding="utf-8-sig")
    print(f"输出目录：{out}")
    print(parameters.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

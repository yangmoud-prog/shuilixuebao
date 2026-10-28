#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize


# ============================================================
# 1. 路径设置
# ============================================================

PROJECT_DIR = Path(r"E:\01-HUST\Nature\03")
DATA_DIR = PROJECT_DIR / "HECRAS" / "data"
DEFAULT_OUT = PROJECT_DIR / "Result" / "routing_model_comparison_result"

DEFAULT_DATA_FILES = [
    DATA_DIR / "Wangjiahe_flood_events.csv",
    DATA_DIR / "Qingjiang_flood_events.csv",
    DATA_DIR / "LAMH-CE.xlsx",
]


# ============================================================
# 2. 常量和数据结构
# ============================================================

EPS = 1.0e-12
DEFAULT_DT_H = 1.0
RUN_SUFFIX = "fixed_tau"
LMC_MODEL_NAME = "LMC_S0_zero"


@dataclass
class Event:
    river: str
    event_id: str
    event_no: int
    split: str
    time_h: np.ndarray
    q_in: np.ndarray
    q_out: np.ndarray
    source_file: str


@dataclass(frozen=True)
class Bounds:
    k: tuple[float, float]
    n: tuple[float, float]
    tau_h: tuple[float, float]


@dataclass(frozen=True)
class MuskingumBounds:
    K_h: tuple[float, float]
    X: tuple[float, float]


# ============================================================
# 3. 评价指标
# ============================================================

def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom <= EPS:
        return float("nan")
    return 1.0 - float(np.sum((sim - obs) ** 2)) / denom


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    if np.std(sim) <= EPS or np.std(obs) <= EPS or abs(np.mean(obs)) <= EPS:
        return float("nan")
    r = float(np.corrcoef(sim, obs)[0, 1])
    alpha = float(np.std(sim) / np.std(obs))
    beta = float(np.mean(sim) / np.mean(obs))
    return 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def rmse(sim: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def mae(sim: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(np.abs(sim - obs)))


def bias(sim: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(sim - obs))


def pbias(sim: np.ndarray, obs: np.ndarray) -> float:
    denom = float(np.sum(obs))
    if abs(denom) <= EPS:
        return float("nan")
    return float(np.sum(sim - obs) / denom * 100.0)


def corr_r(sim: np.ndarray, obs: np.ndarray) -> float:
    if np.std(sim) <= EPS or np.std(obs) <= EPS:
        return float("nan")
    return float(np.corrcoef(sim, obs)[0, 1])


def normalized_sse(sim: np.ndarray, obs: np.ndarray) -> float:
    if not np.all(np.isfinite(sim)):
        return 1.0e12
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom <= EPS:
        return 1.0e12
    return float(np.sum((sim - obs) ** 2) / denom)


# ============================================================
# 4. S0=0 水能互补模型（变量名按公式符号改写）
# ============================================================

def sample_flow(q_in: np.ndarray, time_h: np.ndarray, t_query: float) -> float:
    if t_query <= time_h[0]:
        return float(q_in[0])
    if t_query >= time_h[-1]:
        return float(q_in[-1])
    return float(np.interp(t_query, time_h, q_in))


def lagged_period_mean_inflow(
    q_in: np.ndarray,
    time_h: np.ndarray,
    t_h: float,
    tau_h: float,
    dt_h: float,
) -> tuple[float, float, float]:
    I_tau_now = sample_flow(q_in, time_h, t_h - tau_h)
    I_tau_prev = sample_flow(q_in, time_h, t_h - tau_h - dt_h)
    I_tau_bar = 0.5 * (I_tau_prev + I_tau_now)
    return I_tau_prev, I_tau_now, I_tau_bar


def solve_S_t_by_complementary_equation(S_w: float, S_p: float, n_comp: float) -> float:
    """按互补方程解析求当前槽蓄量 S_t。"""
    if S_w <= 0.0 or S_p <= 0.0 or n_comp <= 0.0:
        return 0.0
    if not np.isfinite(S_w + S_p + n_comp):
        return float("nan")

    m = max(S_w, S_p)
    denom = m * ((S_w / m) ** n_comp + (S_p / m) ** n_comp) ** (1.0 / n_comp)
    return S_w * S_p / denom


def route_lmc_s0_zero(
    q_in: np.ndarray,
    q0: float,
    k: float,
    n_comp: float,
    tau_h: float,
    dt_h: float,
) -> dict[str, np.ndarray]:

    q_in = np.asarray(q_in, dtype=float)
    n_points = len(q_in)

    time_h = np.arange(n_points, dtype=float) * dt_h
    delta_t_seconds = dt_h * 3600.0

    Q_sim = np.zeros(n_points, dtype=float)
    S_t_series = np.zeros(n_points, dtype=float)

    I_tau_prev_series = np.zeros(n_points, dtype=float)
    I_tau_now_series = np.zeros(n_points, dtype=float)
    I_tau_bar_series = np.zeros(n_points, dtype=float)
    S_w_series = np.zeros(n_points, dtype=float)
    S_p_series = np.zeros(n_points, dtype=float)
    O_t_series = np.zeros(n_points, dtype=float)
    before_tau_series = np.zeros(n_points, dtype=bool)

    # 关键：初始槽蓄量固定 S0 = 0
    S_t_minus_1 = 0.0

    for t in range(n_points):
        # t < tau 时滞后入流尚未进入河段，保持原来的无演算处理。
        if time_h[t] < tau_h:
            Q_sim[t] = float(q_in[t])
            S_t_series[t] = S_t_minus_1
            O_t_series[t] = Q_sim[t] * delta_t_seconds / 1.0e6
            before_tau_series[t] = True
            continue

        if t == 0:
            Q_sim[t] = float(q0)
            S_t_series[t] = S_t_minus_1
            continue

        I_tau_prev, I_tau_now, I_tau_bar = lagged_period_mean_inflow(
            q_in=q_in,
            time_h=time_h,
            t_h=time_h[t],
            tau_h=tau_h,
            dt_h=dt_h,
        )

        I_tau_prev = max(0.0, I_tau_prev)
        I_tau_now = max(0.0, I_tau_now)
        I_tau_bar = max(0.0, I_tau_bar)

        # 入流流量 I_tau_bar(m3/s) 转换为本时段入流水量 I_tau_volume(百万 m3)
        I_tau_volume = I_tau_bar * delta_t_seconds / 1.0e6

        # S_w = S_{t-1} + I_{t-τ} Δt
        S_w = max(EPS, S_t_minus_1 + I_tau_volume)

        # S_p = S_{t-1} + k I_{t-τ} Δt
        S_p = max(EPS, S_t_minus_1 + k * I_tau_volume)

        # 由 (S_t/S_w)^n + (S_t/S_p)^n = 1 求 S_t
        S_t = solve_S_t_by_complementary_equation(S_w, S_p, n_comp)

        if not np.isfinite(S_t):
            Q_sim[:] = np.nan
            S_t_series[:] = np.nan
            break

        # O_t = S_w - S_t 为本时段出流水量；Q_t = O_t / Δt 为出流流量
        O_t = S_w - S_t
        Q_t = O_t * 1.0e6 / delta_t_seconds
        Q_t = max(0.0, Q_t)

        Q_sim[t] = Q_t
        S_t_series[t] = S_t

        I_tau_prev_series[t] = I_tau_prev
        I_tau_now_series[t] = I_tau_now
        I_tau_bar_series[t] = I_tau_bar
        S_w_series[t] = S_w
        S_p_series[t] = S_p
        O_t_series[t] = O_t

        S_t_minus_1 = S_t

    return {
        "Q_sim": Q_sim,
        "S_t": S_t_series,
        "I_tau_prev": I_tau_prev_series,
        "I_tau_now": I_tau_now_series,
        "I_tau_bar": I_tau_bar_series,
        "S_w": S_w_series,
        "S_p": S_p_series,
        "O_t": O_t_series,
        "before_tau_h": before_tau_series,
    }


def muskingum_coefficients(K_h: float, X: float, dt_h: float) -> tuple[float, float, float]:
    denominator = 2.0 * K_h * (1.0 - X) + dt_h

    if denominator <= EPS:
        return float("nan"), float("nan"), float("nan")

    C0 = (dt_h - 2.0 * K_h * X) / denominator
    C1 = (dt_h + 2.0 * K_h * X) / denominator
    C2 = (2.0 * K_h * (1.0 - X) - dt_h) / denominator
    return C0, C1, C2


def route_muskingum(
    q_in: np.ndarray,
    q0: float,
    K_h: float,
    X: float,
    dt_h: float,
) -> dict[str, np.ndarray]:

    q_in = np.asarray(q_in, dtype=float)
    n_points = len(q_in)

    Q_sim = np.zeros(n_points, dtype=float)
    S_muskingum = np.zeros(n_points, dtype=float)

    C0, C1, C2 = muskingum_coefficients(K_h, X, dt_h)

    coefficients_are_valid = (
        np.isfinite(C0 + C1 + C2)
        and K_h > 0.0
        and 0.0 <= X <= 0.5
        and C0 >= -EPS
        and C1 >= -EPS
        and C2 >= -EPS
    )

    if not coefficients_are_valid:
        Q_sim[:] = np.nan
        S_muskingum[:] = np.nan
        return {
            "Q_sim": Q_sim,
            "S_muskingum": S_muskingum,
            "C0": np.full(n_points, C0),
            "C1": np.full(n_points, C1),
            "C2": np.full(n_points, C2),
        }

    Q_sim[0] = float(q0)
    S_muskingum[0] = K_h * 3600.0 * (X * q_in[0] + (1.0 - X) * Q_sim[0]) / 1.0e6

    for t in range(1, n_points):
        Q_t = C0 * q_in[t] + C1 * q_in[t - 1] + C2 * Q_sim[t - 1]
        Q_sim[t] = max(0.0, Q_t)
        S_muskingum[t] = K_h * 3600.0 * (
            X * q_in[t] + (1.0 - X) * Q_sim[t]
        ) / 1.0e6

    return {
        "Q_sim": Q_sim,
        "S_muskingum": S_muskingum,
        "C0": np.full(n_points, C0),
        "C1": np.full(n_points, C1),
        "C2": np.full(n_points, C2),
    }


# ============================================================
# 5. 数据读取
# ============================================================

def resolve_data_files(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []

    if args.all_data_files:
        paths.extend(sorted(DATA_DIR.glob("*_flood_events.csv")))
        paths.extend(sorted(DATA_DIR.glob("*.xlsx")))

    if args.data_glob:
        for pattern in args.data_glob:
            pattern_path = Path(pattern)
            search_pattern = str(pattern_path if pattern_path.is_absolute() else PROJECT_DIR / pattern_path)
            paths.extend(Path(p) for p in sorted(glob.glob(search_pattern)))

    if args.data_files:
        paths.extend(Path(p) for p in args.data_files)

    if not paths:
        paths.extend(DEFAULT_DATA_FILES)

    resolved: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        full = path if path.is_absolute() else PROJECT_DIR / path
        full = full.resolve()

        if full in seen:
            continue

        if not full.exists():
            raise FileNotFoundError(
                f"找不到数据文件：{full}\n"
                f"请检查 DATA_DIR 或 DEFAULT_DATA_FILES。"
            )

        resolved.append(full)
        seen.add(full)

    if not resolved:
        raise FileNotFoundError("没有找到任何数据文件。")

    return resolved


def read_two_column_event(path: Path, dt_h: float) -> list[Event]:
    df = pd.read_csv(
        path,
        sep=r"[\s,]+",
        engine="python",
        header=None,
        comment="#",
    )

    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any")

    if df.shape[1] < 2:
        raise ValueError(f"{path} 至少需要两列：入流、出流。")

    q_in = df.iloc[:, 0].to_numpy(dtype=float)
    q_out = df.iloc[:, 1].to_numpy(dtype=float)

    if len(q_in) < 4 or np.nanstd(q_out) <= EPS:
        return []

    return [
        Event(
            river=path.stem,
            event_id=f"{path.stem}_event_01",
            event_no=1,
            split="train",
            time_h=np.arange(len(q_in), dtype=float) * dt_h,
            q_in=q_in,
            q_out=q_out,
            source_file=str(path),
        )
    ]


def normalized_column_name(name: object) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("³", "3")
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_excel_column(columns: Sequence[object], candidates: Sequence[str]) -> object:
    normalized = {col: normalized_column_name(col) for col in columns}

    for col, name in normalized.items():
        if any(candidate in name for candidate in candidates):
            return col

    raise ValueError(f"Excel 表中找不到列：{candidates}")


def read_multi_event_excel(
    path: Path,
    test_ratio: float,
    basins: set[str] | None,
    dt_h: float,
) -> list[Event]:

    workbook = pd.ExcelFile(path)
    events: list[Event] = []

    for sheet_name in workbook.sheet_names:
        river = str(sheet_name).strip()

        if basins is not None and river not in basins:
            continue

        df = pd.read_excel(path, sheet_name=sheet_name)
        df = df.dropna(how="all")

        if df.empty:
            continue

        event_col = find_excel_column(df.columns, ["number", "event", "事件", "编号"])
        q_in_col = find_excel_column(df.columns, ["qin", "入流"])
        q_out_col = find_excel_column(df.columns, ["qout", "出流"])

        df = df[[event_col, q_in_col, q_out_col]].copy()
        df.columns = ["Event", "Qin(m3/s)", "Qout(m3/s)"]

        df["Event"] = pd.to_numeric(df["Event"], errors="coerce")
        df["Qin(m3/s)"] = pd.to_numeric(df["Qin(m3/s)"], errors="coerce")
        df["Qout(m3/s)"] = pd.to_numeric(df["Qout(m3/s)"], errors="coerce")
        df = df.dropna(subset=["Event", "Qin(m3/s)", "Qout(m3/s)"])

        event_numbers = sorted(int(x) for x in df["Event"].dropna().unique().tolist())

        if len(event_numbers) >= 2:
            n_test = max(1, int(round(len(event_numbers) * test_ratio)))
            n_test = min(n_test, len(event_numbers) - 1)
            test_events = set(event_numbers[-n_test:])
        else:
            test_events = set()

        for event_no, event_df in df.groupby("Event", sort=True):
            event_no_int = int(event_no)

            q_in = event_df["Qin(m3/s)"].to_numpy(dtype=float)
            q_out = event_df["Qout(m3/s)"].to_numpy(dtype=float)

            mask = np.isfinite(q_in) & np.isfinite(q_out)
            q_in = q_in[mask]
            q_out = q_out[mask]

            if len(q_in) < 4 or np.nanstd(q_out) <= EPS:
                continue

            split = "test" if event_no_int in test_events else "train"

            events.append(
                Event(
                    river=river,
                    event_id=f"{river}_event_{event_no_int:02d}",
                    event_no=event_no_int,
                    split=split,
                    time_h=np.arange(len(q_in), dtype=float) * dt_h,
                    q_in=q_in,
                    q_out=q_out,
                    source_file=f"{path}::{sheet_name}",
                )
            )

    return events


def read_multi_event_csv(
    path: Path,
    test_ratio: float,
    basins: set[str] | None,
    dt_h: float,
) -> list[Event]:

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return read_two_column_event(path, dt_h)

    required = ["River", "Event", "Step", "Qin(m3/s)", "Qout(m3/s)"]

    if not all(col in df.columns for col in required):
        return read_two_column_event(path, dt_h)

    df = df.copy()
    df["River"] = df["River"].astype(str)
    df["Event"] = pd.to_numeric(df["Event"], errors="coerce")
    df["Step"] = pd.to_numeric(df["Step"], errors="coerce")
    df["Qin(m3/s)"] = pd.to_numeric(df["Qin(m3/s)"], errors="coerce")
    df["Qout(m3/s)"] = pd.to_numeric(df["Qout(m3/s)"], errors="coerce")

    df = df.dropna(subset=["River", "Event", "Step", "Qin(m3/s)", "Qout(m3/s)"])

    events: list[Event] = []

    for river, rg in df.groupby("River", sort=True):
        river = str(river)

        if basins is not None and river not in basins:
            continue

        event_numbers = sorted(int(x) for x in rg["Event"].dropna().unique().tolist())

        if len(event_numbers) >= 2:
            n_test = max(1, int(round(len(event_numbers) * test_ratio)))
            n_test = min(n_test, len(event_numbers) - 1)
            test_events = set(event_numbers[-n_test:])
        else:
            test_events = set()

        for event_no, eg in rg.groupby("Event", sort=True):
            event_no_int = int(event_no)
            eg = eg.sort_values("Step")

            q_in = eg["Qin(m3/s)"].to_numpy(dtype=float)
            q_out = eg["Qout(m3/s)"].to_numpy(dtype=float)

            mask = np.isfinite(q_in) & np.isfinite(q_out)
            q_in = q_in[mask]
            q_out = q_out[mask]

            if len(q_in) < 4 or np.nanstd(q_out) <= EPS:
                continue

            split = "test" if event_no_int in test_events else "train"

            events.append(
                Event(
                    river=river,
                    event_id=f"{river}_event_{event_no_int:02d}",
                    event_no=event_no_int,
                    split=split,
                    time_h=np.arange(len(q_in), dtype=float) * dt_h,
                    q_in=q_in,
                    q_out=q_out,
                    source_file=str(path),
                )
            )

    return events


def read_event_file(
    path: Path,
    test_ratio: float,
    basins: set[str] | None,
    dt_h: float,
) -> list[Event]:

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return read_multi_event_excel(path, test_ratio, basins, dt_h)

    return read_multi_event_csv(path, test_ratio, basins, dt_h)


def read_events(
    paths: Sequence[Path],
    test_ratio: float,
    basins: set[str] | None,
    dt_h: float,
) -> list[Event]:

    events: list[Event] = []

    for path in paths:
        events.extend(read_event_file(path, test_ratio, basins, dt_h))

    if not events:
        raise ValueError("没有读到可用事件。请检查数据文件、列名或 --basins 设置。")

    return events


def filter_event_numbers(events: list[Event], event_numbers: set[int] | None) -> list[Event]:
    if event_numbers is None:
        return events

    selected = [ev for ev in events if ev.event_no in event_numbers]

    if not selected:
        raise ValueError(f"没有匹配到事件编号：{sorted(event_numbers)}")

    return selected


def make_assignment_table(events: list[Event]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "river": ev.river,
                "event_id": ev.event_id,
                "event_no": ev.event_no,
                "split": ev.split,
                "n_points": len(ev.q_in),
                "qin_peak_m3s": float(np.max(ev.q_in)),
                "qout_peak_m3s": float(np.max(ev.q_out)),
                "source_file": ev.source_file,
            }
            for ev in events
        ]
    )


# ============================================================
# 6. 参数率定
# ============================================================

def objective_lmc_for_events(
    params: np.ndarray,
    events: Sequence[Event],
    dt_h: float,
) -> float:

    k, n_comp, tau_h = params

    if k <= 0.0 or n_comp <= 0.0 or tau_h < 0.0:
        return 1.0e12

    losses = []

    for ev in events:
        result = route_lmc_s0_zero(
            q_in=ev.q_in,
            q0=float(ev.q_out[0]),
            k=k,
            n_comp=n_comp,
            tau_h=tau_h,
            dt_h=dt_h,
        )
        losses.append(normalized_sse(result["Q_sim"], ev.q_out))

    return float(np.mean(losses)) if losses else 1.0e12


def objective_muskingum_for_events(
    params: np.ndarray,
    events: Sequence[Event],
    dt_h: float,
) -> float:

    K_h, X = params

    if K_h <= 0.0 or X < 0.0 or X > 0.5:
        return 1.0e12

    losses = []

    for ev in events:
        result = route_muskingum(
            q_in=ev.q_in,
            q0=float(ev.q_out[0]),
            K_h=K_h,
            X=X,
            dt_h=dt_h,
        )
        losses.append(normalized_sse(result["Q_sim"], ev.q_out))

    return float(np.mean(losses)) if losses else 1.0e12


def grid_start_for_muskingum(
    events: Sequence[Event],
    bounds: MuskingumBounds,
    dt_h: float,
) -> tuple[np.ndarray, float]:

    K_values = np.linspace(bounds.K_h[0], bounds.K_h[1], 31)
    X_values = np.linspace(bounds.X[0], bounds.X[1], 21)

    best_params = np.array([bounds.K_h[0], bounds.X[0]], dtype=float)
    best_loss = 1.0e12

    for K_h in K_values:
        for X in X_values:
            loss = objective_muskingum_for_events(np.array([K_h, X]), events, dt_h)

            if loss < best_loss:
                best_loss = loss
                best_params = np.array([K_h, X], dtype=float)

    return best_params, best_loss


def calibrate_lmc(
    events: Sequence[Event],
    bounds: Bounds,
    dt_h: float,
    seed: int,
    maxiter: int,
    popsize: int,
) -> dict[str, float]:

    opt_bounds = [bounds.k, bounds.n, bounds.tau_h]

    de_result = differential_evolution(
        lambda x: objective_lmc_for_events(x, events, dt_h),
        bounds=opt_bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1.0e-8,
        polish=True,
        updating="immediate",
        workers=1,
    )

    local_result = minimize(
        lambda x: objective_lmc_for_events(x, events, dt_h),
        de_result.x,
        method="Nelder-Mead",
        options={
            "maxiter": 5000,
            "xatol": 1.0e-10,
            "fatol": 1.0e-10,
        },
    )

    if local_result.success and local_result.fun <= de_result.fun:
        best = local_result.x
        loss = float(local_result.fun)
    else:
        best = de_result.x
        loss = float(de_result.fun)

    return {
        "k": float(best[0]),
        "n": float(best[1]),
        "tau_h": float(best[2]),
        "S0": 0.0,
        "train_loss": loss,
        "fit_events": len(events),
    }


def calibrate_muskingum(
    events: Sequence[Event],
    bounds: MuskingumBounds,
    dt_h: float,
    seed: int,
    maxiter: int,
    popsize: int,
) -> dict[str, float]:

    opt_bounds = [bounds.K_h, bounds.X]
    grid_best, grid_loss = grid_start_for_muskingum(events, bounds, dt_h)

    de_result = differential_evolution(
        lambda x: objective_muskingum_for_events(x, events, dt_h),
        bounds=opt_bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1.0e-8,
        polish=True,
        updating="immediate",
        workers=1,
    )

    if de_result.fun <= grid_loss:
        local_start = de_result.x
        best_before_local = de_result.x
        best_loss_before_local = float(de_result.fun)
    else:
        local_start = grid_best
        best_before_local = grid_best
        best_loss_before_local = float(grid_loss)

    local_result = minimize(
        lambda x: objective_muskingum_for_events(x, events, dt_h),
        local_start,
        method="Nelder-Mead",
        options={
            "maxiter": 5000,
            "xatol": 1.0e-10,
            "fatol": 1.0e-10,
        },
    )

    if local_result.success and local_result.fun <= best_loss_before_local:
        best = local_result.x
        loss = float(local_result.fun)
    else:
        best = best_before_local
        loss = best_loss_before_local

    C0, C1, C2 = muskingum_coefficients(float(best[0]), float(best[1]), dt_h)

    return {
        "K_h": float(best[0]),
        "X": float(best[1]),
        "C0": float(C0),
        "C1": float(C1),
        "C2": float(C2),
        "train_loss": loss,
        "fit_events": len(events),
    }


def stable_seed(base_seed: int, text: str) -> int:
    value = 0
    for i, ch in enumerate(text):
        value += (i + 1) * ord(ch)
    return int(base_seed + value % 1000000)


def build_fit_groups(events: list[Event], fit_scope: str) -> dict[str, list[Event]]:
    if fit_scope == "all":
        return {"all": events}

    if fit_scope == "river":
        return {
            river: [ev for ev in events if ev.river == river]
            for river in sorted({ev.river for ev in events})
        }

    if fit_scope == "event":
        return {ev.event_id: [ev] for ev in events}

    raise ValueError(f"未知 fit_scope：{fit_scope}")


def training_subset(group_events: list[Event], fit_scope: str) -> list[Event]:
    if fit_scope == "event":
        return group_events

    train_events = [ev for ev in group_events if ev.split == "train"]
    return train_events if train_events else group_events


# ============================================================
# 7. 输出
# ============================================================

def metric_row(
    ev: Event,
    result: dict[str, np.ndarray],
    params: dict[str, float],
    fit_group: str,
    model: str,
) -> dict[str, object]:

    sim = result["Q_sim"]
    obs = ev.q_out
    finite_sim = np.all(np.isfinite(sim))
    storage_series = result.get("S_t", result.get("S_muskingum"))

    return {
        "model": model,
        "fit_group": fit_group,
        "river": ev.river,
        "event_id": ev.event_id,
        "event_no": ev.event_no,
        "split": ev.split,
        "n_points": len(obs),

        "lmc_k": params.get("k", float("nan")),
        "lmc_n": params.get("n", float("nan")),
        "lmc_tau_h": params.get("tau_h", float("nan")),
        "lmc_S0": params.get("S0", float("nan")),
        "muskingum_K_h": params.get("K_h", float("nan")),
        "muskingum_X": params.get("X", float("nan")),
        "muskingum_C0": params.get("C0", float("nan")),
        "muskingum_C1": params.get("C1", float("nan")),
        "muskingum_C2": params.get("C2", float("nan")),

        "NSE": nse(sim, obs),
        "KGE": kge(sim, obs),
        "RMSE": rmse(sim, obs),
        "MAE": mae(sim, obs),
        "Bias": bias(sim, obs),
        "PBIAS_percent": pbias(sim, obs),
        "R": corr_r(sim, obs),

        "qin_peak_m3s": float(np.max(ev.q_in)),
        "qin_peak_time_step": int(np.argmax(ev.q_in) + 1),

        "obs_peak_m3s": float(np.max(obs)),
        "obs_peak_time_step": int(np.argmax(obs) + 1),

        "sim_peak_m3s": float(np.max(sim)) if finite_sim else float("nan"),
        "sim_peak_time_step": int(np.argmax(sim) + 1) if finite_sim else -1,

        "peak_error_m3s": float(np.max(sim) - np.max(obs)) if finite_sim else float("nan"),
        "peak_error_percent": (
            float((np.max(sim) - np.max(obs)) / np.max(obs) * 100.0)
            if finite_sim and abs(float(np.max(obs))) > EPS
            else float("nan")
        ),
        "peak_time_error_step": int(np.argmax(sim) - np.argmax(obs)) if finite_sim else -999,

        "storage_peak_million_m3": (
            float(np.nanmax(storage_series))
            if storage_series is not None and np.any(np.isfinite(storage_series))
            else float("nan")
        ),
        "negative_sim_count": int(np.sum(sim < -EPS)) if finite_sim else 0,
        "nan_sim_count": int(np.sum(~np.isfinite(sim))),
        "source_file": ev.source_file,
    }


def hydro_rows(
    ev: Event,
    result: dict[str, np.ndarray],
    params: dict[str, float],
    fit_group: str,
    model: str,
) -> list[dict[str, object]]:

    rows: list[dict[str, object]] = []

    for i, t_h in enumerate(ev.time_h):
        row = {
            "model": model,
            "fit_group": fit_group,
            "river": ev.river,
            "event_id": ev.event_id,
            "event_no": ev.event_no,
            "split": ev.split,

            "time_step": i + 1,
            "t_h": float(t_h),

            "q_in_m3s": float(ev.q_in[i]),
            "q_out_obs_m3s": float(ev.q_out[i]),
            "Q_sim_m3s": float(result["Q_sim"][i]),
            "error_m3s": float(result["Q_sim"][i] - ev.q_out[i]),

            "lmc_k": params.get("k", float("nan")),
            "lmc_n": params.get("n", float("nan")),
            "lmc_tau_h": params.get("tau_h", float("nan")),
            "lmc_S0": params.get("S0", float("nan")),
            "muskingum_K_h": params.get("K_h", float("nan")),
            "muskingum_X": params.get("X", float("nan")),
            "muskingum_C0": params.get("C0", float("nan")),
            "muskingum_C1": params.get("C1", float("nan")),
            "muskingum_C2": params.get("C2", float("nan")),
        }

        if model == LMC_MODEL_NAME:
            row.update(
                {
                    "I_tau_prev_m3s": float(result["I_tau_prev"][i]),
                    "I_tau_now_m3s": float(result["I_tau_now"][i]),
                    "I_tau_bar_m3s": float(result["I_tau_bar"][i]),
                    "before_tau_h": bool(result["before_tau_h"][i]),
                    "S_w_million_m3": float(result["S_w"][i]),
                    "S_p_million_m3": float(result["S_p"][i]),
                    "S_t_million_m3": float(result["S_t"][i]),
                    "O_t_million_m3": float(result["O_t"][i]),
                    "S_muskingum_million_m3": float("nan"),
                }
            )
        else:
            row.update(
                {
                    "I_tau_prev_m3s": float("nan"),
                    "I_tau_now_m3s": float("nan"),
                    "I_tau_bar_m3s": float("nan"),
                    "before_tau_h": False,
                    "S_w_million_m3": float("nan"),
                    "S_p_million_m3": float("nan"),
                    "S_t_million_m3": float("nan"),
                    "O_t_million_m3": float("nan"),
                    "S_muskingum_million_m3": float(result["S_muskingum"][i]),
                }
            )

        rows.append(row)

    return rows


def summarize(metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if metrics.empty:
        return pd.DataFrame()

    for keys, g in metrics.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: key for col, key in zip(group_cols, keys)}

        nse_series = pd.to_numeric(g["NSE"], errors="coerce")
        kge_series = pd.to_numeric(g["KGE"], errors="coerce")
        rmse_series = pd.to_numeric(g["RMSE"], errors="coerce")
        peak_error = pd.to_numeric(g["peak_error_percent"], errors="coerce")
        peak_time = pd.to_numeric(g["peak_time_error_step"], errors="coerce")

        row.update(
            {
                "events": len(g),
                "mean_NSE": float(nse_series.mean()),
                "median_NSE": float(nse_series.median()),
                "mean_KGE": float(kge_series.mean()),
                "mean_RMSE": float(rmse_series.mean()),
                "median_RMSE": float(rmse_series.median()),
                "mean_abs_peak_error_percent": float(peak_error.abs().mean()),
                "mean_abs_peak_time_error_step": float(peak_time.abs().mean()),
                "NSE_lt_0p6_events": int((nse_series < 0.6).sum()),
                "nan_sim_count": int(pd.to_numeric(g["nan_sim_count"], errors="coerce").sum()),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def safe_output_name(name: str) -> str:
    safe_name = str(name).strip() or "unnamed"

    for bad_char in '<>:"/\\|?*':
        safe_name = safe_name.replace(bad_char, "_")

    return safe_name.replace(" ", "_")


def next_numbered_output_dir(base_dir: Path) -> Path:
    """在默认结果根目录下创建下一个编号运行目录。"""
    base_dir.mkdir(parents=True, exist_ok=True)
    number_pattern = re.compile(r"^(\d+)_")
    existing_numbers = []

    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        match = number_pattern.match(child.name)
        if match:
            existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1

    while True:
        candidate = base_dir / f"{next_number:03d}_{RUN_SUFFIX}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            next_number += 1


def make_parameter_table_for_one_river(
    params_df: pd.DataFrame,
    river_metrics: pd.DataFrame,
) -> pd.DataFrame:

    fit_groups_used_by_this_river = set(river_metrics["fit_group"].astype(str))
    selected = params_df[
        params_df["fit_group"].astype(str).isin(fit_groups_used_by_this_river)
    ].copy()

    return selected


def calibrated_parameter_row(
    model: str,
    fit_group: str,
    fit_scope: str,
    fit_events: list[Event],
    group_events: list[Event],
    params: dict[str, float],
    lmc_bounds: Bounds,
    muskingum_bounds: MuskingumBounds,
) -> dict[str, object]:

    row = {
        "model": model,
        "fit_group": fit_group,
        "fit_scope": fit_scope,
        "fit_events": len(fit_events),
        "eval_events": len(group_events),
        "train_loss": params["train_loss"],

        "lmc_k": params.get("k", float("nan")),
        "lmc_n": params.get("n", float("nan")),
        "lmc_tau_h": params.get("tau_h", float("nan")),
        "lmc_S0": params.get("S0", float("nan")),

        "muskingum_K_h": params.get("K_h", float("nan")),
        "muskingum_X": params.get("X", float("nan")),
        "muskingum_C0": params.get("C0", float("nan")),
        "muskingum_C1": params.get("C1", float("nan")),
        "muskingum_C2": params.get("C2", float("nan")),

        "lmc_k_lower_hit": False,
        "lmc_k_upper_hit": False,
        "lmc_n_lower_hit": False,
        "lmc_n_upper_hit": False,
        "lmc_tau_lower_hit": False,
        "lmc_tau_upper_hit": False,
        "muskingum_K_lower_hit": False,
        "muskingum_K_upper_hit": False,
        "muskingum_X_lower_hit": False,
        "muskingum_X_upper_hit": False,
    }

    if model == LMC_MODEL_NAME:
        row.update(
            {
                "lmc_k_lower_hit": params["k"] <= lmc_bounds.k[0] * 1.001,
                "lmc_k_upper_hit": params["k"] >= lmc_bounds.k[1] * 0.999,
                "lmc_n_lower_hit": params["n"] <= lmc_bounds.n[0] * 1.001,
                "lmc_n_upper_hit": params["n"] >= lmc_bounds.n[1] * 0.999,
                "lmc_tau_lower_hit": params["tau_h"] <= lmc_bounds.tau_h[0] + 1.0e-8,
                "lmc_tau_upper_hit": params["tau_h"] >= lmc_bounds.tau_h[1] * 0.999,
            }
        )
    else:
        row.update(
            {
                "muskingum_K_lower_hit": params["K_h"] <= muskingum_bounds.K_h[0] * 1.001,
                "muskingum_K_upper_hit": params["K_h"] >= muskingum_bounds.K_h[1] * 0.999,
                "muskingum_X_lower_hit": params["X"] <= muskingum_bounds.X[0] + 1.0e-8,
                "muskingum_X_upper_hit": params["X"] >= muskingum_bounds.X[1] * 0.999,
            }
        )

    return row


def write_one_river_result(
    table_dir: Path,
    river: str,
    assignments: pd.DataFrame,
    params_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    hydro_df: pd.DataFrame | None,
) -> Path:

    river_dir = table_dir / "by_river" / safe_output_name(river)
    river_dir.mkdir(parents=True, exist_ok=True)

    river_assignments = assignments[assignments["river"] == river].copy()
    river_metrics = metrics_df[metrics_df["river"] == river].copy()
    river_params = make_parameter_table_for_one_river(params_df, river_metrics)
    river_summary = summarize(river_metrics, ["model", "split"])

    river_assignments.to_csv(
        river_dir / "event_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    river_params.to_csv(
        river_dir / "model_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )
    river_metrics.to_csv(
        river_dir / "event_metrics_by_model.csv",
        index=False,
        encoding="utf-8-sig",
    )
    river_summary.to_csv(
        river_dir / "model_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if hydro_df is not None:
        river_hydro = hydro_df[hydro_df["river"] == river].copy()
        river_hydro.to_csv(
            river_dir / "hydrograph_predictions_by_model.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return river_dir


def stop_if_multi_river_mixed_fit(events: list[Event], fit_scope: str) -> None:
    rivers = sorted({ev.river for ev in events})

    if len(rivers) <= 1 or fit_scope != "all":
        return

    raise ValueError(
        "当前读到了多个流域："
        + ", ".join(rivers)
        + "。不要用 --fit-scope all 把它们混在一起率定；"
        + "请使用默认的 --fit-scope river，或者用 --basins 单独计算某一个流域。"
    )


def plot_event(ev: Event, result: dict[str, np.ndarray], model: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(1, len(ev.q_in) + 1)

    plt.figure(figsize=(10, 5.8), dpi=300)
    plt.plot(x, ev.q_in, linestyle="--", linewidth=1.7, label="Inflow")
    plt.plot(x, ev.q_out, linewidth=2.0, label="Observed outflow")
    plt.plot(x, result["Q_sim"], linewidth=2.0, label=f"{model} simulated outflow")
    plt.xlabel("Time step")
    plt.ylabel("Flow (m3/s)")
    plt.title(f"{model}: {ev.event_id}")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


# ============================================================
# 8. 参数入口
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多流域河道演算模型对比：LMC 水能互补模型（S0=0）和马斯京根模型。"
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="结果根目录；使用默认值时自动建立 NNN_fixed_tau 子目录。",
    )
    parser.add_argument("--data-files", nargs="+", help="一个或多个数据文件。相对路径按 PROJECT_DIR 解析。")
    parser.add_argument("--data-glob", nargs="+", help='通配符，例如 "data/*_flood_events.csv"。')
    parser.add_argument("--all-data-files", action="store_true", help="读取 DATA_DIR 下所有 *_flood_events.csv 和 .xlsx 数据文件。")

    parser.add_argument("--basins", nargs="+", help="只计算指定 River 名称。多个流域用空格分隔。")
    parser.add_argument("--events", nargs="+", type=int, help="只计算指定 Event 编号。多个编号用空格分隔。")
    parser.add_argument("--list-basins", action="store_true", help="只列出流域和事件，不进行率定。")

    parser.add_argument(
        "--fit-scope",
        choices=["all", "river", "event"],
        default="river",
        help="river=每个流域一套参数；event=每场洪水一套参数；all=仅适合单流域，不用于多流域混合率定。",
    )

    parser.add_argument("--dt-h", type=float, default=DEFAULT_DT_H)
    parser.add_argument("--test-ratio", type=float, default=0.2)

    parser.add_argument("--k-bounds", nargs=2, type=float, default=(0.001, 5.0))
    parser.add_argument("--n-bounds", nargs=2, type=float, default=(0.5, 80.0))
    parser.add_argument("--tau-bounds", nargs=2, type=float, default=(0.0, 12.0))
    parser.add_argument("--muskingum-k-bounds", nargs=2, type=float, default=(0.1, 48.0))
    parser.add_argument("--muskingum-x-bounds", nargs=2, type=float, default=(0.0, 0.5))

    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--maxiter", type=int, default=800)
    parser.add_argument("--popsize", type=int, default=15)

    parser.add_argument("--plot", action="store_true", help="输出每场洪水过程线 PNG。")
    parser.add_argument("--no-hydrograph", action="store_true", help="不输出逐时段过程线表，减少文件体积。")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Bounds, MuskingumBounds]:
    if args.dt_h <= 0:
        raise ValueError("--dt-h 必须大于 0。")

    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("--test-ratio 必须在 0 和 1 之间。")

    k_bounds = (float(args.k_bounds[0]), float(args.k_bounds[1]))
    n_bounds = (float(args.n_bounds[0]), float(args.n_bounds[1]))
    tau_bounds = (float(args.tau_bounds[0]), float(args.tau_bounds[1]))
    muskingum_k_bounds = (
        float(args.muskingum_k_bounds[0]),
        float(args.muskingum_k_bounds[1]),
    )
    muskingum_x_bounds = (
        float(args.muskingum_x_bounds[0]),
        float(args.muskingum_x_bounds[1]),
    )

    for name, pair in [
        ("k", k_bounds),
        ("n", n_bounds),
        ("tau", tau_bounds),
        ("muskingum_k", muskingum_k_bounds),
        ("muskingum_x", muskingum_x_bounds),
    ]:
        if pair[1] <= pair[0]:
            raise ValueError(f"{name} 参数范围必须满足 min < max，当前为 {pair}")

    if muskingum_x_bounds[0] < 0.0 or muskingum_x_bounds[1] > 0.5:
        raise ValueError("--muskingum-x-bounds 必须在 0 到 0.5 之间。")

    return (
        Bounds(k=k_bounds, n=n_bounds, tau_h=tau_bounds),
        MuskingumBounds(K_h=muskingum_k_bounds, X=muskingum_x_bounds),
    )


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    args = parse_args()
    lmc_bounds, muskingum_bounds = validate_args(args)

    requested_out = args.out_dir.resolve()
    if requested_out == DEFAULT_OUT.resolve():
        out = next_numbered_output_dir(requested_out)
    else:
        out = requested_out
        out.mkdir(parents=True, exist_ok=True)

    table_dir = out / "02_tables"
    plot_dir = out / "02_plots"

    table_dir.mkdir(parents=True, exist_ok=True)

    data_files = resolve_data_files(args)

    basins = set(args.basins) if args.basins else None
    event_numbers = set(args.events) if args.events else None

    events = read_events(
        paths=data_files,
        test_ratio=args.test_ratio,
        basins=basins,
        dt_h=args.dt_h,
    )

    events = filter_event_numbers(events, event_numbers)

    assignments = make_assignment_table(events)
    assignments.to_csv(
        table_dir / "event_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if args.list_basins:
        view = (
            assignments
            .groupby("river", sort=True)
            .agg(
                events=("event_no", lambda x: ",".join(str(int(v)) for v in sorted(set(x)))),
                event_count=("event_no", "nunique"),
                points=("n_points", "sum"),
            )
            .reset_index()
        )

        print("\n当前读到的流域和事件：")
        print(view.to_string(index=False))
        print(f"\n事件清单已保存：{table_dir / 'event_assignments.csv'}")
        return

    stop_if_multi_river_mixed_fit(events, args.fit_scope)

    fit_groups = build_fit_groups(events, args.fit_scope)

    param_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    all_hydro_rows: list[dict[str, object]] = []

    for fit_group, group_events in fit_groups.items():
        fit_events = training_subset(group_events, args.fit_scope)

        if not fit_events:
            raise ValueError(f"{fit_group} 没有可用于率定的事件。")

        lmc_params = calibrate_lmc(
            events=fit_events,
            bounds=lmc_bounds,
            dt_h=args.dt_h,
            seed=stable_seed(args.seed, f"{fit_group}_{LMC_MODEL_NAME}"),
            maxiter=args.maxiter,
            popsize=args.popsize,
        )

        param_rows.append(
            calibrated_parameter_row(
                model=LMC_MODEL_NAME,
                fit_group=fit_group,
                fit_scope=args.fit_scope,
                fit_events=fit_events,
                group_events=group_events,
                params=lmc_params,
                lmc_bounds=lmc_bounds,
                muskingum_bounds=muskingum_bounds,
            )
        )

        muskingum_params = calibrate_muskingum(
            events=fit_events,
            bounds=muskingum_bounds,
            dt_h=args.dt_h,
            seed=stable_seed(args.seed, f"{fit_group}_Muskingum"),
            maxiter=args.maxiter,
            popsize=args.popsize,
        )

        param_rows.append(
            calibrated_parameter_row(
                model="Muskingum",
                fit_group=fit_group,
                fit_scope=args.fit_scope,
                fit_events=fit_events,
                group_events=group_events,
                params=muskingum_params,
                lmc_bounds=lmc_bounds,
                muskingum_bounds=muskingum_bounds,
            )
        )

        for ev in group_events:
            lmc_result = route_lmc_s0_zero(
                q_in=ev.q_in,
                q0=float(ev.q_out[0]),
                k=lmc_params["k"],
                n_comp=lmc_params["n"],
                tau_h=lmc_params["tau_h"],
                dt_h=args.dt_h,
            )

            metric_rows.append(
                metric_row(
                    ev=ev,
                    result=lmc_result,
                    params=lmc_params,
                    fit_group=fit_group,
                    model=LMC_MODEL_NAME,
                )
            )

            if not args.no_hydrograph:
                all_hydro_rows.extend(
                    hydro_rows(
                        ev=ev,
                        result=lmc_result,
                        params=lmc_params,
                        fit_group=fit_group,
                        model=LMC_MODEL_NAME,
                    )
                )

            muskingum_result = route_muskingum(
                q_in=ev.q_in,
                q0=float(ev.q_out[0]),
                K_h=muskingum_params["K_h"],
                X=muskingum_params["X"],
                dt_h=args.dt_h,
            )

            metric_rows.append(
                metric_row(
                    ev=ev,
                    result=muskingum_result,
                    params=muskingum_params,
                    fit_group=fit_group,
                    model="Muskingum",
                )
            )

            if not args.no_hydrograph:
                all_hydro_rows.extend(
                    hydro_rows(
                        ev=ev,
                        result=muskingum_result,
                        params=muskingum_params,
                        fit_group=fit_group,
                        model="Muskingum",
                    )
                )

            if args.plot:
                one_river_plot_dir = plot_dir / safe_output_name(ev.river)
                one_river_plot_dir.mkdir(parents=True, exist_ok=True)
                safe_name = f"{fit_group}_{ev.event_id}"
                safe_name = safe_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                plot_event(ev, lmc_result, LMC_MODEL_NAME, one_river_plot_dir / f"{safe_name}_{LMC_MODEL_NAME}.png")
                plot_event(ev, muskingum_result, "Muskingum", one_river_plot_dir / f"{safe_name}_Muskingum.png")

    params_df = pd.DataFrame(param_rows)
    metrics_df = pd.DataFrame(metric_rows)
    hydro_df = None

    if not args.no_hydrograph:
        hydro_df = pd.DataFrame(all_hydro_rows)

    summary_by_river = summarize(metrics_df, ["river", "model", "split"])
    summary_by_group = summarize(metrics_df, ["fit_group", "model", "split"])
    river_names = sorted(str(x) for x in metrics_df["river"].dropna().unique())
    river_dirs = [
        write_one_river_result(
            table_dir=table_dir,
            river=river,
            assignments=assignments,
            params_df=params_df,
            metrics_df=metrics_df,
            hydro_df=hydro_df,
        )
        for river in river_names
    ]

    params_df.to_csv(table_dir / "all_rivers_model_parameters.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(table_dir / "all_rivers_event_metrics_by_model.csv", index=False, encoding="utf-8-sig")
    summary_by_river.to_csv(table_dir / "model_comparison_by_river_split.csv", index=False, encoding="utf-8-sig")
    summary_by_group.to_csv(table_dir / "model_comparison_by_fit_group_split.csv", index=False, encoding="utf-8-sig")

    if not args.no_hydrograph:
        hydro_df.to_csv(
            table_dir / "all_rivers_hydrograph_predictions_by_model.csv",
            index=False,
            encoding="utf-8-sig",
        )

    settings = pd.DataFrame(
        [
            {
                "PROJECT_DIR": str(PROJECT_DIR),
                "DATA_DIR": str(DATA_DIR),
                "data_files": ";".join(str(p) for p in data_files),
                "out_dir": str(out),
                "fit_scope": args.fit_scope,
                "basins": ",".join(sorted(basins)) if basins else "ALL",
                "events": ",".join(str(x) for x in sorted(event_numbers)) if event_numbers else "ALL",
                "dt_h": args.dt_h,
                "test_ratio": args.test_ratio,
                "lmc_k_bounds": str(lmc_bounds.k),
                "lmc_n_bounds": str(lmc_bounds.n),
                "lmc_tau_bounds": str(lmc_bounds.tau_h),
                "muskingum_K_h_bounds": str(muskingum_bounds.K_h),
                "muskingum_X_bounds": str(muskingum_bounds.X),
                "seed": args.seed,
                "maxiter": args.maxiter,
                "popsize": args.popsize,
                "models": f"{LMC_MODEL_NAME};Muskingum",
                "LMC_S0": 0.0,
                "LMC_tau_type": "fixed",
                "LMC_before_tau_rule": "Q_sim=q_in, S_t remains zero",
                "LMC_inflow_used": "period-average inflow lagged by fixed tau",
                "result_layout": "numbered run folder, then 02_tables/by_river",
            }
        ]
    )

    settings.to_csv(table_dir / "run_settings.csv", index=False, encoding="utf-8-sig")

    print("\n计算完成。")
    print(f"输出目录：{out}")
    print(f"事件清单：{table_dir / 'event_assignments.csv'}")
    print(f"按流域输出目录：{table_dir / 'by_river'}")
    print(f"全部流域参数总表：{table_dir / 'all_rivers_model_parameters.csv'}")
    print(f"全部流域指标总表：{table_dir / 'all_rivers_event_metrics_by_model.csv'}")
    print(f"模型对比汇总：{table_dir / 'model_comparison_by_river_split.csv'}")

    if not args.no_hydrograph:
        print(f"全部流域过程线总表：{table_dir / 'all_rivers_hydrograph_predictions_by_model.csv'}")

    if args.plot:
        print(f"过程线图目录：{plot_dir}")

    print("\n参数结果：")
    show_cols = [
        "model",
        "fit_group",
        "fit_events",
        "eval_events",
        "lmc_k",
        "lmc_n",
        "lmc_tau_h",
        "muskingum_K_h",
        "muskingum_X",
        "train_loss",
    ]
    print(params_df[show_cols].to_string(index=False))

    print("\n按流域、模型汇总：")
    print(summary_by_river.to_string(index=False))

    print("\n各流域结果文件夹：")
    for river_dir in river_dirs:
        print(f"  {river_dir}")


if __name__ == "__main__":
    main()

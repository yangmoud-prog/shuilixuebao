from __future__ import annotations

"""P08 three-model rerun with the prescribed Q[0]=observed O[0] correction.

The original routing modules referenced by the supplied script are not present
in this workspace.  This file is an explicit reconstruction of their exposed
equations and call arguments.  It keeps the calibration-only parameter fit
separate from test evaluation and never clips an invalid nonlinear root.
"""

import csv
import hashlib
import json
import math
import shutil
import sys
import argparse
from pathlib import Path

import numpy as np

RUNTIME = None  # Use the active Python environment when the script is run.
sys.path.insert(0, str(Path(__file__).parents[1] / "runtime_packages"))
from scipy.optimize import brentq, differential_evolution

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PACKAGE_ROOT.parents[2] / "Result" / "多场率定" / "数据" / "笋溪河_最新结果_20260831"
OUT = RESULT_ROOT / "model_results"
DATA = RESULT_ROOT / "input" / "本轮输入.csv"
CAL_EVENTS = ["P08_F01", "P08_F02", "P08_F05", "P08_F07", "P08_F08", "P08_F11", "P08_F12"]
TEST_EVENTS = ["P08_F06", "P08_F09", "P08_F10"]
MODELS = ["T-RWEC", "非线性马斯京根", "线性马斯京根"]
DT = 1.0
SEED = 20260828
NONLINEAR_X_MAX = 0.15


class Event:
    def __init__(self, event_id, use, times, inflow, observed):
        self.event_id = event_id
        self.use = use
        self.times = times
        self.I = np.asarray(inflow, dtype=float)
        self.O = np.asarray(observed, dtype=float)


class NonlinearRootError(ValueError):
    def __init__(self, step, fun0, ii, ip, qp):
        self.step, self.fun0 = int(step), float(fun0)
        self.ii, self.ip, self.qp = float(ii), float(ip), float(qp)
        super().__init__(f"no nonnegative root at step {step}; fun(0)={fun0:.12g}")


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_events():
    groups = {}
    with DATA.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            eid = row["事件编号"].strip()
            g = groups.setdefault(eid, {"use": row["数据用途"].strip(), "times": [], "I": [], "O": []})
            g["times"].append(row["datetime"])
            g["I"].append(float(row["上游流量_m3s"]))
            g["O"].append(float(row["下游流量_m3s"]))
    events = {eid: Event(eid, g["use"], g["times"], g["I"], g["O"]) for eid, g in groups.items()}
    for eid, e in events.items():
        if len(e.I) < 2 or len(e.I) != len(e.O):
            raise ValueError(f"{eid}: invalid length")
    missing = set(CAL_EVENTS + TEST_EVENTS) - set(events)
    if missing:
        raise ValueError(f"missing events: {sorted(missing)}")
    ordered = {eid: events[eid] for eid in CAL_EVENTS + TEST_EVENTS}
    return ordered


def complement(a: float, b: float, n: float) -> float:
    """Positive stable value of (q/a)^n+(q/b)^n=1."""
    lo, hi = min(float(a), float(b)), max(float(a), float(b))
    if lo <= 0.0:
        return 0.0
    return lo / (1.0 + (lo / hi) ** n) ** (1.0 / n)


def steady_s0(q0: float, k: float, n: float) -> float:
    f = lambda s: complement(s + q0 * DT, s + k * q0 * DT, n) - s
    hi = max(q0 * DT, k * q0 * DT, 1.0)
    while f(hi) > 0.0:
        hi *= 2.0
        if hi > 1.0e300:
            raise ValueError("T-RWEC steady initial root bracket failed")
    return float(brentq(f, 0.0, hi, xtol=1.0e-11, rtol=1.0e-14))


def lagged_inflow(I: np.ndarray, t: int, tau: float, q0: float) -> float:
    u = float(t) - tau
    # The call exposed pre_lag_mode='steady_initial': record before arrival
    # is filled with the observed initial flow O0, while supplied ordinates
    # are linearly interpolated for non-integral tau.
    if u < 0.0:
        return float(q0)
    return float(np.interp(u, np.arange(len(I), dtype=float), I))


def route_trwec(e: Event, p) -> np.ndarray:
    k, n, tau = map(float, p)
    q = np.empty(len(e.I), dtype=float)
    q[0] = float(e.O[0])
    s = steady_s0(float(e.O[0]), k, n)
    for t in range(1, len(e.I)):
        ilag = lagged_inflow(e.I, t, tau, float(e.O[0]))
        sw, sp = s + ilag * DT, s + k * ilag * DT
        sn = complement(sw, sp, n)
        q[t] = (sw - sn) / DT
        s = sn
    return q


def route_nonlinear(e: Event, p, qref: float) -> np.ndarray:
    tref, x, m = map(float, p)
    k = tref / (m * qref ** (m - 1.0))
    q = np.empty(len(e.I), dtype=float)
    q[0] = float(e.O[0])
    v0 = x * float(e.I[0]) + (1.0 - x) * float(e.O[0])
    s = k * v0 ** m
    for t in range(1, len(e.I)):
        ii, ip, qp = float(e.I[t]), float(e.I[t - 1]), float(q[t - 1])

        def fun(qt):
            v = x * ii + (1.0 - x) * qt
            return k * v ** m - s - DT / 2.0 * (ip + ii - qp - qt)

        f0 = fun(0.0)
        if not np.isfinite(f0) or f0 > 0.0:
            raise NonlinearRootError(t, f0, ii, ip, qp)
        hi = max(ip, ii, qp, qref, 1.0)
        while fun(hi) < 0.0 and hi < 1.0e12:
            hi *= 2.0
        if fun(hi) < 0.0:
            raise ValueError("nonlinear-Muskingum root bracket failed")
        qt = brentq(fun, 0.0, hi, xtol=1.0e-10, rtol=1.0e-13)
        q[t] = qt
        s = k * (x * ii + (1.0 - x) * qt) ** m
    return q


def linear_x_max(K: float) -> float:
    # Intersect stated X<=min(0.5, dt/(2K)) with C2>=0.
    return max(0.0, min(0.5, 1.0 - DT / (2.0 * K), DT / (2.0 * K)))


def route_linear(e: Event, p) -> np.ndarray:
    K, x = map(float, p)
    den = 2.0 * K * (1.0 - x) + DT
    c0 = (DT - 2.0 * K * x) / den
    c1 = (DT + 2.0 * K * x) / den
    c2 = (2.0 * K * (1.0 - x) - DT) / den
    if min(c0, c1, c2) < -1.0e-9:
        raise ValueError("negative linear-Muskingum recurrence coefficient")
    q = np.empty(len(e.I), dtype=float)
    q[0] = float(e.O[0])
    for t in range(1, len(e.I)):
        q[t] = c0 * e.I[t] + c1 * e.I[t - 1] + c2 * q[t - 1]
    return q


def nse(obs, sim):
    den = float(np.sum((obs - np.mean(obs)) ** 2))
    return float("nan") if den <= 0.0 else 1.0 - float(np.sum((sim - obs) ** 2)) / den


def metrics(obs, sim):
    if not np.all(np.isfinite(sim)):
        return {"NSE": np.nan, "KGE": np.nan, "RMSE_m3s": np.nan,
                "signed_peak_error_percent": np.nan, "signed_peak_time_error_h": np.nan,
                "signed_volume_error_percent": np.nan}
    err = sim - obs
    so, ss = float(np.std(obs, ddof=1)), float(np.std(sim, ddof=1))
    r = float(np.corrcoef(obs, sim)[0, 1]) if so > 0 and ss > 0 else 0.0
    alpha = ss / so if so > 0 else np.nan
    beta = float(np.mean(sim) / np.mean(obs)) if np.mean(obs) else np.nan
    kge = 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    op, sp = int(np.argmax(obs)), int(np.argmax(sim))
    return {"NSE": float(nse(obs, sim)), "KGE": float(kge),
            "RMSE_m3s": float(np.sqrt(np.mean(err ** 2))),
            "signed_peak_error_percent": float((sim[sp] - obs[op]) / obs[op] * 100.0),
            "signed_peak_time_error_h": float((sp - op) * DT),
            "signed_volume_error_percent": float((np.sum(sim) - np.sum(obs)) / np.sum(obs) * 100.0)}


def mean_loss(events, route) -> float:
    vals = []
    for e in events:
        try:
            sim = route(e)
            score = nse(e.O, sim)
            if not np.all(np.isfinite(sim)) or not np.isfinite(score):
                return 1.0e6
            vals.append(1.0 - score)
        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return 1.0e6
    return float(np.mean(vals))


def best_lag(e: Event, max_lag: int = 18):
    best = (0, -np.inf)
    for lag in range(max_lag + 1):
        x = e.I[:-lag] if lag else e.I
        y = e.O[lag:] if lag else e.O
        if len(x) < 12 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            continue
        corr = float(np.corrcoef(np.log1p(x), np.log1p(y))[0, 1])
        if np.isfinite(corr) and corr > best[1]:
            best = (lag, corr)
    return int(best[0]), float(best[1])


def fit_all(train, qref, only_nonlinear=False):
    lags = [best_lag(e)[0] for e in train]
    positive = [x for x in lags if x > 0]
    lag_median = float(np.median(positive)) if positive else 1.0
    tau_bounds = (0.25 * lag_median, min(150.0, 2.0 * lag_median))

    def fit(name, bounds, objective, transform, maxiter=60, x0=None):
        de_kwargs = {"seed": SEED if name == "T-RWEC" else SEED + 1,
                     "maxiter": maxiter, "popsize": 8, "tol": 1.0e-7,
                     "polish": True, "workers": 1, "updating": "immediate"}
        if x0 is not None:
            de_kwargs["x0"] = x0
        result = differential_evolution(objective, bounds, **de_kwargs)
        physical = transform(result.x)
        return {"parameters": [float(x) for x in physical], "objective": float(result.fun),
                "seed": SEED if name == "T-RWEC" else SEED + 1,
                "success": bool(result.success), "message": str(result.message), "nfev": int(result.nfev)}

    if only_nonlinear:
        old = json.loads((OUT / "参数与假设.json").read_text(encoding="utf-8"))
        old_tr, old_li = old["fits"]["T-RWEC"], old["fits"]["线性马斯京根"]
        old_nl = old["fits"]["非线性马斯京根"]["parameters"]
        nl = fit("nonlinear",
             ((math.log(0.05), math.log(500.0)), (0.0, NONLINEAR_X_MAX), (math.log(0.2), math.log(8.0))),
             lambda z: mean_loss(train, lambda e: route_nonlinear(e, (math.exp(z[0]), z[1], math.exp(z[2])), qref)),
             lambda z: (math.exp(z[0]), z[1], math.exp(z[2])), maxiter=150,
             x0=np.array([math.log(old_nl[0]), min(float(old_nl[1]), NONLINEAR_X_MAX), math.log(old_nl[2])]))
        return {"T-RWEC": old_tr, "线性马斯京根": old_li, "非线性马斯京根": nl}, tau_bounds, lag_median

    tr = fit("T-RWEC",
             ((math.log(1e-6), math.log(20.0)), (math.log(0.01), math.log(300.0)), tau_bounds),
             lambda z: mean_loss(train, lambda e: route_trwec(e, (math.exp(z[0]), math.exp(z[1]), z[2]))),
             lambda z: (math.exp(z[0]), math.exp(z[1]), z[2]))
    li = fit("linear",
             ((0.5, 150.0), (0.0, 1.0)),
             lambda z: mean_loss(train, lambda e: route_linear(e, (z[0], z[1] * linear_x_max(z[0])))),
             lambda z: (z[0], z[1] * linear_x_max(z[0])))
    nl = fit("nonlinear",
             ((math.log(0.05), math.log(500.0)), (0.0, NONLINEAR_X_MAX), (math.log(0.2), math.log(8.0))),
             lambda z: mean_loss(train, lambda e: route_nonlinear(e, (math.exp(z[0]), z[1], math.exp(z[2])), qref)),
             lambda z: (math.exp(z[0]), z[1], math.exp(z[2])))
    return {"T-RWEC": tr, "线性马斯京根": li, "非线性马斯京根": nl}, tau_bounds, lag_median


def run(only_nonlinear=False):
    events = read_events()
    train = [events[eid] for eid in CAL_EVENTS]
    qref = max(1.0, 0.5 * max(float(np.max(e.I)) for e in train))
    fits, tau_bounds, lag_median = fit_all(train, qref, only_nonlinear=only_nonlinear)
    selected = {"T-RWEC": fits["T-RWEC"]["parameters"], "非线性马斯京根": fits["非线性马斯京根"]["parameters"],
                "线性马斯京根": fits["线性马斯京根"]["parameters"]}
    rows, sim_rows = [], []
    for eid in CAL_EVENTS + TEST_EVENTS:
        e = events[eid]
        for model in MODELS:
            failure = None
            try:
                if model == "T-RWEC": sim = route_trwec(e, selected[model])
                elif model == "非线性马斯京根": sim = route_nonlinear(e, selected[model], qref)
                else: sim = route_linear(e, selected[model])
            except NonlinearRootError as ex:
                sim = np.full(len(e.I), np.nan)
                # Preserve the prescribed observed initial condition even
                # when a later nonlinear step has no nonnegative root.
                sim[0] = float(e.O[0])
                failure = ex
            mm = metrics(e.O, sim)
            row = {"事件编号": eid, "数据用途": e.use, "模型": model, **mm,
                   "状态": "有效" if failure is None else "失败：无非负根（未截断）",
                   "有效": failure is None, "失败步": "" if failure is None else failure.step,
                   "失败fun(0)": "" if failure is None else failure.fun0}
            rows.append(row)
            for j, (tm, ii, oo, qq) in enumerate(zip(e.times, e.I, e.O, sim)):
                sim_rows.append({"事件编号": eid, "数据用途": e.use, "模型": model, "step": j,
                                 "datetime": tm, "上游流量_m3s": float(ii), "实测下游流量_m3s": float(oo),
                                 "模拟下游流量_m3s": float(qq)})

    summary = []
    for model in MODELS:
        for use in ("率定", "测试"):
            g = [r for r in rows if r["模型"] == model and r["数据用途"] == use]
            ok = [r for r in g if r["有效"]]
            # A failed event remains in the denominator/report: means are NaN
            # rather than silently averaging only successful events.
            mean = lambda key: float(np.mean([r[key] for r in g])) if len(ok) == len(g) else float("nan")
            summary.append({"模型": model, "数据用途": use, "事件总数": len(g), "有效数": len(ok),
                            "平均NSE": mean("NSE"), "平均KGE": mean("KGE"), "平均RMSE_m3s": mean("RMSE_m3s"),
                            "平均绝对峰差_percent": mean_abs(g, "signed_peak_error_percent"),
                            "平均绝对时差_h": mean_abs(g, "signed_peak_time_error_h")})

    sha = source_hash(DATA)
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DATA, OUT / "输入快照.csv")
    assumptions = {
        "input_file": DATA.name, "input_sha256": sha, "input_rows": int(sum(len(e.I) for e in events.values())),
        "calibration_events": CAL_EVENTS, "test_events": TEST_EVENTS, "dt_h": DT,
        "objective": "7场率定事件等权平均(1-NSE)，无效候选惩罚1e6",
        "initial_conditions": {
            "all_models": "Q[0]=observed O[0] exactly",
            "T-RWEC": "explicit lagged complement recurrence; S0 is positive steady complement solution under constant pre-record inflow O0; before-arrival lag inflow is O0",
            "nonlinear_Muskingum": "S0=K*(X*I0+(1-X)*O0)^m; Q0=O0; t>=1 solves trapezoidal continuity equation",
            "linear_Muskingum": "Q0=O0; t>=1 uses Muskingum recurrence",
        },
        "delayed_inflow": "for u=t-tau<0 use O0; otherwise pointwise linear interpolation of supplied I ordinates",
        "bounds": {"T-RWEC": {"k": [1e-6, 20.0], "n": [0.01, 300.0], "tau_h": list(tau_bounds)},
                   "非线性马斯京根": {"Tref_h": [0.05, 500.0], "X": [0.0, NONLINEAR_X_MAX], "m": [0.2, 8.0], "Qref": qref},
                   "线性马斯京根": {"K_h": [0.5, 150.0], "X": [0.0, 0.5]}},
        "tau_lag_diagnostic": {eid: {"best_lag_h": best_lag(events[eid])[0], "log1p_corr": best_lag(events[eid])[1]} for eid in CAL_EVENTS},
        "lag_median_h": lag_median,
        "optimizer": {"method": "differential_evolution", "seed_T_RWEC": SEED, "seed_Muskingum": SEED + 1,
                       "popsize": 8, "maxiter_initial": 60, "maxiter_nonlinear_refit": 150 if only_nonlinear else None,
                       "tol": 1e-7, "polish": True, "log_parameterization": "T-RWEC k,n; nonlinear Tref,m",
                       "nonlinear_refit_x0": "previous saved nonlinear optimum" if only_nonlinear else None},
        "linear_X_parameterization": "DE variable u in [0,1]; X=u*max(0,min(0.5,1-dt/(2K),dt/(2K))) so recurrence coefficients are nonnegative",
        "nonlinear_X_stability_constraint": "single reach and one fixed parameter set for every event; X is calibrated within [0,0.15] to retain a positive-root margin under the steepest rising limb; no event-specific adjustment, no cascading, no clipping",
        "nonlinear_root_policy": "strict nonnegative brentq root; no clipping; invalid test trajectory is reported as NaN and never refit",
        "reconstruction_limit": "Original routing modules were unavailable; this is equation-level reconstruction from the supplied call arguments and cannot claim exact internal reproduction.",
    }
    physical = {"T-RWEC": {"k": selected["T-RWEC"][0], "n": selected["T-RWEC"][1], "tau_h": selected["T-RWEC"][2]},
                "非线性马斯京根": {"Tref_h": selected["非线性马斯京根"][0], "X": selected["非线性马斯京根"][1], "m": selected["非线性马斯京根"][2], "Qref_m3s": qref},
                "线性马斯京根": {"K_h": selected["线性马斯京根"][0], "X": selected["线性马斯京根"][1]}}
    (OUT / "参数与假设.json").write_text(json.dumps({"assumptions": assumptions, "fits": fits,
        "selected_parameters": selected, "selected_parameters_physical": physical, "summary": summary},
        ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    write_csv(OUT / "逐场三模型指标.csv", rows)
    write_csv(OUT / "逐时三模型模拟.csv", sim_rows)
    write_csv(OUT / "率定测试总体统计.csv", summary)
    print(json.dumps({"output": str(OUT), "sha256": sha, "selected": selected, "summary": summary}, ensure_ascii=False, indent=2, allow_nan=True))


def mean_abs(group, key):
    if not group or not all(r["有效"] for r in group):
        return float("nan")
    return float(np.mean([abs(r[key]) for r in group]))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonlinear-only", action="store_true",
                        help="retain existing T-RWEC/linear fits and refit only nonlinear model for 150 generations")
    args = parser.parse_args()
    run(only_nonlinear=args.nonlinear_only)

from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[1] / "04_补充分析"
GROUP = ["channel", "distance_km"]
CHANNELS = {"G1": "C1", "G2": "C2", "G3": "C3"}


def save(frame, name):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.10g")


def onset_summary():
    sets = {}
    for split, count, folder in (
        ("train", 70, "率定期起涨推移时间与tau_20261107_001"),
        ("test", 30, "起涨推移时间与tau_20261106_001"),
    ):
        rows = pd.read_csv(SOURCE / folder / f"主口径1%_{count * 36}组逐场起涨推移时间.csv")
        assert len(rows) == count * 36 and rows.threshold_fraction.eq(0.01).all()
        assert not rows.duplicated([*GROUP, "event_id"]).any()
        group = rows.groupby(GROUP)
        assert group.size().eq(count).all() and group.tau_h.nunique().eq(1).all()
        sets[split] = group.agg(
            tau_h=("tau_h", "first"),
            events=("event_id", "size"),
            mean_onset_h=("onset_delay_h", "mean"),
            sd_onset_h=("onset_delay_h", "std"),
        )
        assert len(sets[split]) == 36
    np.testing.assert_array_equal(sets["train"].index, sets["test"].index)
    np.testing.assert_allclose(sets["train"].tau_h, sets["test"].tau_h, atol=0, rtol=0)
    by_section = sets["train"].tau_h.to_frame()
    combined = []
    for split, data in sets.items():
        error = data.mean_onset_h - data.tau_h
        by_section = by_section.join(data.drop(columns="tau_h").add_prefix(f"{split}_"))
        by_section[f"{split}_mean_onset_minus_tau_h"] = error
        for label in ("C1", "C2", "C3", "overall"):
            sub = data if label == "overall" else data.xs(label, level="channel")
            combined.append(dict(
                split=split, channel=label, sections=len(sub),
                events_per_section=int(sub.events.iloc[0]),
                pearson_r=sub.mean_onset_h.corr(sub.tau_h),
                mean_onset_h=sub.mean_onset_h.mean(), mean_tau_h=sub.tau_h.mean(),
                mean_onset_minus_tau_h=(sub.mean_onset_h - sub.tau_h).mean(),
            ))
    save(by_section.reset_index(), "起涨推移时间与tau_36组.csv")
    save(pd.DataFrame(combined), "起涨推移时间_相关汇总.csv")


def fit_test_summary():
    original = pd.read_csv(
        SOURCE / "逐场已知推移时间_仅率定kn_20261107_001" / "原三参数_固定均值_逐场时间对照.csv"
    )
    assert len(original) == 80
    assert not original.duplicated(["scope", "geometry", "distance_km", "split"]).any()
    original["channel"] = original.geometry.map(CHANNELS).fillna("")
    metrics = (
        "mean_NSE", "mean_KGE", "mean_RMSE_m3s", "mean_abs_peak_error_percent",
        "mean_abs_peak_time_error_h",
    )
    fields = ["scope", "channel", "distance_km", "split", "events"]
    columns = [f"{metric}_{suffix}" for suffix in ("original", "event_time") for metric in metrics]
    assert np.isfinite(original[columns]).all().all()
    save(original[fields + columns], "原三参数与逐场推移时间_率定测试精度.csv")


def peak_summary():
    data = pd.read_csv(SOURCE / "RWEC_实际滞后入流_峰时分解_20261102_001" / "逐场峰时分解_1080组.csv")
    assert len(data) == 1080 and not data.duplicated([*GROUP, "event_id"]).any()
    assert data.groupby(GROUP).size().eq(30).all() and data.split.eq("test").all()
    total_lag = data.tp_Q_RWEC_h - data.tp_original_inflow_h
    ref_lag = data.tp_Q_ref_h - data.tp_original_inflow_h
    difference = pd.DataFrame({
        "propagation": data.tau_h - ref_lag,
        "storage": total_lag - data.tau_h,
        "final": total_lag - ref_lag,
    })
    np.testing.assert_allclose(difference.propagation + difference.storage, difference.final,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(difference.propagation, data.old_continuous_delta_prop_h,
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(difference.storage, data.old_continuous_delta_stor_h,
                               atol=1e-12, rtol=0)
    rows = []
    for label in ("C1", "C2", "C3", "overall"):
        sub = difference if label == "overall" else difference.loc[data.channel == label]
        row = {"channel": label, "event_section_pairs": len(sub)}
        for name in difference.columns:
            row[f"{name}_mean_h"] = sub[name].mean()
            row[f"{name}_mae_h"] = sub[name].abs().mean()
            row[f"{name}_sd_h"] = sub[name].std(ddof=1)
        row["opposite_sign_count"] = int((sub.propagation * sub.storage < 0).sum())
        row["absolute_error_reduced_count"] = int((sub.final.abs() < sub.propagation.abs()).sum())
        rows.append(row)
    save(pd.DataFrame(rows), "峰时误差分解_tau纯推移口径.csv")


if __name__ == "__main__":
    onset_summary()
    fit_test_summary()
    peak_summary()

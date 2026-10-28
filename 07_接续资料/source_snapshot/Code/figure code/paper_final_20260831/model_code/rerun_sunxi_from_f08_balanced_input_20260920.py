"""Rerun the three models directly from the user-confirmed F08-balanced input.

This script makes no further flow adjustment.  In particular, F08 is not
scaled again: the supplied CSV is copied verbatim as the model input snapshot.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
BASE = HERE / "routing_sunxi_latest_20260901.py"
DATA = Path(
    r"E:\01-HUST\Nature\03\Result\多场率定\数据"
    r"\笋溪河_F08上游洪量配平重跑_20260920\input\9场_F08上游洪量配平.csv"
)
OUT = Path(
    r"E:\01-HUST\Nature\03\Result\多场率定\数据"
    r"\笋溪河_F08上游洪量配平数据直跑_20260920\model_results"
)
USED_INPUT = OUT.parent / "input" / "9场_F08上游洪量配平_去除空行快照.csv"
CAL = ["P08_F01", "P08_F02", "P08_F05", "P08_F06", "P08_F07", "P08_F11", "P08_F12"]
TEST = ["P08_F08", "P08_F09F10"]


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    # The supplied sheet contains nine completely empty trailing rows.  They
    # are not observations and are removed only in this derived read snapshot;
    # the user-supplied CSV is never altered.
    frame = pd.read_csv(DATA, encoding="utf-8-sig", dtype=str)
    required = ["数据用途", "事件编号", "过程类型", "datetime", "上游流量_m3s", "下游流量_m3s"]
    empty = frame[required].isna().all(axis=1)
    if (~empty & frame[required].isna().any(axis=1)).any():
        bad = frame.loc[(~empty) & frame[required].isna().any(axis=1), ["事件编号", "datetime"]]
        raise ValueError(f"Nonblank records with missing fields: {bad.to_dict(orient='records')}")
    USED_INPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[~empty].to_csv(USED_INPUT, index=False, encoding="utf-8-sig")
    spec = importlib.util.spec_from_file_location("sunxi_routing", BASE)
    model = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(model)
    model.DATA = USED_INPUT
    model.OUT = OUT
    model.CAL_EVENTS = CAL
    model.TEST_EVENTS = TEST
    model.NONLINEAR_X_MAX = 0.30
    model.run(only_nonlinear=False)


if __name__ == "__main__":
    main()

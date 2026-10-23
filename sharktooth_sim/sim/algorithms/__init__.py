"""演算法註冊表：run(name, tP, tC, **params) → dict(ts, dhat, valid, ...)。"""
from __future__ import annotations

import time

import numpy as np

from . import baselines, sharktooth

REGISTRY = {
    "tp_only": baselines.tp_only,
    "tc_only": baselines.tc_only,
    "ols_raw": baselines.ols_raw,
    "ols_window": baselines.ols_window,
    "ols_trimmed": baselines.ols_trimmed,
    "min_filter": baselines.min_filter,
    "lower_hull": baselines.lower_hull,
    "sharktooth": sharktooth.run,
    "sharktooth_keep": lambda tP, tC, **kw: sharktooth.run(tP, tC, **{**kw, "keep_last": True}),
    "sharktooth_ff": lambda tP, tC, **kw: sharktooth.run(tP, tC, **{**kw, "keep_last": True,
                                                                     "forget": kw.get("forget_ff", 0.9999)}),
}


def run(name, tP, tC, timing=False, **params):
    f = REGISTRY[name]
    tP = np.ascontiguousarray(tP, float)
    tC = np.ascontiguousarray(tC, float)
    t0 = time.perf_counter()
    out = f(tP, tC, **params)
    if timing:
        out["us_per_pkt"] = (time.perf_counter() - t0) / max(len(tP), 1) * 1e6
    return out

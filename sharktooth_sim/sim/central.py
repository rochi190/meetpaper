"""
C5 / C6：每支 dongle（central）的多連線排程。

  - 同一支 dongle 的連線共用 dongle 時鐘：CI_d = CI · (1 + ρ_dongle)（真實時間）
  - 連線 n 的 anchor 在 CI 內錯開 n · event_len（event_len 由封包長度決定：244 B → 7.5 ms【原文 §IV-A】）
  - 若 Σ event_len > CI，事件重疊：被重疊的 event 依比例被略過（SoftDevice「rescheduled / dropped」）
  - 另外每個 event 以 skip_prob 被略過（掃描、其他角色、時鐘漂移造成的碰撞）
"""
from __future__ import annotations

import numpy as np

from .config import draw


def event_len(bytes_):
    """event 長度（1.25 ms 的倍數）：244 B → 7.5 ms，127 B → 5 ms，17 B → 2.5 ms。"""
    raw = 7.5e-3 * (bytes_ + 30) / (244 + 30)
    return max(2.5e-3, np.ceil(raw / 1.25e-3 - 1e-9) * 1.25e-3)


def assign_dongles(n_nodes, per_dongle):
    return [list(range(i, min(i + per_dongle, n_nodes))) for i in range(0, n_nodes, per_dongle)]


def schedule(cfg, rng):
    """回傳 per-node 排程 dict(anchor, ci, skip, dongle, slot) 與 dongle 列表。"""
    T = cfg["T"]
    CI = cfg["CI"]
    groups = assign_dongles(cfg["n_nodes"], cfg["nodes_per_dongle"])
    elen = event_len(cfg["packet"]["bytes"])
    out = {}
    dongles = []
    for d, members in enumerate(groups):
        rho_d = float(draw(cfg["clock_dongle"]["rho_ppm"], rng)) * 1e-6
        ci_d = CI * (1 + rho_d)
        a0 = rng.uniform(0, CI)
        n_ev = int((T + 5.0) / ci_d) + 10
        overlap = max(0.0, len(members) * elen - CI) / CI
        sc = cfg["sched"]
        p_skip = 0.0
        if sc["enabled"]:
            p_skip = sc["skip_prob"] + sc["skip_prob_per_extra_node"] * (len(members) - 1) + overlap
        for slot, n in enumerate(members):
            anchor = a0 + slot * elen if sc["enabled"] else rng.uniform(0, CI)
            skip = rng.random(n_ev) < min(p_skip, 0.95)
            out[n] = dict(anchor=anchor, ci=ci_d, skip=skip, dongle=d, slot=slot, event_len=elen)
        dongles.append(dict(id=d, members=members, rho_ppm=rho_d * 1e6, ci=ci_d, p_skip=p_skip))
    return out, dongles

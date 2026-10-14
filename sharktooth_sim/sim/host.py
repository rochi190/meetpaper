"""
D、E 類：dongle → USB → PC 作業系統 → Python callback → 打 t_C。

  D1 dongle 韌體處理（常數 + 抖動）
  D2 序列化佇列：同一支 dongle 的所有連線共用一條通道、依序處理（單一伺服器 FIFO）
  D3 USB full-speed：每 1 ms 一個 frame，資料在下一個 frame 才被 PC 收到
  E1 OS 排程延遲（lognormal）；E3 尺度隨節點數放大（per_node_scale）
  E2 共同模式停頓（GC / GIL / 排程）：Poisson 發生，期間所有 callback 都被擋住
  E3 GIL：所有 dongle 的 callback 共用一個直譯器（全域 FIFO 伺服器）
  E4 t_C 的時鐘解析度
  E5 背景負載：週期性停頓
每支 dongle 的 callback 是 FIFO（後到的封包不會比先到的早被處理）。
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def fifo_server(t_arr, service):
    """單一伺服器 FIFO：回傳每個工作的「開始服務時間」。t_arr 必須已排序。"""
    n = t_arr.size
    start = np.empty(n)
    free = -1e30
    for i in range(n):
        s = t_arr[i] if t_arr[i] > free else free
        start[i] = s
        free = s + service[i]
    return start


@njit(cache=True)
def apply_pauses(t, p_start, p_end, extra):
    """t 落在停頓 [p_start, p_end) 中 → 延到停頓結束。p_* 已排序。"""
    out = t.copy()
    j = 0
    o = np.argsort(t)
    for ii in range(t.size):
        i = o[ii]
        while j < p_start.size and p_end[j] <= t[i]:
            j += 1
        if j < p_start.size and p_start[j] <= t[i] < p_end[j]:
            out[i] = p_end[j] + extra[i]
    return out


def make_pauses(cfg, T, rng):
    ps, pe = [], []
    pc = cfg["pauses"]
    if pc["enabled"] and pc["rate_hz"] > 0:
        n = rng.poisson(pc["rate_hz"] * (T + 10))
        s = np.sort(rng.uniform(0, T + 10, n))
        d = rng.uniform(pc["min_s"], pc["max_s"], n)
        ps.append(s)
        pe.append(s + d)
    bg = cfg["background"]
    if bg["enabled"]:
        s = np.arange(rng.uniform(0, bg["period_s"]), T + 10, bg["period_s"])
        ps.append(s)
        pe.append(s + bg["pause_s"])
    if not ps:
        return np.zeros(0), np.zeros(0)
    s = np.concatenate(ps)
    e = np.concatenate(pe)
    if s.size == 0:
        return np.zeros(0), np.zeros(0)
    o = np.argsort(s)
    s, e = s[o], e[o]
    # 合併重疊
    ms, me = [s[0]], [e[0]]
    for a, b in zip(s[1:], e[1:]):
        if a <= me[-1]:
            me[-1] = max(me[-1], b)
        else:
            ms.append(a)
            me.append(b)
    return np.array(ms), np.array(me)


def host_pipeline(cfg, recv, dongles, pc_clock, rng):
    """recv：dict(node, dongle, t_air, bytes)（所有節點、所有已送達 dongle 的封包）。
    回傳 t_C（PC 時鐘讀數）與中間時間（真實時間），順序與輸入相同。"""
    T = cfg["T"]
    n = recv["t_air"].size
    L = cfg["packet"]["bytes"]
    t = recv["t_air"].copy()
    dg = recv["dongle"]
    stages = {}
    # D1
    if cfg["dongle"]["enabled"]:
        base = np.full(n, cfg["dongle"]["base_s"])
        sd = cfg["dongle"].get("base_sd_s", 0.0)
        if sd > 0:          # D 類：不同 dongle（USB hub / 埠）的固定路徑差
            for d in np.unique(dg):
                base[dg == d] += abs(rng.normal(0, sd))
        t = t + base + rng.uniform(0, cfg["dongle"]["jitter_s"], n)
    stages["dongle"] = t.copy()
    # D2：每支 dongle 一條序列化通道
    if cfg["serial"]["enabled"]:
        svc_one = cfg["serial"]["base_s"] + cfg["serial"]["per_byte_s"] * (L + 16)
        for d in np.unique(dg):
            idx = np.flatnonzero(dg == d)
            o = idx[np.argsort(t[idx], kind="stable")]
            st = fifo_server(t[o], np.full(o.size, svc_one))
            t[o] = st + svc_one
    stages["serial"] = t.copy()
    # D3：USB frame（每支 dongle 一個 SOF 相位）
    if cfg["usb"]["enabled"]:
        fr = cfg["usb"]["frame_s"]
        for d in np.unique(dg):
            m = dg == d
            ph = rng.uniform(0, fr)
            t[m] = np.ceil((t[m] - ph) / fr) * fr + ph
    stages["usb"] = t.copy()
    # E1 + E3（尺度）
    if cfg["os"]["enabled"]:
        med = cfg["os"]["median_s"] * (1 + cfg["os"]["per_node_scale"] * (cfg["n_nodes"] - 1))
        t = t + rng.lognormal(np.log(med), cfg["os"]["sigma"], n)
    # 獨立尖峰
    if cfg["indep_spikes"]["enabled"] and cfg["indep_spikes"]["prob"] > 0:
        sp = rng.random(n) < cfg["indep_spikes"]["prob"]
        t = t + sp * rng.uniform(cfg["indep_spikes"]["min_s"], cfg["indep_spikes"]["max_s"], n)
    # E2 / E5 共同模式停頓
    ps, pe = make_pauses(cfg, T, rng)
    if ps.size:
        t = apply_pauses(t, ps, pe, rng.uniform(0, 0.2e-3, n))
    # 每支 dongle 的 callback 執行緒是 FIFO
    for d in np.unique(dg):
        idx = np.flatnonzero(dg == d)
        o = idx[np.argsort(recv["t_air"][idx], kind="stable")]
        t[o] = np.maximum.accumulate(t[o])
    # E3：GIL（全域單一伺服器）
    if cfg["gil"]["enabled"]:
        o = np.argsort(t, kind="stable")
        t[o] = fifo_server(t[o], np.full(n, cfg["gil"]["service_s"]))
    stages["callback"] = t.copy()
    tC = pc_clock.read(t, resolution=cfg["clock_pc"]["tc_resolution"])
    return tC, t, stages, (ps, pe)

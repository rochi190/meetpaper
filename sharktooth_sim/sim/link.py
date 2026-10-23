"""
C 類：BLE 連線層（每個節點一條連線）。

  C1  connection event：anchor + j·CI（CI 以 dongle 時鐘計）；封包要等到下一個 event
  C3  每次嘗試的錯誤率：Gilbert–Elliott 兩態（好 / 壞）；失敗 → 該 event 結束，下一個 event 重傳
  C4  每個 event 最多送幾包 + 韌體流量控制：stack 只有 `window` 個 notification 緩衝，
      送出後要等 ack（下一個 event）回來、callback 再放下一包 → 背壓下每 (1 + ack_delay) 個 event 才能送一包。
      【T0 發現】Fig.6 起始與 Fig.4b 尖峰後的 δT 都以約 −40 ms/包 遞減 = 100.7 − 2×30 ms，正是 window=1、ack_delay=1。
  C5  central 排程略過 event（由 central.py 給 skip 遮罩）
  C7  跳頻：部分頻道受 Wi-Fi 干擾（選配）
  C9  斷線：佇列清空、斷線期間資料遺失、重連後 anchor 改變
  C10 TX 佇列滿 → 丟包
  C11 起始：notification 啟用前（t_enable）不送
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def gilbert_elliott(n, p_g2b, p_b2g, u):
    state = np.zeros(n, np.bool_)
    s = False
    for j in range(n):
        if s:
            s = not (u[j] < p_b2g)
        else:
            s = u[j] < p_g2b
        state[j] = s
    return state


@njit(cache=True)
def link_kernel(t_ready, anchor, ci, n_ev_max, t_end, skip, per_ev, u, q_cap, window, ack_delay,
                max_per_ev, disc_s, disc_e, disc_phase, t_enable, airtime, pre_drop):
    n = t_ready.size
    t_air = np.full(n, np.nan)
    attempts = np.zeros(n, np.int64)
    drop = np.zeros(n, np.int64)          # 0 ok / 1 佇列滿 / 2 斷線 / 3 結束時仍在佇列 / 4 其他遺失
    q = np.empty(n, np.int64)
    qh = 0
    qt = 0
    free_from = np.zeros(window, np.int64)
    p = 0
    base = anchor
    jl = 0
    k = 0
    g = 0
    nd = disc_s.size
    while g < n_ev_max:
        te = base + jl * ci
        if te > t_end:
            break
        if k < nd and te >= disc_s[k]:
            while p < n and t_ready[p] < disc_s[k]:
                if pre_drop[p]:
                    drop[p] = 4
                elif qt - qh >= q_cap:
                    drop[p] = 1
                else:
                    q[qt] = p
                    qt += 1
                p += 1
            while qh < qt:
                drop[q[qh]] = 2
                qh += 1
            while p < n and t_ready[p] < disc_e[k]:
                drop[p] = 2
                p += 1
            base = disc_e[k] + disc_phase[k]
            jl = 0
            k += 1
            for b in range(window):
                free_from[b] = 0
            continue
        while p < n and t_ready[p] <= te:
            if pre_drop[p]:
                drop[p] = 4
            elif qt - qh >= q_cap:
                drop[p] = 1
            else:
                q[qt] = p
                qt += 1
            p += 1
        if te >= t_enable and not skip[g]:
            sent = 0
            while qh < qt and sent < max_per_ev:
                slot = -1
                for b in range(window):
                    if free_from[b] <= g:
                        slot = b
                        break
                if slot < 0:
                    break
                h = q[qh]
                attempts[h] += 1
                if u[g, sent] < per_ev[g]:
                    break                       # CRC 錯 / 沒 ack → event 結束，下個 event 重傳
                t_air[h] = te + (sent + 1) * airtime
                qh += 1
                free_from[slot] = g + 1 + ack_delay
                sent += 1
        jl += 1
        g += 1
    while qh < qt:
        drop[q[qh]] = 3
        qh += 1
    while p < n:
        drop[p] = 3
        p += 1
    return t_air, attempts, drop


def length_scaled_per(per_ref, bytes_, ref_bytes=244, overhead=10):
    """PER 隨封包長度：1 − (1 − PER_ref)^((L+overhead)/(ref+overhead))。"""
    e = (bytes_ + overhead) / (ref_bytes + overhead)
    return 1.0 - (1.0 - per_ref) ** e


def airtime(bytes_):
    """1M PHY：前導 + AA + header + MIC/CRC ≈ 14 B，8 µs/B。"""
    return (bytes_ + 14) * 8e-6


def per_events(cfg, n_ev, n_dongles, rng, node_id=None):
    """每個 event 的嘗試錯誤率（已含封包長度、piconet 數、跳頻）。"""
    pc = cfg["per"]
    L = cfg["packet"]["bytes"]
    if not pc["enabled"]:
        return np.zeros(n_ev), np.zeros(n_ev, bool)
    good = pc["good"] + cfg["piconet_per_add"] * max(n_dongles - 1, 0)
    ov = pc.get("node_good_add") or {}
    if node_id is not None and (node_id in ov or str(node_id) in ov):
        good += float(ov.get(node_id, ov.get(str(node_id))))     # T8：某個節點連線比較差
    bad = pc["bad"]
    if pc.get("length_scaling", True):
        good = length_scaled_per(min(good, 0.999), L, pc["ref_bytes"])
        bad = length_scaled_per(min(bad, 0.999), L, pc["ref_bytes"])
    p_b2g = 1.0 / max(pc["mean_bad_events"], 1.0)
    state = gilbert_elliott(n_ev, pc["p_g2b"], p_b2g, rng.random(n_ev))
    per = np.where(state, bad, good)
    hop = cfg["hop"]
    if hop["enabled"]:
        hop_inc = rng.integers(5, 17)
        ch = (rng.integers(0, 37) + hop_inc * np.arange(n_ev)) % 37
        bad_ch = rng.permutation(37)[: hop["n_bad"]]
        pch = np.where(np.isin(ch, bad_ch), hop["per_bad_ch"], 0.0)
        per = 1 - (1 - per) * (1 - pch)
    return per, state


def simulate_link(cfg, node, sched, n_dongles, rng):
    """一個節點的連線。sched：central.py 給的 dict(anchor, ci, skip)。"""
    T = cfg["T"]
    ci = sched["ci"]
    n_ev = int((T + 5.0) / ci) + 10
    per, state = per_events(cfg, n_ev, n_dongles, rng, node["node_id"])
    fl = cfg["flow"]
    u = rng.random((n_ev, max(fl["max_pkts_per_event"], 1)))
    skip = sched["skip"][:n_ev] if sched["skip"].size >= n_ev else np.r_[
        sched["skip"], np.zeros(n_ev - sched["skip"].size, bool)]
    # 斷線
    ds, de = [], []
    if cfg["disconnect"]["enabled"]:
        for ev in cfg["disconnect"]["events"]:
            if len(ev) >= 3 and int(ev[2]) != node["node_id"]:
                continue
            ds.append(float(ev[0]))
            de.append(float(ev[0]) + float(ev[1]) + cfg["disconnect"]["reconnect_s"])
    o = np.argsort(ds)
    ds = np.array(ds, float)[o]
    de = np.array(de, float)[o]
    phase = rng.uniform(0, ci, ds.size)
    t_enable = -np.inf
    if cfg["startup"]["enabled"]:
        t_enable = node["t_ready"][0] + cfg["startup"]["delay_s"]
    pre_drop = np.zeros(node["t_ready"].size, np.bool_)
    if cfg.get("loss", {}).get("enabled"):
        pre_drop = rng.random(node["t_ready"].size) < cfg["loss"]["prob"]
    t_air, att, drop = link_kernel(node["t_ready"], sched["anchor"], ci, n_ev, T + 5.0, skip, per, u,
                                   int(cfg["tx_queue"]), int(fl["window"]), int(fl["ack_delay_events"]),
                                   int(fl["max_pkts_per_event"]), ds, de, phase, t_enable,
                                   airtime(cfg["packet"]["bytes"]), pre_drop)
    return dict(t_air=t_air, attempts=att, drop=drop, bad_frac=float(state.mean()))

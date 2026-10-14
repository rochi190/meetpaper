"""
§7 完整性檢查（T7 用，SharkTooth 前處理）。

每收到一包（依到達順序）：
  1. 封包編號（1 B，mod 256）：與上一個可信封包差 g → g − 1 包遺失；g = 0 且 t_P 相同 → 重複，丟掉
  2./3. 時間銜接：Δt_P 應 ≈ g · P̂（P̂ = 每包樣本數 × 樣本週期，樣本週期由「首末樣本時間差」欄位估計）；
        差異 < 0.5/f_s → 可信；小幅不符（約 ±k/f_s）→ buffer 讀錯，該包不進 SharkTooth
  4. t_P 倒退或大跳：若剛好差一個欄位週期 → 溢位，unwrap；否則 → 重開機，SharkTooth 狀態重置
  5. 同一 t_P 重複 → 丟棄後者（避免 Eq.9 的 ∞×0）
不可信 / 重複的封包仍然輸出（用目前這段的 α、β），只是不拿來更新 SharkTooth。
"""
from __future__ import annotations

import numpy as np

from . import sharktooth


def classify(pkt_id, tP, first_last_dt, n_samples, fs, wrap_span=None, tol_frac=0.5):
    n = tP.size
    status = np.zeros(n, np.int8)       # 0 可信 / 1 buffer 錯 / 2 重複 / 3 重置點（可信，新段開始）
    seg = np.zeros(n, np.int64)
    tpu = tP.astype(float).copy()
    offset = 0.0
    Ts = np.median(first_last_dt[n_samples > 1] / (n_samples[n_samples > 1] - 1))
    N = int(np.median(n_samples))
    Pk = N * Ts
    tol = tol_frac / fs
    last = 0
    cur = 0
    lost = 0
    n_reset = 0
    n_wrap = 0
    recent = {(int(pkt_id[0]), round(float(tP[0]), 6)): 0}
    for i in range(1, n):
        key = (int(pkt_id[i]), round(float(tP[i]), 6))
        if key in recent and i - recent[key] < 64:
            status[i] = 2                     # 重複（不一定緊接在原封包後面到）
            seg[i] = cur
            tpu[i] = tpu[recent[key]]
            continue
        recent[key] = i
        x = tP[i] + offset
        dtp = x - tpu[last]
        g = (int(pkt_id[i]) - int(pkt_id[last])) % 256
        if wrap_span and dtp < -wrap_span / 2 and abs(dtp + wrap_span - max(g, 1) * Pk) < 1.0 + 256 * Pk:
            offset += wrap_span
            x += wrap_span
            dtp += wrap_span
            n_wrap += 1
        tpu[i] = x
        if g == 0 and abs(dtp) < 1e-9:
            status[i] = 2
            seg[i] = cur
            continue
        gg = max(int(round(dtp / Pk)), 1) if dtp > 0 else g
        if dtp < -1.0 or dtp > 30.0 + g * Pk or (g != gg % 256 and abs(dtp - gg * Pk) > tol):
            # 重開機（或無法解釋的大跳）：新段
            cur += 1
            n_reset += 1
            status[i] = 3
            seg[i] = cur
            last = i
            continue
        if abs(dtp - gg * Pk) < tol:
            last = i
            seg[i] = cur
        else:
            # buffer 讀錯：t_P 偏了約整數個樣本 → 修正後輸出，但不拿來更新 SharkTooth
            status[i] = 1
            seg[i] = cur
            tpu[i] = x - np.round((dtp - gg * Pk) * fs) / fs
    # 遺失：相鄰「收到的」封包編號差 − 1（不含重複、不跨重置點）
    nd = np.flatnonzero(status != 2)
    d_id = (np.diff(pkt_id[nd].astype(np.int64)) % 256) - 1
    ok = (status[nd][1:] != 3) & (d_id < 128)          # 跨重置點、或亂序（差 ≥ 128 視為倒退）不算遺失
    lost = int(np.sum(d_id[ok]))
    return dict(status=status, seg=seg, tP_unwrapped=tpu, lost=lost, n_reset=n_reset, n_wrap=n_wrap,
                n_buf=int((status == 1).sum()), n_dup=int((status == 2).sum()))


def run(p, fs, CI=0.030, wrap_span=None, **st_kw):
    """對已收到的封包 p（received(...) 的輸出）做完整性檢查後跑 SharkTooth。"""
    c = classify(p["pkt_id"], p["t_P"], p["first_last_dt"], p["n_samples"], fs, wrap_span)
    n = p["t_P"].size
    tpu = c["tP_unwrapped"]
    alpha = np.zeros(n)
    beta = np.zeros(n)
    valid = np.zeros(n, bool)
    n_pivot = 0
    for s in np.unique(c["seg"]):
        idx = np.flatnonzero(c["seg"] == s)
        good = idx[(c["status"][idx] == 0) | (c["status"][idx] == 3)]
        if good.size < 2:
            continue
        r = sharktooth.run(tpu[good], p["t_C"][good], CI=CI, **st_kw)
        n_pivot += r["n_pivot"]
        # 每個封包用「到它為止最後一個可信封包」的 α、β
        pos = np.searchsorted(good, idx, side="right") - 1
        ok = pos >= 0
        alpha[idx[ok]] = r["alpha"][pos[ok]]
        beta[idx[ok]] = r["beta"][pos[ok]]
        valid[idx[ok]] = r["valid"][pos[ok]]
    # t̂_S 以「原始」t_P 的時鐘為準：t̂_S = t_P,u + α t_P,u + β（t_P,u 已 unwrap）
    ts = tpu + alpha * tpu + beta
    return dict(ts=ts, valid=valid, alpha=alpha, beta=beta, n_pivot=n_pivot, info=c)

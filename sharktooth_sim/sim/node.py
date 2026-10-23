"""
B 類：節點取樣、打時間戳、讀 buffer、打包、韌體延遲、重開機。

取樣由 ADC 計時器驅動（clock_adc：其週期相對真實時間有 period_err_ppm 的誤差；
開了節點溫度時，誤差再加上 temp_coeff_ppm_per_C × (T − T_start)，RC 振盪器隨板子溫度變），
時間戳 t_P 由 32.768 kHz 晶振時鐘（clock_node）讀出。
"""
from __future__ import annotations

import numpy as np

from .clocks import draw_temp, node_clock
from .config import draw


def adc_sample_times(start, T, fs, adc_err, tc_ppm, temperature):
    """ADC 計時器的取樣時刻。週期誤差 e(t) = adc_err + tc·(T(t) − T(start))；溫度不變時退化成等間隔。"""
    Ts = (1.0 / fs) * (1.0 + adc_err)
    if not tc_ppm or temperature is None:
        n = int((T - start) / Ts)
        return start + Ts * (np.arange(n) + 1)      # 第一筆在開始後一個週期
    g, Tc = temperature
    m = g >= start
    g = np.r_[start, g[m]]
    Tc = np.r_[np.interp(start, *temperature), Tc[m]]
    rate = fs / (1.0 + adc_err + tc_ppm * 1e-6 * (Tc - Tc[0]))
    phase = np.r_[0.0, np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(g))]
    n = int(np.interp(T, g, phase))
    return np.interp(np.arange(1, n + 1), phase, g)


def make_node(cfg, node_id, rng):
    T = cfg["T"]
    pk = cfg["packet"]
    fs = pk["fs"]
    N = int(round(fs * pk["period_nom"]))            # 每包樣本數（50 Hz × 0.1 s = 5）
    cn = dict(cfg["clock_node"])
    if cn.get("temp") and cn["temp"].get("enabled"):
        cn["temp"] = draw_temp(cn["temp"], rng)
    # rho_list / period_err_list：每個節點指定值（T11 用；組合誤差在實驗端先算好）
    rho = float(cn["rho_list"][node_id]) if cn.get("rho_list") is not None else float(draw(cn["rho_ppm"], rng))
    start = float(draw(cn["start_s"], rng))
    clk = node_clock(cn, T, rho, start, rng)
    ca = cfg["clock_adc"]
    adc_err = (float(ca["period_err_list"][node_id]) if ca.get("period_err_list") is not None
               else float(draw(ca["period_err_ppm"], rng))) * 1e-6
    Ts = (1.0 / fs) * (1.0 + adc_err)                # 真實取樣週期（開始時）
    tc = ca.get("temp_coeff_ppm_per_C")
    tc = float(draw(tc, rng)) if (tc is not None and getattr(clk, "temperature", None) is not None) else 0.0
    t_samp = adc_sample_times(start, T, fs, adc_err, tc, getattr(clk, "temperature", None))
    n_samp = t_samp.size

    # B5 重開機：重開機期間沒有樣本；之後節點時鐘從 0 重來
    epoch = np.zeros(n_samp, dtype=np.int64)
    reset_t = []
    if cfg["reset"]["enabled"]:
        for ev in cfg["reset"]["events"]:
            tr = float(ev[0] if isinstance(ev, (list, tuple)) else ev)
            reset_t.append(tr)
        reset_t.sort()
        keep = np.ones(n_samp, bool)
        for k, tr in enumerate(reset_t):
            keep &= ~((t_samp >= tr) & (t_samp < tr + cfg["reset"]["reboot_s"]))
            epoch[t_samp >= tr + cfg["reset"]["reboot_s"]] = k + 1
        t_samp, epoch = t_samp[keep], epoch[keep]

    # B1：ADC 中斷延遲 → 打時間戳的時刻晚於實際取樣
    isr = rng.uniform(0, cfg["isr_jitter"]["max_s"], t_samp.size) if cfg["isr_jitter"]["enabled"] else 0.0
    tp_samp = clk.read(t_samp + isr)
    if reset_t:
        for k, tr in enumerate(reset_t):
            m = epoch == k + 1
            tp_samp[m] = clk.read(t_samp[m] + (isr[m] if np.ndim(isr) else 0.0)) - clk.read(
                tr + cfg["reset"]["reboot_s"])
            tp_samp[m] = np.floor(tp_samp[m] * 32768) / 32768

    # B4 打包
    if pk["mode"] == "count":
        # 每 N 筆一包（重開機邊界重新起算）
        last_idx = []
        for e in np.unique(epoch):
            idx = np.flatnonzero(epoch == e)
            last_idx.extend(idx[N - 1::N].tolist())
        last_idx = np.array(last_idx, dtype=np.int64)
        first_idx = last_idx - (N - 1)
        t_pack = t_samp[last_idx]                     # 最後一筆取樣完成即打包
    elif pk["mode"] == "timer":
        # 節點計時器（晶振）每 period_nom 觸發，打包觸發前所有樣本
        clk_read_exact = clk.exact(t_samp)
        k = np.floor(clk_read_exact / pk["period_nom"]).astype(np.int64)
        boundaries = np.flatnonzero(np.diff(k) > 0)
        last_idx = boundaries
        first_idx = np.r_[0, boundaries[:-1] + 1]
        t_pack = clk.inverse((k[last_idx] + 1) * pk["period_nom"])
    else:
        raise ValueError(pk["mode"])
    n_pkt = last_idx.size
    n_samples = last_idx - first_idx + 1

    # B3：讀 buffer 少抓/多抓 → t_P 對到錯的樣本
    buf_err = np.zeros(n_pkt, bool)
    tp_idx = last_idx.copy()
    if cfg["buf_err"]["enabled"] and cfg["buf_err"]["prob"] > 0:
        buf_err = rng.random(n_pkt) < cfg["buf_err"]["prob"]
        sh = rng.choice([-1, 1], n_pkt) * cfg["buf_err"]["k"]
        tp_idx = np.clip(last_idx + buf_err * sh, 0, t_samp.size - 1)
    tP = tp_samp[tp_idx]

    # B2：RTOS 打包 / 放進 TX 佇列的延遲
    t_ready = t_pack.copy()
    if cfg["fw_delay"]["enabled"]:
        mx = cfg["fw_delay"]["max_s"]
        ov = cfg["fw_delay"].get("node_max_override") or {}
        if node_id in ov or str(node_id) in ov:
            mx = float(ov.get(node_id, ov.get(str(node_id))))            # T8：某個節點韌體比較忙
        t_ready += cfg["fw_delay"]["base_s"] + rng.uniform(0, mx, n_pkt)

    pkt_id = np.zeros(n_pkt, dtype=np.int64)
    pe = epoch[last_idx]
    for e in np.unique(pe):
        m = pe == e
        pkt_id[m] = np.arange(m.sum())

    return dict(
        node_id=node_id, clock=clk, temp=cn.get("temp"), rho_ppm=rho, start_s=start, adc_err_ppm=adc_err * 1e6, adc_tc_ppm=tc, Ts=Ts, N=N,
        t_samp=t_samp, tp_samp=tp_samp, epoch=epoch,
        first_idx=first_idx, last_idx=last_idx, n_samples=n_samples,
        tP=tP, t_ready=t_ready, t_true_last=t_samp[last_idx], pkt_id=pkt_id, buf_err=buf_err,
        reset_t=reset_t,
    )


def signal_values(node, cfg, rng):
    """F：5 Hz 參考正弦（示波器）經 AFE（每節點固定群延遲）→ ADC（加雜訊）。"""
    s = cfg["signal"]
    d_afe = rng.normal(0, s["afe_delay_sd_s"]) if s["afe_delay_sd_s"] > 0 else 0.0
    v = s["amp"] * np.sin(2 * np.pi * s["f"] * (node["t_samp"] - d_afe))
    if s["noise_v"] > 0:
        v = v + rng.normal(0, s["noise_v"], v.size)
    node["afe_delay_s"] = d_afe
    return v

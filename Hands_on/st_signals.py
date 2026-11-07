"""
SharkTooth 實測：每一個訊號都畫一張結果圖。

st_analyze.py 只畫論文 Fig. 6、Fig. 9；這支把算法裡每一個中間訊號、每一塊板子的 ADC 訊號、
每一對板子的 RSE、連線品質都畫出來，並寫一份數字摘要。

用法：
    python st_signals.py runs/20261101_220025 --ci 0.030
    python st_signals.py runs/20261101_220025 --ci 0.030 --paper       # 完全照論文的算法（輸出到 signals_paper/）
    python st_signals.py runs/20261101_220025 --ci 0.030 --gt cc       # RSE 的真實延遲用論文的 10 kHz 互相關

輸出：<資料夾>/signals/（--paper 時是 signals_paper/，非預設 --tilt 時是 signals_tilt<數字>/）

  每塊板子（<node> = NODE_A …）
    <node>_1_timestamp_model.png   論文 Fig. 3：δT 與線性趨勢、放大看一個 CI 內的週期樣式、σ100 對 CI/2
    <node>_2_algorithm_signals.png 算法每個中間訊號（左：開頭 140 s，右：整段）：
                                   δT / δT* / δT_thr / δ̂T（扣掉最後的直線）、δ²T、σ_M、λ0 與 λ、γ0 與 γ1、
                                   α(t)、β(t)、合格點比例與樞紐點
    <node>_3_delay.png             延遲 ε = δT − δ̂T 的分布、到達間隔 Δt_C、封包間隔 Δt_P、每分鐘遲到比例
    <node>_4_waveform.png          ADC 訊號：一秒的樣本與正弦擬合、振幅、DC、SINAD、頻譜、
                                   這塊板子自己的同步殘差（真實延遲 − δ̂T，扣掉常數）

  整次錄音
    pairs_1_rse.png                論文 Fig. 10：每一對節點的 RSE（Eq. 26），600 s 分段、±1 ms 帶
    pairs_2_segments.png           每段每對的 |RSE| 箱形圖；論文 Fig. 12 式 P95|RSE| 對 σ²100
    sync_overlay.png               論文 Fig. 9 在開頭、中段、結尾三個時間點（看收斂）
    link_quality.png               每秒封包數、PC 事件迴圈延遲、累計掉包、每分鐘遲到比例
    summary.md                     上面所有圖的數字摘要

圖的文字用英文（和論文一樣，也避免缺中文字型）。
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from st_analyze import F_REF, LSE_HZ, load, paper_style, run_sharktooth, sample_times, save  # noqa: E402
from st_table2 import N_SEG, SEG_S, epoch_true_offset, rse_epochs  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

C_DOT, C_STAR, C_THR, C_HAT, C_PIV = "#0072BD", "red", "#E07A00", "#7B2CBF", "#00AA00"
PERIOD = 1.0 / F_REF


def wrap(x):
    """弦波每 200 ms 重複：把延遲差換到 ±100 ms 之內。"""
    return (x + PERIOD / 2) % PERIOD - PERIOD / 2


def final_line(d):
    """錄完時的 δ̂T = α·t_P + β；沒有有效輸出（--paper 失敗時）就用 δT 的下緣直線擬合代替。"""
    if d.get("ab_valid"):
        return d["alpha_ppm"] * 1e-6, d["beta_s"]
    tP, dT = d["tP"], d["dT"]
    k = np.polyfit(tP, dT, 1)
    r = dT - np.polyval(k, tP)
    m = r <= np.percentile(r, 5)
    k = np.polyfit(tP[m], dT[m], 1)
    return k[0], k[1]


def rolling_mean(x, n):
    x = np.asarray(x, float)
    c = np.cumsum(np.r_[0.0, np.nan_to_num(x)])
    out = np.full(x.size, np.nan)
    if x.size >= n:
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def lims(y, lo_p=0.5, hi_p=99.5, pad=0.12, min_span=None):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return -1, 1
    lo, hi = np.percentile(y, lo_p), np.percentile(y, hi_p)
    if min_span and hi - lo < min_span:
        c = (hi + lo) / 2
        lo, hi = c - min_span / 2, c + min_span / 2
    s = hi - lo if hi > lo else 1.0
    return lo - pad * s, hi + pad * s


# =========================================================================== 每塊板子
def fig_timestamp_model(d, name, base):
    """論文 Fig. 3：(a) δT 與線性趨勢 (b) 放大一段，看一個 CI 內的週期樣式 (c) σ100 對 CI/2。"""
    tP, dT, CI = d["tP"], d["dT"], d["CI"]
    a, b = final_line(d)
    eps = (dT - (a * tP + b)) * 1e3
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(3.5, 6.4))
    a1.plot(tP, dT, ".", ms=1.0, color=C_DOT, label=r"$\delta T$")
    a1.plot(tP[[0, -1]], a * tP[[0, -1]] + b, "--", color=C_HAT, lw=1.2, label=r"$\alpha t_P + \beta$")
    zw, z0 = 3.0, tP[0] + 0.5 * (tP[-1] - tP[0])
    m = (tP >= z0) & (tP <= z0 + zw)
    y0, y1 = dT[m].min(), dT[m].max()
    bw = max(zw, 0.012 * (tP[-1] - tP[0]))                 # 整段很長時框會太窄，至少畫 1.2 % 寬
    a1.add_patch(Rectangle((z0 + zw / 2 - bw / 2, y0), bw, y1 - y0, fill=False, ec="red", lw=1.2, zorder=5))
    a1.set_ylim(*lims(dT, 0.1, 99.9))
    a1.ticklabel_format(axis="y", useOffset=False)
    a1.set_xlim(tP[0], tP[-1])
    a1.set_ylabel(r"$\delta T$ (s)")
    a1.set_title(f"(a) Timestamp model ({name})")
    a1.legend(loc="best")
    a2.plot(tP[m], eps[m], "-", color=C_DOT, lw=0.8)
    a2.plot(tP[m], eps[m], "*", color="#FF8C00", ms=5)
    a2.axhline(0, color="gray", lw=0.6)
    a2.axhline(CI * 1e3, color="gray", lw=0.6, ls="--")
    a2.set_ylim(-0.1 * CI * 1e3, max(1.25 * CI * 1e3, eps[m].max() * 1.08))
    a2.text(tP[m][-1], CI * 1e3, f"1 CI = {CI * 1e3:.0f} ms ", va="bottom", ha="right", fontsize=7, color="gray")
    a2.set_ylabel(r"$\delta T - (\alpha t_P + \beta)$ (ms)")
    a2.set_title(f"(b) Zoom: {zw:.0f} s from $t_P$ = {z0:.0f} s (stars = packets)", fontsize=9)
    sig = d["sigma"] * 1e3
    a3.plot(tP, sig, color=C_DOT, lw=0.8, label=r"$\sigma_{100}$")
    a3.axhline(CI / 2 * 1e3, color="red", ls="--", lw=1.0, label="CI / 2")
    ok = np.arange(sig.size) >= 100
    a3.set_ylim(*lims(sig[ok], 0.5, 99.9, min_span=4))
    a3.set_xlim(tP[0], tP[-1])
    a3.set_xlabel(r"$t_P$ (s)")
    a3.set_ylabel(r"$\sigma_{100}$ of $\delta^2 T$ (ms)")
    a3.set_title(f"(c) Moving std, mean = {np.nanmean(sig[ok]):.2f} ms")
    a3.legend(loc="upper right")
    fig.tight_layout()
    save(fig, base)


def fig_algorithm_signals(d, name, base, startup_span=140.0):
    """Algorithm 1 + 2 裡每一個訊號隨時間的變化。左欄開頭 140 s，右欄整段。"""
    tP, CI = d["tP"], d["CI"]
    a, b = final_line(d)
    tr = a * tP + b
    det = lambda x: (x - tr) * 1e3                       # 扣掉最後的直線，換成 ms，才看得到 ms 等級的細節
    dT, star, thr = det(d["dT"]), det(d["star"]), det(d["thr"])
    dhat = np.where(d["valid"], det(d["dhat"]), np.nan)
    d2 = np.r_[np.nan, np.diff(d["dT"])] * 1e3
    sig = d["sigma"] * 1e3
    g0 = d["g0"] * 1e6
    g1 = np.where(np.isfinite(d["g1"]), d["g1"] * 1e6, np.nan)
    elig = rolling_mean(d["branch"] == 1, 100) * 100
    piv = np.flatnonzero(d["pivot"])

    rows = 8
    fig, ax = plt.subplots(rows, 2, figsize=(7.16, 13.0), sharex="col")
    spans = [(tP[0], min(tP[0] + startup_span, tP[-1])), (tP[0], tP[-1])]
    for c, (t0, t1) in enumerate(spans):
        m = (tP >= t0) & (tP <= t1)
        A = ax[:, c]
        ms = 2.0 if c == 0 else 0.6
        A[0].plot(tP[m], dT[m], ".", ms=ms, color=C_DOT, label=r"$\delta T$", zorder=2)
        A[0].plot(tP[m], thr[m], "-", lw=0.6, color=C_THR, label=r"$\delta T_{thr}$ (Eq. 9)", zorder=3)
        A[0].plot(tP[m], star[m], "-", lw=1.2, color=C_STAR, label=r"$\delta T_*$ (Eq. 8)", zorder=4)
        A[0].plot(tP[m], dhat[m], "--", lw=1.0, color=C_HAT, label=r"$\delta\hat{T}$ (Eq. 19)", zorder=5)
        pm = piv[(tP[piv] >= t0) & (tP[piv] <= t1)]
        A[0].plot(tP[pm], dT[pm], ls="none", marker=(8, 2, 0), ms=7, mew=1.0, color=C_PIV,
                  label="pivot (Eq. 11)", zorder=6)
        lo = np.nanpercentile(dT[m], 0.3)
        hi = np.nanpercentile(dT[m], 99.0)
        A[0].set_ylim(lo - 0.15 * (hi - lo) - 2, hi + 0.25 * (hi - lo))
        A[1].plot(tP[m], d2[m], ".", ms=ms, color=C_DOT)
        A[1].set_ylim(*lims(d2[m], 0.5, 99.5))
        A[2].plot(tP[m], sig[m], color=C_DOT, lw=0.8, label=r"$\sigma_M$ (Eq. 7)")
        A[2].axhline(CI / 2 * 1e3, color="red", ls="--", lw=0.9, label="CI / 2")
        A[2].set_ylim(0, max(np.nanmax(sig[m]) * 1.1, CI / 2 * 1e3 * 1.3))
        A[3].plot(tP[m], d["lam0"][m], color=C_DOT, lw=0.8, label=r"$\lambda_0$ (Eq. 10)")
        lm = m & np.isfinite(d["lam"])
        A[3].plot(tP[lm], d["lam"][lm], ".", ms=ms + 0.5, color=C_STAR, label=r"$\lambda$ at eligible pts (Eq. 13)")
        A[3].set_ylim(-0.05, 1.1)
        A[4].plot(tP[m], g1[m], "-", lw=0.6, color=C_THR, label=r"$\gamma_1$ (Eq. 14)")
        A[4].plot(tP[m], g0[m], "-", lw=1.1, color=C_STAR, label=r"$\gamma_0$ (Eq. 12)")
        A[4].axhline(a * 1e6, color=C_HAT, ls="--", lw=0.8, label=r"final $\alpha$")
        # 開頭換樞紐點時 γ0 會到上萬 ppm；y 軸固定在最後的 α 附近，才看得到收斂後的鋸齒
        up = np.nanpercentile(g1[m] - a * 1e6, 99) if np.isfinite(g1[m]).any() else 100
        A[4].set_ylim(a * 1e6 - 60, a * 1e6 + float(np.clip(up * 1.1, 100, 400)))
        A[5].plot(tP[m], d["alpha_t"][m] * 1e6, color=C_HAT, lw=1.0)
        A[5].axhline(a * 1e6, color="gray", ls="--", lw=0.7)
        A[5].set_ylim(*lims(d["alpha_t"][m] * 1e6, 1, 99.5, min_span=5))
        A[6].plot(tP[m], d["beta_t"][m] * 1e3, color=C_HAT, lw=1.0)
        A[6].axhline(b * 1e3, color="gray", ls="--", lw=0.7)
        A[6].set_ylim(*lims(d["beta_t"][m] * 1e3, 1, 99.5, min_span=1))
        A[6].ticklabel_format(axis="y", useOffset=False)
        for r in (5, 6):
            if np.isfinite((d["alpha_t"] if r == 5 else d["beta_t"])[m]).sum() < 2:
                A[r].text(0.5, 0.5, r"no $\alpha,\beta$ output", transform=A[r].transAxes, ha="center",
                          va="center", color="gray")
        A[7].plot(tP[m], elig[m], color=C_DOT, lw=0.8, label="eligible pts, last 100 (%)")
        for t in tP[pm]:
            A[7].axvline(t, color=C_PIV, lw=0.8)
        inval = m & ~d["valid"]
        if inval.any():
            A[7].fill_between(tP[m], 0, 100, where=inval[m], color="gray", alpha=0.25, lw=0,
                              label=r"no $\alpha,\beta$ output")
        A[7].set_ylim(0, 100)
        A[7].set_xlabel(r"$t_P$ (s)")
        A[7].set_xlim(t0, t1)
        A[0].set_title(("Startup (first %.0f s)" % (t1 - t0)) if c == 0 else "Whole recording")
    labels = [r"$\delta T-(\alpha t_P+\beta)$ (ms)", r"$\delta^2 T$ (ms)", r"$\sigma_M$ (ms)", r"$\lambda$",
              r"$\gamma$ (ppm)", r"$\alpha(t)$ (ppm)", r"$\beta(t)$ (ms)", "eligible (%)"]
    for r, lab in enumerate(labels):
        ax[r, 0].set_ylabel(lab)
    for r in (0, 2, 3, 4, 7):
        ax[r, 0].legend(loc="upper right", fontsize=6, ncol=2 if r == 0 else 1)
    fig.suptitle(f"SharkTooth internal signals ({name}):  final α = {a * 1e6:+.1f} ppm, β = {b * 1e3:.2f} ms, "
                 f"{int(d['pivot'].sum())} pivots, output {100 * d['valid'].mean():.1f} %".replace("-", "−"),
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save(fig, base)


def eps_s(d):
    """每一包的延遲 ε = δT − δ̂T（秒）。SharkTooth 沒有輸出的包（例如 --paper 失敗時）改用下緣直線代替 δ̂T。"""
    a, b = final_line(d)
    return d["dT"] - np.where(d["valid"], d["dhat"], a * d["tP"] + b)


def delay_stats(d):
    CI = d["CI"]
    eps = eps_s(d) * 1e3
    return dict(eps=eps, p10=np.percentile(eps, 10), p50=np.percentile(eps, 50), p90=np.percentile(eps, 90),
                p99=np.percentile(eps, 99), late1=100 * np.mean(eps > CI * 1e3), late2=100 * np.mean(eps > 2 * CI * 1e3))


def fig_delay(d, name, base):
    """延遲 ε 的分布、到達間隔、封包間隔、遲到比例。"""
    CI = d["CI"] * 1e3
    st = delay_stats(d)
    eps = st["eps"]
    tP = d["tP"]
    fig, ax = plt.subplots(2, 2, figsize=(7.16, 5.2))
    a = ax[0, 0]
    a.hist(eps, bins=np.arange(np.floor(eps.min()) - 1, max(eps.max(), 3 * CI) + 1, 0.5), color=C_DOT)
    for k in range(4):
        a.axvline(k * CI, color="red" if k else "gray", ls="--", lw=0.8)
    a.set_yscale("log")
    a.set_xlabel(r"$\varepsilon = \delta T - \delta\hat{T}$ (ms)")
    a.set_ylabel("packets")
    a.set_title(f"(a) Delay: P50 {st['p50']:.1f}, P99 {st['p99']:.1f} ms; "
                f">1 CI {st['late1']:.2f} %, >2 CI {st['late2']:.3f} %", fontsize=8.5)
    a = ax[0, 1]
    dtc = np.diff(np.sort(d["t_C"])) * 1e3
    a.hist(dtc, bins=np.arange(0, min(dtc.max(), 400) + 1, 1.0), color=C_DOT)
    for k in range(1, 14):
        a.axvline(k * CI, color="red", lw=0.4, alpha=0.5)
    a.set_yscale("log")
    a.set_xlabel(r"$\Delta t_C$ (ms)")
    a.set_title(f"(b) Arrival interval (red: multiples of CI = {CI:.0f} ms)", fontsize=8.5)
    a = ax[1, 0]
    dtp = np.diff(tP)
    P = np.median(dtp)
    dev = (dtp - P) * 1e6
    tick = 1e6 / LSE_HZ
    a.hist(dev, bins=np.arange(-5.5, 5.5) * tick, color=C_DOT, rwidth=0.85)
    a.set_yscale("log")
    a.set_xlabel(r"$\Delta t_P$ $-$ median ($\mu$s, bins = 1 LSE tick)")
    a.set_ylabel("packets")
    a.set_title(f"(c) Packet period {P * 1e3:.3f} ms (sample rate {4 / np.median(d['fl']):.2f} Hz)", fontsize=8.5)
    a = ax[1, 1]
    e_all = eps_s(d) * 1e3
    edges = np.arange(tP[0], tP[-1], 60.0)
    idx = np.digitize(tP, edges) - 1
    r1, r2 = [], []
    for i in range(edges.size):
        x = e_all[(idx == i)]
        x = x[np.isfinite(x)]
        r1.append(100 * np.mean(x > CI) if x.size >= 300 else np.nan)          # 最後不滿半分鐘的不畫
        r2.append(100 * np.mean(x > 2 * CI) if x.size >= 300 else np.nan)
    a.plot(edges + 30, r1, ".-", ms=3, lw=0.8, color=C_DOT, label="> 1 CI (retransmission / late)")
    a.plot(edges + 30, r2, ".-", ms=3, lw=0.8, color="red", label="> 2 CI")
    a.set_xlabel(r"$t_P$ (s)")
    a.set_ylabel("% of packets / min")
    a.set_title("(d) Late packets over time", fontsize=8.5)
    a.legend(loc="upper right", fontsize=7)
    fig.suptitle(f"Delay and timing ({name})", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, base)
    return st


def node_epochs(d):
    """每 1 秒（t_P 時間軸）擬合 5 Hz 正弦：振幅、DC、殘差、真實延遲 τ_n；同一秒 δ̂T 的平均。"""
    t_samp, _ = sample_times(d)
    t, v = t_samp.ravel(), d["s"].ravel()
    o = np.argsort(t)
    t, v = t[o], v[o]
    edges = np.arange(math.ceil(t[0]), math.floor(t[-1]) - 1 + 1e-9, 1.0)
    R = np.full(edges.size, np.nan)
    C = R.copy()
    rms = R.copy()
    for i, e in enumerate(edges):
        lo, hi = np.searchsorted(t, [e, e + 1])
        if hi - lo < 40:
            continue
        w = 2 * np.pi * F_REF * t[lo:hi]
        X = np.stack([np.cos(w), np.sin(w), np.ones_like(w)], axis=1)
        coef, *_ = np.linalg.lstsq(X, v[lo:hi], rcond=None)
        R[i], C[i] = math.hypot(coef[0], coef[1]), coef[2]
        rms[i] = np.sqrt(np.mean((v[lo:hi] - X @ coef) ** 2))
    tau = epoch_true_offset(t, v, edges)
    grid = edges[:, None] + np.arange(0, 1, 1 / 50.0)[None, :]
    est = np.interp(grid.ravel(), d["tP"], d["dhat"]).reshape(grid.shape).mean(axis=1)
    ok = np.interp(grid.ravel(), d["tP"], d["valid"].astype(float)).reshape(grid.shape).min(axis=1) >= 1
    res = wrap(tau - est)
    res[~ok] = np.nan
    if np.isfinite(res).any():
        res = wrap(res - np.nanmedian(res))
    return dict(t=t, v=v, edges=edges, R=R, C=C, rms=rms, sinad=10 * np.log10((R ** 2 / 2) / rms ** 2), res=res)


def remove_common_drift(eps):
    """同步殘差裡有一條所有節點共用的斜線：函數產生器時基相對 PC 時鐘的漂移（幾 ppm）。
    用各節點斜率的中位數當共同漂移，每個節點扣掉「共同斜率 + 自己的常數」。
    只有一塊板子時分不出來，就扣掉它自己的斜率（標題會註明）。"""
    slopes = []
    for ep in eps.values():
        m = np.isfinite(ep["res"])
        if m.sum() > 60:
            slopes.append(np.polyfit(ep["edges"][m], ep["res"][m], 1)[0])
    k = float(np.median(slopes)) if slopes else 0.0
    for ep in eps.values():
        r = ep["res"] - k * ep["edges"]
        ep["res_raw"] = ep["res"]
        ep["res"] = r - np.nanmedian(r) if np.isfinite(r).any() else r
        ep["drift_ppm"] = k * 1e6
        ep["drift_own"] = len(eps) < 2
    return k


def fig_waveform(d, name, base, ep):
    t, v = ep["t"], ep["v"]
    fig, ax = plt.subplots(6, 1, figsize=(3.5, 10.2))
    e = ep["edges"]
    t_at = math.floor(t[0] + 0.8 * (t[-1] - t[0]))
    i_at = int(np.clip(np.searchsorted(e, t_at), 0, e.size - 1))
    m = (t >= e[i_at]) & (t < e[i_at] + 1)
    w = 2 * np.pi * F_REF * t[m]
    X = np.stack([np.cos(w), np.sin(w), np.ones_like(w)], axis=1)
    coef, *_ = np.linalg.lstsq(X, v[m], rcond=None)
    tf = np.linspace(e[i_at], e[i_at] + 1, 500)
    wf = 2 * np.pi * F_REF * tf
    ax[0].plot(tf, coef[0] * np.cos(wf) + coef[1] * np.sin(wf) + coef[2], "-", color="gray", lw=0.8, label="5 Hz fit")
    ax[0].plot(t[m], v[m], ".", ms=4, color=C_DOT, label="ADC samples")
    ax[0].set_xlabel(r"$t_P$ (s)")
    ax[0].set_ylabel("Signal (V)")
    ax[0].set_title(f"(a) ADC samples, 1 s at $t_P$ = {e[i_at]:.0f} s")
    ax[0].legend(loc="upper right", fontsize=6.5)
    for a, y, lab, tit, fmt in ((ax[1], 2 * ep["R"], "Vpp (V)", "(b) Amplitude per 1-s epoch", ".3f"),
                                (ax[2], ep["C"], "DC (V)", "(c) Offset per 1-s epoch", ".3f"),
                                (ax[3], ep["sinad"], "SINAD (dB)", "(d) Fit SINAD per 1-s epoch", ".1f")):
        a.plot(e, y, color=C_DOT, lw=0.7)
        a.set_ylim(*lims(y, 0.5, 99.5, min_span=0.01 if "V" in lab else 2))
        a.set_ylabel(lab)
        a.set_title(f"{tit}: median {np.nanmedian(y):{fmt}}")
        a.set_xlim(e[0], e[-1])
    o = np.argsort(t)
    fs = 1.0 / np.median(np.diff(t[o]))
    x = v[o] - v.mean()
    n = x.size
    X = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / fs)
    db = 20 * np.log10(X / X.max() + 1e-12)
    ax[4].plot(f, db, color=C_DOT, lw=0.5)
    for k in range(1, 5):
        if k * F_REF < fs / 2:
            ax[4].axvline(k * F_REF, color="red", lw=0.5, ls="--")
    ax[4].set_ylim(-140, 5)
    ax[4].set_xlim(0, fs / 2)
    ax[4].set_xlabel("Frequency (Hz)")
    ax[4].set_ylabel("dB re peak")
    ax[4].set_title(f"(e) Spectrum (whole record, fs = {fs:.2f} Hz)")
    r = ep["res"] * 1e3
    ax[5].plot(e, (ep["res_raw"] - np.nanmedian(ep["res_raw"])) * 1e3, "-", lw=0.6, color="#BBBBBB",
               label="before removing generator drift", zorder=1)
    ax[5].plot(e, r, ".", ms=1.2, color=C_HAT, label="after", zorder=2)
    ax[5].axhline(0, color="gray", lw=0.6)
    ax[5].set_ylim(*lims(r, 0.5, 99.5, min_span=0.4))
    ax[5].set_xlim(e[0], e[-1])
    ax[5].set_xlabel(r"$t_P$ (s)")
    ax[5].set_ylabel("ms")
    rr = r[np.isfinite(r)]
    src = "own slope" if ep.get("drift_own") else "common to all nodes"
    ax[5].set_title(f"(f) True delay $-$ $\\delta\\hat{{T}}$, std {rr.std():.3f} ms\n"
                    f"(generator drift {ep.get('drift_ppm', 0):+.2f} ppm removed, {src})" if rr.size
                    else "(f) True delay − δ̂T: no output", fontsize=9)
    ax[5].legend(loc="lower right", fontsize=6)
    fig.suptitle(f"Reference signal ({name})", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, base)


# =========================================================================== 整次錄音
def fig_rse(edges, rse, base):
    """論文 Fig. 10。"""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.16, 4.8), sharex=True)
    allv = np.concatenate([x[np.isfinite(x)] for x in rse.values()]) * 1e3
    for (n, m), x in rse.items():
        lab = f"{n}$-${m}".replace("NODE_", "")
        for a in (a1, a2):
            a.plot(edges, x * 1e3, lw=0.6, label=lab)
    for a in (a1, a2):
        for s in range(1, N_SEG):
            if s * SEG_S < edges[-1]:
                a.axvline(s * SEG_S, color="#1f4fbf", ls="--", lw=0.8)
        a.set_ylabel("RSE (ms)")
    a1.axhspan(-1, 1, color="#FFE680", alpha=0.5, lw=0, zorder=0)
    a1.set_ylim(-2, 2)
    a1.set_title("(a) RSE of every pair (Eq. 26), paper scale; yellow = ±1 ms target")
    a1.legend(loc="upper right", ncol=len(rse), fontsize=7)
    a2.set_ylim(*lims(allv, 0.2, 99.8, min_span=0.4))
    a2.axhline(0, color="gray", lw=0.6)
    a2.set_title("(b) Zoom")
    a2.set_xlabel(r"$t_P$ (s, common grid)")
    a2.set_xlim(edges[0], edges[-1])
    fig.tight_layout()
    save(fig, base)


def fig_segments(edges, rse, nodes, base):
    """每段每對 |RSE| 箱形圖；論文 Fig. 12：P95|RSE| 對 σ²100。"""
    pairs = list(rse)
    nseg = int(min(N_SEG, math.ceil((edges[-1] + 1) / SEG_S)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 3.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    wbox = 0.8 / len(pairs)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rows = []
    for s in range(nseg):
        m = (edges >= s * SEG_S) & (edges + 1 <= (s + 1) * SEG_S)
        best, top = None, -1
        stats = {}
        for j, p in enumerate(pairs):
            y = np.abs(rse[p][m])
            y = y[np.isfinite(y)] * 1e3
            if y.size < 30:
                continue
            bp = a1.boxplot([y], positions=[s + 1 - 0.4 + (j + 0.5) * wbox], widths=wbox * 0.85, showfliers=False,
                            patch_artist=True)
            for k in ("boxes",):
                for art in bp[k]:
                    art.set_facecolor(cols[j % len(cols)])
                    art.set_alpha(0.7)
            p95 = np.percentile(y, 95)
            sig2 = np.mean([np.nanmean((nodes[q]["sigma"][(nodes[q]["tP"] >= s * SEG_S) &
                                                         (nodes[q]["tP"] < (s + 1) * SEG_S)] * 1e3) ** 2) for q in p])
            stats[p] = (y.mean(), p95, sig2)
            if p95 > top:
                best, top = p, p95
        for j, p in enumerate(pairs):
            if p not in stats:
                continue
            mean_abs, p95, sig2 = stats[p]
            a2.scatter(sig2, p95, s=36 if p == best else 16, color=cols[j % len(cols)],
                       edgecolor="black" if p == best else "none", marker="os^dv<>"[s % 7], zorder=3)
            rows.append((s + 1, p, mean_abs, p95, sig2, p == best))
    a1.set_xticks(range(1, nseg + 1), [str(k) for k in range(1, nseg + 1)])
    a1.set_xlim(0.5, nseg + 0.5)
    a1.set_xlabel("Acquisition segment (600 s)")
    a1.set_ylabel("|RSE| (ms)")
    a1.set_title("(a) |RSE| per segment and pair")
    hs = [plt.Rectangle((0, 0), 1, 1, color=cols[j % len(cols)], alpha=0.7) for j in range(len(pairs))]
    a1.legend(hs, [f"{n}$-${m}".replace("NODE_", "") for n, m in pairs], fontsize=7, loc="upper right")
    a2.set_xlabel(r"$\sigma^2_{100}$ (ms$^2$, mean of the pair)")
    a2.set_ylabel(r"P95 |RSE| (ms)")
    a2.set_title("(b) Fig. 12 style (marker = segment,\nblack edge = worst pair, Eq. 27)", fontsize=8.5)
    fig.tight_layout()
    save(fig, base)
    return rows


def fig_sync_overlay(nodes, base):
    """論文 Fig. 9 在三個時間點：開頭（剛有輸出）、中段、結尾。"""
    t_end = min(d["tP"][-1] for d in nodes.values())
    t_beg = max(d["tP"][0] for d in nodes.values())
    first_valid = max(d["tP"][np.argmax(d["valid"])] if d["valid"].any() else t_beg for d in nodes.values())
    ats = [min(first_valid + 2, t_end - 2), 0.5 * (t_beg + t_end), t_end - 2.5]
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 4.0))
    ref = next(iter(nodes.values()))
    t_p0, t_s0 = sample_times(ref)
    for c, at in enumerate(ats):
        ts_at = np.interp(at, t_p0.ravel(), t_s0.ravel())
        for name, d in nodes.items():
            t_p, t_s = sample_times(d)
            v = d["s"]
            m = (t_p >= at) & (t_p <= at + 1)
            o = np.argsort(t_p[m])
            ax[0, c].plot(t_p[m][o], v[m][o], lw=1.1, label=name)
            m = (t_s >= ts_at) & (t_s <= ts_at + 1)
            o = np.argsort(t_s[m])
            ax[1, c].plot(t_s[m][o], v[m][o], lw=1.1, label=name)
        ax[0, c].set_title(f"OFF, $t_P$ ≈ {at:.0f} s", fontsize=8.5)
        ax[1, c].set_title("ON", fontsize=8.5)
        ax[0, c].set_xlabel(r"$t_P$ (s)")
        ax[1, c].set_xlabel(r"$\hat{t}_S$ (s)")
        for a in ax[:, c]:
            a.margins(x=0)
            a.ticklabel_format(axis="x", useOffset=False)
    ax[0, 0].set_ylabel("Signal (V)")
    ax[1, 0].set_ylabel("Signal (V)")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, base)


def read_stats(folder: Path):
    f = folder / "events.csv"
    if not f.exists():
        return None
    t, lag, rate = [], [], {}
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["event"] != "stats":
                continue
            t.append(float(r["t_C"]))
            mm = re.search(r"loop_lag_max_ms=([\d.]+)", r["detail"])
            lag.append(float(mm.group(1)) if mm else np.nan)
            for n, x in re.findall(r"(\w+):\s*([\d.]+)/s", r["detail"]):
                rate.setdefault(n, {})[len(t) - 1] = float(x)
    t = np.array(t)
    rates = {n: np.array([v.get(i, np.nan) for i in range(t.size)]) for n, v in rate.items()}
    return t, np.array(lag), rates


def fig_link(nodes, folder, base):
    ev = read_stats(folder)
    fig, ax = plt.subplots(4, 1, figsize=(7.16, 7.0), sharex=True)
    for n, d in nodes.items():                            # 每分鐘的封包數（接收程式記的 5 秒統計只有 50/51 包兩種值，看不出東西）
        e = np.arange(d["t_C"][0], d["t_C"][-1] - 30, 60.0)
        c = np.histogram(d["t_C"], bins=np.r_[e, e[-1] + 60])[0] / 60.0
        c[-1] = np.nan if d["t_C"][-1] < e[-1] + 60 else c[-1]
        ax[0].plot(e + 30, c, ".-", ms=3, lw=0.8, label=f"{n} (expected {1 / np.median(np.diff(d['tP'])):.3f})")
    if ev is not None and ev[0].size:
        t, lag, rates = ev
        ax[1].semilogy(t, np.maximum(lag, 0.01), ".", ms=2, color="black")
        ax[1].axhline(5, color="red", ls="--", lw=0.8)
        ax[1].text(t[0], 5, " 5 ms", color="red", va="bottom", fontsize=7)
    else:
        ax[0].text(0.5, 0.5, "no events.csv", transform=ax[0].transAxes, ha="center")
    ax[0].set_ylabel("packets / s")
    ax[0].set_title("(a) Packet rate per minute (expected = 1 / packet period on the node clock)")
    ax[0].legend(loc="lower right", ncol=len(nodes), fontsize=6.5)
    ax[1].set_ylabel("max lag (ms)")
    ax[1].set_title("(b) PC event-loop lag (t_C may be delayed when high)")
    for n, d in nodes.items():
        gaps = (np.diff(d["pkt_id"]) % 256) - 1
        ax[2].plot(d["t_C"][1:], np.cumsum(gaps), lw=0.9, label=f"{n} lost")
        q = (d["qdrop"] - d["qdrop"][0]) % 256
        if q.any():
            ax[2].plot(d["t_C"], q, ls="--", lw=0.9, label=f"{n} node-side skipped")
        e = eps_s(d)
        edges = np.arange(d["t_C"][0], d["t_C"][-1], 60.0)
        idx = np.digitize(d["t_C"], edges) - 1
        r = [np.mean(x[np.isfinite(x)] > d["CI"]) * 100 if np.isfinite(x).sum() >= 300 else np.nan
             for x in (e[idx == i] for i in range(edges.size))]
        ax[3].plot(edges + 30, r, ".-", ms=3, lw=0.8, label=n)
    if all(((np.diff(d["pkt_id"]) % 256) == 1).all() for d in nodes.values()):
        ax[2].text(0.5, 0.75, "no lost packets", transform=ax[2].transAxes, ha="center", va="center", color="gray")
    ax[2].set_ylabel("cumulative")
    ax[2].set_title("(c) Lost packets (from packet ID gaps)")
    ax[2].legend(loc="upper left", fontsize=7, ncol=3)
    ax[3].set_ylabel("% > 1 CI")
    ax[3].set_title("(d) Late / retransmitted packets per minute")
    ax[3].legend(loc="upper right", fontsize=7, ncol=len(nodes))
    ax[3].set_xlabel(r"$t_C$ (s)")
    fig.tight_layout()
    save(fig, base)


# =========================================================================== 主程式
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="st_receiver.py 建的資料夾")
    ap.add_argument("--ci", type=float, default=0.030, help="實際的 connection interval（秒）")
    ap.add_argument("--tilt", type=float, default=100.0, help="預傾斜（ppm），0 = 論文原版")
    ap.add_argument("--paper", action="store_true", help="完全照論文：--tilt 0，穩定期內 α = β = 0")
    ap.add_argument("--tick-us", type=float, default=None, help="每個 tick 當成幾 µs（預設精確的 30.5176）")
    ap.add_argument("--gt", choices=("fit", "cc"), default="fit", help="RSE 的真實延遲：正弦擬合或論文的 10 kHz 互相關")
    a = ap.parse_args()

    paper_style()
    folder = Path(a.data)
    if a.paper:
        a.tilt = 0.0
    tag = "_paper" if a.paper else ("" if a.tilt == 100.0 else f"_tilt{a.tilt:g}")
    out = folder / f"signals{tag}"
    out.mkdir(exist_ok=True)
    nodes = load(folder, tick_s=(a.tick_us * 1e-6) if a.tick_us else 1.0 / LSE_HZ)
    for d in nodes.values():
        run_sharktooth(d, a.ci, a.tilt, keep_last=not a.paper)

    md = [f"# 訊號結果圖摘要：`{folder.name}`", "",
          f"參數：CI = {a.ci * 1e3:.2f} ms、預傾斜 {a.tilt:g} ppm、{'論文原版（穩定期 α = β = 0）' if a.paper else '穩定期保留上一組 α、β'}、"
          f"RSE 真實延遲 = {'10 kHz 互相關' if a.gt == 'cc' else '正弦擬合'}。由 `st_signals.py` 產生。", "",
          "## 每塊板子", "",
          "| 節點 | 封包 | 掉包 | 樞紐點 | 有輸出 | α (ppm) | β (ms) | σ100 平均 (ms) | ε P50 / P99 (ms) | > 1 CI / > 2 CI (%) "
          "| Vpp (V) | SINAD (dB) | 同步殘差 std (ms) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    eps = {name: node_epochs(d) for name, d in nodes.items()}
    drift = remove_common_drift(eps)
    for name, d in nodes.items():
        print(f"{name}：畫 4 張圖")
        fig_timestamp_model(d, name, out / f"{name}_1_timestamp_model")
        fig_algorithm_signals(d, name, out / f"{name}_2_algorithm_signals")
        st = fig_delay(d, name, out / f"{name}_3_delay")
        ep = eps[name]
        fig_waveform(d, name, out / f"{name}_4_waveform", ep)
        al, be = final_line(d)
        lost = int(np.sum((np.diff(d["pkt_id"]) % 256) - 1))
        res = ep["res"][np.isfinite(ep["res"])] * 1e3
        md.append(f"| {name} | {d['tP'].size} | {lost} | {int(d['pivot'].sum())} | {100 * d['valid'].mean():.1f} % "
                  f"| {al * 1e6:+.1f}{'' if d.get('ab_valid') else '（下緣擬合）'} | {be * 1e3:.2f} "
                  f"| {np.nanmean(d['sigma'][100:]) * 1e3:.2f} | {st['p50']:.1f} / {st['p99']:.1f} "
                  f"| {st['late1']:.2f} / {st['late2']:.3f} | {2 * np.nanmedian(ep['R']):.3f} "
                  f"| {np.nanmedian(ep['sinad']):.1f} | {res.std() if res.size else float('nan'):.3f} |")
    md += ["", "同步殘差（`<node>_4_waveform.png` 的 (f)）= 每秒在 t_P 軸上量到的真實延遲（正弦相位）減掉 SharkTooth 的 δ̂T，"
           "再扣掉中位數（中位數包含 ADC 取樣時刻與時間戳的固定差，量不到）。它是單一節點的同步誤差，不需要另一塊板子。"
           f"原始殘差還有一條所有板子共用的斜線 = 函數產生器時基相對 PC 時鐘的漂移，這次是 {drift * 1e6:+.2f} ppm"
           f"（{'只有一塊板子，用它自己的斜率' if len(nodes) < 2 else '各板斜率的中位數'}），已扣掉；它在 RSE 裡本來就會相消。", ""]

    if len(nodes) >= 2:
        print("節點對：RSE、分段、同步疊圖")
        edges, rse, _, _ = rse_epochs(nodes, a.gt)
        fig_rse(edges, rse, out / "pairs_1_rse")
        rows = fig_segments(edges, rse, nodes, out / "pairs_2_segments")
        fig_sync_overlay(nodes, out / "sync_overlay")
        md += ["## 每一對節點（RSE，Eq. 26）", "",
               "| 節點對 | 有效秒數 | mean abs (ms) | std (ms) | P95 abs (ms) | 跳過前 600 s 的 mean abs (ms) |",
               "|---|---|---|---|---|---|"]
        for (n, m), x in rse.items():
            y = x[np.isfinite(x)] * 1e3
            z = x[(edges >= SEG_S) & np.isfinite(x)] * 1e3
            if y.size == 0:
                md.append(f"| {n}−{m} | 0 | – | – | – | – |")
                continue
            md.append(f"| {n}−{m} | {y.size} | {np.abs(y).mean():.3f} | {y.std():.3f} | "
                      f"{np.percentile(np.abs(y), 95):.3f} | {np.abs(z).mean() if z.size else float('nan'):.3f} |")
        md += ["", "### 每段（600 s）：Eq. 27 只取最差的一對 vs 所有節點對平均", "",
               "| 段 | 最差的一對（P95 最大） | 最差那對 mean abs (ms) | 所有對平均 mean abs (ms) | 最差那對 P95 (ms) |",
               "|---|---|---|---|---|"]
        for s in sorted({r[0] for r in rows}):
            rs = [r for r in rows if r[0] == s]
            w = next((r for r in rs if r[5]), None)
            if w is None:
                continue
            md.append(f"| {s} | {w[1][0]}−{w[1][1]} | {w[2]:.3f} | {np.mean([r[2] for r in rs]):.3f} | {w[3]:.3f} |")
        md += ["", "論文的 Table II 只報最差的一對（Eq. 27）。節點越多、節點對越多，「最差的一對」本身就會越大，"
               "所以節點數的效應有一部分是統計上的取最大值，不全是網路壅塞。", ""]
    print("連線品質")
    fig_link(nodes, folder, out / "link_quality")

    md += ["## 圖", ""]
    for p in sorted(out.glob("*.png")):
        md.append(f"- [`{p.name}`]({p.name})")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n已輸出到 {out}：{len(list(out.glob('*.png')))} 張圖 + summary.md")


if __name__ == "__main__":
    main()

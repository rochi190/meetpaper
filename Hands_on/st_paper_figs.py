"""
SharkTooth 實測：用你的資料重現論文的每一張結果圖，並逐條檢查論文的說法。

每張圖左邊是論文的原圖數據（sharktooth_sim/data/digitized/，從 PDF 向量圖取出），右邊是你的資料，
圖下方寫論文的原話、你的資料量到的數字、支不支持。

  每塊板子（<node> = NODE_A …）
    fig03_<node>.png   論文 Fig. 3：低延遲情況下的 δT、放大看一個 CI 內的週期樣式、σ100 ≈ CI/2
    fig04_<node>.png   論文 Fig. 4：雜訊多（重傳）時的 δT 尖峰、σ100 變大（取你資料裡 σ100 最大的那 126 秒）
    fig06_<node>.png   論文 Fig. 6：δT*（紅線）貼著 δT 的下緣、換樞紐點後幾秒穩定
  節點對
    fig09.png          論文 Fig. 9：同步前（t_P）錯開、同步後（t̂_S）對齊
    fig10.png          論文 Fig. 10：每一對的 RSE，開頭暫態、之後收斂、±1 ms
    fig11.png          論文 Fig. 11：三個指標（平均、P95、標準差）隨 6 段的趨勢，附論文 A2、A4
    fig12.png          論文 Fig. 12：每段 P95|RSE| 對 σ²100，紅虛線 (CI/2)² = 225 ms²、黑線 = 迴歸
    table2.png         論文 Table II：論文 A2、A4 和你的 A3 放在同一張表
  summary.md           每張圖的「論文說法 → 你的資料 → 結論」

用法：
    python st_paper_figs.py runs/20261101_220025 --ci 0.030
    python st_paper_figs.py runs/20261101_220025 --ci 0.030 --paper    # 完全照論文的算法 → paper_figs_paper/
    python st_paper_figs.py runs/20261101_220025 --ci 0.030 --debug    # 另外輸出算法中間訊號、ADC 訊號（論文沒有的圖）

輸出：<資料夾>/paper_figs/（--paper 時是 paper_figs_paper/，非預設 --tilt 時是 paper_figs_tilt<數字>/）
圖的文字用英文（和論文一樣）。
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from st_analyze import F_REF, LSE_HZ, load, paper_style, run_sharktooth, sample_times, save  # noqa: E402
from st_table2 import (MY_BYTES, MY_ID, N_SEG, SEG_S, draw_table, epoch_true_offset,  # noqa: E402
                       rse_epochs, segment_metrics)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

C_DOT, C_STAR, C_THR, C_HAT, C_PIV = "#0072BD", "red", "#E07A00", "#7B2CBF", "#00AA00"
PERIOD = 1.0 / F_REF
ROOT = Path(__file__).resolve().parent.parent
DIG = ROOT / "sharktooth_sim" / "data" / "digitized"
PAPER_T2 = ROOT / "sharktooth_sim" / "data" / "paper_table2.csv"
WIN = 126.0                     # 論文 Fig. 3 / 4 的時間窗（約 126 秒）
ZOOM = 6.0                      # 論文 Fig. 3b / 4b 的放大窗（約 6 秒）
CLAIMS = []                     # (圖, 節點, 論文說法, 你的資料, 結論) → summary.md
WORST_VS_ALL = []               # (段, 最差那對, 最差那對平均 |RSE|, 所有對的平均) → summary.md（Eq. 27 的偏差）
VERDICT_COLOR = {"SUPPORTED": "#1a7f37", "PARTLY": "#b26a00", "NOT SUPPORTED": "#c62828", "NOT TESTABLE": "#666666"}
VERDICT_ZH = {"SUPPORTED": "支持", "PARTLY": "部分支持", "NOT SUPPORTED": "不支持", "NOT TESTABLE": "無法檢驗"}


# =========================================================================== 輔助（final_line 等）與 --debug 的圖
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


# =========================================================================== 共用
def paper_csv(name):
    """論文圖的數位化數據（x, y）；找不到就回傳 None（圖的左欄改成提示文字）。"""
    f = DIG / f"{name}.csv"
    return np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2) if f.exists() else None


def no_paper(ax):
    ax.text(0.5, 0.5, f"paper data not found\n({DIG})", transform=ax.transAxes, ha="center", va="center",
            fontsize=7, color="gray")


def wrap_text(txt, width):
    """依空白換行，但不在 $...$（數學式）裡面斷開。width 以「顯示出來的字數」估計。"""
    lines, cur, n = [], "", 0
    for w in txt.split(" "):
        if cur.count("$") % 2 == 1:                       # 還在數學式裡，不能斷
            cur += " " + w
            continue
        vis = len(w.replace("$", "").replace("\\", "")) * (0.6 if "\\" in w else 1.0)
        if cur and n + vis + 1 > width:
            lines.append(cur)
            cur, n = w, vis
        else:
            cur, n = (cur + " " + w if cur else w), n + vis + 1
    if cur:
        lines.append(cur)
    return lines


def verdict_box(fig, fig_id, node, claim_en, result_en, verdict, claim_zh, result_zh, h_in):
    """圖下方：論文的原話、你的資料、結論；同時記下來寫進 summary.md。回傳要保留的下緣比例。"""
    w_in = fig.get_size_inches()[0]
    width = int(w_in * 17.5) - 14                         # 7.2 pt 字大約每吋 17.5 個字，右邊留給結論
    lines = wrap_text("Paper: " + claim_en, width) + wrap_text("This data: " + result_en, width)
    fig.text(0.012, 0.012, "\n".join(lines), ha="left", va="bottom", fontsize=7.2, linespacing=1.35)
    fig.text(0.988, 0.012, verdict, ha="right", va="bottom", fontsize=9, weight="bold",
             color=VERDICT_COLOR[verdict])
    CLAIMS.append((fig_id, node, claim_zh, result_zh, verdict))
    return (0.08 + 0.135 * len(lines)) / h_in


def eps_ms(d):
    """每一包相對於「最後的直線 α·t_P + β」的延遲（ms）；下緣 ≈ 0。"""
    a, b = final_line(d)
    return (d["dT"] - (a * d["tP"] + b)) * 1e3


def pick_window(d, noisy):
    """找 σ100 平均最小（Fig. 3：低延遲）或最大（Fig. 4：雜訊多）的 126 秒。跳過前 60 秒。"""
    tP, sig = d["tP"], d["sigma"]
    starts = np.arange(tP[0] + 60, tP[-1] - WIN, 5.0)
    if starts.size == 0:
        return tP[0]
    c = np.cumsum(np.r_[0, np.nan_to_num(sig)])
    i0 = np.searchsorted(tP, starts)
    i1 = np.searchsorted(tP, starts + WIN)
    mean = (c[i1] - c[i0]) / np.maximum(i1 - i0, 1)
    return float(starts[np.argmax(mean) if noisy else np.argmin(mean)])


# =========================================================================== 每塊板子：Fig. 3、4、6
def fig_3_4(d, name, base, which):
    """論文 Fig. 3（which = 3，低延遲）或 Fig. 4（which = 4，雜訊多）：左論文、右你的資料。"""
    CI = d["CI"] * 1e3
    tP, dT, sig = d["tP"], d["dT"], d["sigma"] * 1e3
    pa, pb_line, pb_mark, pc = (paper_csv(f"fig{which}{k}") for k in ("a_line", "b_line", "b_xmark", "c_line"))
    H = 6.9
    fig, ax = plt.subplots(3, 2, figsize=(7.16, H))
    # ---- 左：論文
    if pa is not None:
        ax[0, 0].plot(pa[:, 0], pa[:, 1], lw=0.6, color=C_DOT)
        if pb_mark is not None:
            y0, y1 = pb_mark[:, 1].min(), pb_mark[:, 1].max()
            ax[0, 0].add_patch(Rectangle((pb_mark[0, 0], y0), pb_mark[-1, 0] - pb_mark[0, 0], y1 - y0,
                                         fill=False, ec="red", lw=1.1))
        ax[1, 0].plot(pb_line[:, 0], pb_line[:, 1], lw=0.8, color=C_DOT)
        ax[1, 0].plot(pb_mark[:, 0], pb_mark[:, 1], "*", ms=5, color="#FF8C00")
        ax[2, 0].plot(pc[:, 0], pc[:, 1], lw=0.8, color=C_DOT)
        ax[2, 0].axhline(15.0, color="red", ls="--", lw=0.9, label="CI / 2 (paper CI = 30 ms)")
        ax[2, 0].legend(loc="upper right", fontsize=6.5)
        for a in ax[:, 0]:
            a.ticklabel_format(axis="y", useOffset=False)
    else:
        for a in ax[:, 0]:
            no_paper(a)
    # ---- 右：你的資料
    w0 = pick_window(d, noisy=(which == 4))
    m = (tP >= w0) & (tP <= w0 + WIN)
    z0 = w0 + WIN / 2 - ZOOM / 2
    if which == 4:                                        # 放大 σ 最大的那一段（尖峰在裡面）
        j = np.flatnonzero(m)[np.nanargmax(sig[m])]
        z0 = float(np.clip(tP[j] - ZOOM, w0, w0 + WIN - ZOOM))
    mz = (tP >= z0) & (tP <= z0 + ZOOM)
    ax[0, 1].plot(tP[m], dT[m], lw=0.6, color=C_DOT)
    y0, y1 = dT[mz].min(), dT[mz].max()
    ax[0, 1].add_patch(Rectangle((z0, y0), ZOOM, y1 - y0, fill=False, ec="red", lw=1.1))
    ax[1, 1].plot(tP[mz], dT[mz], lw=0.8, color=C_DOT)
    ax[1, 1].plot(tP[mz], dT[mz], "*", ms=5, color="#FF8C00")
    a, b = final_line(d)
    for k, ls in ((0, "-"), (1, "--")):                   # 下緣與下緣 + 1 CI：週期樣式應落在這兩條線之間
        ax[1, 1].plot(tP[mz], a * tP[mz] + b + k * CI / 1e3, ls=ls, color="gray", lw=0.7)
    ax[1, 1].text(tP[mz][-1], a * tP[mz][-1] + b + CI / 1e3, "lower bound + 1 CI ", ha="right", va="bottom",
                  fontsize=6.5, color="gray")
    ax[2, 1].plot(tP[m], sig[m], lw=0.8, color=C_DOT)
    ax[2, 1].axhline(CI / 2, color="red", ls="--", lw=0.9, label=f"CI / 2 (CI = {CI:.0f} ms)")
    ax[2, 1].legend(loc="upper right", fontsize=6.5)
    for a_ in ax[:, 1]:
        a_.ticklabel_format(axis="y", useOffset=False)
    # ---- 軸、標題
    tag = "minimum delay" if which == 3 else "noisy communication"
    ax[0, 0].set_title(f"Paper Fig. {which} ({tag})", fontsize=9)
    ax[0, 1].set_title(f"{name}: {WIN:.0f} s with the {'lowest' if which == 3 else 'highest'} "
                       r"mean $\sigma_{100}$", fontsize=9)
    for c in range(2):
        ax[0, c].set_ylabel(r"$\delta T$ (s)")
        ax[1, c].set_ylabel(r"$\delta T$ (s)")
        ax[2, c].set_ylabel(r"$\sigma_{100}$ (ms)")
        ax[2, c].set_xlabel(r"$t_P$ (s)")
        for r, lab in enumerate("abc"):
            ax[r, c].text(0.01, 0.97, f"({lab})", transform=ax[r, c].transAxes, va="top", fontsize=8, weight="bold")
    # 兩欄的 σ 用同一個 y 範圍，雜訊大小才能直接比
    hi = max([np.nanmax(sig[m]), CI / 2] + ([pc[:, 1].max()] if pc is not None else []))
    lo = min([np.nanmin(sig[m]), CI / 2] + ([pc[:, 1].min()] if pc is not None else []))
    for c in range(2):
        ax[2, c].set_ylim(lo - 0.15 * (hi - lo) - 0.5, hi + 0.3 * (hi - lo) + 0.5)

    # ---- 論文說法 vs 你的資料
    e = eps_ms(d)[m]
    if which == 3:
        inside = 100 * np.mean((e >= -1) & (e <= CI + 1))
        med = float(np.nanmedian(sig[m]))
        ok = inside >= 95 and abs(med - CI / 2) <= 0.15 * CI / 2
        v = "SUPPORTED" if ok else ("PARTLY" if inside >= 90 or abs(med - CI / 2) <= 0.25 * CI / 2 else "NOT SUPPORTED")
        claim_en = ('"the periodic pattern lies within the range of one CI"; '
                    '"$\\sigma_{100}$ fluctuates around half of the CI value" (Sec. III-A)')
        res_en = (f"{inside:.1f} % of packets lie within one CI above the lower bound; "
                  f"median $\\sigma_{{100}}$ = {med:.2f} ms vs CI/2 = {CI / 2:.1f} ms")
        claim_zh = "週期樣式落在一個 CI 的範圍內；σ100 在 CI 的一半附近擺動（Sec. III-A）"
        res_zh = f"{inside:.1f} % 的封包落在下緣上方一個 CI 內；σ100 中位數 {med:.2f} ms，CI/2 = {CI / 2:.1f} ms"
    else:
        wc = pick_window(d, noisy=False)
        mc = (tP >= wc) & (tP <= wc + WIN)
        late_n = 100 * np.mean(e > CI)
        late_c = 100 * np.mean(eps_ms(d)[mc] > CI)
        smax = float(np.nanmax(sig[m]))
        s_c = float(np.nanmean(sig[mc]))
        s_n = float(np.nanmean(sig[m]))
        pmax = float(pc[:, 1].max()) if pc is not None else float("nan")
        up = late_n > late_c and s_n > s_c
        v = ("PARTLY" if smax < 0.6 * pmax else "SUPPORTED") if up else "NOT SUPPORTED"
        claim_en = ('"retransmissions and software delays ... cause spikes that exceed the CI threshold", '
                    '"leading to an increased standard deviation of $\\delta T$ differences" (Sec. III-A)')
        res_en = (f"> 1 CI: {late_n:.1f} % here vs {late_c:.1f} % in the calmest window; mean $\\sigma_{{100}}$ "
                  f"{s_n:.1f} vs {s_c:.1f} ms. Max $\\sigma_{{100}}$ {smax:.1f} ms (paper Fig. 4: up to {pmax:.0f} ms)")
        claim_zh = "重傳與軟體延遲造成超過一個 CI 的尖峰，使 δ²T 的標準差變大（Sec. III-A）"
        res_zh = (f"這段超過 1 CI 的封包 {late_n:.1f} %（最平靜的一段 {late_c:.1f} %），σ100 平均 {s_n:.1f} ms"
                  f"（平靜段 {s_c:.1f} ms）；最大 {smax:.1f} ms，論文 Fig. 4 最大 {pmax:.0f} ms"
                  + ("，方向一致但雜訊比論文小很多" if v == "PARTLY" else ""))
    fb = verdict_box(fig, f"Fig. {which}", name, claim_en, res_en, v, claim_zh, res_zh, H)
    fig.tight_layout(rect=(0, fb, 1, 1))
    save(fig, base)


def local_lower_bound(d, half=5.0):
    """δT 的局部下緣：每一包前後 ±5 秒（約 100 包）內的最小值。"""
    tP, dT = d["tP"], d["dT"]
    lo = np.searchsorted(tP, tP - half)
    hi = np.searchsorted(tP, tP + half, side="right")
    return np.array([dT[i:j].min() for i, j in zip(lo, hi)])


def settle_time(d, tol_ms=1.0, hold_s=10.0):
    """最後一個樞紐點之後，δT* 要多久才進入 δT 局部下緣 ±tol_ms，並且連續 hold_s 秒都在裡面（秒）。
    回傳 (穩定時間, 最後樞紐點的 t_P, 穩定後 |δT* − 局部下緣| 的 P95 ms)。"""
    piv = np.flatnonzero(d["pivot"])
    if piv.size == 0:
        return float("nan"), float("nan"), float("nan")
    i0 = piv[-1]
    if "lower" not in d:
        d["lower"] = local_lower_bound(d)
    t = d["tP"][i0:]
    dev = np.abs(d["star"][i0:] - d["lower"][i0:]) * 1e3
    bad_t = t[dev > tol_ms]
    for k in range(t.size):
        nxt = bad_t[np.searchsorted(bad_t, t[k]):]
        if dev[k] <= tol_ms and (nxt.size == 0 or nxt[0] > t[k] + hold_s) and t[k] + hold_s <= t[-1]:
            return float(t[k] - t[0]), float(t[0]), float(np.percentile(dev[k:], 95))
    return float("nan"), float(t[0]), float("nan")


def fig_6(d, name, base, span=140.0):
    """論文 Fig. 6：左論文、右你的資料（各自的前 140 秒）。"""
    H = 3.9
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.16, H))
    dots, red, star = paper_csv("fig6_dots"), paper_csv("fig6_red"), paper_csv("fig6_star")
    if dots is not None:
        a0.plot(dots[:, 0], dots[:, 1], ".", ms=2.2, color=C_DOT, label=r"Input $\delta T$")
        a0.plot(red[:, 0], red[:, 1], "-", lw=1.8, color=C_STAR, label=r"Output $\delta T_{*}$")
        a0.plot(star[:, 0], star[:, 1], ls="none", marker=(8, 2, 0), ms=9, mew=1.2, color="#00CC00",
                label="Pivot point")
        a0.legend(loc="upper right")
    else:
        no_paper(a0)
    m = d["tP"] <= d["tP"][0] + span
    a1.plot(d["tP"][m], d["dT"][m], ".", ms=2.2, color=C_DOT, label=r"Input $\delta T$")
    a1.plot(d["tP"][m], d["star"][m], "-", lw=1.8, color=C_STAR, label=r"Output $\delta T_{*}$")
    pv = np.flatnonzero(m & d["pivot"])[-1:]
    a1.plot(d["tP"][pv], d["dT"][pv], ls="none", marker=(8, 2, 0), ms=9, mew=1.2, color="#00CC00", label="Pivot point")
    lo, hi = np.percentile(d["dT"][m], 0.5), np.percentile(d["dT"][m], 99.5)
    a1.set_ylim(lo - 0.15 * (hi - lo), max(hi, lo + 0.05) + 0.25 * (hi - lo))
    a1.legend(loc="upper right")
    a0.set_title("Paper Fig. 6", fontsize=9)
    a1.set_title(f"{name}, first {span:.0f} s", fontsize=9)
    for a in (a0, a1):
        a.set_xlabel(r"$t_P$ (s)")
        a.set_ylabel(r"$\delta T$ (s)")
        a.ticklabel_format(axis="y", useOffset=False)
    # ---- 論文說法 vs 你的資料：紅線貼下緣（穩定後很少點在紅線下方）、換樞紐點後幾秒穩定
    st, t_piv, p95_dev = settle_time(d)
    after = d["tP"] > (t_piv + max(st, 0) if np.isfinite(st) else np.inf)
    below = 100 * np.mean((d["dT"][after] - d["star"][after]) * 1e3 < -1.0) if after.any() else float("nan")
    n_piv = int(d["pivot"].sum())
    out = 100 * d["valid"].mean()
    if not np.isfinite(st) or out < 50:
        v = "NOT SUPPORTED"
        res_en = (f"{n_piv} pivots, output only {out:.1f} % of the time: $\\delta T_*$ never settles "
                  f"(last pivot at $t_P$ = {t_piv:.0f} s)")
        res_zh = f"樞紐點 {n_piv} 次，只有 {out:.1f} % 的時間有輸出：δT* 一直沒有穩定（最後一個樞紐點在 {t_piv:.0f} s）"
    else:
        v = ("SUPPORTED" if st <= 10 else "PARTLY" if st <= 30 else "NOT SUPPORTED") if below < 1.0 else "PARTLY"
        res_en = (f"{st:.1f} s after the last pivot ($t_P$ = {t_piv:.1f} s) $\\delta T_*$ is within 1 ms of the "
                  f"local lower bound (min over ±5 s) for 10 s; afterwards P95 distance {p95_dev:.2f} ms and "
                  f"{below:.2f} % of points lie > 1 ms below $\\delta T_*$")
        res_zh = (f"最後一個樞紐點（{t_piv:.1f} s）之後 {st:.1f} 秒，δT* 進入 δT 局部下緣（前後 5 秒的最小值）±1 ms "
                  f"並維持 10 秒；之後兩者距離的 P95 = {p95_dev:.2f} ms，只有 {below:.2f} % 的點在 δT* 下方超過 1 ms")
    claim_en = ('"$\\delta T_*$ represents the linear trend that is tangent to the lower bound of the $\\delta T$ '
                'distribution"; "following the identification of the pivot point, $\\delta T_*$ requires a few '
                'seconds to stabilize" (Sec. III-C)')
    claim_zh = "δT* 是貼著 δT 分布下緣的直線；找到樞紐點後 δT* 需要幾秒才穩定（Sec. III-C）"
    fb = verdict_box(fig, "Fig. 6", name, claim_en, res_en, v, claim_zh, res_zh, H)
    fig.tight_layout(rect=(0, fb, 1, 1))
    save(fig, base)


# =========================================================================== 節點對：Fig. 9、10、11、12、Table II
def fig_9(nodes, edges, est, tru, rse, base, span=1.0):
    """論文 Fig. 9：上 t_P（同步前）、下 t̂_S（同步後），取錄音 80 % 處的 1 秒。"""
    H = 5.2
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, H))
    ok = np.all(np.isfinite(np.array(list(rse.values()))), axis=0) if rse else np.zeros(edges.size, bool)
    cand = np.flatnonzero(ok & (edges >= edges[0] + 0.8 * (edges[-1] - edges[0])))
    k = int(cand[0]) if cand.size else int(0.8 * edges.size)
    at = float(edges[k])
    ref = next(iter(nodes.values()))
    t_p0, t_s0 = sample_times(ref)
    ts_at = np.interp(at, t_p0.ravel(), t_s0.ravel())
    for name, d in nodes.items():
        t_p, t_s = sample_times(d)
        v = d["s"]
        mm = (t_p >= at) & (t_p <= at + span)
        o = np.argsort(t_p[mm])
        a1.plot(t_p[mm][o], v[mm][o], lw=1.3, label=name)
        mm = (t_s >= ts_at) & (t_s <= ts_at + span)
        o = np.argsort(t_s[mm])
        a2.plot(t_s[mm][o], v[mm][o], lw=1.3, label=name)
    a1.set_title("Synchronization OFF")
    a1.set_xlabel(r"$t_P$ (s)")
    a2.set_title("Synchronization ON")
    a2.set_xlabel(r"$\hat{t}_S$ (s)")
    for a in (a1, a2):
        a.set_ylabel("Signal (V)")
        a.margins(x=0)
        a.ticklabel_format(axis="x", useOffset=False)
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), fontsize=7)
    names = list(nodes)
    before = max(abs(wrap(tru[n][k] - tru[m][k])) for i, n in enumerate(names) for m in names[i + 1:]) * 1e3
    after = max(abs(rse[p][k]) for p in rse) * 1e3 if rse else float("nan")
    v = "SUPPORTED" if np.isfinite(after) and after < 1.0 and before > 5 * after else (
        "NOT TESTABLE" if not np.isfinite(after) else "NOT SUPPORTED")
    claim_en = '"misaligned when plotted against $t_P$ ... in $\\hat{t}_S$ the signals exhibit clear alignment"'
    claim_zh = "用 t_P 畫時各節點的訊號錯開；換到 t̂_S 後明顯對齊（Sec. IV-B）"
    if np.isfinite(after):
        res_en = f"at $t_P$ = {at:.0f} s: max pair offset {before:.1f} ms before, {after:.3f} ms after"
        res_zh = f"t_P = {at:.0f} s 那一秒：節點對最大的時間差，同步前 {before:.1f} ms，同步後 {after:.3f} ms"
    else:
        res_en = f"at $t_P$ = {at:.0f} s: max pair offset {before:.1f} ms before; no SharkTooth output (t̂_S = t_P)"
        res_zh = f"t_P = {at:.0f} s 那一秒：同步前節點對最大差 {before:.1f} ms；SharkTooth 沒有輸出（t̂_S = t_P），沒有同步"
    fig.text(0.02, 0.012, f"Paper: {claim_en}\nThis data: {res_en}", fontsize=6.3, va="bottom", linespacing=1.35,
             wrap=True)
    fig.text(0.98, 0.075, v, ha="right", va="bottom", fontsize=8.5, weight="bold", color=VERDICT_COLOR[v])
    CLAIMS.append(("Fig. 9", "all", claim_zh, res_zh, v))
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    save(fig, base)


def fig_10(edges, rse, base):
    """論文 Fig. 10：左論文（4 個節點、6 對），右你的資料，同一個 y 範圍。"""
    H = 3.9
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.16, H), sharey=True)
    polys = sorted(DIG.glob("fig10_poly_*.csv")) if DIG.exists() else []
    for f in polys:
        p = np.loadtxt(f, delimiter=",", skiprows=1)
        rgb = tuple(float(x) for x in f.stem.replace("fig10_poly_", "").split("-"))
        a0.plot(p[:, 0], p[:, 1], lw=0.5, color=rgb)
    if not polys:
        no_paper(a0)
    for (n, m), x in rse.items():
        a1.plot(edges, x * 1e3, lw=0.5, label=f"{n}$-${m}".replace("NODE_", ""))
    for a, t_end in ((a0, 3700), (a1, max(edges[-1], 600))):
        a.axhspan(-1, 1, color="#FFE680", alpha=0.55, lw=0, zorder=0)
        for s in range(1, 7):
            if s * SEG_S < t_end:
                a.axvline(s * SEG_S, color="#1f4fbf", ls="--", lw=0.8)
        a.set_xlabel("Output timestamp (s)")
    a0.set_ylabel("RSE (ms)")
    a0.set_ylim(-3, 2)
    a0.set_title("Paper Fig. 10 (4 nodes, 6 pairs)", fontsize=9)
    a1.set_title(f"This recording ({len(rse)} pair{'s' if len(rse) > 1 else ''})", fontsize=9)
    a1.legend(loc="lower right", fontsize=6.5, ncol=len(rse))
    a1.set_xlim(0, max(edges[-1] + 1, 600))
    allv = np.array(list(rse.values())) * 1e3
    seg1 = allv[:, edges < SEG_S]
    rest = allv[:, edges >= SEG_S]
    p1 = np.nanpercentile(np.abs(seg1), 95) if np.isfinite(seg1).any() else float("nan")
    p2 = np.nanpercentile(np.abs(rest), 95) if np.isfinite(rest).any() else float("nan")
    inside = 100 * np.nanmean(np.abs(allv[np.isfinite(allv)]) <= 1.0) if np.isfinite(allv).any() else float("nan")
    if not np.isfinite(p1):
        v, res_en = "NOT TESTABLE", "no SharkTooth output, RSE cannot be computed"
        res_zh = "SharkTooth 沒有輸出，算不出 RSE"
    elif not np.isfinite(p2):
        v = "PARTLY" if inside >= 95 else "NOT SUPPORTED"
        res_en = f"only the first segment recorded; P95|RSE| = {p1:.2f} ms, {inside:.1f} % of epochs within ±1 ms"
        res_zh = f"只錄到第 1 段：P95|RSE| = {p1:.2f} ms，{inside:.1f} % 的秒落在 ±1 ms 內；看不到之後是否收斂"
    else:
        v = "SUPPORTED" if p2 < p1 and inside >= 95 else "PARTLY"
        res_en = (f"P95|RSE| {p1:.2f} ms in segment 1 → {p2:.2f} ms afterwards; "
                  f"{inside:.1f} % of epochs within ±1 ms")
        res_zh = f"P95|RSE| 第 1 段 {p1:.2f} ms → 之後 {p2:.2f} ms；{inside:.1f} % 的秒落在 ±1 ms 內"
    claim_en = ('"the RSE sequences ... exhibit an initial transient phase, which gradually dissipates"; '
                'yellow band = "±1 ms target accuracy" (Sec. IV-B)')
    claim_zh = "RSE 一開始有暫態，之後逐漸消失；目標精度 ±1 ms（Sec. IV-B）"
    fb = verdict_box(fig, "Fig. 10", "all pairs", claim_en, res_en, v, claim_zh, res_zh, H)
    fig.tight_layout(rect=(0, fb, 1, 1))
    save(fig, base)


def paper_t2():
    """論文 Table II 的 A2、A4（17 B、2 / 4 個節點）：{(metric, config): (med[6], iqr[6])}。"""
    out = {}
    if not PAPER_T2.exists():
        return out
    with open(PAPER_T2, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["config"] in ("A2", "A4"):
                out[(r["metric"], r["config"])] = (np.array([float(r[f"seg{s}"]) for s in range(1, 7)]),
                                                   np.array([float(r[f"seg{s}_iqr"]) for s in range(1, 7)]))
    return out


def fig_11(edges, rse, base):
    """論文 Fig. 11：Absolute average、95th percentile、Standard deviation 隨 6 段的變化。"""
    avg, sd, p95, worst = segment_metrics(edges, rse)
    for s_ in range(N_SEG):
        if not np.isfinite(avg[s_]):
            continue
        m = (edges >= s_ * SEG_S) & (edges + 1 <= (s_ + 1) * SEG_S)
        allp = [np.nanmean(np.abs(x[m])) * 1e3 for x in rse.values() if np.isfinite(x[m]).any()]
        WORST_VS_ALL.append((s_ + 1, worst[s_], avg[s_], float(np.mean(allp))))
    ref = paper_t2()
    H = 3.5
    fig, ax = plt.subplots(1, 3, figsize=(7.16, H))
    seg = np.arange(1, N_SEG + 1)
    cols = {"A2": "#1f77b4", "A4": "#2ca02c"}
    for a, mine, key, title in ((ax[0], avg, "absavg", "Absolute average"), (ax[1], p95, None, "95th percentile"),
                                (ax[2], sd, "std", "Standard deviation")):
        for cfg in ("A2", "A4"):
            if key and (key, cfg) in ref:
                med, iqr = ref[(key, cfg)]
                a.plot(seg, med, "o-", ms=3, lw=1.0, color=cols[cfg], label=f"paper {cfg} (17 B, {cfg[1]} nodes)")
                a.fill_between(seg, med - iqr / 2, med + iqr / 2, color=cols[cfg], alpha=0.15, lw=0)
        a.plot(seg, mine, "s-", ms=4, lw=1.4, color="#d62728", label=f"this data {MY_ID} (worst pair, Eq. 27)")
        a.set_xticks(seg)
        a.set_xlabel("Acquisition segment (600 s)")
        a.set_title(title, fontsize=9)
        a.set_ylim(bottom=0)
    ax[0].set_ylabel("RSE (ms)")
    ax[1].text(0.5, 0.97, "paper: not in Table II\n(Fig. 11 curves not digitized)", transform=ax[1].transAxes,
               ha="center", va="top", fontsize=6.5, color="gray")
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, fontsize=7)
    n_ok = int(np.isfinite(avg).sum())
    if n_ok < 2:
        v = "NOT TESTABLE"
        res_en = f"only {n_ok} complete 600-s segment(s); a trend needs at least 2 (record ≥ 1 h)"
        res_zh = (f"只有 {n_ok} 個完整的 600 秒段，看不出趨勢（要錄滿 1 小時）" if np.isfinite(
            np.array(list(rse.values()))).any() else "SharkTooth 沒有輸出，算不出 RSE")
    else:
        last = np.flatnonzero(np.isfinite(avg))[-1]
        dec = avg[last] < avg[0] and sd[last] < sd[0]
        sd_drop = sd[1] < 0.6 * sd[0] if np.isfinite(sd[1]) else False
        v = "SUPPORTED" if dec and sd_drop else ("PARTLY" if dec else "NOT SUPPORTED")
        res_en = (f"abs. avg {avg[0]:.2f} → {avg[last]:.2f} ms, std {sd[0]:.2f} → {sd[1]:.2f} (seg 2) → "
                  f"{sd[last]:.2f} ms, P95 {p95[0]:.2f} → {p95[last]:.2f} ms")
        res_zh = (f"平均 {avg[0]:.2f} → {avg[last]:.2f} ms；標準差 {sd[0]:.2f} →（第 2 段）{sd[1]:.2f} → "
                  f"{sd[last]:.2f} ms；P95 {p95[0]:.2f} → {p95[last]:.2f} ms")
    claim_en = ('"we observe a general decreasing trend"; "the standard deviation drops substantially from '
                'second time segment onward in all configurations" (Sec. V)')
    claim_zh = "三個指標隨時間大致下降；標準差從第 2 段起在所有設定都大幅下降（Sec. V）"
    fb = verdict_box(fig, "Fig. 11", "worst pair", claim_en, res_en, v, claim_zh, res_zh, H)
    fig.tight_layout(rect=(0, fb, 1, 0.9))
    save(fig, base)
    return avg, sd, p95, worst


def fig_12(edges, rse, nodes, CI, base, win=60.0):
    """論文 Fig. 12：每段 P95|RSE| 對 σ²100。論文每個點是一個設定的 20 次中位數；
    這裡只有一個設定，所以把每段再切成 1 分鐘、每一對各一個點（小點），600 秒的值畫大點。"""
    nseg = int(min(N_SEG, max(1, math.floor((edges[-1] + 1) / SEG_S))))
    H = 5.0 if nseg > 3 else 2.9
    rows = 2 if nseg > 3 else 1
    ncol = 3 if nseg > 1 else 1
    fig, ax = plt.subplots(rows, ncol, figsize=(7.16 if ncol > 1 else 3.5, H), squeeze=False)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    thr = (CI * 1e3 / 2) ** 2
    pairs = list(rse)
    rs, slopes, below_hi, allX = [], [], [], []

    def sig2(node, t0, t1):
        d = nodes[node]
        m = (d["tP"] >= t0) & (d["tP"] < t1)
        return float(np.nanmean((d["sigma"][m] * 1e3) ** 2)) if m.any() else np.nan

    for s in range(rows * ncol):
        a = ax.flat[s]
        if s >= nseg:
            a.axis("off")
            continue
        X, Y = [], []
        for j, p in enumerate(pairs):
            for t0 in np.arange(s * SEG_S, (s + 1) * SEG_S, win):
                m = (edges >= t0) & (edges < t0 + win)
                y = np.abs(rse[p][m])
                y = y[np.isfinite(y)] * 1e3
                if y.size < 0.8 * win:
                    continue
                X.append(np.mean([sig2(q, t0, t0 + win) for q in p]))
                Y.append(np.percentile(y, 95))
                a.plot(X[-1], Y[-1], ".", ms=3, color=cols[j % len(cols)], alpha=0.7)
            m = (edges >= s * SEG_S) & (edges < (s + 1) * SEG_S)
            y = np.abs(rse[p][m])
            y = y[np.isfinite(y)] * 1e3
            if y.size:
                a.plot(np.mean([sig2(q, s * SEG_S, (s + 1) * SEG_S) for q in p]), np.percentile(y, 95), "o",
                       ms=7, mec="black", color=cols[j % len(cols)], label=f"{p[0]}$-${p[1]}".replace("NODE_", ""))
        a.axvline(thr, color="red", ls="--", lw=0.9)
        X, Y = np.array(X), np.array(Y)
        ok = np.isfinite(X) & np.isfinite(Y)
        allX += list(X[ok])
        if ok.sum() >= 5 and np.ptp(X[ok]) > 0:
            k = np.polyfit(X[ok], Y[ok], 1)
            r = float(np.corrcoef(X[ok], Y[ok])[0, 1])
            xs = np.array([X[ok].min(), X[ok].max()])
            a.plot(xs, np.polyval(k, xs), "-", color="black", lw=1.2)
            a.text(0.03, 0.96, f"slope {k[0] * 1e3:+.1f} µs/ms²\nr = {r:+.2f} (n = {ok.sum()})",
                   transform=a.transAxes, va="top", fontsize=6.5)
            rs.append(r)
            slopes.append(k[0])
            lo, hi = Y[ok][X[ok] <= thr], Y[ok][X[ok] > thr]
            if lo.size >= 3 and hi.size >= 3:
                below_hi.append((np.median(lo), np.median(hi)))
        a.set_title(f"Segment {s + 1}", fontsize=8.5)
        a.set_xlabel(r"$\sigma^2_{100}$ (ms$^2$)", fontsize=8)
        if s % ncol == 0:
            a.set_ylabel("P95 |RSE| (ms)", fontsize=8)
    h, lab = ax.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab) + 1, fontsize=7,
               title="big = 600-s value of each pair, small = 1-min windows; red dashed = (CI/2)$^2$", title_fontsize=6.5)
    if not rs:
        v = "NOT TESTABLE"
        res_en = "not enough RSE values to fit a line"
        res_zh = "RSE 太少，畫不出迴歸線"
    else:
        npos = int(sum(r > 0 for r in rs))
        x5, x95 = np.percentile(allX, [5, 95])
        narrow = x95 - x5 < 0.3 * thr                     # σ² 只在門檻附近一小段：斜率檢驗不出來
        v = "NOT TESTABLE" if narrow else (
            "SUPPORTED" if npos == len(rs) and np.median(rs) > 0.3 else ("PARTLY" if npos > len(rs) / 2 else "NOT SUPPORTED"))
        bh = (f"; median P95 below/above {thr:.0f} ms²: " +
              ", ".join(f"{l:.2f}/{h_:.2f}" for l, h_ in below_hi)) if below_hi else ""
        res_en = (f"$\\sigma^2_{{100}}$ only spans {x5:.0f}–{x95:.0f} ms$^2$ (one configuration; paper: 18 configurations); "
                  f"slope > 0 in {npos}/{len(rs)} segments, r = " + ", ".join(f"{r:+.2f}" for r in rs) + f"{bh} ms")
        res_zh = (f"σ²100 只分布在 {x5:.0f}–{x95:.0f} ms²（只有一種設定；論文跨 18 種設定），範圍太窄"
                  + ("，檢驗不出斜率" if narrow else "") + f"；{npos}/{len(rs)} 段斜率為正，r = "
                  + "、".join(f"{r:+.2f}" for r in rs) + "（1 分鐘窗當點）" +
                  ("；σ² 低於 / 高於門檻時 P95 中位數 " + "、".join(f"{l:.2f}/{h_:.2f}" for l, h_ in below_hi) + " ms"
                   if below_hi else ""))
    claim_en = ('"as variance increases, RSE also increases, indicated by the positive slope of the fitted black '
                'regression lines"; points below $\\sigma^2_{100}$ = 225 ms$^2$ "exhibit relatively low errors" (Sec. V-A)')
    claim_zh = "σ²100 越大 RSE 越大（每段迴歸線斜率為正）；σ²100 低於 225 ms² 的點誤差較低（Sec. V-A）"
    fb = verdict_box(fig, "Fig. 12", "all pairs", claim_en, res_en, v, claim_zh, res_zh, H)
    fig.tight_layout(rect=(0, fb, 1, 0.88 if rows > 1 else 0.8))
    save(fig, base)


def table_2(avg, sd, p95, npr, base):
    """論文 Table II：論文的 A2、A4 和你的 A3 放在同一張表（P95 論文沒有列在表裡）。"""
    ref = paper_t2()
    cell = lambda x: [None if not np.isfinite(v) else (float(v), 0.0) for v in x]
    paper_rows = lambda key: [(cfg, 17, int(cfg[1]), [(float(m), float(i)) for m, i in zip(*ref[(key, cfg)])])
                              for cfg in ("A2", "A4") if (key, cfg) in ref]
    blocks = [("Absolute\nAverage", paper_rows("absavg") + [(MY_ID, MY_BYTES, npr, cell(avg))], (0.85, 0.15, 0.55)),
              ("Standard\nDeviation", paper_rows("std") + [(MY_ID, MY_BYTES, npr, cell(sd))], (0.15, 0.45, 0.85)),
              ("95th\nPercentile", [(MY_ID, MY_BYTES, npr, cell(p95))], (0.15, 0.65, 0.35))]
    draw_table(blocks, base.parent / (base.name + ".png"),
               f"A2, A4: paper Table II (17 B, nRF52840 dongles, n = 20). {MY_ID}: this recording "
               f"(B-L475E-IOT01A + PC Bluetooth, {MY_BYTES} B, {npr} peripherals, n = 1, IQR not available). "
               "Paper 95th percentile is only shown in Fig. 11/12.")
    if ("absavg", "A2") in ref and np.isfinite(avg).any():
        a2, a4 = ref[("absavg", "A2")][0], ref[("absavg", "A4")][0]
        ok = np.isfinite(avg)
        inside = bool(np.all(avg[ok] <= 0.5))
        v = "SUPPORTED" if inside else "NOT SUPPORTED"
        res_zh = ("A3 每段平均 " + "、".join(f"{x:.2f}" for x in avg[ok]) + " ms；論文 A2 "
                  + "、".join(f"{x:.2f}" for x in a2[ok]) + "、A4 " + "、".join(f"{x:.2f}" for x in a4[ok]) + " ms")
        CLAIMS.append(("Table II", "worst pair", "17 B、少節點的設定 median 平均 |RSE| 都低於 0.5 ms（Sec. V）",
                       res_zh, v))


def write_summary(out, folder, args, nodes):
    md = [f"# 論文圖重現：`{folder.name}`", "",
          f"CI = {args.ci * 1e3:.2f} ms、預傾斜 {args.tilt:g} ppm、"
          f"{'論文原版（穩定期 α = β = 0）' if args.paper else '穩定期保留上一組 α、β'}、"
          f"RSE 的真實延遲 = {'10 kHz 互相關（論文的做法）' if args.gt == 'cc' else '正弦擬合'}。"
          "由 `st_paper_figs.py` 產生；每張圖左邊是論文原圖的數位化數據，右邊是這次的資料。", "",
          "| 圖 | 節點 | 論文的說法 | 這次的資料 | 結論 |", "|---|---|---|---|---|"]
    for fig_id, node, claim, res, v in CLAIMS:
        f = {"Fig. 3": "fig03", "Fig. 4": "fig04", "Fig. 6": "fig06"}.get(fig_id)
        link = f"[{fig_id}]({f}_{node}.png)" if f else \
            f"[{fig_id}]({ {'Fig. 9': 'fig09', 'Fig. 10': 'fig10', 'Fig. 11': 'fig11', 'Fig. 12': 'fig12', 'Table II': 'table2'}[fig_id]}.png)"
        md.append(f"| {link} | {node} | {claim} | {res} | **{VERDICT_ZH[v]}** |")
    if WORST_VS_ALL:
        md += ["", "## Eq. 27 只取最差的一對：和所有節點對的平均差多少", "",
               "| 段 | 最差的一對（P95 最大） | 最差那對 mean abs (ms) | 所有對平均 mean abs (ms) | 差 |", "|---|---|---|---|---|"]
        md += [f"| {s_} | {w.replace('NODE_', '')} | {a_:.3f} | {b_:.3f} | {100 * (a_ / b_ - 1):+.0f} % |"
               for s_, w, a_, b_ in WORST_VS_ALL]
        md += ["", "節點越多、節點對越多（2 節點 1 對 … 12 節點 66 對），「最差的一對」本身就會越大，"
               "所以論文裡節點數的效應有一部分是取最大值造成的。"]
    md += ["", "結論的判斷標準（程式裡寫死，可以自己改）：",
           "- Fig. 3：≥ 95 % 的封包落在下緣上方一個 CI 內，且 σ100 中位數在 CI/2 的 ±15 % 內。",
           "- Fig. 4：最吵的一段比最平靜的一段遲到比例高、σ100 高 → 支持；但最大 σ100 不到論文的 60 % → 部分支持。",
           "- Fig. 6：δT* 進入局部下緣（前後 5 秒的最小值）±1 ms 並維持 10 秒所需的時間：≤ 10 秒（論文說「幾秒」）→ 支持，"
           "≤ 30 秒（論文自己設的穩定期）→ 部分支持；另外之後要 < 1 % 的點在 δT* 下方超過 1 ms。",
           "- Fig. 9：同步後節點對的時間差 < 1 ms，且比同步前小 5 倍以上。",
           "- Fig. 10：第 1 段之後的 P95|RSE| 比第 1 段小，且 ≥ 95 % 的秒在 ±1 ms 內。",
           "- Fig. 11：最後一段的平均與標準差都比第 1 段小，且第 2 段標準差 < 第 1 段的 60 %。",
           "- Fig. 12：每一段的迴歸斜率都為正、r 中位數 > 0.3；但如果 σ²100 的 5–95 % 範圍不到門檻的 30 %（只在門檻附近），"
           "斜率檢驗不出來 → 無法檢驗。",
           "- Table II：每段平均 |RSE| ≤ 0.5 ms（論文說 A2 一直低於 0.5 ms）。", "",
           "Fig. 12 的限制：論文每個點是一種設定（封包長度 × 節點數）20 次的中位數，跨 18 種設定看 σ²100 和誤差的關係；"
           "這裡只有一種設定，只能看同一種設定裡、1 分鐘窗之間的關係，兩者不是同一件事。", ""]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# =========================================================================== 主程式
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="st_receiver.py 建的資料夾")
    ap.add_argument("--ci", type=float, default=0.030, help="實際的 connection interval（秒）")
    ap.add_argument("--tilt", type=float, default=100.0, help="預傾斜（ppm），0 = 論文原版")
    ap.add_argument("--paper", action="store_true", help="完全照論文：--tilt 0，穩定期內 α = β = 0")
    ap.add_argument("--tick-us", type=float, default=None, help="每個 tick 當成幾 µs（預設精確的 30.5176）")
    ap.add_argument("--gt", choices=("fit", "cc"), default="fit", help="RSE 的真實延遲：正弦擬合或論文的 10 kHz 互相關")
    ap.add_argument("--debug", action="store_true", help="另外輸出算法中間訊號、ADC 訊號（論文沒有的圖）")
    a = ap.parse_args()

    paper_style()
    folder = Path(a.data)
    if a.paper:
        a.tilt = 0.0
    tag = "_paper" if a.paper else ("" if a.tilt == 100.0 else f"_tilt{a.tilt:g}")
    out = folder / f"paper_figs{tag}"
    out.mkdir(exist_ok=True)
    if not DIG.exists():
        print(f"注意：找不到論文的數位化數據 {DIG}，圖的左欄會是空的")
    nodes = load(folder, tick_s=(a.tick_us * 1e-6) if a.tick_us else 1.0 / LSE_HZ)
    for d in nodes.values():
        run_sharktooth(d, a.ci, a.tilt, keep_last=not a.paper)

    for name, d in nodes.items():
        print(f"{name}：Fig. 3、4、6")
        fig_3_4(d, name, out / f"fig03_{name}", 3)
        fig_3_4(d, name, out / f"fig04_{name}", 4)
        fig_6(d, name, out / f"fig06_{name}")
    if len(nodes) >= 2:
        print("節點對：Fig. 9、10、11、12、Table II")
        edges, rse, est, tru = rse_epochs(nodes, a.gt)
        fig_9(nodes, edges, est, tru, rse, out / "fig09")
        fig_10(edges, rse, out / "fig10")
        avg, sd, p95, _ = fig_11(edges, rse, out / "fig11")
        fig_12(edges, rse, nodes, a.ci, out / "fig12")
        table_2(avg, sd, p95, len(nodes), out / "table2")
    else:
        print("只有一塊板子：Fig. 9–12、Table II 需要至少兩塊")
    if a.debug:
        print("--debug：算法中間訊號、ADC 訊號")
        eps = {name: node_epochs(d) for name, d in nodes.items()}
        remove_common_drift(eps)
        for name, d in nodes.items():
            fig_algorithm_signals(d, name, out / f"debug_{name}_algorithm_signals")
            fig_waveform(d, name, out / f"debug_{name}_waveform", eps[name])
    write_summary(out, folder, a, nodes)
    print(f"\n已輸出到 {out}：{len(list(out.glob('*.png')))} 張圖 + summary.md")
    for fig_id, node, _, res, v in CLAIMS:
        print(f"  {fig_id:9} {node:11} {VERDICT_ZH[v]:5}  {res}")


if __name__ == "__main__":
    main()

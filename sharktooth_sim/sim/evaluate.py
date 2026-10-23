"""
評估：
  1. δT 序列的描述統計（給 T0 目標與模擬比對用）：下緣、殘差分位數、離群比例、起始尖峰、鋸齒、σ_M
  2. 真值誤差（模擬專用，直接和上帝時鐘比）
  3. 論文式 RSE（§IV-B，Eq.23–27）：50 Hz 重取樣 → 1 s epoch → 5 Hz 參考正弦互相關 → 600 s 分段 → 最差一對

互相關的實作說明（F1）：
  論文把 50 Hz 訊號上取樣到 10 kHz 再和 5 Hz 參考正弦做互相關取最大值。
  1 s epoch 剛好含 5 個週期，若用 FFT 上取樣（頻帶限制內插），上取樣後訊號在 epoch 內是週期的，
  互相關對 lag 是一個純正弦 → 最大值 = 該 epoch 在 5 Hz 的 DFT 相位，再量化到 0.1 ms（10 kHz 格點）。
  所以這裡直接用 DFT 相位 + 0.1 ms 量化，數學上等價、快約 1000 倍；`cc_delay_bruteforce` 是逐 lag 的
  直接實作，tests/test_equivalence.py 會比對兩者。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# 1. δT 描述統計
# ---------------------------------------------------------------------------


def lower_line_lp(x, y):
    """Moon–Skelly–Towsley：在所有點下方、且使 Σ(y − line) 最小的直線（= 在 x̄ 處支撐下凸包的邊）。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x0 = x.mean()
    res = linprog(c=[0.0, -1.0], A_ub=np.c_[x - x0, np.ones_like(x)], b_ub=y,
                  bounds=[(None, None), (None, None)], method="highs")
    if not res.success:  # 退回：用最低 5 % 點的 OLS
        return quantile_line(x, y, 0.05)
    a, b = res.x
    return float(a), float(b - a * x0)


def quantile_line(x, y, q=0.05):
    """分位數迴歸（線性規劃）。點數多時先在時間上分箱取分位數，避免 LP 過大。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) > 4000:
        edges = np.linspace(x.min(), x.max(), 400)
        idx = np.digitize(x, edges)
        xs, ys = [], []
        for k in np.unique(idx):
            m = idx == k
            if m.sum() >= 5:
                xs.append(x[m].mean())
                ys.append(np.quantile(y[m], q))
        x, y = np.array(xs), np.array(ys)
        q = 0.5
    n = len(x)
    x0 = x.mean()
    c = np.r_[0.0, 0.0, q * np.ones(n), (1 - q) * np.ones(n)]
    from scipy.sparse import hstack, csr_matrix, identity
    A = hstack([csr_matrix(np.c_[x - x0, np.ones(n)]), identity(n), -identity(n)]).tocsr()
    res = linprog(c, A_eq=A, b_eq=y, bounds=[(None, None)] * 2 + [(0, None)] * (2 * n), method="highs")
    a, b = res.x[:2]
    return float(a), float(b - a * x0)


def band_stats(tp, dt, line=None, skip_start_s=1.0, outlier_ms=45.0):
    """殘差 r = δT − 下緣（ms）的分位數與離群比例。前 skip_start_s 秒（起始尖峰）不計。"""
    tp = np.asarray(tp, float)
    dt = np.asarray(dt, float)
    m = tp > tp[0] + skip_start_s
    if line is None:
        line = lower_line_lp(tp[m], dt[m])
    a, b = line
    r = (dt[m] - (a * tp[m] + b)) * 1e3
    q = np.percentile(r, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return dict(slope_ppm=a * 1e6, intercept=b,
                P1=q[0], P5=q[1], P10=q[2], P25=q[3], P50=q[4], P75=q[5], P90=q[6], P95=q[7], P99=q[8],
                max=float(r.max()), outlier_frac=float((r > outlier_ms).mean()),
                frac_gt_CI=float((r > 32.5).mean()), frac_gt_2CI=float((r > 62.5).mean()),
                n=int(m.sum())), r


def startup_spike(tp, dt, line, n_first=10):
    """前 n_first 點的 δT 與下緣的差（ms）。"""
    a, b = line
    k = min(n_first, len(tp))
    ex = (np.asarray(dt[:k]) - (a * np.asarray(tp[:k]) + b)) * 1e3
    return dict(first_excess_ms=float(ex[0]), max_excess_ms=float(ex.max()),
                excess_ms=[float(v) for v in ex])


def sigma_M(dt, M=100):
    """Eq.(7)：δ²T 的移動標準差（長度 M）。回傳和 dt 同長、前 M 個為 NaN。"""
    d2 = np.diff(np.asarray(dt, float))
    out = np.full(len(dt), np.nan)
    if len(d2) < M:
        return out
    c1 = np.cumsum(np.r_[0, d2])
    c2 = np.cumsum(np.r_[0, d2 * d2])
    s1 = c1[M:] - c1[:-M]
    s2 = c2[M:] - c2[:-M]
    var = np.maximum(s2 / M - (s1 / M) ** 2, 0)
    out[M:] = np.sqrt(var)
    return out


def sawtooth_metrics(tp, dt, CI=0.030):
    """Fig.3 式鋸齒：每個鋸齒週期的封包數（層數）、週期、峰對峰、δ²T 的兩個主值。"""
    tp = np.asarray(tp, float)
    dt = np.asarray(dt, float)
    d2 = np.diff(dt)
    dtp = np.diff(tp)
    ok = dtp < 1.5 * np.median(dtp)          # 只用相鄰封包
    d2 = d2[ok]
    up = d2 > CI / 3                          # 往上跳（wrap）
    idx = np.flatnonzero(up)
    runs = np.diff(idx) if len(idx) > 2 else np.array([np.nan])
    a, b = np.polyfit(tp, dt, 1)
    det = (dt - (a * tp + b)) * 1e3
    return dict(packets_per_cycle=float(np.mean(runs)), period_s=float(np.mean(runs) * np.median(dtp)),
                p2p_ms=float(np.percentile(det, 99) - np.percentile(det, 1)),
                step_down_ms=float(np.median(d2[~up]) * 1e3) if (~up).any() else np.nan,
                step_up_ms=float(np.median(d2[up]) * 1e3) if up.any() else np.nan,
                pkt_period_ms=float(np.median(dtp) * 1e3),
                pkt_period_mean_ms=float(np.mean(np.diff(tp)[ok]) * 1e3),
                sigma100_ms=float(np.nanmean(sigma_M(dt, 100)) * 1e3) if len(dt) > 101 else np.nan,
                frac_missing=float(1 - ok.mean()))


def lower_edge_roughness(tp, dt, win_s=5.0):
    """下緣粗糙度（ms）：每 win_s 秒取最小值，扣掉直線擬合後的 P95 − P5。
    平滑斜線 → 約 1 ms；相位鎖定的階梯 → 約 gcd(P, CI)。"""
    tp = np.asarray(tp, float)
    dt = np.asarray(dt, float)
    edges = np.arange(tp[0], tp[-1], win_s)
    mins = []
    for lo in edges[:-1]:
        m = (tp >= lo) & (tp < lo + win_s)
        if m.sum() >= 3:
            mins.append((lo + win_s / 2, dt[m].min()))
    mins = np.array(mins)
    if len(mins) < 4:
        return np.nan
    a, b = np.polyfit(mins[:, 0], mins[:, 1], 1)
    r = (mins[:, 1] - (a * mins[:, 0] + b)) * 1e3
    return float(np.percentile(r, 95) - np.percentile(r, 5))


def staircase_index(tp, dt, win_s=5.0):
    """下緣「階梯度」：每 win_s 秒取最小值形成下緣，下緣一階差分的最大跳躍 / (平均斜率 × win_s)。
    平滑斜線 ≈ 1；階梯 >> 1。"""
    tp = np.asarray(tp, float)
    dt = np.asarray(dt, float)
    edges = np.arange(tp[0], tp[-1], win_s)
    mins = []
    for lo in edges[:-1]:
        m = (tp >= lo) & (tp < lo + win_s)
        if m.sum() >= 3:
            mins.append((lo + win_s / 2, dt[m].min()))
    mins = np.array(mins)
    if len(mins) < 4:
        return np.nan
    slope = np.polyfit(mins[:, 0], mins[:, 1], 1)[0]
    jumps = np.abs(np.diff(mins[:, 1]))
    denom = abs(slope) * win_s
    return float(jumps.max() / denom) if denom > 0 else np.inf


# ---------------------------------------------------------------------------
# 2. 真值誤差
# ---------------------------------------------------------------------------


def true_error(est_ts, true_pc_ts):
    """e(i) = t̂_S(i) − clock_PC(t_true(i))。回傳 (誤差, 常數偏差=中位數)。"""
    e = np.asarray(est_ts, float) - np.asarray(true_pc_ts, float)
    good = np.isfinite(e)
    bias = float(np.median(e[good])) if good.any() else np.nan
    return e, bias


def segment_stats(t, e, seg_s=600.0, T=3600.0):
    """依 t 分段（預設 600 s），每段 mean|e|、std、P95|e|（ms）。"""
    out = []
    for k in range(int(np.ceil(T / seg_s))):
        m = (t >= k * seg_s) & (t < (k + 1) * seg_s) & np.isfinite(e)
        if m.sum() < 2:
            out.append(dict(seg=k + 1, absavg=np.nan, std=np.nan, p95=np.nan, n=int(m.sum())))
            continue
        v = e[m] * 1e3
        out.append(dict(seg=k + 1, absavg=float(np.mean(np.abs(v))), std=float(np.std(v)),
                        p95=float(np.percentile(np.abs(v), 95)), n=int(m.sum())))
    return out


def convergence_time(t, e, thr=1e-3, hold_s=60.0):
    """誤差首次「持續 hold_s 秒」|e| < thr 的時間。"""
    t = np.asarray(t, float)
    bad = ~(np.abs(e) < thr)
    if not bad.any():
        return float(t[0])
    last_bad_t = None
    # 從頭掃：找到第一個 i 使得 [t_i, t_i + hold] 內全 good
    bad_t = t[bad]
    j = 0
    for i in range(len(t)):
        if bad[i]:
            continue
        while j < len(bad_t) and bad_t[j] < t[i]:
            j += 1
        if j >= len(bad_t) or bad_t[j] > t[i] + hold_s:
            return float(t[i])
        last_bad_t = bad_t[j]
    return float("nan") if last_bad_t is not None else float(t[0])


# ---------------------------------------------------------------------------
# 3. 論文式 RSE（§IV-B）
# ---------------------------------------------------------------------------

FS = 50.0
F_REF = 5.0
CC_FS = 10_000.0


def resample_node(sample_tp, sample_val, sample_ts, grid, method="linear"):
    """把一個節點的 (t_P, 訊號, t̂_S) 重取樣到共同的 t_P 格點（Eq.23 前的步驟）。"""
    o = np.argsort(sample_tp, kind="stable")
    tp = sample_tp[o]
    keep = np.r_[True, np.diff(tp) > 0]
    tp = tp[keep]
    val = sample_val[o][keep]
    ts = sample_ts[o][keep]
    inside = (grid >= tp[0]) & (grid <= tp[-1])
    v = np.full(grid.size, np.nan)
    s = np.full(grid.size, np.nan)
    if method == "linear":
        v[inside] = np.interp(grid[inside], tp, val)
    elif method == "nearest":
        j = np.clip(np.searchsorted(tp, grid[inside]), 1, len(tp) - 1)
        left = grid[inside] - tp[j - 1] < tp[j] - grid[inside]
        v[inside] = np.where(left, val[j - 1], val[j])
    else:
        raise ValueError(method)
    s[inside] = np.interp(grid[inside], tp, ts)
    # 缺樣本（掉包）處：若格點兩側的原始樣本相距 > 1.5 個取樣週期，標 NaN
    j = np.searchsorted(tp, grid[inside])
    j = np.clip(j, 1, len(tp) - 1)
    gap = (tp[j] - tp[j - 1]) > 1.5 / FS
    vi = v[inside]
    si = s[inside]
    vi[gap] = np.nan
    si[gap] = np.nan
    v[inside] = vi
    s[inside] = si
    return v, s


def cc_delay_epochs(grid, v, fs=FS, f=F_REF, cc_fs=CC_FS):
    """每 1 s epoch 的 τ_n^k：訊號相對參考正弦 sin(2π f t_P) 的延遲（s），量化到 1/cc_fs，
    範圍 (−T/2, T/2]。等價於 FFT 上取樣到 cc_fs 後的循環互相關最大值。"""
    n_per = int(round(fs))
    n_ep = len(grid) // n_per
    g = grid[: n_ep * n_per].reshape(n_ep, n_per)
    x = v[: n_ep * n_per].reshape(n_ep, n_per)
    valid = np.isfinite(x).all(axis=1)
    x = np.where(np.isfinite(x), x, 0.0)
    x = x - x.mean(axis=1, keepdims=True)
    w = 2 * np.pi * f
    # 訊號 ≈ A sin(w (t − τ))  → 用投影求相位
    S = (x * np.sin(w * g)).sum(axis=1)
    C = (x * np.cos(w * g)).sum(axis=1)
    # x = A sin(wt − wτ) = A[sin wt cos wτ − cos wt sin wτ] → S ∝ cos wτ, C ∝ −sin wτ
    tau = np.arctan2(-C, S) / w
    tau = np.round(tau * cc_fs) / cc_fs
    tau[~valid] = np.nan
    return tau


def cc_delay_bruteforce(t_epoch, x_epoch, fs=FS, f=F_REF, cc_fs=CC_FS):
    """逐 lag 的直接實作（給等價性測試）：FFT 上取樣到 cc_fs，與參考正弦做循環互相關。"""
    from scipy.signal import resample
    n_up = int(round(len(x_epoch) * cc_fs / fs))
    xu = resample(x_epoch - np.mean(x_epoch), n_up)
    tu = t_epoch[0] + np.arange(n_up) / cc_fs
    half = int(cc_fs / f / 2)
    lags = np.arange(-half + 1, half + 1)
    best, best_l = -np.inf, 0
    for L in lags:
        ref = np.sin(2 * np.pi * f * (tu - L / cc_fs))
        c = float(np.dot(xu, ref))
        if c > best:
            best, best_l = c, L
    return best_l / cc_fs


def wrap(x, period=1.0 / F_REF):
    return (x + period / 2) % period - period / 2


def paper_rse(nodes, T=3600.0, seg_s=600.0, method="linear", truth=None):
    """完整照 §IV-B。

    nodes：list of dict，每個節點要有
        sample_tp  每個樣本的 t_P（節點時鐘，s）
        sample_val 訊號值
        sample_ts  每個樣本的 t̂_S（演算法輸出；NaN = 無效）
        sample_true_pc（選配）樣本真正被取樣時的 PC 時鐘讀數 → 產生真值版 RSE
    回傳 dict(seg=[...], rse_cc=..., rse_true=...)。
    """
    lo = max(np.nanmin(n["sample_tp"]) for n in nodes)
    hi = min(np.nanmax(n["sample_tp"]) for n in nodes)
    lo = np.ceil(lo)
    grid = np.arange(lo, hi, 1.0 / FS)
    n_ep = len(grid) // int(FS)
    rs = []
    for n in nodes:
        v, s = resample_node(n["sample_tp"], n["sample_val"], n["sample_ts"], grid, method)
        # 互相關 lag 的定義是「訊號落後參考」：s(g) = ref(g − lag) → lag = g − t_true(g)。
        # τ̂_nm = t̂_Sn − t̂_Sm ≈ t_n − t_m，所以 τ 要取 −lag 才同號（論文沒寫 lag 的正負號約定；照字面會變成兩倍而非相消）。
        tau = -cc_delay_epochs(grid, v)
        s_ep = s[: n_ep * int(FS)].reshape(n_ep, int(FS))
        tau_hat = np.mean(s_ep - grid[: n_ep * int(FS)].reshape(n_ep, int(FS)), axis=1)  # δT̂ 的 epoch 平均
        rec = dict(tau=tau, dhat=tau_hat)
        if "sample_true_pc" in n:
            _, tt = resample_node(n["sample_tp"], n["sample_val"], n["sample_true_pc"], grid, method)
            rec["true_off"] = np.mean(tt[: n_ep * int(FS)].reshape(n_ep, int(FS))
                                      - grid[: n_ep * int(FS)].reshape(n_ep, int(FS)), axis=1)
        rs.append(rec)
    ep_t = grid[: n_ep * int(FS)].reshape(n_ep, int(FS))[:, 0] - lo
    pairs = [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    rse_cc = {}
    rse_true = {}
    for i, j in pairs:
        tau_hat = rs[i]["dhat"] - rs[j]["dhat"]                      # Eq.(23)(24)
        tau = rs[i]["tau"] - rs[j]["tau"]                            # Eq.(25)
        rse_cc[(i, j)] = wrap(tau_hat - tau)                         # Eq.(26)
        if "true_off" in rs[i]:
            rse_true[(i, j)] = tau_hat - (rs[i]["true_off"] - rs[j]["true_off"])
    return dict(epoch_t=ep_t, rse_cc=rse_cc, rse_true=rse_true, pairs=pairs,
                seg=worst_pair_metrics(ep_t, rse_cc, seg_s, T),
                seg_true=worst_pair_metrics(ep_t, rse_true, seg_s, T) if rse_true else None)


def worst_pair_metrics(ep_t, rse, seg_s=600.0, T=3600.0):
    """Eq.(27)：每段取 P95(|RSE|) 最大的那一對 → mean|RSE|、std、P95|RSE|（ms）。"""
    out = []
    for k in range(int(round(T / seg_s))):
        m = (ep_t >= k * seg_s) & (ep_t < (k + 1) * seg_s)
        best, best_p = None, -1
        for p, r in rse.items():
            v = r[m]
            v = v[np.isfinite(v)]
            if len(v) < 10:
                continue
            p95 = np.percentile(np.abs(v), 95)
            if p95 > best_p:
                best_p, best = p95, (p, v)
        if best is None:
            out.append(dict(seg=k + 1, absavg=np.nan, std=np.nan, p95=np.nan, pair=None))
            continue
        v = best[1] * 1e3
        out.append(dict(seg=k + 1, absavg=float(np.mean(np.abs(v))), std=float(np.std(v)),
                        p95=float(np.percentile(np.abs(v), 95)), pair=list(best[0]),
                        mean=float(np.mean(v))))
    return out

"""
先前方法（規格 §6）。全部是「線上」版本：第 i 包的輸出只用到第 0..i 包。

  tp_only      t̂_S = t_P + 第一包的 δT（只修 offset）
  tc_only      t̂_S = t_C（主機打時間；含全部延遲抖動）
  ols_raw      對全部 (t_P, δT) 做遞迴 OLS                                  [25]
  ols_window   最近 W 秒的 OLS                                              FTSP 風格 [9]
  ols_trimmed  線上剔除殘差 > kσ 的點再擬合
  min_filter   每 W 秒取 δT 最小值，對這些最小值做 OLS
  lower_hull   (t_P, δT) 的下凸包；LP 解 = 在所有點下方、Σ距離最小的直線
               （Moon, Skelly, Towsley 1999 的 clock-skew 估計；線上用單調鏈維護凸包）
"""
from __future__ import annotations

import numpy as np
from numba import njit


def tp_only(tP, tC, **_):
    off = tC[0] - tP[0]
    dhat = np.full(tP.size, off)
    return dict(ts=tP + dhat, dhat=dhat, valid=np.ones(tP.size, bool))


def tc_only(tP, tC, **_):
    return dict(ts=tC.astype(float).copy(), dhat=tC - tP, valid=np.ones(tP.size, bool))


@njit(cache=True)
def _ols_raw(x, y):
    n = x.size
    out = np.empty(n)
    valid = np.zeros(n, np.bool_)
    x0 = x[0]
    S = 0.0
    Sx = 0.0
    Sy = 0.0
    Sxx = 0.0
    Sxy = 0.0
    a = 0.0
    b = y[0]
    for i in range(n):
        xi = x[i] - x0
        S += 1
        Sx += xi
        Sy += y[i]
        Sxx += xi * xi
        Sxy += xi * y[i]
        det = S * Sxx - Sx * Sx
        if det > 1e-12:
            a = (S * Sxy - Sx * Sy) / det
            b = (Sy - a * Sx) / S
            valid[i] = True
        else:
            a = 0.0
            b = Sy / S
            valid[i] = i > 0
        out[i] = a * xi + b
    return out, valid


def ols_raw(tP, tC, **_):
    dhat, valid = _ols_raw(tP.astype(float), (tC - tP).astype(float))
    return dict(ts=tP + dhat, dhat=dhat, valid=valid)


@njit(cache=True)
def _ols_window(x, y, W):
    n = x.size
    out = np.empty(n)
    valid = np.zeros(n, np.bool_)
    x0 = x[0]
    S = 0.0
    Sx = 0.0
    Sy = 0.0
    Sxx = 0.0
    Sxy = 0.0
    j = 0
    for i in range(n):
        xi = x[i] - x0
        S += 1
        Sx += xi
        Sy += y[i]
        Sxx += xi * xi
        Sxy += xi * y[i]
        while x[j] < x[i] - W:
            xj = x[j] - x0
            S -= 1
            Sx -= xj
            Sy -= y[j]
            Sxx -= xj * xj
            Sxy -= xj * y[j]
            j += 1
        det = S * Sxx - Sx * Sx
        if S >= 3 and det > 1e-9 * S * S:
            a = (S * Sxy - Sx * Sy) / det
            b = (Sy - a * Sx) / S
            valid[i] = True
        else:
            a = 0.0
            b = Sy / S
        out[i] = a * xi + b
    return out, valid


def ols_window(tP, tC, W=60.0, **_):
    dhat, valid = _ols_window(tP.astype(float), (tC - tP).astype(float), W)
    return dict(ts=tP + dhat, dhat=dhat, valid=valid)


@njit(cache=True)
def _ols_trimmed(x, y, k, warm):
    n = x.size
    out = np.empty(n)
    valid = np.zeros(n, np.bool_)
    x0 = x[0]
    S = 0.0
    Sx = 0.0
    Sy = 0.0
    Sxx = 0.0
    Sxy = 0.0
    a = 0.0
    b = y[0]
    var = 0.0
    nacc = 0
    for i in range(n):
        xi = x[i] - x0
        r = y[i] - (a * xi + b)
        accept = i < warm or abs(r) <= k * np.sqrt(var)
        if accept:
            S += 1
            Sx += xi
            Sy += y[i]
            Sxx += xi * xi
            Sxy += xi * y[i]
            nacc += 1
            var = var + (r * r - var) / nacc if nacc < 200 else 0.995 * var + 0.005 * r * r
        det = S * Sxx - Sx * Sx
        if det > 1e-12:
            a = (S * Sxy - Sx * Sy) / det
            b = (Sy - a * Sx) / S
            valid[i] = True
        out[i] = a * xi + b
    return out, valid


def ols_trimmed(tP, tC, k=2.0, warm=50, **_):
    dhat, valid = _ols_trimmed(tP.astype(float), (tC - tP).astype(float), k, warm)
    return dict(ts=tP + dhat, dhat=dhat, valid=valid)


@njit(cache=True)
def _min_filter(x, y, W):
    n = x.size
    out = np.empty(n)
    valid = np.zeros(n, np.bool_)
    x0 = x[0]
    S = 0.0
    Sx = 0.0
    Sy = 0.0
    Sxx = 0.0
    Sxy = 0.0
    a = 0.0
    b = y[0]
    win_start = x[0]
    cur_min = np.inf
    cur_x = 0.0
    have_fit = False
    for i in range(n):
        if x[i] >= win_start + W:
            if np.isfinite(cur_min):
                xi = cur_x - x0
                S += 1
                Sx += xi
                Sy += cur_min
                Sxx += xi * xi
                Sxy += xi * cur_min
                det = S * Sxx - Sx * Sx
                if det > 1e-12:
                    a = (S * Sxy - Sx * Sy) / det
                    b = (Sy - a * Sx) / S
                    have_fit = True
                else:
                    a = 0.0
                    b = Sy / S
            while x[i] >= win_start + W:
                win_start += W
            cur_min = np.inf
        if y[i] < cur_min:
            cur_min = y[i]
            cur_x = x[i]
        if S == 0:
            out[i] = cur_min
        else:
            out[i] = a * (x[i] - x0) + b
        valid[i] = have_fit
    return out, valid


def min_filter(tP, tC, W=10.0, **_):
    dhat, valid = _min_filter(tP.astype(float), (tC - tP).astype(float), W)
    return dict(ts=tP + dhat, dhat=dhat, valid=valid)


@njit(cache=True)
def _lower_hull(x, y):
    """線上下凸包（x 遞增；單調鏈）。LP 最佳線 = 支撐下凸包於 x̄ 的邊。"""
    n = x.size
    out = np.empty(n)
    valid = np.zeros(n, np.bool_)
    hx = np.empty(n)
    hy = np.empty(n)
    h = 0
    x0 = x[0]
    sx = 0.0
    for i in range(n):
        xi = x[i] - x0
        yi = y[i]
        sx += xi
        add = True
        if h > 0 and xi <= hx[h - 1]:
            if yi < hy[h - 1]:
                h -= 1          # 重複 / 倒退的 t_P：只留較低的點
            else:
                add = False
        if add:
            while h >= 2:
                cross = (hx[h - 1] - hx[h - 2]) * (yi - hy[h - 2]) - (hy[h - 1] - hy[h - 2]) * (xi - hx[h - 2])
                if cross <= 0:
                    h -= 1
                else:
                    break
            hx[h] = xi
            hy[h] = yi
            h += 1
        if h == 1:
            out[i] = hy[0]
            continue
        xbar = sx / (i + 1)
        lo = 0
        hi = h - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if hx[mid] <= xbar:
                lo = mid
            else:
                hi = mid
        a = (hy[lo + 1] - hy[lo]) / (hx[lo + 1] - hx[lo])
        b = hy[lo] - a * hx[lo]
        out[i] = a * xi + b
        valid[i] = True
    return out, valid


def lower_hull(tP, tC, **_):
    dhat, valid = _lower_hull(tP.astype(float), (tC - tP).astype(float))
    return dict(ts=tP + dhat, dhat=dhat, valid=valid)

"""
SharkTooth（論文 Algorithm 1 + 2，Eq.(3)–(22)）。

兩個實作：
  SharkTooth        純 Python 參考版，逐行對照論文（易讀、給實測資料即時用）
  sharktooth_batch  numba 版，數值上與參考版相同（tests/test_sharktooth.py 比對），給大量模擬用

選項（規格 T5 / T9）：
  M          Eq.(7) 視窗長度
  lam_max    Eq.(13) 最大權重 0.5
  g1_inc     Eq.(14) γ1 增量 1e-6
  settle_s   Algorithm 1 穩定期 30 s
  keep_last  穩定期輸出：False = 論文字面（α = β = 0）；True = 保留上一組 α、β
  forget     遞迴 OLS 的遺失因子（1.0 = 論文；0.999 / 0.9999 = T5 加做）
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


class SharkTooth:
    def __init__(self, CI, M=100, lam_max=0.5, g1_inc=1e-6, settle_s=30.0, keep_last=False, forget=1.0):
        self.CI, self.M, self.lam_max, self.g1_inc = CI, M, lam_max, g1_inc
        self.settle, self.keep_last, self.forget = settle_s, keep_last, forget
        # Eq.(21)
        self.dT0, self.tP0, self.g0 = math.inf, 0.0, 0.0
        # Eq.(22)
        self.dT1, self.tP1, self.g1 = math.inf, 0.0, math.inf
        self.prev_dT = None
        self.prev_star = None
        self.d2 = []
        self.A = np.zeros((2, 2))
        self.b = np.zeros(2)
        self.alpha = self.beta = 0.0
        self.valid = False
        self.n_pivot = 0

    def step(self, tP, tC):
        # Algorithm 2 -------------------------------------------------------
        dT = tC - tP                                           # Eq.(3)
        if self.prev_dT is not None:
            self.d2.append(dT - self.prev_dT)                  # Eq.(5)
            if len(self.d2) > self.M:
                self.d2.pop(0)
        self.prev_dT = dT
        sigma = float(np.std(self.d2)) if len(self.d2) >= 2 else 0.0      # Eq.(7)
        lam0 = 1.0 if sigma == 0 else min(self.CI / (2 * sigma), 1.0)     # Eq.(10)

        if dT < self.dT0:                                      # Eq.(11) 新樞紐點
            self.dT0, self.g0, self.tP0, self.g1 = dT, 0.0, tP, math.inf
            self.n_pivot += 1
        else:
            dtp1 = tP - self.tP1
            thr = self.dT1 + self.g1 * dtp1 if dtp1 != 0 else self.dT1   # Eq.(9)，避開 ∞×0
            if dT < thr:                                       # Eq.(12)
                star_prev = self.prev_star if self.prev_star is not None else dT
                lam = self.lam_max * lam0 if dT >= star_prev else self.lam_max   # Eq.(13)
                if tP != self.tP0:
                    self.g0 = (1 - lam) * self.g0 + lam * (dT - self.dT0) / (tP - self.tP0)
                self.g1 = self.g0
                self.dT1 = self.dT0 + self.g0 * (tP - self.tP0)   # δT1 = δT*^i
                self.tP1 = tP
            else:
                self.g1 = self.g1 + self.g1_inc                # Eq.(14)
        star = self.dT0 + self.g0 * (tP - self.tP0)            # Eq.(8)
        self.prev_star = star

        # Algorithm 1 -------------------------------------------------------
        if tP > self.tP0 + self.settle:
            x = tP - self.tP0                                  # 以 tP0 為原點（數值穩定，等價）
            self.A = self.forget * self.A + np.array([[1.0, -x], [-x, x * x]])   # Eq.(16)
            self.b = self.forget * self.b + np.array([x * star, star])           # Eq.(17)
            det = self.A[0, 0] * self.A[1, 1] - self.A[0, 1] * self.A[1, 0]
            if det > 0:
                a, b0 = (self.A @ self.b) / det                # Eq.(15)
                self.alpha, self.beta = a, b0 - a * self.tP0
                self.valid = True
        else:                                                  # Eq.(18)
            self.A[:] = 0.0
            self.b[:] = 0.0
            if not self.keep_last:
                self.alpha = self.beta = 0.0
                self.valid = False
        dhat = self.alpha * tP + self.beta                     # Eq.(19)
        return dict(dT=dT, star=star, dhat=dhat, tS=tP + dhat, valid=self.valid,
                    alpha=self.alpha, beta=self.beta)          # Eq.(20)


@njit(cache=True)
def sharktooth_batch(tP, tC, CI, M, lam_max, g1_inc, settle, keep_last, forget):
    n = tP.size
    dT_out = np.empty(n)
    star_out = np.empty(n)
    dhat_out = np.empty(n)
    alpha_out = np.empty(n)
    beta_out = np.empty(n)
    valid_out = np.zeros(n, np.bool_)
    pivot_out = np.zeros(n, np.bool_)
    dT0 = np.inf
    tP0 = 0.0
    g0 = 0.0
    dT1 = np.inf
    tP1 = 0.0
    g1 = np.inf
    ring = np.zeros(M)
    nr = 0
    pos = 0
    prev_dT = 0.0
    have_prev = False
    prev_star = 0.0
    have_star = False
    A00 = 0.0
    A01 = 0.0
    A11 = 0.0
    b0 = 0.0
    b1 = 0.0
    alpha = 0.0
    beta = 0.0
    valid = False
    n_pivot = 0
    for i in range(n):
        dT = tC[i] - tP[i]
        if have_prev:
            ring[pos] = dT - prev_dT
            pos = (pos + 1) % M
            if nr < M:
                nr += 1
        prev_dT = dT
        have_prev = True
        sigma = 0.0
        if nr >= 2:
            mu = 0.0
            for k in range(nr):
                mu += ring[k]
            mu /= nr
            v = 0.0
            for k in range(nr):
                v += (ring[k] - mu) ** 2
            sigma = math.sqrt(v / nr)
        lam0 = 1.0 if sigma == 0 else min(CI / (2 * sigma), 1.0)
        if dT < dT0:
            dT0 = dT
            g0 = 0.0
            tP0 = tP[i]
            g1 = np.inf
            n_pivot += 1
            pivot_out[i] = True
        else:
            dtp1 = tP[i] - tP1
            thr = dT1 + g1 * dtp1 if dtp1 != 0 else dT1
            if dT < thr:
                sp = prev_star if have_star else dT
                lam = lam_max * lam0 if dT >= sp else lam_max
                if tP[i] != tP0:
                    g0 = (1 - lam) * g0 + lam * (dT - dT0) / (tP[i] - tP0)
                g1 = g0
                dT1 = dT0 + g0 * (tP[i] - tP0)
                tP1 = tP[i]
            else:
                g1 = g1 + g1_inc
        star = dT0 + g0 * (tP[i] - tP0)
        prev_star = star
        have_star = True
        if tP[i] > tP0 + settle:
            x = tP[i] - tP0
            A00 = forget * A00 + 1.0
            A01 = forget * A01 - x
            A11 = forget * A11 + x * x
            b0 = forget * b0 + x * star
            b1 = forget * b1 + star
            det = A00 * A11 - A01 * A01
            if det > 0:
                a = (A00 * b0 + A01 * b1) / det
                c0 = (A01 * b0 + A11 * b1) / det
                alpha = a
                beta = c0 - a * tP0
                valid = True
        else:
            A00 = 0.0
            A01 = 0.0
            A11 = 0.0
            b0 = 0.0
            b1 = 0.0
            if not keep_last:
                alpha = 0.0
                beta = 0.0
                valid = False
        dT_out[i] = dT
        star_out[i] = star
        dhat_out[i] = alpha * tP[i] + beta
        alpha_out[i] = alpha
        beta_out[i] = beta
        valid_out[i] = valid
    return dT_out, star_out, dhat_out, alpha_out, beta_out, valid_out, pivot_out


def run(tP, tC, CI=0.030, M=100, lam_max=0.5, g1_inc=1e-6, settle_s=30.0, keep_last=False, forget=1.0,
        tilt_ppm=0.0, **_):
    """tilt_ppm（T7e 的修正，非論文）：先把 δT 加上已知的正斜率 tilt·t_P 再跑，輸出時扣回。
    讓 α + tilt > 0，避開負漂移時「每個新低點都是新樞紐點」的問題。"""
    tP = np.ascontiguousarray(tP, float)
    tC = np.ascontiguousarray(tC, float)
    k = tilt_ppm * 1e-6
    tCt = tC + k * (tP - tP[0]) if k else tC
    dT, star, dhat, a, b, valid, piv = sharktooth_batch(tP, tCt, CI, int(M), lam_max, g1_inc, settle_s,
                                                        keep_last, forget)
    if k:
        dhat = dhat - k * (tP - tP[0])
        a = a - k
        b = b + k * tP[0]
        dT = dT - k * (tP - tP[0])
        star = star - k * (tP - tP[0])
    return dict(ts=tP + dhat, dhat=dhat, dT=dT, star=star, alpha=a, beta=b, valid=valid, pivot=piv,
                n_pivot=int(piv.sum()))

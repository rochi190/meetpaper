"""SharkTooth 實作：逐字轉錄自 claude/SharkTooth 導讀教材.pdf 第 7 部分 7.2（p.47–50）。

只修正 PDF 排版造成的縮排與斷行，邏輯一行都沒改，目的是測試「教材裡的程式」本身。
單位一律是秒。每個節點一個物件。
"""
import numpy as np
from collections import deque


class SharkTooth:
    def __init__(self, CI, M=100, lam_max=0.5, gamma_step=1e-6, settle=30.0,
                 hold_during_settling=False):
        self.CI, self.M = CI, M                 # CI: 連線間隔 [s]；M: Eq.(7) 視窗（論文未明說，推測 100）
        self.lam_max, self.step, self.settle = lam_max, gamma_step, settle
        self.hold = hold_during_settling        # False = 照 Eq.(18) 字面：穩定期 α=β=0
        self.dT0, self.tP0, self.g0 = np.inf, 0.0, 0.0          # Eq.(21)
        self.dT1, self.tP1, self.g1 = np.inf, 0.0, np.inf       # Eq.(22)
        self.prev_dT = None
        self.prev_dTs = None
        self.d2 = deque(maxlen=M)
        self.A = np.zeros((2, 2))
        self.b = np.zeros(2)
        self.alpha, self.beta = 0.0, 0.0
        self.n_pivots = 0

    # ---------- Algorithm 2：不可預測延遲修正 ----------
    def _delay_correction(self, tP, tC):
        dT = tC - tP                                                     # Eq.(3)
        if self.prev_dT is not None:
            self.d2.append(dT - self.prev_dT)                            # Eq.(5)
        self.prev_dT = dT
        sigma = float(np.std(self.d2)) if len(self.d2) > 1 else 0.0      # Eq.(7)
        lam0 = 1.0 if sigma == 0.0 else min(self.CI / (2 * sigma), 1.0)  # Eq.(10)

        if dT < self.dT0:                                                # Eq.(11)
            self.dT0, self.g0, self.tP0, self.g1 = dT, 0.0, tP, np.inf
            self.n_pivots += 1
        else:
            thr = self.dT1 + self.g1 * (tP - self.tP1)                   # Eq.(9)
            if dT < thr:                                                 # Eq.(12)
                lam = self.lam_max * lam0 if dT >= self.prev_dTs else self.lam_max  # Eq.(13)
                self.g0 = (1 - lam) * self.g0 + lam * (dT - self.dT0) / (tP - self.tP0)
                self.g1 = self.g0
                self.dT1 = self.dT0 + self.g0 * (tP - self.tP0)          # = δT_*^i
                self.tP1 = tP
            else:
                self.g1 += self.step                                     # Eq.(14)
        dTs = self.dT0 + self.g0 * (tP - self.tP0)                       # Eq.(8)
        self.prev_dTs = dTs
        return dT, dTs

    # ---------- Algorithm 1：一次迭代 ----------
    def update(self, tP, tC):
        dT, dTs = self._delay_correction(tP, tC)
        if tP > self.tP0 + self.settle:                                  # Alg.1 line 5
            r = np.array([[1.0, -tP], [-tP, tP * tP]])                   # Eq.(16)
            v = np.array([tP * dTs, dTs])                                # Eq.(17)
            if self.A[0, 0] == 0:
                self.A, self.b = r, v
            else:
                self.A, self.b = self.A + r, self.b + v
            det = self.A[0, 0] * self.A[1, 1] - self.A[0, 1] * self.A[1, 0]
            if det != 0:                                                 # Eq.(15)
                self.alpha, self.beta = (self.A @ self.b) / det
        else:                                                            # Eq.(18)
            self.A = np.zeros((2, 2)); self.b = np.zeros(2)
            if not self.hold:
                self.alpha, self.beta = 0.0, 0.0
        dT_hat = self.alpha * tP + self.beta                             # Eq.(19)
        return tP + dT_hat, dT, dTs                                      # Eq.(20)


class SharkToothBiased:
    """【推測】第 6 部分 6.2-1 的修正：先加斜率 B 讓趨勢為正，再減回。非論文內容。"""

    def __init__(self, CI, B=100e-6, **kw):
        self.B, self.core = B, SharkTooth(CI, hold_during_settling=True, **kw)

    def update(self, tP, tC):
        tS_hat, dT, dTs = self.core.update(tP, tC + self.B * tP)
        return tS_hat - self.B * tP


if __name__ == "__main__":
    # 用法：每個節點一個物件，每收到一個封包就呼叫一次
    st = SharkTooth(CI=0.030)
    for tP, tC in [(0.0, 0.305), (1.0, 1.3252), (2.0, 2.3154), (3.0, 3.3056)]:
        tS_hat, dT, dTs = st.update(tP, tC)
        print(f"tP={tP:4.1f} dT={dT*1e3:7.2f} ms dT*={dTs*1e3:7.3f} ms tS_hat={tS_hat:.4f}")

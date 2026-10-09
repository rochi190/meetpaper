"""SharkTooth 實作：逐字轉錄自 gemini/SharkTooth (IEEE IoT-J 2025) 導讀教材 第 7 部分。

只修正 PDF 排版造成的縮排，邏輯沒改，用來和論文 Eq.(11)(12)(21)(22) 對照。
"""
import numpy as np


class SharkToothSync:
    def __init__(self, ci_ms=30.0, m_window=100, settling_sec=30.0):
        self.CI = ci_ms / 1000.0
        self.M = m_window
        self.settling_sec = settling_sec

        self.delta_T_prev = None
        self.d2_T_buffer = []

        self.delta_T_0 = float('inf')
        self.t_P0 = 0.0
        self.gamma_0 = 0.0

        self.delta_T_1 = float('inf')
        self.t_P1 = 0.0
        self.gamma_1 = 1e-8

        self.delta_T_star_prev = 0.0

        self.A = np.zeros((2, 2))
        self.b = np.zeros((2, 1))
        self.alpha = 0.0
        self.beta = 0.0
        self.n_pivots = 0  # 額外加的計數器，只供測試用

    def update(self, t_P, t_C):
        delta_T = t_C - t_P

        if self.delta_T_prev is not None:
            d2_T = delta_T - self.delta_T_prev
            self.d2_T_buffer.append(d2_T)
            if len(self.d2_T_buffer) > self.M:
                self.d2_T_buffer.pop(0)
            sigma_M = np.std(self.d2_T_buffer) if len(self.d2_T_buffer) > 1 else (self.CI / 2.0)
        else:
            sigma_M = self.CI / 2.0
        self.delta_T_prev = delta_T

        lambda_0 = min(self.CI / (2.0 * max(sigma_M, 1e-6)), 1.0)

        if delta_T < self.delta_T_0:
            self.delta_T_0 = delta_T
            self.t_P0 = t_P
            self.gamma_0 = 0.0
            self.gamma_1 = 1e-8
            self.A.fill(0.0)
            self.b.fill(0.0)
            self.alpha = 0.0
            self.beta = 0.0
            self.n_pivots += 1  # 測試用
        else:
            delta_T_thr = self.delta_T_1 + self.gamma_1 * (t_P - self.t_P1)
            if delta_T < delta_T_thr:
                lam = 0.5 * lambda_0 if delta_T >= self.delta_T_star_prev else 0.5
                denom = max(t_P - self.t_P0, 1e-6)
                self.gamma_0 = (1.0 - lam) * self.gamma_0 + lam * ((delta_T - self.delta_T_0) / denom)
                self.gamma_1 = 1e-8
                self.delta_T_1 = delta_T
                self.t_P1 = t_P
            else:
                self.gamma_1 += 1e-6

        delta_T_star = self.delta_T_0 + self.gamma_0 * (t_P - self.t_P0)
        self.delta_T_star_prev = delta_T_star

        if t_P > (self.t_P0 + self.settling_sec):
            dA = np.array([[1.0, -t_P], [-t_P, t_P**2]])
            db = np.array([[t_P * delta_T_star], [delta_T_star]])
            self.A += dA
            self.b += db

            det = self.A[0, 0] * self.A[1, 1] - self.A[0, 1] * self.A[1, 0]
            if abs(det) > 1e-9:
                res = np.dot(self.A, self.b) / det
                self.alpha = res[0, 0]
                self.beta = res[1, 0]

        delta_T_hat = self.alpha * t_P + self.beta if abs(self.alpha) > 0 else delta_T_star
        t_S_hat = t_P + delta_T_hat
        # 額外回傳 δT* 以便測試（原程式只回傳 t_S_hat）
        self.last_dTs = delta_T_star
        return t_S_hat

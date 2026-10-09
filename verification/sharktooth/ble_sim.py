"""自寫的 BLE 連線模式時序模擬器，用來檢查教材第 5、6 部分「模擬」結論的機制。

教材沒有公開它的通道模型參數，所以這裡的數值不可能逐位對上；只看機制與量級是否成立。
模型（全部用真實時間 t，PC 時鐘 = 真實時間）：
  節點時鐘   t_P = (1 + ρ_n) t + θ_n，每 P（節點時間）產生一個封包，t_P = 最後一筆樣本時間
  連線事件   e_j = anchor_n + j · CI / (1 + ρ_d)（dongle 時鐘）
  等待       封包在下一個連線事件送出；每次嘗試以機率 p_loss 失敗，失敗就等下一個事件
  到 PC     t_C = 事件時間 + d_min + 指數分布 USB/OS 抖動 (+ 機率 p_spike 的軟體尖峰)
因為 PC 時鐘就是真實時間，正確答案 t_S = t（產生時刻），δT 的真實斜率 α ≈ −ρ_n。
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class Channel:
    p_loss: float = 0.02
    p_spike: float = 0.01
    spike_lo: float = 0.020
    spike_hi: float = 0.120
    d_min: float = 0.002
    jitter_mean: float = 0.0002


def simulate_node(rng, rho_n, theta_n, rho_d, anchor, ch: Channel,
                  P=0.100, CI=0.030, duration=3600.0):
    """回傳 (t_P, t_C, t_true)，皆為 numpy 陣列。"""
    n = int(duration / P * (1 + rho_n)) - 1
    tP0 = theta_n + 0.05  # 節點開機後第一筆
    tP = tP0 + np.arange(n) * P
    t_true = (tP - theta_n) / (1 + rho_n)
    CI_true = CI / (1 + rho_d)
    j = np.ceil((t_true - anchor) / CI_true)
    k = rng.geometric(1 - ch.p_loss, n) - 1  # 重傳次數
    t_event = anchor + (j + k) * CI_true
    jitter = rng.exponential(ch.jitter_mean, n)
    spike = np.where(rng.random(n) < ch.p_spike,
                     rng.uniform(ch.spike_lo, ch.spike_hi, n), 0.0)
    tC = t_event + ch.d_min + jitter + spike
    return tP, tC, t_true


class DirectOLS:
    """消融對照：不做延遲修正，直接對 δT 做遞迴 OLS（教材第 5 部分 5.5 的「直接 OLS」）。"""

    def __init__(self):
        self.A = np.zeros((2, 2)); self.b = np.zeros(2)
        self.alpha = self.beta = 0.0

    def update(self, tP, tC):
        dT = tC - tP
        self.A += np.array([[1.0, -tP], [-tP, tP * tP]])
        self.b += np.array([tP * dT, dT])
        det = self.A[0, 0] * self.A[1, 1] - self.A[0, 1] * self.A[1, 0]
        if det != 0:
            self.alpha, self.beta = (self.A @ self.b) / det
        return tP + self.alpha * tP + self.beta


def run_sync(sync, tP, tC):
    out = np.empty_like(tP)
    for i in range(len(tP)):
        r = sync.update(tP[i], tC[i])
        out[i] = r[0] if isinstance(r, tuple) else r
    return out


def worst_pair_by_segment(errors, t_trues, n_seg=6, seg_len=600):
    """教材 Eq.(24)(27)：每秒平均 → 每對 RSE → 每 600 s 取 P95|RSE| 最大的一對 → 回傳其平均 |RSE|（秒）。"""
    K = n_seg * seg_len
    per_epoch = []
    for e, t in zip(errors, t_trues):
        k = np.floor(t).astype(int)
        keep = (k >= 0) & (k < K)
        s = np.bincount(k[keep], weights=e[keep], minlength=K)
        c = np.bincount(k[keep], minlength=K)
        per_epoch.append(s / np.maximum(c, 1))
    out = []
    for s in range(n_seg):
        seg = slice(s * seg_len, (s + 1) * seg_len)
        best_val, best_mean = -1, None
        for a, b in combinations(range(len(per_epoch)), 2):
            r = per_epoch[a][seg] - per_epoch[b][seg]
            p95 = np.percentile(np.abs(r), 95)
            if p95 > best_val:
                best_val, best_mean = p95, np.mean(np.abs(r))
        out.append(best_mean)
    return np.array(out)

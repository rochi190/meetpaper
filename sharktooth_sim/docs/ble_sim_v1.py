"""
BLE 多節點時間戳模擬 + SharkTooth 演算法（依論文 Eq.(3)–(22) 重寫）

用途：產生 (t_P, t_C) 序列，檢查是否能重現論文 Fig.3 / Fig.4 / Fig.6 的樣子，
並觀察各種 BLE 延遲／掉包／緩衝區錯誤對演算法的影響。

時間基準：以 PC 的時鐘當作「真實時間」t（單位：秒）。
所有參數都是自訂的示意值，不是論文量到的數字。
"""
import numpy as np
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. 參數
# ---------------------------------------------------------------------------
@dataclass
class Cfg:
    T: float = 3600.0          # 擷取長度 (s)
    # --- 節點時鐘 (Eq.1) ---
    rho_ppm: float = -200.0    # 節點晶振頻率誤差；負 = 節點走得慢 → δT 往上飄
    theta: float = -1.0        # 節點開機偏移 (s)：t_P = (1+ρ) t + θ
    tick_hz: float = 32768.0   # 節點計時器解析度（32.768 kHz → 30.5 μs）
    # --- 取樣與打包 ---
    fs: float = 50.0           # 節點取樣率 (Hz)（論文驗證用 50 Hz）
    pkt_period: float = 0.1    # 每 100 ms（節點時鐘）送一包
    fw_delay_max: float = 0.5e-3   # 韌體打包、放進 TX 佇列的延遲上限 (s)
    buf_err_prob: float = 0.0  # 抓 buffer 少抓/多抓一筆的機率 → t_P 對應到錯的樣本
    # --- BLE 連線 ---
    CI: float = 0.030          # connection interval (s)
    anchor: float = None       # 這個節點 connection event 的相位；None = 隨機
    per_good: float = 0.0      # 通道「好」狀態的封包錯誤率（每次嘗試）
    per_bad: float = 0.0       # 通道「壞」狀態的封包錯誤率（干擾、身體遮蔽）
    p_g2b: float = 0.0         # 每個 event 由好轉壞的機率（Gilbert–Elliott 模型）
    p_b2g: float = 0.2         # 每個 event 由壞轉好的機率
    tx_queue: int = 8          # 節點 TX 佇列容量；滿了 → 新封包被丟掉（真正的掉包）
    disconnect: tuple = ()     # [(開始 s, 長度 s), ...] 斷線期間佇列清空、資料遺失
    # --- dongle / USB / PC ---
    dongle_delay: float = 0.3e-3   # dongle 處理 (s)
    usb_frame: float = 1e-3        # USB full-speed 每 1 ms 輪詢一次 → 到達時間被量化
    os_median: float = 0.4e-3      # 作業系統 + Python callback 延遲中位數 (s)
    os_sigma: float = 0.6          # lognormal 的寬度
    spike_prob: float = 0.0        # 偶發的大延遲（GC、執行緒排程、其他程式）機率
    spike_max: float = 0.08        # 大延遲上限 (s)
    seed: int = 0


# ---------------------------------------------------------------------------
# 2. 產生一個節點的 (t_P, t_C, packet_id)
# ---------------------------------------------------------------------------
def simulate_node(c: Cfg):
    rng = np.random.default_rng(c.seed)
    rho = c.rho_ppm * 1e-6
    n_pkt = int(c.T / c.pkt_period)
    k = np.arange(1, n_pkt + 1)

    # 節點時鐘到 k·P 時打包（最後一筆樣本的本地時間）
    tP_ideal = k * c.pkt_period
    t_gen = (tP_ideal - c.theta) / (1 + rho)          # 對應的真實時間

    # 計時器量化 + buffer 抓錯（t_P 對應到前一筆或後一筆樣本）
    tP = np.floor(tP_ideal * c.tick_hz) / c.tick_hz
    bad = rng.random(n_pkt) < c.buf_err_prob
    tP = tP + bad * rng.choice([-1, 1], n_pkt) / c.fs

    t_ready = t_gen + rng.uniform(0, c.fw_delay_max, n_pkt)

    # connection events：anchor + j·CI；通道狀態用兩態馬可夫鏈
    anchor = rng.uniform(0, c.CI) if c.anchor is None else c.anchor
    n_ev = int(c.T / c.CI) + 200
    state = np.zeros(n_ev, dtype=bool)   # True = 壞
    s = False
    u = rng.random(n_ev)
    for j in range(n_ev):
        s = (u[j] < c.p_g2b) if not s else not (u[j] < c.p_b2g)
        state[j] = s
    per = np.where(state, c.per_bad, c.per_good)
    ok = rng.random(n_ev) >= per                        # 該次 event 傳送是否成功

    def in_disc(t):
        return any(a <= t < a + d for a, d in c.disconnect)

    t_air = np.full(n_pkt, np.nan)      # 成功送達 dongle 的時間
    departures = []                     # 已送出的時間，用來算佇列長度
    next_ev_free = 0                    # 一個 event 只送一包（244 B 需要 7.5 ms）
    head = 0
    for i in range(n_pkt):
        tr = t_ready[i]
        # 佇列裡還沒送出的封包數
        while head < len(departures) and departures[head] <= tr:
            head += 1
        if len(departures) - head >= c.tx_queue or in_disc(tr):
            continue                                    # 丟包
        j = max(int(np.ceil((tr - anchor) / c.CI)), next_ev_free)
        while j < n_ev and (not ok[j] or in_disc(anchor + j * c.CI)):
            if in_disc(anchor + j * c.CI):             # 斷線：佇列清空
                j = n_ev; break
            j += 1                                      # 失敗 → 下一個 event 重傳
        if j >= n_ev:
            continue
        t_air[i] = anchor + j * c.CI
        departures.append(t_air[i])
        next_ev_free = j + 1

    keep = ~np.isnan(t_air)
    t = t_air[keep] + c.dongle_delay
    t = np.ceil(t / c.usb_frame) * c.usb_frame          # USB 1 ms 輪詢
    os_lat = rng.lognormal(np.log(c.os_median), c.os_sigma, t.size)
    spike = rng.random(t.size) < c.spike_prob
    os_lat += spike * rng.uniform(0.005, c.spike_max, t.size)
    tC = t + os_lat
    return dict(tP=tP[keep], tC=tC, pid=k[keep], t_gen=t_gen[keep],
                n_sent=n_pkt, n_recv=int(keep.sum()), buf_err=bad[keep])


# ---------------------------------------------------------------------------
# 3. SharkTooth（Algorithm 1 + 2）
# ---------------------------------------------------------------------------
class SharkTooth:
    def __init__(self, CI, M=100, settle=30.0, keep_last=True):
        self.CI, self.M, self.settle, self.keep_last = CI, M, settle, keep_last
        # Eq.(21)(22) 初始化
        self.dT0, self.tP0, self.g0 = np.inf, 0.0, 0.0
        self.dT1, self.tP1, self.g1 = np.inf, 0.0, np.inf
        self.prev_dT, self.prev_star = None, None
        self.d2 = []
        self.A = np.zeros((2, 2)); self.b = np.zeros(2)
        self.alpha = self.beta = 0.0
        self.n_pivot = 0

    def step(self, tP, tC):
        dT = tC - tP                                    # Eq.(3)
        if self.prev_dT is not None:
            self.d2.append(dT - self.prev_dT)           # Eq.(5)
            self.d2 = self.d2[-self.M:]
        self.prev_dT = dT
        sigma = np.std(self.d2) if len(self.d2) >= 2 else 0.0   # Eq.(7)
        lam0 = 1.0 if sigma == 0 else min(self.CI / (2 * sigma), 1.0)  # Eq.(10)

        if dT < self.dT0:                               # Eq.(11) 新樞紐點
            self.dT0, self.g0, self.tP0, self.g1 = dT, 0.0, tP, np.inf
            self.n_pivot += 1
        else:
            dtp1 = tP - self.tP1
            thr = self.dT1 + (self.g1 * dtp1 if dtp1 != 0 else 0.0)   # Eq.(9)，避開 ∞×0
            if dT < thr:                                # Eq.(12)
                star_prev = self.prev_star if self.prev_star is not None else dT
                lam = 0.5 * lam0 if dT >= star_prev else 0.5          # Eq.(13)
                if tP != self.tP0:
                    chord = (dT - self.dT0) / (tP - self.tP0)
                    self.g0 = (1 - lam) * self.g0 + lam * chord
                self.g1 = self.g0
                self.dT1 = self.dT0 + self.g0 * (tP - self.tP0)
                self.tP1 = tP
            else:
                self.g1 = self.g1 + 1e-6                # Eq.(14)
        star = self.dT0 + self.g0 * (tP - self.tP0)     # Eq.(8)
        self.prev_star = star

        # Algorithm 1：穩定期 / 遞迴 OLS
        if tP > self.tP0 + self.settle:
            x = tP - self.tP0                           # 先減參考值，避免大數相減
            self.A += np.array([[1, -x], [-x, x * x]])  # Eq.(16)
            self.b += np.array([x * star, star])        # Eq.(17)
            det = self.A[0, 0] * self.A[1, 1] - self.A[0, 1] * self.A[1, 0]
            if det != 0:
                a, b0 = (self.A @ self.b) / det         # Eq.(15)，b0 是 x = 0 處的截距
                self.alpha, self.beta = a, b0 - a * self.tP0
        else:                                           # Eq.(18)
            self.A[:] = 0; self.b[:] = 0
            if not self.keep_last:
                self.alpha = self.beta = 0.0
        dhat = self.alpha * tP + self.beta              # Eq.(19)
        return dT, star, dhat, tP + dhat                # Eq.(20)


def run_sharktooth(d, CI, **kw):
    st = SharkTooth(CI, **kw)
    out = np.array([st.step(p, c) for p, c in zip(d['tP'], d['tC'])])
    return dict(dT=out[:, 0], star=out[:, 1], dhat=out[:, 2], tS=out[:, 3], n_pivot=st.n_pivot)


def moving_std_d2(dT, M=100):
    d2 = np.diff(dT)
    out = np.full(dT.size, np.nan)
    for i in range(M, d2.size + 1):
        out[i] = np.std(d2[i - M:i])
    return out

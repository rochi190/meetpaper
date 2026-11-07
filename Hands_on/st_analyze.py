"""
SharkTooth 實測：分析與畫圖（對應論文 Fig. 6 與 Fig. 9）

用法：
    pip install numpy matplotlib
    python st_analyze.py runs/20261031_153000 --ci 0.030      # st_receiver.py 建的資料夾
    python st_analyze.py run.csv --ci 0.030                   # 舊版的單一 CSV 也可以
    python st_analyze.py runs/... --ci 0.030 --fig6-node NODE_B --fig9-at 1800   # 只畫 NODE_B 的 Fig. 6

輸出（放在資料所在的資料夾）：
    unpredictable_delay_correction_<node>.png   論文 Fig. 6 樣式：δT（藍點）、δT*（紅線）、樞紐點（綠色星號），
                                                標題寫是哪一塊板子的資料
    synchronization_example.png                 論文 Fig. 9 樣式：上 Synchronization OFF（t_P）、下 Synchronization ON（t̂_S）
    終端機摘要             每個節點的封包數、遺失、封包間隔、δT 斜率；節點對的殘餘延遲
圖的尺寸是 IEEE 單欄寬（3.5 吋），字型用 Times（沒有的話用相近的襯線字），只輸出 PNG（300 dpi）。

SharkTooth 的實作逐行對照論文 Eq.(3)–(22)，和 sharktooth_sim/sim/algorithms/sharktooth.py
的參考版相同（只用 numpy，不需要 numba）。
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def paper_style():
    """接近 IEEE 期刊圖的樣式：襯線字、9 pt、細框線。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
        "legend.framealpha": 1.0, "legend.edgecolor": "black", "legend.fancybox": False,
        "savefig.dpi": 300,
    })


def save(fig, base: Path):
    """只存 PNG。檔名直接接 .png（不用 with_suffix，否則 _tilt0.5 這種檔名會被截掉）。"""
    fig.savefig(base.parent / (base.name + ".png"))
    plt.close(fig)

LSE_HZ = 32768.0
VREF = 3.3
ADC_FS = 4095.0
F_REF = 5.0            # 參考正弦頻率（Hz）


# --------------------------------------------------------------------------- 讀檔
def load(path, tick_s=1.0 / LSE_HZ):
    """path 可以是 st_receiver.py 建的資料夾（讀裡面的 packets.csv），或一個 CSV 檔。
    tick_s：每個 LSE tick 換算成幾秒。預設是精確的 1/32768；--tick-us 30.51 可以模擬論文韌體的近似常數。"""
    path = Path(path)
    if path.is_dir():
        path = path / "packets.csv"
    raw = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw[r["node"]].append(r)
    nodes = {}
    for name, rows in raw.items():
        g = lambda k, t=float: np.array([t(r[k]) for r in rows])
        tp_ticks = g("tP_ticks", int).astype(np.int64)
        # 32 位元 tick 約 36 小時溢位一次；保險起見展開
        tp_ticks = tp_ticks + np.r_[0, np.cumsum(np.diff(tp_ticks) < -(1 << 31))] * (1 << 32)
        d = dict(
            t_C=g("t_C"), pkt_id=g("pkt_id", int), tP=tp_ticks * tick_s,
            fl=g("fl_ticks", int) * tick_s, n=g("n", int),
            s=np.stack([g(f"s{k}", int) for k in range(5)], axis=1) * VREF / ADC_FS,
            qdrop=g("qdrop", int),
        )
        nodes[name] = clean(name, d)
    return dict(sorted(nodes.items()))


def clean(name, d):
    """簡化版完整性檢查（模擬 §7 的子集）：
    - 同一個 t_P 出現兩次 → 重複封包，丟掉後者
    - t_P 倒退 → 節點重開機或 notify 重新打開（t_P 歸零），只留最長的一段"""
    keep = np.ones(d["tP"].size, bool)
    seen = set()
    for i, t in enumerate(d["tP"]):
        if t in seen:
            keep[i] = False
        seen.add(t)
    d = {k: v[keep] for k, v in d.items()}
    seg = np.r_[0, np.cumsum(np.diff(d["tP"]) < -1.0)]
    if seg[-1] > 0:
        best = np.bincount(seg).argmax()
        print(f"  {name}: t_P 倒退 {seg[-1]} 次（重開機或 notify 重開），只用第 {best + 1} 段")
        d = {k: v[seg == best] for k, v in d.items()}
    # 到達順序可能和 t_P 順序略有不同（BLE 會依序送，正常不會發生）；SharkTooth 依到達順序處理
    return d


# --------------------------------------------------------------------------- SharkTooth
class SharkTooth:
    """論文 Algorithm 1 + 2。step(tP, tC) 每來一包呼叫一次。"""

    def __init__(self, CI, M=100, lam_max=0.5, g1_inc=1e-6, settle_s=30.0, keep_last=True):
        self.CI, self.lam_max, self.g1_inc = CI, lam_max, g1_inc
        self.settle, self.keep_last = settle_s, keep_last
        self.dT0, self.tP0, self.g0 = math.inf, 0.0, 0.0          # Eq.(21)
        self.dT1, self.tP1, self.g1 = math.inf, 0.0, math.inf     # Eq.(22)
        self.prev_dT = None
        self.prev_star = None
        self.d2 = deque(maxlen=M)
        self.A = np.zeros((2, 2))
        self.b = np.zeros(2)
        self.alpha = self.beta = 0.0
        self.valid = False
        # 每一步的中間訊號（給 st_signals.py 畫圖；不影響算法）
        self.sigma = self.lam0 = self.lam = self.thr = math.nan
        self.branch = 0                                           # 0 = 新樞紐點、1 = 合格點（Eq.12）、2 = 不合格（Eq.14）

    def step(self, tP, tC):
        dT = tC - tP                                              # Eq.(3)
        if self.prev_dT is not None:
            self.d2.append(dT - self.prev_dT)                     # Eq.(5)
        self.prev_dT = dT
        sigma = float(np.std(self.d2)) if len(self.d2) >= 2 else 0.0          # Eq.(7)
        lam0 = 1.0 if sigma == 0 else min(self.CI / (2 * sigma), 1.0)         # Eq.(10)
        pivot = False
        self.sigma, self.lam0, self.lam, self.thr = sigma, lam0, math.nan, math.nan
        if dT < self.dT0:                                         # Eq.(11) 新樞紐點
            self.dT0, self.g0, self.tP0, self.g1 = dT, 0.0, tP, math.inf
            pivot = True
            self.branch = 0
        else:
            dtp1 = tP - self.tP1
            thr = self.dT1 + self.g1 * dtp1 if dtp1 != 0 else self.dT1        # Eq.(9)
            self.thr = thr
            if dT < thr:                                          # Eq.(12)
                sp = self.prev_star if self.prev_star is not None else dT
                lam = self.lam_max * lam0 if dT >= sp else self.lam_max       # Eq.(13)
                if tP != self.tP0:
                    self.g0 = (1 - lam) * self.g0 + lam * (dT - self.dT0) / (tP - self.tP0)
                self.g1 = self.g0
                self.dT1 = self.dT0 + self.g0 * (tP - self.tP0)
                self.tP1 = tP
                self.lam, self.branch = lam, 1
            else:
                self.g1 += self.g1_inc                            # Eq.(14)
                self.branch = 2
        star = self.dT0 + self.g0 * (tP - self.tP0)               # Eq.(8)
        self.prev_star = star
        if tP > self.tP0 + self.settle:                           # Algorithm 1
            x = tP - self.tP0
            self.A += np.array([[1.0, -x], [-x, x * x]])          # Eq.(16)
            self.b += np.array([x * star, star])                  # Eq.(17)
            det = self.A[0, 0] * self.A[1, 1] - self.A[0, 1] ** 2
            if det > 0:
                a, b0 = (self.A @ self.b) / det                   # Eq.(15)
                self.alpha, self.beta = a, b0 - a * self.tP0
                self.valid = True
        else:                                                     # Eq.(18)
            self.A[:] = 0.0
            self.b[:] = 0.0
            if not self.keep_last:
                self.alpha = self.beta = 0.0
                self.valid = False
        return dT, star, self.alpha * tP + self.beta, self.valid, pivot     # Eq.(19)


def run_sharktooth(d, CI, tilt_ppm, keep_last=True):
    """tilt_ppm：先把 δT 加上已知正斜率再跑、輸出扣回（模擬 T7 的修正）。
    晶振的漂移可能是負的；負斜率時每個新低點都會變成樞紐點，原版幾乎沒有輸出。"""
    k = tilt_ppm * 1e-6
    tP, tC = d["tP"], d["t_C"]
    st = SharkTooth(CI, keep_last=keep_last)
    out, tr = [], []
    for p, c in zip(tP, tC):
        out.append(st.step(p, c + k * (p - tP[0])))
        tr.append((st.sigma, st.lam0, st.lam, st.thr, st.g0, st.g1, st.alpha, st.beta, st.branch))
    out = np.array(out, dtype=float)
    tr = np.array(tr, dtype=float)
    tilt = k * (tP - tP[0])
    d["dT"] = out[:, 0] - tilt
    d["star"] = out[:, 1] - tilt
    d["dhat"] = out[:, 2] - tilt
    d["valid"] = out[:, 3].astype(bool)
    d["pivot"] = out[:, 4].astype(bool)
    d["alpha_ppm"] = (st.alpha - k) * 1e6
    d["beta_s"] = st.beta + k * tP[0]              # 預傾斜是 k·(t_P − t_P[0])，扣回後截距要補 k·t_P[0]
    d["ab_valid"] = bool(st.valid)                 # 錄完時 α、β 是否有效（--paper 且還在 30 s 穩定期時為 False）
    # 每一包的中間訊號（全部扣回預傾斜，單位和論文一樣：秒、s/s）
    d["sigma"], d["lam0"], d["lam"] = tr[:, 0], tr[:, 1], tr[:, 2]   # Eq.(7)、(10)、(13)（λ 只在合格點有值）
    d["thr"] = tr[:, 3] - tilt                                        # Eq.(9) δT_thr（樞紐點那一包是 nan）
    d["g0"], d["g1"] = tr[:, 4] - k, tr[:, 5] - k                     # Eq.(8) γ0、Eq.(9) γ1（γ1 可能是 ∞）
    d["alpha_t"] = np.where(d["valid"], tr[:, 6] - k, np.nan)         # 每一包當下的 α、β（沒有輸出時是 nan）
    d["beta_t"] = np.where(d["valid"], tr[:, 7] + k * tP[0], np.nan)
    d["branch"] = tr[:, 8].astype(int)                                # 0 樞紐點、1 合格、2 不合格
    d["CI"] = CI


# --------------------------------------------------------------------------- 樣本時間
def sample_times(d):
    """每包 5 個樣本的時間：最後一個是 t_P，往前每個間隔 fl / (n − 1)。"""
    n = d["n"][:, None]
    Ts = d["fl"][:, None] / np.maximum(n - 1, 1)
    k = np.arange(5)[None, :]
    t_samp = d["tP"][:, None] - (n - 1 - k) * Ts
    dhat = np.interp(t_samp.ravel(), d["tP"], d["dhat"]).reshape(t_samp.shape)
    return t_samp, t_samp + dhat          # t_P 下的時間、t̂_S 下的時間


# --------------------------------------------------------------------------- 圖
def fig6(d, span, base: Path, node=""):
    """論文 Fig. 6：Unpredictable delay correction。標題第二行寫是哪一塊板子的資料。"""
    m = d["tP"] <= d["tP"][0] + span
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(d["tP"][m], d["dT"][m], ".", ms=2.2, color="#0072BD", label=r"Input $\delta T$", zorder=2)
    ax.plot(d["tP"][m], d["star"][m], "-", lw=1.8, color="red", label=r"Output $\delta T_{*}$", zorder=3)
    pv = np.flatnonzero(m & d["pivot"])[-1:]       # 只標最後一個樞紐點（紅線目前掛在它上面）
    ax.plot(d["tP"][pv], d["dT"][pv], ls="none", marker=(8, 2, 0), ms=9, mew=1.2,
            color="#00CC00", label="Pivot point", zorder=4)
    lo = np.percentile(d["dT"][m], 0.5)
    hi = np.percentile(d["dT"][m], 99.5)
    ax.set_ylim(lo - 0.15 * (hi - lo), max(hi, lo + 0.05) + 0.25 * (hi - lo))
    ax.set_xlim(0, d["tP"][m][-1])
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_xlabel(r"$t_P$ (s)")
    ax.set_ylabel(r"$\delta T$ (s)")
    ax.set_title("Unpredictable delay correction" + (f"\n({node})" if node else ""))
    ax.legend(loc="upper right")
    if d.get("ab_valid"):                          # 圖下方：整段錄音最後估出的 α、β（δ̂T = α·t_P + β）
        ab = (rf"$\alpha$ = {d['alpha_ppm']:+.1f} ppm,   $\beta$ = {d['beta_s']:.4f} s").replace("-", "\u2212")
    else:
        ab = r"$\alpha$, $\beta$: not available (still in 30 s settling)"
    fig.text(0.5, 0.015, ab, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, base)


def fig6_annotated(d, span, base: Path, CI, node=""):
    """Fig. 6 加註版：標出 α、β、開頭的突降、帶寬、重傳。數字全部由資料算出來。"""
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "DejaVu Sans"]
    m = d["tP"] <= d["tP"][0] + span
    tp, dT, star = d["tP"][m], d["dT"][m], d["star"][m]
    a = d["alpha_ppm"] * 1e-6
    b = float(d["dhat"][-1] - a * d["tP"][-1])                # 最後輸出的直線：δ̂ = α·t_P + β
    pv = np.flatnonzero(m & d["pivot"])
    t_piv = d["tP"][pv[-1]] if pv.size else tp[0]
    t_settle = min(t_piv + 6.0, tp[-1])                       # 找到最後樞紐點後再幾秒，紅線就貼住下緣
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_axes([0.07, 0.13, 0.48, 0.78])
    ax.plot(tp, dT, ".", ms=2.2, color="#0072BD", label="輸入 δT（每一包）", zorder=2)
    ax.plot(tp, star, "-", lw=1.8, color="red", label="SharkTooth 追的下緣 δT*", zorder=3)
    ax.plot(d["tP"][pv[-1:]], d["dT"][pv[-1:]], ls="none", marker=(8, 2, 0), ms=10, mew=1.3, color="#00CC00",
            label="樞紐點", zorder=5)
    xs = np.array([0.0, tp[-1]])
    ax.plot(xs, a * xs + b, "--", lw=1.5, color="#7B2CBF", label="α·t_P + β（最後輸出的直線）", zorder=4)
    ax.plot(0, b, "o", ms=8, mfc="white", mew=2, color="#7B2CBF", zorder=6, clip_on=False)
    lo, hi = np.percentile(dT, 0.5), np.percentile(dT, 99.5)
    ax.set_ylim(lo - 0.35 * (hi - lo), max(hi, lo + 0.05) + 0.25 * (hi - lo))
    ax.set_xlim(0, tp[-1])
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.axvspan(0, t_settle, color="#E07A00", alpha=0.13, lw=0)
    t1, t2 = 0.3 * tp[-1], 0.85 * tp[-1]
    y1, y2 = a * t1 + b, a * t2 + b
    ax.plot([t1, t1, t2], [y1, y2, y2], color="#0B6E4F", lw=2, zorder=6)
    box = lambda c: dict(boxstyle="round,pad=0.25", fc="white", ec=c, lw=1.1)
    ax.text((t1 + t2) / 2, y2, f"{t2 - t1:.0f} s", color="#0B6E4F", ha="center", va="top", fontsize=9, zorder=7)
    ax.text(t1, (y1 + y2) / 2, f"{(y2 - y1) * 1e3:+.1f} ms ", color="#0B6E4F", ha="right", va="center", fontsize=9,
            bbox=box("#0B6E4F"), zorder=7)
    tb = 0.95 * tp[-1]
    ax.annotate("", (tb, a * tb + b), (tb, a * tb + b + CI), arrowprops=dict(arrowstyle="<->", lw=1.6, color="#444444"))
    for txt, x, y, c in (("①", t_settle / 2, ax.get_ylim()[1], "#E07A00"), ("②", 0, b, "#7B2CBF"),
                         ("③", (t1 + t2) / 2, y2 - 0.25 * (hi - lo), "#0B6E4F"),
                         ("④", tb, a * tb + b + CI / 2, "#444444")):
        ax.text(x, y, txt, color=c, fontsize=12, ha="center", va="center", weight="bold", bbox=box(c), zorder=8)
    ax.set_xlabel("t_P（s）")
    ax.set_ylabel("δT（s）")
    if node:
        ax.set_title(f"Unpredictable delay correction（{node} 的資料）", fontsize=11)
    ax.legend(loc="upper right", fontsize=7.5)
    tx = fig.add_axes([0.58, 0.05, 0.41, 0.9])
    tx.axis("off")
    lines = [
        ("#E07A00", f"① 開頭突降（0～{t_settle:.0f} s）：SharkTooth 在找起點",
         "第一包當起點，之後更低的點就換成新樞紐點（Eq. 11），紅線一路往下；"
         f"最後一次換樞紐點在 {t_piv:.1f} s（綠星），再幾秒斜率調穩就貼住下緣。"),
        ("#7B2CBF", f"② β = {b:.4f} s：虛線在 t_P = 0 的高度",
         "兩支鐘的起點差：程式開始（t_C = 0）到 notify 打開（t_P = 0）隔的時間，"
         "加上最快那一包的傳輸時間。每次錄都不同，不影響同步。"),
        ("#0B6E4F", f"③ α = {d['alpha_ppm']:+.1f} ppm：虛線的斜率",
         f"{t2 - t1:.0f} 秒變化 {(y2 - y1) * 1e3:+.1f} ms。負 = 板子的晶振比電腦的鐘快；正 = 比較慢。"),
        ("#444444", f"④ 帶子厚度 ≈ 1 個 CI = {CI * 1e3:.0f} ms",
         "每包等 connection event 的時間；上方零星的點是重傳（多等 1～2 個 CI）。"),
    ]
    y = 0.98
    for c, head, body in lines:
        tx.text(0, y, head, color=c, fontsize=10.5, weight="bold", va="top", transform=tx.transAxes)
        tx.text(0.03, y - 0.075, "\n".join(body[i:i + 34] for i in range(0, len(body), 34)), fontsize=8.8,
                va="top", transform=tx.transAxes, color="#222222")
        y -= 0.25
    save(fig, base)
    paper_style()


def fig9(nodes, at, base: Path, legend=True, span=1.0):
    """論文 Fig. 9：上 Synchronization OFF（各自的 t_P），下 Synchronization ON（t̂_S）。"""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.4))
    ref = next(iter(nodes.values()))
    t_p0, t_s0 = sample_times(ref)
    ts_at = np.interp(at, t_p0.ravel(), t_s0.ravel())   # 下圖從同一個瞬間開始
    for name, d in nodes.items():
        t_p, t_s = sample_times(d)
        v = d["s"]
        m = (t_p >= at) & (t_p <= at + span)
        o = np.argsort(t_p[m])
        a1.plot(t_p[m][o], v[m][o], lw=1.4, label=name)
        m = (t_s >= ts_at) & (t_s <= ts_at + span)
        o = np.argsort(t_s[m])
        a2.plot(t_s[m][o], v[m][o], lw=1.4, label=name)
    thousands = FuncFormatter(lambda x, _: f"{x:,.1f}".rstrip("0").rstrip("."))
    a1.set_title("Synchronization OFF")
    a1.set_xlabel(r"$t_P$ (s)")
    a2.set_title("Synchronization ON")
    a2.set_xlabel(r"$\hat{t}_S$ (s)")
    for a in (a1, a2):
        a.set_ylabel("Signal (V)")
        a.xaxis.set_major_formatter(thousands)
        a.margins(x=0)
    if legend:                                          # 圖例放在最上面一排，不會擋到波形
        h, lab = a1.get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=len(lab), fontsize=7.5,
                   bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.2)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    save(fig, base)


# --------------------------------------------------------------------------- 摘要
def epoch_phase_delay(d, t0, t1):
    """在 t̂_S 時間軸上，每 1 秒（剛好 5 個週期）算一次 5 Hz 分量的相位 → 換成延遲（s）。"""
    _, t_s = sample_times(d)
    t_s, v = t_s.ravel(), d["s"].ravel()
    edges = np.arange(math.ceil(t0), math.floor(t1))
    out = np.full(edges.size, np.nan)
    for i, e in enumerate(edges):
        m = (t_s >= e) & (t_s < e + 1)
        if m.sum() < 40:
            continue
        # 最小平方擬合 a·cos + b·sin + c（樣本數不一定剛好 50，用 DFT 會漏頻、相位偏掉）
        w = 2 * np.pi * F_REF * t_s[m]
        X = np.stack([np.cos(w), np.sin(w), np.ones_like(w)], axis=1)
        a, b, _ = np.linalg.lstsq(X, v[m], rcond=None)[0]
        out[i] = -math.atan2(a, b) / (2 * np.pi * F_REF)       # v ≈ A·sin(w + φ)，延遲 = −φ/ω
    return edges, out


def estimate_ci(tC, P=None, win=5.0):
    """從封包「到達時間」推估 connection interval。
    封包只能在 connection event 送出，所以 t_C 會落在間隔 = CI 的格點上（加上 PC 端約 1 ms 的抖動）。
    每 5 秒一段，試 7.5～100 ms（BLE 的 CI 一定是 1.25 ms 的倍數），看哪個間隔讓到達時間最整齊。
    CI/2、CI/3 也會一樣整齊，所以最後取「整齊程度差不多」的最大倍數。
    P = 封包週期：CI 很短時，到達時間幾乎就是每 P 一包，所以只在 0.9·P 以下找，避免把 P 誤認成 CI。"""
    t = np.sort(tC)
    cands = np.arange(6, 81) * 1.25e-3
    if P:
        cands = cands[cands <= 0.9 * P]
    coh = np.zeros(cands.size)
    n = 0
    for e in np.arange(t[0], t[-1] - win, win):
        x = t[(t >= e) & (t < e + win)]
        if x.size < 10:
            continue
        coh += np.abs(np.exp(2j * np.pi * x[:, None] / cands[None, :]).mean(axis=0))
        n += 1
    if n == 0:
        return float("nan"), 0.0
    coh /= n
    i = int(np.argmax(coh))
    best = i
    for k in range(2, 81):                      # 往上找倍數：真正的 CI 是「一樣整齊」的最大那個
        j = (i + 6) * k - 6
        if j >= cands.size:
            break
        if coh[j] >= 0.9 * coh[i]:
            best = j
    if coh[best] < 0.6:                         # 不夠整齊（CI 太短、PC 端抖動太大）就不要亂猜
        return float("nan"), float(coh[best])
    return float(cands[best]), float(coh[best])


def summary(nodes, CI):
    print(f"\n{'節點':6} {'封包':>7} {'遺失':>6} {'節點丟':>6} {'封包間隔(ms)':>12} "
          f"{'δT 斜率(ppm)':>12} {'樞紐點':>6} {'有輸出':>7}")
    for name, d in nodes.items():
        lost = int(np.sum((np.diff(d["pkt_id"]) % 256) - 1))
        dur_pkts = np.diff(d["tP"])
        P = np.median(dur_pkts[dur_pkts < 0.15]) * 1e3
        qd = int((d["qdrop"][-1] - d["qdrop"][0]) % 256)
        print(f"{name:6} {d['tP'].size:7d} {lost:6d} {qd:6d} {P:12.3f} {d['alpha_ppm']:12.1f} "
              f"{int(d['pivot'].sum()):6d} {100 * d['valid'].mean():6.1f}%")
    for name, d in nodes.items():
        dtp = np.diff(d["tP"])
        ci_est, c = estimate_ci(d["t_C"], P=float(np.median(dtp[dtp < 0.15])))
        if not np.isfinite(ci_est):
            print(f"  {name}：從到達時間判斷不出 CI（整齊度 {c:.2f}），請看板子序列埠印的 interval")
            continue
        flag = "" if abs(ci_est - CI) < 1e-4 else f"  ← 和 --ci {CI * 1e3:.2f} ms 不同，建議改用 --ci {ci_est:.4g}"
        print(f"  {name}：從到達時間推估的 CI ≈ {ci_est * 1e3:.2f} ms（整齊度 {c:.2f}）{flag}")
    print("  封包間隔 ≠ 100 ms 的部分 = 取樣節拍（MSI）相對晶振的誤差；論文是 100.64–100.93 ms")
    print(f"  δT 斜率 = 節點晶振和 PC 時鐘的速度差；SharkTooth 用的 CI = {CI * 1e3:.2f} ms")

    names = list(nodes)
    if len(names) < 2:
        return
    t0 = max(d["t_C"][0] for d in nodes.values()) + 60.0      # 跳過前 60 秒（收斂中）
    t1 = min(d["t_C"][-1] for d in nodes.values()) - 2.0
    if t1 - t0 < 30:
        print("\n資料太短，不算節點對的殘餘延遲（至少要 90 秒）")
        return
    ph = {n: epoch_phase_delay(nodes[n], t0, t1)[1] for n in names}
    print("\n同步後節點對的殘餘延遲（每 1 秒一個值，t̂_S 時間軸；包含類比路徑差，不只是同步誤差）")
    print(f"{'節點對':14} {'平均(ms)':>9} {'std(ms)':>8} {'P95|·|(ms)':>11}")
    period = 1.0 / F_REF
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x = ph[names[i]] - ph[names[j]]
            x = (x + period / 2) % period - period / 2
            x = x[np.isfinite(x)] * 1e3
            print(f"{names[i] + '-' + names[j]:14} {x.mean():9.3f} {x.std():8.3f} "
                  f"{np.percentile(np.abs(x), 95):11.3f}")
    print("  論文 Table II（17 B、2–4 節點、穩態）：mean|RSE| 約 0.21–0.29 ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="st_receiver.py 建的資料夾，或一個 CSV 檔")
    ap.add_argument("--ci", type=float, default=0.030, help="實際的 connection interval（秒）")
    ap.add_argument("--tilt", type=float, default=100.0, help="預傾斜（ppm），0 = 論文原版")
    ap.add_argument("--fig6-node", default=None, help="只畫某個節點的 Fig. 6（預設：每個節點各畫一張）")
    ap.add_argument("--fig6-span", type=float, default=140.0, help="Fig.6 畫前幾秒")
    ap.add_argument("--fig9-at", type=float, default=None, help="Fig.9 取哪個 t_P（秒）開始")
    ap.add_argument("--fig9-span", type=float, default=1.0, help="Fig.9 畫幾秒（論文約 1 秒）")
    ap.add_argument("--no-legend", action="store_true", help="Fig.9 不畫哪條線是哪塊板子的圖例（和論文一樣）")
    ap.add_argument("--paper", action="store_true",
                    help="完全照論文：不加預傾斜（--tilt 0），穩定期內 α = β = 0（Eq. 18）")
    ap.add_argument("--annotate", action="store_true", help="另外輸出加註版 Fig. 6（標出 α、β、開頭突降、帶寬）")
    ap.add_argument("--tick-us", type=float, default=None,
                    help="每個 tick 當成幾 µs（預設精確的 30.5176）。填 30.51 = 模擬論文韌體的換算誤差"
                         "（t_P 慢約 248 ppm → δT 斜率多 +248 ppm）")
    a = ap.parse_args()

    paper_style()
    src = Path(a.data)
    outdir = src if src.is_dir() else src.parent
    nodes = load(src, tick_s=(a.tick_us * 1e-6) if a.tick_us else 1.0 / LSE_HZ)
    if a.tick_us:
        print(f"注意：tick 換算用 {a.tick_us} µs（精確值 {1e6 / LSE_HZ:.4f} µs），"
              f"等於 t_P 有 {(a.tick_us * 1e-6 * LSE_HZ - 1) * 1e6:+.0f} ppm 的換算誤差"
              f"（δT 斜率多 {(1 - a.tick_us * 1e-6 * LSE_HZ) * 1e6:+.0f} ppm）")
    print("節點：", ", ".join(f"{k}（{v['tP'].size} 包）" for k, v in nodes.items()))
    if a.paper:
        a.tilt = 0.0
        print("--paper：完全照論文（不加預傾斜；穩定期內 α = β = 0）")
    for d in nodes.values():
        run_sharktooth(d, a.ci, a.tilt, keep_last=not a.paper)

    n6s = [a.fig6_node] if a.fig6_node else list(nodes)
    tag = "_paper" if a.paper else ("" if a.tilt == 100.0 else f"_tilt{a.tilt:g}")           # --tilt 0 的圖另存，不會蓋掉預設的
    for n6 in n6s:
        fig6(nodes[n6], a.fig6_span, outdir / f"unpredictable_delay_correction_{n6}{tag}", node=n6)
        if a.annotate:
            fig6_annotated(nodes[n6], a.fig6_span, outdir / f"unpredictable_delay_correction_{n6}{tag}_annotated",
                           a.ci, node=n6)
    at = a.fig9_at if a.fig9_at is not None else 0.8 * min(d["tP"][-1] for d in nodes.values())
    fig9(nodes, at, outdir / "synchronization_example", legend=not a.no_legend, span=a.fig9_span)
    summary(nodes, a.ci)
    print(f"\n已輸出到 {outdir}：" + "、".join(f"unpredictable_delay_correction_{n}{tag}.png" for n in n6s)
          + "、synchronization_example.png")


if __name__ == "__main__":
    main()

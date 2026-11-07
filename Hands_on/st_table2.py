"""
仿論文 Table II：RSE（relative synchronization error）的 median (IQR)，每 600 秒一段，共 6 段。
表裡只有你的設置 A3（20 B 封包、3 個節點）。

用法：
    python st_table2.py runs/20261102_090000                       # 一次錄音（IQR 會是 0）
    python st_table2.py runs/20261102_090000 runs/20261102_110000 runs/20261102_130000   # 多次錄音 = 論文的重複次數
    python st_table2.py runs/2026110* --ci 0.030

每個資料夾 = 一次三塊板子一起錄的資料（st_receiver.py 錄的），最好錄滿 1 小時（6 段 × 600 秒）；
不滿 600 秒的段落顯示「–」。

    python st_table2.py runs/20261102_090000 --gt cc               # 真實延遲改用論文的互相關（0.1 ms 解析度）

輸出：只給一個資料夾時是 <資料夾>/rse_table.png；給多個時是 runs/rse_table.png（第一個資料夾的上一層）；
     也可以用 --out 指定。終端機也會印出 A3 的數值。

計算方式（論文 Section IV-B，Eq. 23–27）：
    1. 每塊板子跑 SharkTooth，得到 δ̂T。
    2. 在共同的 t_P 格點（50 Hz）上，估計延遲 τ̂_nm = δ̂T_n − δ̂T_m，每 1 秒平均一次（Eq. 23–24）。
    3. 真實延遲 τ_nm = τ_n − τ_m：每 1 秒，在各自的 t_P 時間軸上量 5 Hz 弦波相對理想正弦的延遲（Eq. 25）。
       論文用升頻到 10 kHz 的互相關（解析度 0.1 ms）；預設用 5 Hz 正弦的最小平方擬合（--gt fit），
       對純正弦結果相同，但沒有 0.1 ms 的量化。--gt cc = 照論文的互相關。
    4. RSE_nm = τ̂_nm − τ_nm（Eq. 26），每 1 秒一個值。SharkTooth 還沒有輸出 α、β 的 epoch（開頭約 30 秒）不算。
    5. 每 600 秒一段，取 P95(|RSE|) 最大的那一對節點（Eq. 27），算 |RSE| 的平均（Absolute Average）、
       RSE 的標準差（Standard Deviation）和 P95(|RSE|)（論文的第三個指標，畫在 Fig. 12，不在 Table II）。
    6. 多次錄音之間取 median 和 IQR（第 75 百分位 − 第 25 百分位）。
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from st_analyze import F_REF, load, paper_style, run_sharktooth, sample_times  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

FS = 50.0           # 取樣率
SEG_S = 600.0       # 每段 600 秒
N_SEG = 6           # 6 段
MY_ID = "A3"
MY_BYTES = 20



# --------------------------------------------------------------------------- 一次錄音 → 每段的指標
def epoch_true_offset(t, v, edges):
    """每 1 秒，在 t_P 時間軸上擬合 v ≈ a·cos(ωt) + b·sin(ωt) + c = R·sin(ωt + φ) + c。
    φ/ω = 這個節點的訊號相對理想正弦「提前」多少 = 這段時間的真實 δT（mod 200 ms）。"""
    out = np.full(edges.size, np.nan)
    for i, e in enumerate(edges):
        m = (t >= e) & (t < e + 1)
        if m.sum() < 40:
            continue
        w = 2 * np.pi * F_REF * t[m]
        X = np.stack([np.cos(w), np.sin(w), np.ones_like(w)], axis=1)
        a, b, _ = np.linalg.lstsq(X, v[m], rcond=None)[0]
        out[i] = math.atan2(a, b) / (2 * np.pi * F_REF)
    return out


def epoch_true_offset_cc(t, v, edges, fu=10_000.0):
    """論文 Eq. 25 的做法：每 1 秒把訊號線性升頻到 10 kHz，和軟體產生的 5 Hz 正弦做互相關，
    取讓互相關最大的位移（解析度 1/fu = 0.1 ms，範圍 ±半個週期）。
    參考訊號是解析的正弦，所以 CC(lag) = cos(ω·lag)·Σ s·sin(ωt) − sin(ω·lag)·Σ s·cos(ωt)，不用真的掃 2000 個位移。
    回傳的符號和 epoch_true_offset 相同（訊號「提前」多少）。"""
    out = np.full(edges.size, np.nan)
    period = 1.0 / F_REF
    lags = np.arange(-period / 2, period / 2, 1.0 / fu)
    wl = 2 * np.pi * F_REF * lags
    for i, e in enumerate(edges):
        m = (t >= e - 0.05) & (t < e + 1.05)
        if m.sum() < 40:
            continue
        tu = e + np.arange(int(fu)) / fu
        su = np.interp(tu, t[m], v[m])
        su = su - su.mean()
        w = 2 * np.pi * F_REF * tu
        S, C = su @ np.sin(w), su @ np.cos(w)
        out[i] = -lags[np.argmax(np.cos(wl) * S - np.sin(wl) * C)]         # s(t) ≈ ref(t − lag) → 提前 = −lag
    return out


def rse_epochs(nodes, gt="fit"):
    """Eq. 23–26：每 1 秒一個 RSE，每一對節點一條。nodes 要先跑過 run_sharktooth。
    gt：真實延遲怎麼量，"fit" = 正弦最小平方擬合（預設），"cc" = 論文的 10 kHz 互相關。
    回傳 edges（每個 epoch 的起點，共同 t_P 格點）、rse {(n, m): 秒}、est / tru {n: 秒}。"""
    names = list(nodes)
    t_lo = max(d["tP"][0] for d in nodes.values())
    t_hi = min(d["tP"][-1] for d in nodes.values())
    edges = np.arange(math.ceil(t_lo), math.floor(t_hi) - 1 + 1e-9, 1.0)
    grid = np.arange(edges[0], edges[-1] + 1, 1.0 / FS)                   # Eq. 23 前的重新取樣（50 Hz）
    k_of = np.floor(grid - edges[0]).astype(int)

    est, tru = {}, {}
    cnt = np.bincount(k_of, minlength=edges.size)[:edges.size]
    for n, d in nodes.items():
        dhat = np.interp(grid, d["tP"], d["dhat"])                         # δ̂T 在共同格點上
        est[n] = np.bincount(k_of, dhat, minlength=edges.size)[:edges.size] / cnt   # Eq. 24 的每秒平均
        # SharkTooth 還沒有輸出 α、β 的時間（開頭約 30 秒）不算：那時 δ̂T = 0，不是同步誤差
        ok = np.interp(grid, d["tP"], d["valid"].astype(float)) >= 1.0
        est[n][np.bincount(k_of, ~ok, minlength=edges.size)[:edges.size] > 0] = np.nan
        t_samp, _ = sample_times(d)
        f = epoch_true_offset_cc if gt == "cc" else epoch_true_offset
        tru[n] = f(t_samp.ravel(), d["s"].ravel(), edges)

    period = 1.0 / F_REF
    rse = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n, m = names[i], names[j]
            x = (est[n] - est[m]) - (tru[n] - tru[m])                      # Eq. 26
            rse[(n, m)] = (x + period / 2) % period - period / 2           # 弦波每 200 ms 重複 → 取最近的
    return edges, rse, est, tru


def segment_metrics(edges, rse):
    """Eq. 27 + 論文的三個指標：每 600 秒一段，取 P95(|RSE|) 最大的那一對，
    算 |RSE| 平均（Absolute Average）、標準差（Standard Deviation）、P95(|RSE|)。單位 ms。"""
    avg, sd, p95 = (np.full(N_SEG, np.nan) for _ in range(3))
    worst = [""] * N_SEG
    for s in range(N_SEG):
        m = (edges >= s * SEG_S) & (edges + 1 <= (s + 1) * SEG_S)
        if m.sum() < 0.9 * SEG_S:                                          # 這一段資料不滿
            continue
        best, top = None, -1.0
        for pair, x in rse.items():                                        # Eq. 27：P95(|RSE|) 最大的那一對
            y = x[m][np.isfinite(x[m])]
            if y.size and np.percentile(np.abs(y), 95) > top:
                best, top = pair, np.percentile(np.abs(y), 95)
        y = rse[best][m]
        y = y[np.isfinite(y)] * 1e3
        avg[s], sd[s], p95[s] = np.abs(y).mean(), y.std(), np.percentile(np.abs(y), 95)
        worst[s] = f"{best[0]}-{best[1]}"
    return avg, sd, p95, worst


def run_metrics(folder, CI, tilt, gt="fit"):
    """回傳 (avg, sd, p95, worst, 節點數)：各 N_SEG 個值（ms），沒有資料的段落是 nan。"""
    nodes = load(folder)
    if len(nodes) < 2:
        raise SystemExit(f"{folder}：只有 {len(nodes)} 塊板子，算 RSE 至少要 2 塊")
    for d in nodes.values():
        run_sharktooth(d, CI, tilt)
    edges, rse, _, _ = rse_epochs(nodes, gt)
    avg, sd, p95, worst = segment_metrics(edges, rse)
    return avg, sd, p95, worst, len(nodes)


# --------------------------------------------------------------------------- 畫表
def draw_table(blocks, out: Path, note: str):
    """blocks：[(標籤, rows, rgb), ...]；rows：[(config_id, bytes, peripherals, [(median, iqr) or None] * 6), ...]"""
    paper_style()
    cols = [0.0, 1.15, 2.25, 3.2, 4.25]                    # 欄位左邊界（單位：吋）
    seg_w = 0.92
    W = cols[-1] + N_SEG * seg_w + 0.1
    row_h = 0.32                                          # 兩行的「Absolute Average」要放得下
    n_rows = sum(len(b[1]) for b in blocks)
    H = 0.95 + n_rows * row_h + 0.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    fs = 7.2
    y = 0.12
    ax.text(W / 2, y, "TABLE II", ha="center", va="center", fontsize=8.5)
    ax.text(W / 2, y + 0.17, "MEDIAN AND IQR OF THE RSE METRICS ACROSS "
            "DIFFERENT CONFIGURATIONS AND ACQUISITION SEGMENTS", ha="center", va="center", fontsize=7.5)
    y = 0.45
    ax.plot([0, W], [y, y], color="black", lw=1.0)
    hy = y + 0.2
    for x, t in ((cols[0] + 0.55, "RSE metrics\n(ms)"), (cols[1] + 0.52, "Configuration\nID"),
                 (cols[2] + 0.45, "Packet\nlength\n(bytes)"), (cols[3] + 0.5, "#Peripherals")):
        ax.text(x, hy, t, ha="center", va="center", fontsize=fs, linespacing=1.1)
    ax.text(cols[4] + N_SEG * seg_w / 2, y + 0.1, "Acquisition segment", ha="center", va="center", fontsize=fs + 0.5)
    for s in range(N_SEG):
        ax.text(cols[4] + (s + 0.5) * seg_w, y + 0.3, str(s + 1), ha="center", va="center", fontsize=fs)
    y = 0.85
    ax.plot([0, W], [y, y], color="black", lw=0.6)
    ax.text(cols[4] + N_SEG * seg_w / 2, y + 0.08, "median (interquartile range) ms", ha="center", va="center",
            fontsize=fs, weight="bold")
    y += 0.17
    ax.plot([0, W], [y, y], color="black", lw=0.6)

    for block, (label, rows, rgb) in enumerate(blocks):
        vals = [c[0] for r in rows for c in r[3] if c is not None]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        y0 = y + 0.03
        for i, (cid, nb, npr, cells) in enumerate(rows):
            yc = y0 + (i + 0.5) * row_h
            kw = dict(ha="center", va="center", fontsize=fs)
            ax.text(cols[1] + 0.52, yc, cid, **kw)
            ax.text(cols[2] + 0.45, yc, str(nb), **kw)
            ax.text(cols[3] + 0.5, yc, str(npr), **kw)
            for s, c in enumerate(cells):
                x0 = cols[4] + s * seg_w
                if c is None:
                    ax.text(x0 + seg_w / 2, yc, "–", **kw)
                    continue
                a = 0.08 + 0.62 * (c[0] - lo) / (hi - lo if hi > lo else 1.0)     # 越大越深（同論文）
                ax.add_patch(Rectangle((x0 + 0.02, yc - row_h / 2), seg_w - 0.04, row_h,
                                       facecolor=(*rgb, a), edgecolor="none"))
                ax.text(x0 + seg_w / 2, yc, f"{c[0]:.2f} ({c[1]:.2f})", **kw)
        ax.text(cols[0] + 0.55, y0 + len(rows) * row_h / 2, label, ha="center", va="center", fontsize=fs + 0.3)
        y = y0 + len(rows) * row_h + 0.03
        ax.plot([0, W], [y, y], color="black", lw=1.0 if block == len(blocks) - 1 else 0.6)
    for s in range(1, N_SEG):                                # 段落之間的直線
        x = cols[4] + s * seg_w
        ax.plot([x, x], [0.45 + 0.22, 0.45 + 0.38], color="black", lw=0.5)
    ax.plot([cols[4] - 0.03] * 2, [0.45, y], color="black", lw=0.6)
    ax.text(0.02, y + 0.18, note, ha="left", va="center", fontsize=6.5)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+", help="一個或多個實驗資料夾（每個 = 一次三塊板子一起錄）")
    ap.add_argument("--ci", type=float, default=0.030, help="實際的 connection interval（秒）")
    ap.add_argument("--tilt", type=float, default=100.0, help="預傾斜（ppm），和 st_analyze.py 一樣")
    ap.add_argument("--gt", choices=("fit", "cc"), default="fit",
                    help="真實延遲：fit = 正弦擬合（預設），cc = 論文的 10 kHz 互相關（0.1 ms 解析度）")
    ap.add_argument("--out", default=None,
                    help="輸出的 PNG；預設：只給一個資料夾時放在那個資料夾裡，多個時放在第一個資料夾的上一層")
    a = ap.parse_args()

    res = {"avg": [], "sd": [], "p95": []}
    npers = set()
    for f in a.data:
        avg, sd, p95, worst, npr = run_metrics(f, a.ci, a.tilt, a.gt)
        for k, v in (("avg", avg), ("sd", sd), ("p95", p95)):
            res[k].append(v)
        npers.add(npr)
        print(f"{f}：{npr} 塊板子")
        for s in range(N_SEG):
            if np.isfinite(avg[s]):
                print(f"  第 {s + 1} 段：最差的一對 {worst[s]}，|RSE| 平均 {avg[s]:.3f} ms，"
                      f"標準差 {sd[s]:.3f} ms，P95 {p95[s]:.3f} ms")
            else:
                print(f"  第 {s + 1} 段：資料不滿 600 秒，略過")

    def med_iqr(x):
        x = np.array(x)
        out = []
        for s in range(N_SEG):
            y = x[:, s][np.isfinite(x[:, s])]
            out.append(None if y.size == 0 else
                       (float(np.median(y)), float(np.percentile(y, 75) - np.percentile(y, 25))))
        return out

    npr = max(npers)
    rows = {k: (MY_ID, MY_BYTES, npr, med_iqr(v)) for k, v in res.items()}
    labels = (("avg", "Absolute Average", "Absolute\nAverage", (0.85, 0.15, 0.55)),
              ("sd", "Standard Deviation", "Standard\nDeviation", (0.15, 0.45, 0.85)),
              ("p95", "95th Percentile", "95th\nPercentile", (0.15, 0.65, 0.35)))

    print(f"\n{MY_ID}（{MY_BYTES} B、{npr} 塊板子、{len(a.data)} 次錄音、真實延遲 = {a.gt}）median (IQR) ms：")
    for k, name, _, _ in labels:
        print(f"  {name:18} " + "  ".join("–" if c is None else f"{c[0]:.2f} ({c[1]:.2f})" for c in rows[k][3]))
    print("  論文 Table II 只列前兩個指標；P95 在論文 Fig. 12（Section V-A）。")

    if a.out:
        out = Path(a.out)
    elif a.gt == "cc" and len(a.data) == 1 and Path(a.data[0]).is_dir():
        out = Path(a.data[0]) / "rse_table_cc.png"
    elif len(a.data) == 1 and Path(a.data[0]).is_dir():
        out = Path(a.data[0]) / "rse_table.png"
    else:
        out = Path(a.data[0]).resolve().parent / "rse_table.png"
    note = (f"{MY_ID}: B-L475E-IOT01A + PC Bluetooth, {MY_BYTES} B, {npr} peripherals, "
            f"n = {len(a.data)} recording{'s' if len(a.data) > 1 else ''}, ground truth = "
            f"{'10 kHz cross-correlation' if a.gt == 'cc' else 'sine fit'}. 95th percentile: paper Fig. 12, not Table II.")
    draw_table([(lab, [rows[k]], rgb) for k, _, lab, rgb in labels], out, note)
    print(f"\n已輸出：{out}")


if __name__ == "__main__":
    main()

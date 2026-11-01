"""
仿論文 Table II：RSE（relative synchronization error）的 median (IQR)，每 600 秒一段，共 6 段。
表裡只有你的設置 A3（20 B 封包、3 個節點）。

用法：
    python st_table2.py runs/20261102_090000                       # 一次錄音（IQR 會是 0）
    python st_table2.py runs/20261102_090000 runs/20261102_110000 runs/20261102_130000   # 多次錄音 = 論文的重複次數
    python st_table2.py runs/2026110* --ci 0.030

每個資料夾 = 一次三塊板子一起錄的資料（st_receiver.py 錄的），最好錄滿 1 小時（6 段 × 600 秒）；
不滿 600 秒的段落顯示「–」。

輸出：runs/rse_table.png（放在第一個資料夾的上一層），終端機也會印出 A3 的數值。

計算方式（論文 Section IV-B，Eq. 23–27）：
    1. 每塊板子跑 SharkTooth，得到 δ̂T。
    2. 在共同的 t_P 格點（50 Hz）上，估計延遲 τ̂_nm = δ̂T_n − δ̂T_m，每 1 秒平均一次（Eq. 23–24）。
    3. 真實延遲 τ_nm = τ_n − τ_m：每 1 秒，在各自的 t_P 時間軸上量 5 Hz 弦波相對理想正弦的延遲（Eq. 25）。
       論文用升頻到 10 kHz 的互相關（解析度 0.1 ms）；這裡用 5 Hz 正弦的最小平方擬合，
       對純正弦結果相同，但沒有 0.1 ms 的量化。
    4. RSE_nm = τ̂_nm − τ_nm（Eq. 26），每 1 秒一個值。SharkTooth 還沒有輸出 α、β 的 epoch（開頭約 30 秒）不算。
    5. 每 600 秒一段，取 P95(|RSE|) 最大的那一對節點（Eq. 27），算 |RSE| 的平均（Absolute Average）和
       RSE 的標準差（Standard Deviation）。
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


def run_metrics(folder, CI, tilt):
    """回傳 (avg, sd)：各 N_SEG 個值（ms），沒有資料的段落是 nan。"""
    nodes = load(folder)
    names = list(nodes)
    if len(names) < 2:
        raise SystemExit(f"{folder}：只有 {len(names)} 塊板子，算 RSE 至少要 2 塊")
    for d in nodes.values():
        run_sharktooth(d, CI, tilt)

    # 共同的 t_P 格點（Eq. 23 前的重新取樣）與 1 秒 epoch
    t_lo = max(d["tP"][0] for d in nodes.values())
    t_hi = min(d["tP"][-1] for d in nodes.values())
    edges = np.arange(math.ceil(t_lo), math.floor(t_hi) - 1 + 1e-9, 1.0)
    grid = np.arange(edges[0], edges[-1] + 1, 1.0 / FS)
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
        tru[n] = epoch_true_offset(t_samp.ravel(), d["s"].ravel(), edges)

    period = 1.0 / F_REF
    rse = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n, m = names[i], names[j]
            x = (est[n] - est[m]) - (tru[n] - tru[m])                      # Eq. 26
            rse[(n, m)] = (x + period / 2) % period - period / 2           # 弦波每 200 ms 重複 → 取最近的

    avg = np.full(N_SEG, np.nan)
    sd = np.full(N_SEG, np.nan)
    worst = [""] * N_SEG
    for s in range(N_SEG):
        m = (edges >= s * SEG_S) & (edges + 1 <= (s + 1) * SEG_S)
        if m.sum() < 0.9 * SEG_S:                                          # 這一段資料不滿
            continue
        best, p95 = None, -1.0
        for pair, x in rse.items():                                        # Eq. 27：P95(|RSE|) 最大的那一對
            y = x[m][np.isfinite(x[m])]
            if y.size and np.percentile(np.abs(y), 95) > p95:
                best, p95 = pair, np.percentile(np.abs(y), 95)
        y = rse[best][m]
        y = y[np.isfinite(y)] * 1e3
        avg[s], sd[s] = np.abs(y).mean(), y.std()
        worst[s] = f"{best[0]}-{best[1]}"
    return avg, sd, worst, len(names)


# --------------------------------------------------------------------------- 畫表
def draw_table(rows_avg, rows_sd, out: Path, note: str):
    """rows_*：[(config_id, bytes, peripherals, [(median, iqr) or None] * 6), ...]"""
    paper_style()
    cols = [0.0, 1.15, 2.25, 3.2, 4.25]                    # 欄位左邊界（單位：吋）
    seg_w = 0.92
    W = cols[-1] + N_SEG * seg_w + 0.1
    row_h = 0.32                                          # 兩行的「Absolute Average」要放得下
    n_rows = len(rows_avg) + len(rows_sd)
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
            "AND ACQUISITION SEGMENTS", ha="center", va="center", fontsize=7.5)
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

    for block, (label, rows, rgb) in enumerate((("Absolute\nAverage", rows_avg, (0.85, 0.15, 0.55)),
                                                ("Standard\nDeviation", rows_sd, (0.15, 0.45, 0.85)))):
        vals = [c[0] for r in rows for c in r[3] if c is not None]
        lo, hi = min(vals), max(vals)
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
        ax.plot([0, W], [y, y], color="black", lw=1.0 if block else 0.6)
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
    a = ap.parse_args()

    avgs, sds, npers = [], [], set()
    for f in a.data:
        avg, sd, worst, npr = run_metrics(f, a.ci, a.tilt)
        avgs.append(avg)
        sds.append(sd)
        npers.add(npr)
        print(f"{f}：{npr} 塊板子")
        for s in range(N_SEG):
            if np.isfinite(avg[s]):
                print(f"  第 {s + 1} 段：最差的一對 {worst[s]}，|RSE| 平均 {avg[s]:.3f} ms，標準差 {sd[s]:.3f} ms")
            else:
                print(f"  第 {s + 1} 段：資料不滿 600 秒，略過")
    avgs, sds = np.array(avgs), np.array(sds)

    def med_iqr(x):
        out = []
        for s in range(N_SEG):
            y = x[:, s][np.isfinite(x[:, s])]
            out.append(None if y.size == 0 else
                       (float(np.median(y)), float(np.percentile(y, 75) - np.percentile(y, 25))))
        return out

    npr = max(npers)
    mine_avg = (MY_ID, MY_BYTES, npr, med_iqr(avgs))
    mine_sd = (MY_ID, MY_BYTES, npr, med_iqr(sds))

    print(f"\n{MY_ID}（{MY_BYTES} B、{npr} 塊板子、{len(a.data)} 次錄音）median (IQR) ms：")
    for name, r in (("Absolute Average", mine_avg), ("Standard Deviation", mine_sd)):
        print(f"  {name:18} " + "  ".join("–" if c is None else f"{c[0]:.2f} ({c[1]:.2f})" for c in r[3]))

    out = Path(a.data[0]).resolve().parent / "rse_table.png"
    note = (f"{MY_ID}: B-L475E-IOT01A + PC Bluetooth, {MY_BYTES} B, {npr} peripherals, "
            f"n = {len(a.data)} recording{'s' if len(a.data) > 1 else ''}.")
    draw_table([mine_avg], [mine_sd], out, note)
    print(f"\n已輸出：{out}")


if __name__ == "__main__":
    main()

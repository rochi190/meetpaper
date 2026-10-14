"""交付物：results/fig_compare.png（Fig.3、4、6 與模擬並排）、results/ablation_T4.md。"""
import json

import numpy as np

from .. import digitize as dg
from .. import evaluate as ev
from .. import figures as F
from ..config import load_config, set_path
from ..runner import RESULTS, ROOT, node_series, run_node_algos, write_text
from ..simulate import simulate


def _align(tp, dt, ref_tp, ref_dt, x0):
    a_p, b_p = ev.lower_line_lp(ref_tp, ref_dt)
    m = (tp > x0) & (tp < x0 + 130)
    a_s, b_s = ev.lower_line_lp(tp[m], dt[m])
    return dt + (a_p * x0 + b_p) - (a_s * x0 + b_s)


def run(cfg_path):
    fig, axs = F.plt.subplots(3, 2, figsize=(15, 11))
    # Fig.3
    c1 = load_config(ROOT / "config" / "T1_fig3.yaml")
    best1 = json.loads((RESULTS / "T1_fig3" / "summary.json").read_text())["best"]["os.median_s"]
    p = node_series(simulate(set_path(c1, "os.median_s", best1), 0))
    tp, dt = p["t_P"], p["t_C"] - p["t_P"]
    la = dg.load("fig3a", "line")
    dt = _align(tp, dt, la[:, 0], la[:, 1], 2888)
    A = F.PAPER_AX["fig3a"]
    for ax, (x, y), t in [(axs[0, 0], (la[:, 0], la[:, 1]), "論文 Fig.3a（最小延遲）"),
                          (axs[0, 1], (tp, dt), f"模擬 T1（單節點、E1 中位數 {best1*1e3:g} ms）")]:
        m = (x >= A["x"][0]) & (x <= A["x"][1])
        ax.plot(x[m], y[m], "-", lw=0.5, color=F.BLUE)
        ax.set_xlim(*A["x"])
        ax.set_ylim(*A["y"])
        ax.set_title(t)
        ax.set_ylabel("δT (s)")
    # Fig.4
    c3 = load_config(ROOT / "config" / "T3_fig4.yaml")
    b3 = json.loads((RESULTS / "T3_fig4" / "summary.json").read_text())["best"]
    for k in ["per.good", "per.bad", "per.mean_bad_events", "per.p_g2b"]:
        c3 = set_path(c3, k, b3[k])
    c3 = set_path(set_path(c3, "pauses.rate_hz", b3["spike_prob"] / 0.1), "T", 3600.0)
    p = node_series(simulate(c3, 7))
    tp, dt = p["t_P"], p["t_C"] - p["t_P"]
    la = dg.load("fig4a", "line")
    dt = _align(tp, dt, la[:, 0], la[:, 1], 3366)
    A = F.PAPER_AX["fig4a"]
    for ax, (x, y), t in [(axs[1, 0], (la[:, 0], la[:, 1]), "論文 Fig.4a（雜訊）"),
                          (axs[1, 1], (tp, dt), "模擬 T3（Gilbert–Elliott 突發 + 流量控制）")]:
        m = (x >= A["x"][0]) & (x <= A["x"][1])
        ax.plot(x[m], y[m], "-", lw=0.5, color=F.BLUE)
        ax.set_xlim(*A["x"])
        ax.set_ylim(*A["y"])
        ax.set_title(t)
        ax.set_ylabel("δT (s)")
    # Fig.6
    c4 = load_config(ROOT / "config" / "T4_fig6.yaml")
    for k, v in json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"].items():
        c4 = set_path(c4, k, v)
    sim = simulate(set_path(c4, "T", 3600.0), 11)
    p = node_series(sim, 0)
    tp, dt = p["t_P"], p["t_C"] - p["t_P"]
    st = run_node_algos(p, c4, ["sharktooth"])["sharktooth"]
    d = dg.load("fig6", "dots")
    d = d[np.argsort(d[:, 0])]
    red = dg.load("fig6", "red")
    m = tp < 145
    a_p, b_p = ev.lower_line_lp(d[d[:, 0] > 6, 0], d[d[:, 0] > 6, 1])
    a_s, b_s = ev.lower_line_lp(tp[m & (tp > 1)], dt[m & (tp > 1)])
    off = b_p - b_s
    toff = d[0, 0] - tp[0]
    F.fig6_panel(axs[2, 0], d[:, 0], d[:, 1], None, dg.load("fig6", "star").ravel(), "論文 Fig.6")
    axs[2, 0].plot(red[:, 0], red[:, 1], "-", color="r", lw=1.6, label="Output δT*")
    axs[2, 0].legend(loc="upper right", fontsize=10)
    pv = np.flatnonzero(st["pivot"][m])
    F.fig6_panel(axs[2, 1], tp[m] + toff, dt[m] + off, st["star"][m] + off, (tp[pv[-1]] + toff, dt[pv[-1]] + off),
                 "模擬 T4（4 節點 / 1 dongle，擬合參數）")
    for ax in axs[:2].ravel():
        ax.set_xlabel("t_P (s)")
    F.save(fig, RESULTS / "fig_compare.png")
    # ablation 表
    abl = (RESULTS / "T4_fig6" / "ablation_table.md").read_text(encoding="utf-8")
    write_text(RESULTS / "ablation_T4.md", f"""# 延遲從何而來：Fig.6 每個特徵的來源（T4 消融）

做法：以 T4 擬合出的完整模型（4 節點 / 1 dongle、244 B）為基準，每次關掉一個模組，3 seeds × 1200 s × 4 節點，量 δT 的特徵。
「下緣」= δT 的 LP 下界線；殘差 = δT − 下緣。

{abl}

## 一句話版（簡報用）

| Fig.6 的特徵 | 來源 | 證據（關掉後） |
|---|---|---|
| 約 30 ms 厚的連續帶 | **C1 等 connection event**（0 ~ 1 CI） | 帶寬 27 → 3 ms |
| 帶內分布均勻、下緣是平滑斜線（不是階梯） | **封包間隔 100.78 ms 與 CI 30 ms 不成整數比**，相位連續掃過 CI | 改回剛好 100 ms：下緣粗糙度 2.7 → 8 ms（階梯） |
| 帶上方 30–60 ms 的稀疏點（約 5 % > 45 ms） | **C3 重傳一次**（+1 CI；每次嘗試失敗率約 8 %） | 離群 5.2 → 0.1 % |
| 起始的垂直尖峰（約 120 ms） | **C11 第一包等連線 / notification 就緒** | 尖峰 122 → 19 ms |
| 尖峰後 3–4 點的斜坡（每包 −40 ms） | **C4 韌體流量控制**：一次只有一包在 stack 裡，背壓時每 2 個 event 才送一包 | 第 2 點 83 → 28 ms |
| δT 的斜率（≈ 244 ppm） | 時鐘：節點晶振 + 【推測】韌體 tick 換算常數（A8） | 只影響斜率 |
| PC / dongle / USB / OS（D、E 類） | 合計約 1 ms 量級，看不出來 | 關掉任一個，帶寬與分位數變化 < 0.5 ms |
""")

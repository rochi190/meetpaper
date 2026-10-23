"""畫圖：與論文相同的座標範圍與刻度；字體 ≥ 12 pt；中文用文泉驛正黑。"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "WenQuanYi Zen Hei", "Noto Sans TC"],   # 西文 / 數學用 DejaVu，中文自動退回文泉驛（沒裝就用 Noto Sans TC）
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "axes.formatter.use_mathtext": False,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

BLUE = (0.0, 0.447, 0.741)
ORANGE = (0.851, 0.325, 0.098)
PAIR_COLORS = [(0.0, 0.447, 0.741), (0.851, 0.325, 0.098), (0.929, 0.694, 0.125),
               (0.494, 0.184, 0.557), (0.467, 0.675, 0.188), (0.302, 0.745, 0.933)]

# 論文各圖的座標範圍
PAPER_AX = {
    "fig3a": dict(x=(2888, 3017), y=(1.015, 1.065)),
    "fig3b": dict(x=(2951.2, 2957.3), y=(1.024, 1.062)),
    "fig3c": dict(x=(2888, 3017), y=(14.2, 14.8)),
    "fig4a": dict(x=(3366, 3494), y=(1.16, 1.41)),
    "fig4b": dict(x=(3427.8, 3433.9), y=(1.165, 1.335)),
    "fig4c": dict(x=(3366, 3494), y=(28, 55)),
    "fig6": dict(x=(0, 145), y=(0.28, 0.42)),
    "fig10": dict(x=(0, 4000), y=(-4, 2)),
}


def align_offset(tp, dt, target_intercept, slope_ppm=None):
    """β 是任意常數（PC 與節點的開機時間差）：把模擬 δT 平移，使下緣截距等於論文的值。"""
    from .evaluate import lower_line_lp
    m = tp > tp[0] + 1.0
    a, b = lower_line_lp(tp[m], dt[m])
    return target_intercept - b


def fig6_panel(ax, tp, dt, star=None, pivot=None, title=None, xr=(0, 145), yr=(0.28, 0.42)):
    ax.plot(tp, dt, ".", ms=2.5, color=BLUE, label="Input δT")
    if star is not None:
        ax.plot(tp, star, "-", color="r", lw=1.6, label="Output δT*")
    if pivot is not None:
        ax.plot([pivot[0]], [pivot[1]], "*", color="lime", ms=12, mec="g", label="Pivot point")
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_xlabel("t_P (s)")
    ax.set_ylabel("δT (s)")
    ax.legend(loc="upper right", fontsize=10)
    if title:
        ax.set_title(title)


def three_panel(axs, tp, dt, sig, zoom, which="fig3"):
    """Fig.3 / Fig.4 格式：(a) δT、(b) 放大、(c) σ_100。"""
    A = PAPER_AX[which + "a"]
    B = PAPER_AX[which + "b"]
    C = PAPER_AX[which + "c"]
    m = (tp >= A["x"][0]) & (tp <= A["x"][1])
    axs[0].plot(tp[m], dt[m], "-", lw=0.5, color=BLUE)
    axs[0].set_xlim(*A["x"])
    axs[0].set_ylim(*A["y"])
    axs[0].set_ylabel("δT (s)")
    axs[0].add_patch(plt.Rectangle((B["x"][0], B["y"][0]), B["x"][1] - B["x"][0], B["y"][1] - B["y"][0],
                                   fill=False, ec="r", lw=1.5))
    m = (tp >= B["x"][0]) & (tp <= B["x"][1])
    axs[1].plot(tp[m], dt[m], "-", lw=0.6, color=BLUE)
    axs[1].plot(tp[m], dt[m], "x", ms=5, color=ORANGE, mew=1.5)
    axs[1].set_xlim(*B["x"])
    axs[1].set_ylim(*B["y"])
    axs[1].set_ylabel("δT (s)")
    for s in axs[1].spines.values():
        s.set_edgecolor("r")
        s.set_linewidth(1.5)
    m = (tp >= C["x"][0]) & (tp <= C["x"][1])
    axs[2].plot(tp[m], sig[m] * 1e3, "-", lw=0.8, color=BLUE)
    axs[2].set_xlim(*C["x"])
    axs[2].set_ylim(*C["y"])
    axs[2].set_ylabel("σ100 (ms)")
    axs[2].set_xlabel("t_P (s)")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def paper_vs_sim_threepanel(path, paper, sim, which="fig3", sim_label="模擬"):
    """左：論文（數位化），右：模擬。paper/sim = dict(tp, dt, sig)。"""
    fig, axs = plt.subplots(3, 2, figsize=(13, 8.5), gridspec_kw=dict(height_ratios=[1, 1, 1]))
    three_panel(axs[:, 0], paper["tp"], paper["dt"], paper["sig"], None, which)
    three_panel(axs[:, 1], sim["tp"], sim["dt"], sim["sig"], None, which)
    axs[0, 0].set_title(f"論文 {which.upper()}（數位化）")
    axs[0, 1].set_title(sim_label)
    save(fig, path)

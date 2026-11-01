"""
只畫收到的原始資料（不跑 SharkTooth）：每一包一個點，δT = t_C − t_P。

用法：
    python st_rawplot.py runs/20261031_163700
    python st_rawplot.py runs/20261031_163700 --span 140      # 只畫前 140 秒

輸出（放在同一個資料夾）：
    raw_<節點>.png
      上圖：原始 δT（秒），就是 t_C − t_P，沒有任何處理
      下圖：同一批點減掉最小值（ms），看得出延遲的分布：帶寬、重傳、斜率

δT 的絕對值是什麼：
    t_C 和 t_P 的零點不同，δT 的絕對值 ≈ 兩個零點的差（論文的 β），不是傳輸延遲。
    st_receiver.py 2026-11-01b 起：t_C 從「全部連好、送出 notify 前」算起，β 約零點幾秒（和論文一樣）。
    更早的版本：t_C 從「程式開始」算起，β 會多出搜尋 + 連線的好幾秒。
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from st_analyze import load, paper_style, save  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def read_events(folder: Path):
    ev = {}
    f = folder / "events.csv"
    if not f.exists():
        return ev
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["event"] in ("connected", "notify_request", "notify_on") and r["node"]:
                ev.setdefault(r["node"], {})[r["event"]] = float(r["t_C"])
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="st_receiver.py 建的資料夾")
    ap.add_argument("--span", type=float, default=None, help="只畫前幾秒（預設全部）")
    a = ap.parse_args()

    paper_style()
    folder = Path(a.data)
    nodes = load(folder)
    ev = read_events(folder)
    for name, d in nodes.items():
        tP, tC = d["tP"], d["t_C"]
        dT = tC - tP
        m = np.ones(tP.size, bool) if a.span is None else tP <= tP[0] + a.span
        base = dT[m].min()

        e = ev.get(name, {})
        print(f"\n{name}：{tP.size} 包")
        print(f"  第一包：t_P = {tP[0]:.4f} s，t_C = {tC[0]:.4f} s，δT = {dT[0]:.4f} s")
        if "connected" in e:
            print(f"  程式開始 → 連上 {name}：{e['connected']:.3f} s")
        if "notify_request" in e:
            print(f"  程式開始 → 送出 notify（板子 t_P 歸零）：{e['notify_request']:.3f} s")
        if "notify_on" in e:
            print(f"  程式開始 → notify 打開完成：{e['notify_on']:.3f} s")
        print(f"  δT 最小值 {base:.4f} s ≈ 兩個時鐘的零點差（論文的 β），不是延遲")
        rel = (dT - base) * 1e3
        print(f"  減掉最小值後：中位數 {np.median(rel[m]):.1f} ms，95% 的點在 {np.percentile(rel[m], 95):.1f} ms 以內，"
              f"最大 {rel[m].max():.1f} ms")

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True)
        a1.plot(tP[m], dT[m], ".", ms=2.0, color="#0072BD")
        a1.ticklabel_format(axis="y", useOffset=False)
        a1.set_ylabel(r"$\delta T = t_C - t_P$ (s)")
        a1.set_title(f"Raw packets ({name})")
        a2.plot(tP[m], rel[m], ".", ms=2.0, color="#0072BD")
        a2.set_ylabel(r"$\delta T - \min$ (ms)")
        a2.set_xlabel(r"$t_P$ (s)")
        a2.set_ylim(-3, max(np.percentile(rel[m], 99.8) * 1.15, 40))
        a2.set_xlim(0, tP[m][-1])
        fig.tight_layout()
        save(fig, folder / f"raw_{name}")
    print(f"\n已輸出到 {folder}：raw_<節點>.png")


if __name__ == "__main__":
    main()

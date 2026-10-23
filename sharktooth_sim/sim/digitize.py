"""
T0：從論文 PDF 數位化 Fig.3 / Fig.4 / Fig.6 / Fig.10 / Fig.12。

論文的圖是向量圖（pgfplots 輸出），所以不需要用顏色門檻去「猜」像素：
直接讀 PDF 的繪圖指令（PyMuPDF `page.get_drawings()`），
  - 座標軸：白色填滿的矩形（每個子圖一個）
  - 刻度：灰色短線（落在座標軸下緣 / 左緣），刻度數值由下面的 FIGS 表手動給定（從頁面影像讀出）
  - 資料：藍色折線（Fig.3/4）、藍色圓點（Fig.6）、紅色折線（Fig.6 δT*）、綠色星號（樞紐點）、
          橘色叉號（Fig.3b/4b 的封包點）、彩色填滿多邊形（Fig.10 的 RSE 曲線）
像素↔座標是線性轉換，以刻度最小平方擬合，殘差會寫進 targets.json 供檢查。

用法：python -m sim.digitize   （輸出到 data/digitized/ 與 results/T0_digitize/）
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT.parent / ("SharkTooth_A_Scalable_Real-Time_Algorithm_for_BLE-Based_"
                     "Wireless_Body_Sensor_Networks_Synchronization.pdf")
OUT = ROOT / "data" / "digitized"

BLUE = (0.0, 0.447, 0.741)
ORANGE = (0.851, 0.325, 0.098)
RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
GRAY = (0.5, 0.5, 0.5)
WHITE = (1.0, 1.0, 1.0)

# 子圖：頁碼（0 起算）、第幾個白色座標框（依 (y, x) 排序）、x / y 刻度值
FIGS = {
    "fig3a": dict(page=5, axes=0, xt=[2900, 2920, 2940, 2960, 2980, 3000], yt=[1.02, 1.04, 1.06]),
    "fig3b": dict(page=5, axes=2, xt=[2952, 2953, 2954, 2955, 2956, 2957], yt=[1.03, 1.04, 1.05, 1.06]),
    "fig3c": dict(page=5, axes=4, xt=[2900, 2920, 2940, 2960, 2980, 3000], yt=[14.2, 14.4, 14.6, 14.8]),
    "fig4a": dict(page=5, axes=1, xt=[3380, 3400, 3420, 3440, 3460, 3480], yt=[1.2, 1.3, 1.4]),
    "fig4b": dict(page=5, axes=3, xt=[3428, 3429, 3430, 3431, 3432, 3433], yt=[1.2, 1.25, 1.3]),
    "fig4c": dict(page=5, axes=5, xt=[3380, 3400, 3420, 3440, 3460, 3480], yt=[30, 40, 50]),
    "fig6": dict(page=7, axes=0, xt=[0, 20, 40, 60, 80, 100, 120, 140], yt=[0.3, 0.35, 0.4]),
    "fig10": dict(page=11, axes=0, xt=[0, 1000, 2000, 3000, 4000], yt=[-4, -2, 0, 2]),
}
# Fig.12：6 個分段的 σ²100（x, ms²）vs RSE P95（y, ms）
FIG12 = {
    1: dict(axes=0, xt=[200, 400, 600, 800]), 2: dict(axes=1, xt=[200, 400, 600, 800, 1000]),
    3: dict(axes=2, xt=[200, 400, 600, 800]), 4: dict(axes=3, xt=[200, 400, 600]),
    5: dict(axes=4, xt=[200, 400, 600, 800]), 6: dict(axes=5, xt=[200, 300, 400, 500]),
}
FIG12_COLORS = {(0.0, 0.447, 0.741): 2, (0.851, 0.325, 0.098): 4, (0.929, 0.694, 0.125): 6,
                (0.494, 0.184, 0.557): 8, (0.467, 0.675, 0.188): 10, (0.302, 0.745, 0.933): 12}


def _rgb(c):
    return tuple(round(v, 3) for v in c) if c else ()


def _close(a, b, tol=0.01):
    return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))


def _axes_rects(drawings):
    rects = [d["rect"] for d in drawings
             if d["type"] == "f" and _close(_rgb(d.get("fill")), WHITE)
             and len(d["items"]) == 1 and d["items"][0][0] == "re" and d["rect"].width > 100]
    # 依列（y）再依欄（x）排序：同一列的 y 差在 5 pt 以內
    rects.sort(key=lambda r: (round(r.y0 / 20), r.x0))
    return rects


def _ticks(drawings, ax):
    """回傳 x 刻度像素位置（下緣）與 y 刻度像素位置（左緣）。"""
    xs, ys = [], []
    for d in drawings:
        if not _close(_rgb(d.get("color")), GRAY) or len(d["items"]) != 1 or d["items"][0][0] != "l":
            continue
        p, q = d["items"][0][1], d["items"][0][2]
        if abs(p.x - q.x) < 0.05 and abs(max(p.y, q.y) - ax.y1) < 0.6 and ax.x0 - 0.5 <= p.x <= ax.x1 + 0.5:
            xs.append(p.x)
        if abs(p.y - q.y) < 0.05 and abs(min(p.x, q.x) - ax.x0) < 0.6 and ax.y0 - 0.5 <= p.y <= ax.y1 + 0.5:
            ys.append(p.y)
    if not xs and not ys:
        # Fig.10：刻度畫成灰色填滿的細長矩形
        hy = []
        for d in drawings:
            if d["type"] != "f" or not _close(_rgb(d.get("fill")), GRAY):
                continue
            for it in d["items"]:
                if it[0] == "re":
                    r = it[1]
                    if r.width < 0.5 and abs(r.y1 - ax.y1) < 0.6:
                        xs.append((r.x0 + r.x1) / 2)
                elif it[0] == "l":
                    p, q = it[1], it[2]
                    if abs(p.y - q.y) < 0.01 and abs(min(p.x, q.x) - ax.x0) < 0.6 and abs(p.x - q.x) > 1:
                        hy.append(p.y)
        hy.sort()
        groups = []
        for v in hy:
            if groups and v - groups[-1][-1] < 0.5:
                groups[-1].append(v)
            else:
                groups.append([v])
        ys = [float(np.mean(g)) for g in groups]
    return sorted(set(round(v, 2) for v in xs)), sorted(set(round(v, 2) for v in ys), reverse=True)


def _fit(pix, vals):
    A = np.vstack([pix, np.ones(len(pix))]).T
    coef, *_ = np.linalg.lstsq(A, np.asarray(vals, float), rcond=None)
    resid = A @ coef - vals
    return coef, float(np.max(np.abs(resid))) if len(vals) else np.nan


def _inside(r, ax, pad=3.0):
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    return ax.x0 - pad <= cx <= ax.x1 + pad and ax.y0 - pad <= cy <= ax.y1 + pad


def digitize_all(pdf=PDF):
    import pymupdf

    doc = pymupdf.open(pdf)
    out, calib = {}, {}
    cache = {}
    for name, spec in FIGS.items():
        page = doc[spec["page"]]
        if spec["page"] not in cache:
            cache[spec["page"]] = page.get_drawings()
        drs = cache[spec["page"]]
        ax = _axes_rects(drs)[spec["axes"]]
        xt_pix, yt_pix = _ticks(drs, ax)
        if len(xt_pix) != len(spec["xt"]) or len(yt_pix) != len(spec["yt"]):
            raise RuntimeError(f"{name}: 刻度數量不符 x {len(xt_pix)}/{len(spec['xt'])} "
                               f"y {len(yt_pix)}/{len(spec['yt'])}")
        (ax_, bx), ex = _fit(np.array(xt_pix), spec["xt"])
        (ay_, by), ey = _fit(np.array(yt_pix), spec["yt"])
        X = lambda px: ax_ * np.asarray(px) + bx  # noqa: E731
        Y = lambda py: ay_ * np.asarray(py) + by  # noqa: E731
        calib[name] = dict(axes_pt=[ax.x0, ax.y0, ax.x1, ax.y1], x_tick_pt=xt_pix, y_tick_pt=yt_pix,
                           x_fit_maxerr=ex, y_fit_maxerr=ey,
                           x_per_pt=float(ax_), y_per_pt=float(ay_))
        series = defaultdict(list)
        # 圖例框（白底、深灰框）之後畫、且落在框內的是圖例符號，不是資料（框之前畫的資料點保留）
        legends = [(i, d["rect"]) for i, d in enumerate(drs) if d["type"] == "fs"
                   and _close(_rgb(d.get("fill")), WHITE) and _close(_rgb(d.get("color")), (0.15, 0.15, 0.15))]
        for i, d in enumerate(drs):
            if not _inside(d["rect"], ax) or any(i > li and _inside(d["rect"], lg, pad=0) for li, lg in legends):
                continue
            col, fill = _rgb(d.get("color")), _rgb(d.get("fill"))
            items = d["items"]
            if d["type"] == "s" and _close(col, BLUE) and len(items) > 10:          # 折線
                pts = [it[1] for it in items] + [items[-1][2]]
                series["line"].append(np.array([[X(p.x), Y(p.y)] for p in pts]))
            elif d["type"] == "fs" and _close(fill, BLUE) and len(items) == 4:       # 圓點
                r = d["rect"]
                series["dots"].append([X((r.x0 + r.x1) / 2), Y((r.y0 + r.y1) / 2)])
            elif d["type"] == "s" and _close(col, RED) and len(items) > 10:           # δT*
                pts = [it[1] for it in items] + [items[-1][2]]
                series["red"].append(np.array([[X(p.x), Y(p.y)] for p in pts]))
            elif d["type"] == "s" and _close(col, ORANGE) and len(items) == 1:        # 叉號
                r = d["rect"]
                series["xmark"].append((round((r.x0 + r.x1) / 2, 2), round((r.y0 + r.y1) / 2, 2)))
            elif d["type"] == "s" and _close(col, GREEN) and len(items) == 1:         # 星號
                r = d["rect"]
                series["star"].append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
            elif d["type"] == "f" and len(items) > 1000 and fill:                    # Fig.10 粗線
                pts = np.array([[it[1].x, it[1].y] for it in items])
                series["poly:" + ",".join(f"{v:.3f}" for v in fill)].append(
                    np.column_stack([X(pts[:, 0]), Y(pts[:, 1])]))
        rec = {}
        if series["line"]:
            rec["line"] = max(series["line"], key=len)
        if series["dots"]:
            rec["dots"] = np.array(series["dots"])
        if series["red"]:
            rec["red"] = np.vstack(series["red"])
        if series["xmark"]:
            uniq = sorted(set(series["xmark"]))
            rec["xmark"] = np.array([[X(px), Y(py)] for px, py in uniq])
        if series["star"]:
            s = np.array(series["star"])
            rec["star"] = np.array([[X(s[:, 0].mean()), Y(s[:, 1].mean())]])
        for k, v in series.items():
            if k.startswith("poly:"):
                rec[k] = v[0]
        out[name] = rec
    return out, calib


def digitize_fig12(pdf=PDF):
    """回傳 list of dict(seg, config, sigma2_100, rse_p95)。圓 = A、菱形 = B、三角形 = C。"""
    import pymupdf
    drs = pymupdf.open(pdf)[14].get_drawings()
    axes = _axes_rects(drs)
    out = []
    for seg, spec in FIG12.items():
        ax = axes[spec["axes"]]
        xt_pix, yt_pix = _ticks(drs, ax)
        (axx, bx), _ = _fit(np.array(xt_pix), spec["xt"])
        (ayy, by), _ = _fit(np.array(yt_pix), [1, 2, 3, 4])
        for d in drs:
            if not _inside(d["rect"], ax, pad=1.0):
                continue
            fill = _rgb(d.get("fill"))
            col = next((n for c, n in FIG12_COLORS.items() if _close(fill, c)), None)
            if col is None:
                continue
            ops = "".join(it[0] for it in d["items"])
            shape = {"cccc": "A", "llll": "B", "lll": "C"}.get(ops)
            if shape is None:
                continue
            r = d["rect"]
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if shape == "C":   # 三角形的重心
                pts = [it[1] for it in d["items"]]
                cx = float(np.mean([p.x for p in pts]))
                cy = float(np.mean([p.y for p in pts]))
            out.append(dict(seg=seg, config=f"{shape}{col}", sigma2_100=float(axx * cx + bx),
                            rse_p95=float(ayy * cy + by)))
    return out


def save(out, calib, outdir=OUT):
    outdir.mkdir(parents=True, exist_ok=True)
    for name, rec in out.items():
        for k, arr in rec.items():
            fn = outdir / f"{name}_{k.replace(':', '_').replace(',', '-')}.csv"
            np.savetxt(fn, np.asarray(arr), delimiter=",", header="x,y", comments="", fmt="%.7g")
    (outdir / "calibration.json").write_text(json.dumps(calib, indent=1))


def load(name, key):
    fn = OUT / f"{name}_{key}.csv"
    return np.loadtxt(fn, delimiter=",", skiprows=1)




# ---------------------------------------------------------------------------
# 目標統計量（targets.json）
# ---------------------------------------------------------------------------
def build_targets():
    from . import evaluate as ev
    T = {}
    # Fig.6
    d = load("fig6", "dots")
    d = d[np.argsort(d[:, 0])]
    tp, dt = d[:, 0], d[:, 1]
    stats, r = ev.band_stats(tp, dt, skip_start_s=1.0)
    line = (stats["slope_ppm"] * 1e-6, stats["intercept"])
    qline = ev.quantile_line(tp[tp > tp[0] + 1], dt[tp > tp[0] + 1], 0.05)
    piv = load("fig6", "star").ravel()
    red = load("fig6", "red")
    T["fig6"] = dict(
        n_points=int(len(tp)), tp_first=float(tp[0]), tp_last=float(tp[-1]),
        pkt_period_ms=float(np.median(np.diff(tp)) * 1e3),
        pkt_period_mean_ms=float((tp[-1] - tp[0]) / (len(tp) - 1) * 1e3),
        lost_packets=int(np.sum(np.round(np.diff(tp) / np.median(np.diff(tp))) - 1)),
        lower_edge_lp=dict(slope_ppm=stats["slope_ppm"], intercept=stats["intercept"]),
        lower_edge_q05=dict(slope_ppm=qline[0] * 1e6, intercept=qline[1]),
        residual_ms={k: stats[k] for k in ["P1", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P99", "max"]},
        outlier_frac_gt45ms=stats["outlier_frac"], frac_gt_CI=stats["frac_gt_CI"], frac_gt_2CI=stats["frac_gt_2CI"],
        startup=ev.startup_spike(tp, dt, line, 10),
        startup_drain_ms_per_pkt=float(np.mean(np.diff(dt[:4])) * 1e3),
        pivot=dict(tp=float(piv[0]), dT=float(piv[1])),
        dTstar_end=float(red[-1, 1]), dTstar_slope_last_80s_ppm=float(
            np.polyfit(red[red[:, 0] > 60, 0], red[red[:, 0] > 60, 1], 1)[0] * 1e6),
        residual_hist_2p5ms=np.histogram(r, bins=np.arange(0, 105, 2.5))[0].tolist(),
    )
    # Fig.3 / Fig.4
    for fig in ["fig3", "fig4"]:
        la = load(fig + "a", "line")
        lb = load(fig + "b", "xmark")
        lc = load(fig + "c", "line")
        st_a = ev.sawtooth_metrics(la[:, 0], la[:, 1])
        st_b = ev.sawtooth_metrics(lb[:, 0], lb[:, 1])
        bs, rr = ev.band_stats(la[:, 0], la[:, 1], skip_start_s=0)
        # 尖峰：殘差 > 1.5 CI 的局部最大值高度
        is_pk = (rr[1:-1] > 45) & (rr[1:-1] >= rr[:-2]) & (rr[1:-1] >= rr[2:])
        pk = rr[1:-1][is_pk]
        d2 = np.diff(la[:, 1]) * 1e3
        drain = d2[(d2 < -30) & (d2 > -50)]
        T[fig] = dict(
            pkt_period_ms_zoom=st_b["pkt_period_mean_ms"], pkt_period_ms=st_a["pkt_period_mean_ms"],
            packets_per_cycle_zoom=st_b["packets_per_cycle"], sawtooth_period_s_zoom=st_b["period_s"],
            step_down_ms=st_b["step_down_ms"], step_up_ms=st_b["step_up_ms"],
            p2p_ms_zoom=st_b["p2p_ms"], p2p_ms=st_a["p2p_ms"],
            slope_ppm=bs["slope_ppm"],
            residual_ms={k: bs[k] for k in ["P10", "P50", "P90", "P95", "P99", "max"]},
            frac_gt_CI=bs["frac_gt_CI"], frac_gt_2CI=bs["frac_gt_2CI"],
            sigma100_ms=dict(mean=float(lc[:, 1].mean()), min=float(lc[:, 1].min()), max=float(lc[:, 1].max()),
                             p5=float(np.percentile(lc[:, 1], 5)), p95=float(np.percentile(lc[:, 1], 95))),
            spike_heights_ms=dict(n=int(len(pk)), per_100s=float(len(pk) / (la[-1, 0] - la[0, 0]) * 100),
                                  q=np.percentile(pk, [10, 50, 90]).tolist() if len(pk) else []),
            drain_step_ms=float(np.median(drain)) if len(drain) else None,
            n_drain_steps=int(len(drain)),
        )
    # Fig.10
    import glob
    names = {"0.000-0.447-0.741": "1-2", "0.851-0.325-0.098": "1-3", "0.929-0.694-0.125": "1-4",
             "0.494-0.184-0.557": "2-3", "0.467-0.675-0.188": "2-4", "0.302-0.745-0.933": "3-4"}
    f10 = {}
    for fn in sorted(glob.glob(str(OUT / "fig10_poly_*.csv"))):
        key = Path(fn).stem.replace("fig10_poly_", "")
        p = np.loadtxt(fn, delimiter=",", skiprows=1)
        segs = []
        for k in range(6):
            m = (p[:, 0] >= 600 * k) & (p[:, 0] < 600 * (k + 1))
            segs.append(dict(median=float(np.median(p[m, 1])), p5=float(np.percentile(p[m, 1], 5)),
                             p95=float(np.percentile(p[m, 1], 95))))
        f10[names.get(key, key)] = dict(first_t=float(p[:, 0].min()), segments=segs)
    T["fig10"] = dict(pairs=f10, steady_offset_range_ms=[
        min(v["segments"][5]["median"] for v in f10.values()),
        max(v["segments"][5]["median"] for v in f10.values())])
    # Table II 的穩態推論：A2 後段 mean|RSE| ≈ 0.8σ → 幾乎全是零均值雜訊
    import csv
    rows = list(csv.DictReader(open(ROOT / "data" / "paper_table2.csv")))
    T["table2_source"] = "data/paper_table2.csv"
    T["table2_steady_floor_std_ms"] = {
        g: float(np.mean([float(r["seg6"]) for r in rows if r["metric"] == "std" and r["config"].startswith(g)]))
        for g in "ABC"}
    return T


def overlay_check(outpng):
    """把數位化的點疊回 PDF 頁面影像（肉眼對齊檢查）。"""
    import pymupdf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    doc = pymupdf.open(PDF)
    calib = json.loads((OUT / "calibration.json").read_text())
    panels = [("fig3a", "line"), ("fig3b", "xmark"), ("fig4a", "line"), ("fig4b", "xmark"),
              ("fig6", "dots"), ("fig10", None)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for axp, (name, key) in zip(axs.ravel(), panels):
        c = calib[name]
        x0, y0, x1, y1 = c["axes_pt"]
        page = doc[FIGS[name]["page"]]
        zoom = 4
        clip = pymupdf.Rect(x0 - 2, y0 - 2, x1 + 2, y1 + 2)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]
        axp.imshow(img, extent=[clip.x0, clip.x1, clip.y1, clip.y0])
        # 資料座標 → pt：反解線性轉換
        (axx, bx) = np.polyfit(c["x_tick_pt"], FIGS[name]["xt"], 1)
        (ayy, by) = np.polyfit(c["y_tick_pt"], FIGS[name]["yt"], 1)
        if key is None:
            import glob
            for fn in glob.glob(str(OUT / "fig10_poly_*.csv")):
                p = np.loadtxt(fn, delimiter=",", skiprows=1)[::7]
                axp.plot((p[:, 0] - bx) / axx, (p[:, 1] - by) / ayy, ",", color="k", alpha=0.6)
        else:
            p = load(name, key)
            axp.plot((p[:, 0] - bx) / axx, (p[:, 1] - by) / ayy, "+", ms=3, color="lime", mew=0.6)
        if name == "fig6":
            r = load("fig6", "red")
            axp.plot((r[:, 0] - bx) / axx, (r[:, 1] - by) / ayy, "-", color="k", lw=0.6)
        axp.set_title(f"{name}: 原圖 + 數位化點（綠/黑）", fontsize=11)
        axp.set_xticks([])
        axp.set_yticks([])
    fig.tight_layout()
    fig.savefig(outpng, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    o, c = digitize_all()
    save(o, c)
    for n, r in o.items():
        print(n, {k: np.asarray(v).shape for k, v in r.items()},
              "fit err x/y:", f"{c[n]['x_fit_maxerr']:.2e}", f"{c[n]['y_fit_maxerr']:.2e}")

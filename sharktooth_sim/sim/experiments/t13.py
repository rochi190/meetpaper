"""T13：暖機能不能解釋論文 Table II / Fig.10–12「第 1 段（前 10 分鐘）比較大、之後收斂」？

T6 的 18 個配置，沿用 T6 的校正（每個配置的失敗率、V4 量測模型），只在節點時鐘加暖機：
  off          = 不加（T6 V4）；
  bench / body = T12 的物理模型（ABS04W 拋物線 + 自熱 / 皮膚溫度、PC 與 ADC 計時器溫度）；
  dX           = 不管物理，每個節點的頻率在暖機時直接變 U(−X, +X) ppm（τ 3–10 min），找「要多大才看得出來」。
兩種「前期誤差大」的來源有不同的特徵：
  暖機          — 同一張桌子、同一批板子，每個配置都一樣 → Fig.12 第 1 段迴歸線整條往上（截距變大）；
  SharkTooth 收斂 — 連線雜訊越大收斂越慢 → 第 1 段迴歸線變陡（斜率變大）。
"""
import copy
import glob
import json

import numpy as np

from .. import figures as F
from ..config import load_config, set_path
from ..digitize import OUT as DIG
from ..digitize import digitize_fig12
from ..runner import RESULTS, outdir, pmap, write_json, write_text
from . import t12
from .t6 import _main_job, cfg_for, paper_table

LETTER = {17: "A", 127: "B", 244: "C"}
SCEN_PLOT = ["off", "bench", "d10"]                                     # 圖上只畫三種，其餘在表格
SCEN_COLOR = {"off": "#2a78d6", "bench": "#eb6834", "d10": "#1baf7a"}  # 參考色盤 1–3 格
SCEN_SHORT = {"off": "不加暖機", "bench": "桌上（物理）", "d10": "每節點 ±10 ppm"}
PAIR_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # 參考色盤 1–6 格
INK, INK2 = "#0b0b0b", "#52514e"
FIG10_FILES = {"0.000-0.447-0.741": "1-2", "0.851-0.325-0.098": "1-3", "0.929-0.694-0.125": "1-4",
               "0.494-0.184-0.557": "2-3", "0.467-0.675-0.188": "2-4", "0.302-0.745-0.933": "3-4"}


def base_cfg(cfg_path):
    cfg0 = load_config(cfg_path)
    bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
    for k, v in bp.items():
        if not k.startswith("per."):
            cfg0 = set_path(cfg0, k, v)
    cfg0["t12"] = load_config(cfg_path.parent / cfg0["physical_from"])
    return cfg0


def job_cfg(cfg0, scen, seed, c):
    """配置 c（T6 校正後）+ 情境 → 這次擷取的設定。"""
    sc = cfg0["scenarios"][scen]
    if sc.get("phys"):
        t = cfg0["t12"]
        c = dict(c, **{k: t[k] for k in ("room", "crystal", "pc", "adc_temp_coeff_ppm_per_C", "scenarios")})
        return t12.job_cfg(c, sc["phys"], seed)
    if sc.get("drift_ppm"):
        c = copy.deepcopy(c)
        x = float(sc["drift_ppm"])
        # ρ += k1 (T − T_start)，T 從 0 升到 1：頻率在 τ 內變 k1 ppm
        c["clock_node"]["temp"] = dict(enabled=True, T_start=0.0, dT_C=1.0, k_ppm_per_C2=0.0,
                                       tau_s=cfg0["warm_tau_s"], k1_ppm_per_C={"uniform": [-x, x]})
    return c


def fig10_dev(pairs_t, pairs_v, wins, ref):
    """每一對在 wins 各時間窗的平均，相對最後 ref 窗的偏移；回傳各窗「最大的那一對」的 |偏移|（ms）。"""
    out = []
    for a, b in wins:
        d = []
        for t, v in zip(pairs_t, pairs_v):
            m = (t >= a) & (t < b)
            r = (t >= ref[0]) & (t < ref[1])
            if m.any() and r.any():
                d.append(abs(np.nanmean(v[m]) - np.nanmean(v[r])))
        out.append(float(np.max(d)) if d else np.nan)
    return out


def paper_fig10(cfg0):
    ts, vs, pairs = [], [], {}
    for fn in sorted(glob.glob(str(DIG / "fig10_poly_*.csv"))):
        key = fn.split("fig10_poly_")[1][:-4]
        p = np.loadtxt(fn, delimiter=",", skiprows=1)
        o = np.argsort(p[:, 0])
        ts.append(p[o, 0])
        vs.append(p[o, 1])
        pairs[FIG10_FILES.get(key, key)] = (p[o, 0], p[o, 1])
    return fig10_dev(ts, vs, cfg0["paper_fig10_windows_s"], cfg0["paper_fig10_ref_s"]), pairs


def _job(args):
    cfg, seed, v4, wins, ref = args
    r = _main_job((cfg, seed, {"V4": v4}))
    out = dict(seg=r["V4"]["cc"], sigma2=r["sigma2"])
    if "pairs" in r["V4"]:
        pv = [np.array(v, float) * 1e3 for v in r["V4"]["pairs"].values()]   # s → ms
        pt = [np.arange(len(v)) * 10.0 + 5.0 for v in pv]
        out["fig10"] = fig10_dev(pt, pv, wins, ref)
        if seed == 0:
            out["pairs"] = r["V4"]["pairs"]
    return out


def fig12_lines(points):
    """每段：P95 = a + b σ²100 的最小平方線；回傳 a、b、以及 σ²100 = 225 ms² 時的值。"""
    out = []
    for seg in range(1, 7):
        x = np.array([p["sigma2_100"] for p in points if p["seg"] == seg])
        y = np.array([p["rse_p95"] for p in points if p["seg"] == seg])
        b, a = np.polyfit(x, y, 1)
        out.append(dict(a=float(a), b100=float(b * 100), at225=float(a + b * 225)))
    return out


def run(cfg_path):
    cfg0 = base_cfg(cfg_path)
    od = outdir(cfg0["name"])
    S6 = json.loads((RESULTS / "T6_tableII" / "summary.json").read_text())
    calib, v4 = S6["calibration"], S6["variants"]["V4"]
    confs = [(n, L) for L in cfg0["bytes_list"] for n in cfg0["nodes_list"]]
    names = [f"{LETTER[L]}{n}" for (n, L) in confs]
    scens = list(cfg0["scenarios"])
    wins, ref = cfg0["paper_fig10_windows_s"], cfg0["paper_fig10_ref_s"]
    jobs = []
    for sc in scens:
        for (n, L), nm in zip(confs, names):
            c = cfg_for(cfg0, n, L, calib[nm]["p"])
            c = set_path(c, "dongle.base_sd_s", cfg0["dongle_path_sd_s"])
            jobs += [(job_cfg(cfg0, sc, s, c), s, v4, wins, ref) for s in cfg0["seeds"]]
    res = pmap(_job, jobs)
    ns = len(cfg0["seeds"])
    out = {}
    i = 0
    for sc in scens:
        tab, f10, pairs0 = {}, [], {}
        for nm in names:
            rr = res[i:i + ns]
            i += ns
            t = {met: np.nanmedian(np.array([[s[met] for s in r["seg"]] for r in rr], float), axis=0).tolist()
                 for met in ("absavg", "std", "p95")}
            t["sigma2"] = np.median([r["sigma2"] for r in rr], axis=0).tolist()
            tab[nm] = t
            if "fig10" in rr[0]:
                f10 += [r["fig10"] for r in rr]
                pairs0[nm] = rr[0]["pairs"]
        f10 = np.array(f10)
        out[sc] = dict(table=tab, pairs_seed0=pairs0,
                       fig10=dict(med=np.nanmedian(f10, axis=0).tolist(),
                                  p5=np.nanpercentile(f10, 5, axis=0).tolist(),
                                  p95=np.nanpercentile(f10, 95, axis=0).tolist()))
    summ = summarize(cfg0, out, names)
    write_json(od / "summary.json", dict(scenarios=out, summary=summ))
    figures(od, cfg0, out, summ, names)
    write_readme(od, cfg0, summ)


def report_only(cfg_path):
    """從 summary.json 重畫圖與 README（不重跑模擬）。"""
    cfg0 = base_cfg(cfg_path)
    od = outdir(cfg0["name"])
    S = json.loads((od / "summary.json").read_text())
    names = list(S["scenarios"]["off"]["table"])
    summ = summarize(cfg0, S["scenarios"], names)
    figures(od, cfg0, S["scenarios"], summ, names)
    write_readme(od, cfg0, summ)


def summarize(cfg0, out, names):
    P = paper_table()
    f12 = digitize_fig12()
    groups = {"2 節點": [n for n in names if n[1:] == "2"], "12 節點": [n for n in names if n[1:] == "12"],
              "全部": names}

    def excess(get):
        """第 1 段 − 第 2–6 段平均。"""
        return {g: float(np.mean([get(nm)[0] - np.mean(get(nm)[1:]) for nm in ns])) for g, ns in groups.items()}

    paper = dict(excess={met: excess(lambda nm, met=met: P[(met, nm)]) for met in ("absavg", "std")},
                 fig12=fig12_lines(f12), fig10=paper_fig10(cfg0)[0])
    s2_paper = {(p["config"], p["seg"]): p["sigma2_100"] for p in f12}
    sc_sum = {}
    for sc, o in out.items():
        tab = o["table"]
        e_all, e1, e26 = [], [], []
        for nm in names:
            for met in ("absavg", "std"):
                d = np.array(tab[nm][met]) - np.array(P[(met, nm)])
                e_all += list(d)
                e1.append(d[0])
                e26 += list(d[1:])
        rms = lambda e: float(np.sqrt(np.nanmean(np.square(e))))
        # Fig.12：模擬的 P95 對論文的 σ²100（每配置每段）；也給模擬自己的 σ²100 版
        pts = [dict(seg=s + 1, sigma2_100=s2_paper.get((nm, s + 1), tab[nm]["sigma2"][s]), rse_p95=tab[nm]["p95"][s])
               for nm in names for s in range(6)]
        sc_sum[sc] = dict(rms_all=rms(e_all), rms_seg1=rms(e1), rms_seg26=rms(e26), bias_seg1=float(np.nanmean(e1)),
                          excess={met: excess(lambda nm, met=met: tab[nm][met]) for met in ("absavg", "std")},
                          fig12=fig12_lines(pts), fig10=o["fig10"],
                          seg_curve={nm: tab[nm]["absavg"] for nm in ("A2", "A12", "C2", "C4", "C12")})
    return dict(paper=paper, scenarios=sc_sum)


def figures(od, cfg0, out, summ, names):
    plt = F.plt
    P = paper_table()
    # 1. 各段 mean|RSE|：論文 vs 不加暖機 / 桌上 / ±10 ppm，三個代表配置
    show = ["A2", "C4", "C12"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, nm in zip(axs, show):
        ax.plot(range(1, 7), P[("absavg", nm)], "-s", color=INK, lw=2, ms=7, label="論文 Table II")
        for sc in SCEN_PLOT:
            ax.plot(range(1, 7), out[sc]["table"][nm]["absavg"], "-o", color=SCEN_COLOR[sc], lw=2, ms=8,
                    label=f"模擬：{SCEN_SHORT[sc]}")
        ax.set_title(nm, color=INK)
        ax.set_xlabel("段（每段 10 分鐘）")
        ax.set_xticks(range(1, 7))
    ymax = max(max(P[("absavg", nm)] + [x for sc in SCEN_PLOT for x in out[sc]["table"][nm]["absavg"]]) for nm in show)
    axs[0].set_ylim(0, ymax * 1.08)                 # 三格共用 y 軸：用三個配置裡最大的值
    axs[0].set_ylabel("最差一對 mean|RSE| (ms，中位數)")
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle("暖機讓後段變大；論文是後段變小", color=INK)
    fig.tight_layout()
    fig.savefig(od / "segment_curves.png")
    plt.close(fig)

    # 2. Fig.10 式：論文 vs 模擬 C4（seed 0）不加暖機 / ±10 ppm
    _, pp = paper_fig10(cfg0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for (k, (t, v)), c in zip(sorted(pp.items()), PAIR_COLOR):
        axs[0].plot(t, v, "-", color=c, lw=0.5, label=f"Apx {k}")
    for ax, sc in zip(axs[1:], ["off", "d10"]):
        for (k, v), c in zip(out[sc]["pairs_seed0"]["C4"].items(), PAIR_COLOR):
            v = np.array(v, float) * 1e3
            ax.plot(np.arange(len(v)) * 10 + 5, v, "-", color=c, lw=1, label=f"Apx {k}")
    for ax, ttl in zip(axs, ["論文 Fig.10（數位化）", "模擬 C4：不加暖機", "模擬 C4：每節點 ±10 ppm"]):
        ax.axhspan(-1, 1, color="#eda100", alpha=0.12, lw=0)
        for x in range(600, 3601, 600):
            ax.axvline(x, color=INK2, ls="--", lw=0.6)
        ax.set_xlim(0, 3700)
        ax.set_ylim(-3, 2)
        ax.set_title(ttl, color=INK)
        ax.set_xlabel("Output timestamp (s)")
    axs[0].set_ylabel("RSE (ms)")
    axs[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(od / "fig10_compare.png")
    plt.close(fig)


def write_readme(od, cfg0, summ):
    pa, S = summ["paper"], summ["scenarios"]
    lab = {k: v["label"] for k, v in cfg0["scenarios"].items()}
    order = list(cfg0["scenarios"])
    g3 = ["2 節點", "12 節點", "全部"]
    P = paper_table()
    o, b, d3, d10 = S["off"], S["bench"], S["d3"], S["d10"]

    def rms_row(sc):
        r = S[sc]
        return f"| {lab[sc]} | {r['rms_all']:.3f} | {r['rms_seg1']:.3f}（{r['bias_seg1']:+.2f}） | {r['rms_seg26']:.3f} |"

    def exc_row(name, e):
        return f"| {name} | " + " | ".join(f"{e['absavg'][g]:+.2f} / {e['std'][g]:+.2f}" for g in g3) + " |"

    def f12_row(name, f):
        return f"| {name} | " + " | ".join(f"{f[k]['at225']:.2f} / {f[k]['b100']:.2f}" for k in range(6)) + " |"

    def f10_row(sc):
        f = S[sc]["fig10"]
        return f"| {lab[sc]} | " + " | ".join(f"{m:.2f}（{a:.2f} ～ {c:.2f}）" for m, a, c in zip(f["med"], f["p5"], f["p95"])) + " |"

    def curve(vals):
        return " / ".join(f"{x:.2f}" for x in vals)

    curve_rows = "\n".join(f"| {lab[sc]} | " + " | ".join(curve(S[sc]["seg_curve"][nm]) for nm in ("A2", "C4", "C12")) + " |"
                           for sc in order)
    paper_curve = "| **論文** | " + " | ".join(curve(P[("absavg", nm)]) for nm in ("A2", "C4", "C12")) + " |"
    (w0, w1), ref = cfg0["paper_fig10_windows_s"], cfg0["paper_fig10_ref_s"]
    dT = lambda x: (x / 0.034) ** 0.5            # ABS04W 典型 −0.034 ppm/°C²：從轉折溫度升多少度才變 x ppm
    ex = lambda r, met, g: r["excess"][met][g]
    a12 = lambda sc: S[sc]["seg_curve"]["A12"]
    text = f"""# T13　暖機能不能解釋 Table II / Fig.10–12「前 10 分鐘比較大」？

**問題**：論文 Table II、Fig.11–13 的第 1 段（前 10 分鐘）誤差都比後面大，之後收斂；Fig.10 各對 RSE 在前 10 分鐘慢慢移到穩定值。
這是晶振暖機（頻率還在變）造成的，還是 SharkTooth 本身的起始收斂？

**設定**（`config/T13_warmup_tableII.yaml`）：T6 的 18 個配置，完全沿用 T6 的校正（每個配置的連線失敗率、V4 量測模型），只在節點時鐘加暖機。
每個情境 × 配置跑 {len(cfg0['seeds'])} 次 × 3600 s，取中位數（論文是 20 次的中位數）。

| 情境 | 節點頻率在暖機時怎麼變 |
|---|---|
| 不加暖機 | 不變（T6 的設定） |
| 桌上（物理模型） | T12 bench：ABS04W 拋物線、自熱 +0.5～2 °C、空調 ±0.5 °C，加 PC 與 ADC 計時器溫度 |
| 貼身（物理模型） | T12 body：升到皮膚溫度 31～34 °C |
| 每節點 ±3 / ±10 / ±30 ppm | 不管物理，每個節點的頻率在 τ = 3–10 min 內直接變 U(−X, +X) ppm，找「要多大才看得出來」 |

ABS04W（−0.034 ppm/°C²）從轉折溫度升溫 {dT(3):.0f} / {dT(10):.0f} / {dT(30):.0f} °C 才會變 3 / 10 / 30 ppm；桌上自熱只有 0.5–2 °C。

## 1. 第 1 段比第 2–6 段平均多多少（mean|RSE| / std，ms）

兩種原因的特徵不同。暖機在每個配置都一樣（同一張桌子、同一批板子）；SharkTooth 的收斂則是連線雜訊越大越慢，所以節點越多，第 1 段多出越多。

| | 2 節點（A2、B2、C2） | 12 節點 | 18 個配置平均 |
|---|---|---|---|
{exc_row('**論文**', pa['excess'])}
{chr(10).join(exc_row(lab[sc], S[sc]['excess']) for sc in order)}

## 2. 各段 mean|RSE|（第 1–6 段，ms）

| | A2 | C4 | C12 |
|---|---|---|---|
{paper_curve}
{curve_rows}

![各段](segment_curves.png)

## 3. 和 Table II 的整體差距（18 配置 × 6 段 × {{mean|RSE|, std}}，RMS，ms）

| 情境 | 全部 | 第 1 段（平均差） | 第 2–6 段 |
|---|---|---|---|
{chr(10).join(rms_row(sc) for sc in order)}

## 4. Fig.12：RSE P95 對 σ²100 的迴歸線（每段：σ²100 = 225 ms² 時的 P95 / 每 100 ms² 的斜率，ms）

模擬的點用論文 Fig.12 的 σ²100（每配置每段）配模擬的 P95。

| | 第 1 段 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
{f12_row('**論文**', pa['fig12'])}
{chr(10).join(f12_row(lab[sc], S[sc]['fig12']) for sc in order)}

## 5. Fig.10：前 10 分鐘各對 RSE 離穩定值多遠（最偏的那一對，ms）

穩定值 = 最後 10 分鐘（{ref[0]:.0f}–{ref[1]:.0f} s）的平均。模擬是 A4 / B4 / C4 各 {len(cfg0['seeds'])} 次的中位數（括號 5–95 %）；論文 Fig.10 只有 1 次。

| | {w0[0]:.0f}–{w0[1]:.0f} s | {w1[0]:.0f}–{w1[1]:.0f} s |
|---|---|---|
| **論文 Fig.10** | {pa['fig10'][0]:.2f} | {pa['fig10'][1]:.2f} |
{chr(10).join(f10_row(sc) for sc in order)}

![Fig.10 比較](fig10_compare.png)

## 結論

1. **論文「前 10 分鐘大、之後收斂」不加暖機就做得出來**，來源是 SharkTooth 的起始收斂。
   - 第 1 段多出的量：論文 2 節點 {ex(pa, 'absavg', '2 節點'):+.2f}、12 節點 {ex(pa, 'absavg', '12 節點'):+.2f} ms；不加暖機的模擬 {ex(o, 'absavg', '2 節點'):+.2f}、{ex(o, 'absavg', '12 節點'):+.2f} ms（std：論文 {ex(pa, 'std', '12 節點'):+.2f}，模擬 {ex(o, 'std', '12 節點'):+.2f}）。
   - 一開始資料少：δT* 的斜率還在收斂，累積 OLS 也只有幾分鐘的點，連線雜訊越大越慢收斂。
     所以第 1 段的 Fig.12 迴歸線特別陡：論文每 100 ms² {pa['fig12'][0]['b100']:.2f} ms，模擬 {o['fig12'][0]['b100']:.2f} ms；後段論文 {min(f['b100'] for f in pa['fig12'][1:]):.2f}–{max(f['b100'] for f in pa['fig12'][1:]):.2f}、模擬 {min(f['b100'] for f in o['fig12'][1:]):.2f}–{max(f['b100'] for f in o['fig12'][1:]):.2f}。
   - Fig.10 前 10 分鐘的偏移（論文 {pa['fig10'][0]:.2f} / {pa['fig10'][1]:.2f} ms）落在不加暖機的範圍內（{o['fig10']['p5'][0]:.2f}–{o['fig10']['p95'][0]:.2f} / {o['fig10']['p5'][1]:.2f}–{o['fig10']['p95'][1]:.2f} ms）。
2. **物理上合理的暖機（桌上）幾乎不改變任何數字**：整體 RMS {o['rms_all']:.3f} → {b['rms_all']:.3f} ms。
   - 晶振在轉折溫度附近，升溫 1–2 °C 頻率只變約 0.1 ppm。
   - 板子一起升溫的部分，在兩兩相減（RSE 是節點對）時抵消，只剩節點之間「不一樣」的部分。
3. **暖機大到看得出來時，形狀和論文相反**：後段變大，而且不會回落。
   - 每節點 ±10 ppm：A12 各段 {curve(a12('d10'))}（論文 {curve(P[('absavg', 'A12')])}）；第 1 段多出的量變成 {ex(d10, 'absavg', '全部'):+.2f} ms（後段反而比較大）。
   - 原因：論文用累積 OLS（Eq.16–17 一直往上加），暖機期間 δT 的彎曲會一直留在迴歸裡，變成斜率偏差，暖機結束後誤差還在（T5 也看到同樣的現象）。
   - 論文的後段是最小、最穩的，所以論文的資料反而限制了暖機：每節點 ±3 ppm 時第 2–6 段已開始變差（RMS {o['rms_seg26']:.3f} → {d3['rms_seg26']:.3f} ms），±10 ppm 就差很多（{d10['rms_seg26']:.3f} ms）。
     ABS04W 要從轉折溫度升溫約 {dT(3):.0f} °C 才會變 3 ppm；桌上自熱只有 0.5–2 °C。
4. **仍然沒對上的，暖機也補不起來**：
   - 模擬的第 1 段整體偏高（平均差 {o['bias_seg1']:+.2f} ms，A 組最明顯，T6 已記錄）。暖機沒有讓它變低：桌上 {b['bias_seg1']:+.2f}（差別在抽樣誤差內）、±10 ppm {d10['bias_seg1']:+.2f} ms。
     可能的原因：論文從 notification 開始後才計時，或第 1 段的連線雜訊比 Fig.12 的平均低。
   - C12 第 2 段偏低（論文 {P[('absavg', 'C12')][1]:.2f}、模擬 {o['seg_curve']['C12'][1]:.2f} ms）。±10 ppm 能把第 2 段拉到 {d10['seg_curve']['C12'][1]:.2f}，
     但第 3–6 段跟著變成 {min(d10['seg_curve']['C12'][2:]):.2f}–{max(d10['seg_curve']['C12'][2:]):.2f}（論文 {min(P[('absavg', 'C12')][2:]):.2f}–{max(P[('absavg', 'C12')][2:]):.2f}），所以也不是暖機。
     論文 C12 第 2 段的 IQR 高達 0.79 ms，比較像少數幾次擷取出了狀況（例如斷線重連）。

**限制**：
- 只模擬了「節點頻率隨溫度變」。溫度如果也改變 BLE 連線品質或延遲，這裡沒有包含（論文沒有這方面的資料）。
- ±X ppm 情境的 τ 用 3–10 min；τ 更長時彎曲更晚出現，後段的偏差更大。
"""
    write_text(od / "README.md", text)

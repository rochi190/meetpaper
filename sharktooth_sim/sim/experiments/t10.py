"""T10：δT 的向上飄移從哪來——拿掉手動設定的斜率，只留物理時鐘誤差。"""
import json

import numpy as np

from .. import figures as F
from ..algorithms import sharktooth
from ..config import deep_merge, load_config, set_path
from ..evaluate import lower_line_lp
from ..runner import RESULTS, algo_params, node_series, outdir, pmap, write_json, write_text
from ..simulate import simulate

RED = (0.8, 0.1, 0.1)


def variants(tilt):
    return {"論文字面": dict(keep_last=False),
            "保留 α、β": dict(keep_last=True),
            f"保留 + 預傾斜 {tilt:g} ppm": dict(keep_last=True, tilt_ppm=tilt)}


def lower_slope(tp, dt, bin_s=10.0):
    """δT 下緣斜率：每 bin_s 秒取最小值，再用 LP 下緣線（Moon–Skelly–Towsley）。回傳 (斜率, 截距)。"""
    b = np.floor((tp - tp[0]) / bin_s).astype(int)
    order = np.lexsort((dt, b))
    first = order[np.r_[True, b[order][1:] != b[order][:-1]]]
    return lower_line_lp(tp[first], dt[first])


def _job(args):
    cfg, seed, vs, bad_err = args
    sim = simulate(cfg, seed)
    T = cfg["T"]
    grid = np.arange(0, T, 1.0) + 0.5
    prm = algo_params(cfg)
    nodes, E = [], {v: [] for v in vs}
    for k in range(cfg["n_nodes"]):
        p = node_series(sim, k)
        tp, dt = p["t_P"], p["t_C"] - p["t_P"]
        a_meas, _ = lower_slope(tp, dt)
        a_true = np.polyfit(tp, p["t_pc_true"] - tp, 1)[0]
        row = dict(slope_meas_ppm=a_meas * 1e6, slope_true_ppm=a_true * 1e6)
        for name, kw in vs.items():
            r = sharktooth.run(tp, p["t_C"], **{**prm, **kw})
            e = np.where(r["valid"], r["ts"] - p["t_pc_true"], np.nan)
            tt = p["t_true_last_sample"]
            steady = r["valid"] & (tt > T / 3)
            c = np.nanmedian(e[steady]) if steady.any() else np.nan
            err = e - c
            bad = ~r["valid"] | ~(np.abs(err) <= bad_err)
            good = ~bad
            row[name] = dict(n_pivot=r["n_pivot"], bad=float(bad.mean()),
                             p95=float(np.percentile(np.abs(err[good]), 95) * 1e3) if good.any() else np.nan)
            E[name].append(np.interp(grid, tt, e, left=np.nan, right=np.nan))
        nodes.append(row)
    # 節點對（論文 RSE 量的東西；不扣常數）：各段 mean|e_n − e_m|，無效輸出不計入平均，另計壞輸出比例
    neg = [n["slope_true_ppm"] < 0 for n in nodes]
    pairs = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pr = dict(has_neg=bool(neg[i] or neg[j]))
            for name in vs:
                d = E[name][i] - E[name][j]
                seg = [float(np.nanmean(np.abs(d[(grid >= 600 * s) & (grid < 600 * (s + 1))])) * 1e3)
                       if np.isfinite(d[(grid >= 600 * s) & (grid < 600 * (s + 1))]).any() else np.nan
                       for s in range(int(T // 600))]
                pr[name] = dict(seg=seg, bad=float(np.mean(~(np.abs(d) <= bad_err))))
            pairs.append(pr)
    return dict(nodes=nodes, pairs=pairs)


def case_cfg(cfg0, case):
    c = {k: v for k, v in case.items() if k != "label"}
    return deep_merge(cfg0, c)


def example(cfg, want, seeds, min_abs=15.0):
    """找一個斜率符號符合 want（+1 / −1）、且 |斜率| ≥ min_abs ppm 的節點，回傳 (tp, dt, a, b)。"""
    for s in seeds:
        sim = simulate(cfg, s)
        for k in range(cfg["n_nodes"]):
            p = node_series(sim, k)
            tp, dt = p["t_P"], p["t_C"] - p["t_P"]
            a, b = lower_slope(tp, dt)
            if np.sign(a) == want and abs(a) * 1e6 >= min_abs:
                return tp, dt, a, b
    raise RuntimeError("找不到符合條件的節點")


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    if cfg0.get("use_T4_best"):
        bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
        for k, v in bp.items():
            cfg0 = set_path(cfg0, k, v)
    od = outdir(cfg0["name"])
    vs = variants(cfg0["tilt_ppm"])
    vnames = list(vs)
    cases = cfg0["cases"]
    cfgs = {c: case_cfg(cfg0, cases[c]) for c in cases}
    jobs = [(cfgs[c], s, vs, cfg0["bad_err_s"]) for c in cases for s in cfg0["seeds"]]
    res = pmap(_job, jobs)
    ns = len(cfg0["seeds"])
    by_case = {c: res[i * ns:(i + 1) * ns] for i, c in enumerate(cases)}

    # ---- 匯總 ----
    groups = {}          # 節點群組：論文設定 / 物理 α>0 / 物理 α<0
    slopes = {}
    pair_groups = {}
    for c, rr in by_case.items():
        nodes = [n for r in rr for n in r["nodes"]]
        pairs = [p for r in rr for p in r["pairs"]]
        slopes[c] = dict(meas=[n["slope_meas_ppm"] for n in nodes], true=[n["slope_true_ppm"] for n in nodes])
        if c == "paper":
            groups["論文設定"] = nodes
            pair_groups["論文設定"] = pairs
        else:
            groups["物理時鐘，δT 往上（α > 0）"] = [n for n in nodes if n["slope_true_ppm"] >= 0]
            groups["物理時鐘，δT 往下（α < 0）"] = [n for n in nodes if n["slope_true_ppm"] < 0]
            pair_groups["物理時鐘，兩個節點都往上"] = [p for p in pairs if not p["has_neg"]]
            pair_groups["物理時鐘，至少一個往下"] = [p for p in pairs if p["has_neg"]]

    def node_stat(nodes, v):
        return dict(n=len(nodes),
                    n_pivot=float(np.median([n[v]["n_pivot"] for n in nodes])),
                    bad=float(np.median([n[v]["bad"] for n in nodes])),
                    p95=float(np.nanmedian([n[v]["p95"] for n in nodes])))

    def pair_stat(pairs, v):
        return dict(n=len(pairs),
                    seg=np.nanmedian([p[v]["seg"] for p in pairs], axis=0).tolist(),
                    bad=float(np.median([p[v]["bad"] for p in pairs])))

    node_tab = {g: {v: node_stat(n, v) for v in vnames} for g, n in groups.items()}
    pair_tab = {g: {v: pair_stat(p, v) for v in vnames} for g, p in pair_groups.items()}
    slope_sum = {}
    for c, s in slopes.items():
        m = np.array(s["meas"])
        t = np.array(s["true"])
        slope_sum[c] = dict(label=cases[c]["label"], n=m.size, min=m.min(), p5=np.percentile(m, 5), median=np.median(m),
                            p95=np.percentile(m, 95), max=m.max(), neg_frac=float(np.mean(t < 0)),
                            in_paper_range=float(np.mean((m >= cfg0["paper_slope_ppm"][0] - 5) &
                                                         (m <= cfg0["paper_slope_ppm"][1] + 5))),
                            meas_minus_true_med=float(np.median(m - t)))
    write_json(od / "summary.json", dict(slopes=slope_sum, nodes=node_tab, pairs=pair_tab,
                                         raw_slopes=slopes))

    # ---- 圖 1：斜率分布 + 兩種設定下的 δT 長相（簡報主圖）----
    lo, hi = cfg0["paper_slope_ppm"]
    fig, axs = F.plt.subplots(1, 3, figsize=(17, 5.0), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    bins = np.arange(-100, 305, 5)
    axs[0].hist(slopes["physical"]["meas"], bins=bins, color=F.BLUE, alpha=0.85, label="只有物理時鐘誤差")
    axs[0].hist(slopes["paper"]["meas"], bins=bins, color=F.ORANGE, alpha=0.85, label="論文設定（手動 −248 ppm）")
    axs[0].axvline(0, color="k", lw=1)
    ymax = axs[0].get_ylim()[1] * 1.45
    axs[0].set_ylim(0, ymax)
    for xv in (lo, hi):
        axs[0].axvline(xv, color="k", ls="--", lw=1.2, zorder=5)
    axs[0].text((lo + hi) / 2, ymax * 0.97, f"論文實測\n{lo:.0f}–{hi:.0f} ppm", fontsize=11, ha="center", va="top")
    neg = slope_sum["physical"]["neg_frac"]
    axs[0].text(-96, ymax * 0.97, f"往下飄：\n{neg:.0%} 的節點", fontsize=11, color=RED, va="top")
    axs[0].set_xlabel("δT 下緣斜率 (ppm)")
    axs[0].set_ylabel(f"節點數（每種設定 {slope_sum['paper']['n']} 個）")
    axs[0].set_title("(a) δT 斜率：物理誤差做不出 +245 ppm")
    axs[0].legend(loc="center", bbox_to_anchor=(0.5, 0.62), fontsize=10)

    seeds = cfg0["seeds"]
    tp, dt, a, b = example(cfgs["paper"], +1, seeds)
    axs[1].plot(tp / 60, (dt - b - a * tp[0]) * 1e3, ".", ms=1.2, alpha=0.35, color=F.ORANGE)
    axs[1].plot(tp / 60, a * (tp - tp[0]) * 1e3, "-", color="k", lw=1.5, label=f"下緣 {a * 1e6:+.0f} ppm")
    axs[1].set_title("(b) 論文設定：一路往上")
    axs[1].set_xlabel("t_P (min)")
    axs[1].set_ylabel("δT − 起點 (ms)")
    axs[1].legend(loc="upper left", fontsize=10)
    axs[1].set_ylim(-60, 1000)

    for want, col in ((+1, F.BLUE), (-1, RED)):
        tp, dt, a, b = example(cfgs["physical"], want, seeds)
        axs[2].plot(tp / 60, (dt - b - a * tp[0]) * 1e3, ".", ms=1.2, alpha=0.25, color=col)
        axs[2].plot(tp / 60, a * (tp - tp[0]) * 1e3, "-", color=col, lw=2,
                    label=f"{'往上' if want > 0 else '往下'}：下緣 {a * 1e6:+.0f} ppm")
    axs[2].set_title("(c) 只有物理時鐘：可能往上，也可能往下")
    axs[2].set_xlabel("t_P (min)")
    axs[2].set_ylabel("δT − 起點 (ms)")
    axs[2].legend(loc="upper left", fontsize=10)
    axs[2].set_ylim(-120, 170)
    F.save(fig, od / "slope_compare.png")

    # ---- 圖 2：這個斜率對 SharkTooth 的影響 ----
    fig, axs = F.plt.subplots(1, 2, figsize=(15, 4.8))
    gnames = list(node_tab)
    w = 0.26
    cols = [F.PAIR_COLORS[0], F.PAIR_COLORS[1], F.PAIR_COLORS[4]]
    for k, v in enumerate(vnames):
        vals = [node_tab[g][v]["bad"] * 100 for g in gnames]
        x = np.arange(len(gnames)) + (k - 1) * w
        axs[0].bar(x, vals, w, color=cols[k], label=v)
        for xi, yi in zip(x, vals):
            axs[0].text(xi, yi + 1, f"{yi:.0f}" if yi >= 1 else f"{yi:.1f}", ha="center", fontsize=10)
    axs[0].set_xticks(range(len(gnames)))
    axs[0].set_xticklabels([g.replace("，", "\n") for g in gnames], fontsize=11)
    axs[0].set_ylabel("壞輸出比例 (%)（無效或誤差 > 10 ms）")
    axs[0].set_title("(a) 每個節點：δT 往下時，論文字面版大部分時間沒有輸出")
    axs[0].legend(fontsize=10)
    axs[0].set_ylim(0, 100)
    x = np.arange(1, 7)
    # 顏色與 (a) 的處理方式一致；灰色虛線 = 兩個節點都往上（對照）
    show = [("物理時鐘，兩個節點都往上", vnames[0], "0.45", "s--", "兩個節點都往上｜論文字面（對照）")]
    show += [("物理時鐘，至少一個往下", v, cols[k], "o-", f"至少一個往下｜{v}") for k, v in enumerate(vnames)]
    for g, v, col, ls, lab in show:
        bad = pair_tab[g][v]["bad"] * 100
        axs[1].plot(x, pair_tab[g][v]["seg"], ls, color=col, lw=2, label=f"{lab}（壞 {bad:.0f} %）")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("600 s 分段")
    axs[1].set_ylabel("節點對 mean|e| (ms)（只算有效輸出）")
    axs[1].set_title("(b) 物理時鐘下的節點對誤差（真值）")
    axs[1].set_ylim(0.04, 20)
    axs[1].legend(loc="upper right", fontsize=9.5)
    F.save(fig, od / "sharktooth_effect.png")

    # ---- README ----
    fmt = lambda v: " / ".join("—" if not np.isfinite(x) else f"{x:.2f}" for x in v)  # noqa: E731
    ts = "\n".join(f"| {s['label']} | {s['n']} | {s['min']:+.0f} / {s['median']:+.0f} / {s['max']:+.0f} | "
                   f"{s['p5']:+.0f} ～ {s['p95']:+.0f} | {s['neg_frac']:.0%} | {s['in_paper_range']:.0%} |"
                   for s in slope_sum.values())
    tn = "\n".join(f"| {g} | {node_tab[g][vnames[0]]['n']} | " +
                   " | ".join(f"{node_tab[g][v]['n_pivot']:.0f} / {node_tab[g][v]['bad'] * 100:.1f} % / {node_tab[g][v]['p95']:.2f}"
                              for v in vnames) + " |" for g in gnames)
    tp_ = "\n".join(f"| {g} | {pair_tab[g][vnames[0]]['n']} | " +
                    " | ".join(f"{fmt(pair_tab[g][v]['seg'])}（壞 {pair_tab[g][v]['bad'] * 100:.0f} %）" for v in vnames) + " |"
                    for g in pair_tab)
    P, R = slope_sum["paper"], slope_sum["physical"]
    up, dn = node_tab["物理時鐘，δT 往上（α > 0）"], node_tab["物理時鐘，δT 往下（α < 0）"]
    pdn = pair_tab["物理時鐘，至少一個往下"]
    write_text(od / "README.md", f"""# T10　δT 的向上飄移從哪來

**問題**：論文 Fig.3 / 4 / 6 的 δT 下緣斜率都是 +{lo:.0f}～+{hi:.0f} ppm（T0）。T1 / T4 能做出一樣的圖，是因為設定檔**手動**放了
A8 = −248 ppm（`T4_fig6.yaml`）或 ρ = −245 ppm（`T1_fig3.yaml`），而且把 PC 時鐘設成理想。這個測試把這兩個設定拿掉，看 δT 會不會自己往上飄。

**設定**（`config/T10_drift_origin.yaml`）：T4 擬合出的延遲參數、4 節點 / 1 dongle、3600 s × {ns} seeds（共 {P['n']} 個節點 / 每種設定）。兩種時鐘設定：

- **論文設定**：節點晶振 ±20 ppm（均勻）＋ A8 = −248 ppm，PC 時鐘理想（= T4）。
- **只有物理時鐘誤差**：節點晶振 ±20 ppm（論文的 ABS04W 規格）、PC 時鐘 ±50 ppm（均勻）、A8 = 0。

δT 斜率用「每 10 s 取最小值 → LP 下緣線」量（與 T0 量論文圖的方法相同）；量到的值與真值差中位數 {R['meas_minus_true_med']:+.2f} ppm。

## 1. δT 斜率

| 設定 | 節點數 | 最小 / 中位數 / 最大 (ppm) | 5–95 % | δT 往下（α < 0） | 落在論文範圍 ±5 ppm |
|---|---|---|---|---|---|
{ts}

![斜率比較](slope_compare.png)

## 2. 對 SharkTooth 的影響

每個節點（中位數）：**樞紐點數 / 壞輸出比例（無效或真值誤差扣常數後 > 10 ms）/ 好輸出的誤差 P95 (ms)**。
「保留 α、β」= 穩定期內沿用上一組 α、β（論文字面是輸出 α = β = 0）；「預傾斜」= T7e 的修正，先把 δT 加上 +{cfg0['tilt_ppm']:g} ppm 再跑、輸出時扣回。

| 群組 | 節點數 | {' | '.join(vnames)} |
|---|---|---|---|---|
{tn}

節點對（論文 RSE 量的東西，不扣常數）：各段 mean|e_nm| (ms)，第 1–6 段，只算兩邊都有效的輸出；括號內是壞輸出比例。

| 群組 | 節點對數 | {' | '.join(vnames)} |
|---|---|---|---|---|
{tp_}

![對 SharkTooth 的影響](sharktooth_effect.png)

## 結論

1. **物理時鐘誤差做不出論文的向上飄移。** 只留晶振與 PC 時鐘誤差時，δT 斜率在 {R['min']:+.0f}～{R['max']:+.0f} ppm（中位數 {R['median']:+.0f}），
   落在論文 {lo:.0f}–{hi:.0f} ppm 附近的比例是 {R['in_paper_range']:.0%}；而且 **{R['neg_frac']:.0%} 的節點 δT 是往下走的**。
   論文那條 +245 ppm 需要一個「所有節點共有、約 −248 ppm」的額外因素。T1 / T4 的圖之所以對得上，是因為這個值是設定進去的，不是模擬推出來的。
2. **這個斜率的來源論文沒有說明。** 候選：(a) 韌體把 tick 換成時間時用 30.51 µs（正確值 1/32768 = 30.5176 µs，剛好 −248 ppm）；
   (b) PC 時鐘快 245 ppm（一天快 21 s，不太合理）；(c) 時間戳不是用 32.768 kHz 晶振打的（但論文 §II-A 明寫晶振是 time source）。三個都**未驗證**。
3. **斜率的正負對 SharkTooth 影響很大。** δT 往上的節點，論文字面版壞輸出只有 {up[vnames[0]]['bad'] * 100:.1f} %（樞紐點 {up[vnames[0]]['n_pivot']:.0f} 個）；
   往下的節點壞輸出 {dn[vnames[0]]['bad'] * 100:.0f} %（樞紐點 {dn[vnames[0]]['n_pivot']:.0f} 個）：每個新低點都變成新樞紐點、迴歸一直被清空。
   只「保留 α、β」能把無效輸出補起來，但輸出的是過時的 α、β（壞輸出 {dn[vnames[1]]['bad'] * 100:.0f} %，好輸出的 P95 卻變成 {dn[vnames[1]]['p95']:.2f} ms）；
   **預傾斜 +{cfg0['tilt_ppm']:g} ppm 讓往下的節點回到 {dn[vnames[2]]['bad'] * 100:.1f} %、P95 {dn[vnames[2]]['p95']:.2f} ms**，和往上的節點一樣。
   節點對：含往下節點的對，第 6 段 mean|e| 從 {fmt([pdn[vnames[0]]['seg'][-1]])}（壞 {pdn[vnames[0]]['bad'] * 100:.0f} %）降到 {fmt([pdn[vnames[2]]['seg'][-1]])} ms（壞 {pdn[vnames[2]]['bad'] * 100:.0f} %）。
4. **論文的系統剛好有一個天然的預傾斜。** 換成一般開發板（沒有那個 −248 ppm），大約 {R['neg_frac']:.0%} 的節點會碰到這個問題。
   所以開發板實測時，**第一個要量的就是 δT 斜率的正負**；如果是負的，SharkTooth 要先加預傾斜。

## 開發板實測時的預測（實測前寫下）

| 量到的 δT 斜率 | 代表 |
|---|---|
| 幾十 ppm 以內，各板正負不一 | 和本測試的「物理時鐘」一致；論文的 +245 ppm 是他們系統特有的 |
| 約 +248 ppm | 韌體用了類似 30.51 µs 的換算常數（可直接檢查程式碼） |
| 其他很大的值（> 100 ppm） | 時間戳的時鐘源不是 32.768 kHz 晶振，或 PC 時鐘有問題 |

PC 端時間戳的時鐘源也會直接改變斜率：Windows `time.perf_counter()`（QPC）與 Linux `CLOCK_MONOTONIC`（受 NTP 調速）不同，實測時要記下用的是哪一個。
""")

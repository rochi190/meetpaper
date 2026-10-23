"""T11：δT 向上飄移的候選來源——把所有找得到的時鐘 / 時間戳誤差組合起來，找出最像論文的組合。

每個節點的時鐘誤差在一次擷取內是固定值（不會每個封包亂跳）；隨機的只有「抽到哪一顆晶振 / 哪一塊板子 / 哪一台 PC」。
δT 斜率 ≈ ρ_C − ρ_P − conv（ρ_C = PC 時鐘、ρ_P = t_P 的時鐘、conv = 韌體換算常數誤差）。
"""
import itertools
import json

import numpy as np

from .. import figures as F
from ..algorithms import sharktooth
from ..config import deep_merge, draw, load_config, set_path
from ..evaluate import lower_line_lp, sawtooth_metrics, sigma_M
from ..runner import RESULTS, algo_params, node_series, outdir, pmap, write_json, write_text
from ..simulate import simulate
from .t10 import lower_slope

SRC_LABEL = {"xt": "32.768 kHz 晶振（論文寫的）", "hfadj": "HFRC 鎖在晶振（HFADJ）",
             "hfrc": "HFRC（RC，和 ADC 同一顆）", "lfrc": "LFRC（RC）"}
PC_LABEL = {"ntp": "PC 時鐘有 NTP 校正", "ntp_15ms": "同左，但 time.time() 15.6 ms 解析度",
            "qpc": "perf_counter（PC 晶振 ±50 ppm）", "ntp_slew": "NTP / w32time 調速",
            "pc_bad": "PC 時鐘很差（±300 ppm）"}
PULL_LABEL = {"nocap": "不接電容（≈2.2 pF）", "matched": "相符（4 pF）", "ext7": "外接 7 pF（≈5.7 pF）",
              "CL8": "≈8 pF（外接約 12 pF）", "ext15": "外接 15 pF（≈9.7 pF）"}
CONV_LABEL = {"exact": "換算正確", "t30.51us": "× 30.51 µs", "t30.5us": "× 30.5 µs"}


def combos(lv):
    """t_P 不是由晶振計時時，負載電容與換算常數不影響 t_P → 只留一個水準。"""
    out = []
    for src in lv["src"]:
        pulls = list(lv["pull"]) if src in ("xt", "hfadj") else ["matched"]
        convs = list(lv["conv"]) if src == "xt" else ["exact"]
        for pull, conv, pc in itertools.product(pulls, convs, lv["pc"]):
            out.append(dict(src=src, pull=pull, conv=conv, pc=pc))
    return out


def combo_name(c):
    return f"{c['src']}|{c['pull']}|{c['conv']}|{c['pc']}"


def draw_clocks(cfg0, c, rng, n):
    """回傳 (ρ_P ppm, ADC 週期誤差 ppm, conv ppm)，每個節點一組。"""
    X = cfg0["crystal"]
    tol = rng.uniform(-X["tol_ppm"], X["tol_ppm"], n)
    C0 = rng.uniform(*X["C0_pF"], n)
    C1 = rng.normal(*X["C1_fF"], n) * 1e-3                       # pF
    CL = cfg0["levels"]["pull"][c["pull"]] + rng.normal(0, X["CL_board_sd_pF"], n)
    pull = 1e6 * C1 / 2 * (1 / (C0 + CL) - 1 / (C0 + X["CL_spec_pF"]))
    adc = rng.uniform(*cfg0["adc_err_ppm"], n)
    xt = tol + pull
    rho = {"xt": xt, "hfadj": xt + cfg0["hfadj_ppm"],
           "hfrc": -adc / (1 + adc * 1e-6),                       # 和 ADC 計時器同一顆振盪器
           "lfrc": rng.uniform(-cfg0["lfrc_ppm"], cfg0["lfrc_ppm"], n)}[c["src"]]
    return rho, adc, cfg0["levels"]["conv"][c["conv"]]


def slope_ppm(rho_p, rho_c, conv):
    return ((1 + rho_c * 1e-6) / ((1 + rho_p * 1e-6) * (1 + conv * 1e-6)) - 1) * 1e6


def analytic(cfg0, c, n):
    rng = np.random.default_rng(12345)
    rho_p, _, conv = draw_clocks(cfg0, c, rng, n)
    rho_c = draw(cfg0["levels"]["pc"][c["pc"]]["rho"], rng, n)
    return slope_ppm(rho_p, rho_c, conv)


def slews(cfg0, T, rng):
    s = cfg0["slew"]
    out = []
    for _ in range(rng.poisson(s["per_hour"] * T / 3600)):
        t0 = rng.uniform(0, T)
        dur = rng.uniform(*s["dur_s"])
        off = rng.uniform(-s["offset_ms"], s["offset_ms"]) * 1e-3
        out.append([t0, t0 + dur, off / dur * 1e6])
    return out


def job_cfg(cfg0, c, seed):
    rng = np.random.default_rng([seed, 11])
    n = cfg0["n_nodes"]
    rho_p, adc, conv = draw_clocks(cfg0, c, rng, n)
    pl = cfg0["levels"]["pc"][c["pc"]]
    cp = dict(rho_ppm=float(draw(pl["rho"], rng)), tc_resolution=pl["res"],
              ntp_slews=slews(cfg0, cfg0["T"], rng) if pl["slew"] else [])
    return deep_merge(cfg0, dict(clock_node=dict(rho_list=rho_p.tolist(), scale_err_ppm=conv),
                                 clock_adc=dict(period_err_list=adc.tolist()), clock_pc=cp))


def _job(args):
    cfg, seed, bad_err = args
    sim = simulate(cfg, seed)
    T = cfg["T"]
    grid = np.arange(0, T, 1.0) + 0.5
    prm = algo_params(cfg)
    nodes, E = [], []
    for k in range(cfg["n_nodes"]):
        p = node_series(sim, k)
        tp, dt = p["t_P"], p["t_C"] - p["t_P"]
        a, b = lower_slope(tp, dt)
        seq = p["pkt_seq"]
        row = dict(slope=a * 1e6, interval_ms=(tp[-1] - tp[0]) / (seq[-1] - seq[0]) * 1e3)
        w = (tp >= 5) & (tp <= 145)                                # 和 Fig.6 一樣的時間窗
        a6, b6 = lower_line_lp(tp[w], dt[w])
        res = (dt[w] - (a6 * tp[w] + b6)) * 1e3
        row.update(P50=float(np.percentile(res, 50)), P90=float(np.percentile(res, 90)),
                   slope_140s=a6 * 1e6)
        r = sharktooth.run(tp, p["t_C"], **prm)
        e = np.where(r["valid"], r["ts"] - p["t_pc_true"], np.nan)
        tt = p["t_true_last_sample"]
        steady = r["valid"] & (tt > T / 3)
        cst = np.nanmedian(e[steady]) if steady.any() else np.nan
        row["bad"] = float(np.mean(~r["valid"] | ~(np.abs(e - cst) <= bad_err)))
        row["n_pivot"] = int(r["n_pivot"])
        E.append(np.interp(grid, tt, e, left=np.nan, right=np.nan))
        nodes.append(row)
    seg6 = []
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            d = np.abs(E[i] - E[j])[grid >= T - 600]
            seg6.append(float(np.nanmean(d) * 1e3) if np.isfinite(d).any() else np.nan)
    return dict(nodes=nodes, seg6=seg6)


def _fig3_job(args):
    """Fig.3 的乾淨通道（T1 設定）下，t_C 解析度對 σ100 與鋸齒的影響。"""
    cfg, seed = args
    sim = simulate(cfg, seed)
    p = node_series(sim, 0)
    m = sawtooth_metrics(p["t_P"], p["t_C"] - p["t_P"], cfg["CI"])
    return dict(sigma100=m["sigma100_ms"], step_down=m["step_down_ms"])


def summarize(cfg0, c, rr, ana):
    P = cfg0["paper"]
    lo, hi = P["slope_ppm"]
    nodes = [n for r in rr for n in r["nodes"]]
    sl = np.array([n["slope"] for n in nodes])
    seg6 = np.array([v for r in rr for v in r["seg6"]])
    s = dict(c, name=combo_name(c),
             p_in=float(np.mean((ana >= lo) & (ana <= hi))), neg=float(np.mean(ana < 0)),
             a_p5=float(np.percentile(ana, 5)), a_med=float(np.median(ana)), a_p95=float(np.percentile(ana, 95)),
             sim_slope_med=float(np.median(sl)),
             interval=float(np.median([n["interval_ms"] for n in nodes])),
             P50=float(np.median([n["P50"] for n in nodes])), P90=float(np.median([n["P90"] for n in nodes])),
             bad=float(np.median([n["bad"] for n in nodes])), bad_max=float(np.max([n["bad"] for n in nodes])),
             fail_nodes=float(np.mean([n["bad"] > 0.05 for n in nodes])),
             seg6=float(np.nanmedian(seg6)) if np.isfinite(seg6).any() else np.nan)
    s["L3"] = s["p_in"] ** 3
    checks = dict(
        slope=lo <= s["a_med"] <= hi,
        no_neg=s["neg"] <= 0.01,
        interval=P["interval_ms"][0] <= s["interval"] <= P["interval_ms"][1],
        resid=abs(s["P50"] - P["resid_ms"]["P50"]) <= 3 and abs(s["P90"] - P["resid_ms"]["P90"]) <= 4,
        sharktooth=s["fail_nodes"] <= 0.1 and np.isfinite(s["seg6"]) and s["seg6"] <= 1.0,
    )
    s["checks"] = {k: bool(v) for k, v in checks.items()}
    s["n_pass"] = int(sum(checks.values()))
    return s


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    if cfg0.get("use_T4_best"):
        bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
        for k, v in bp.items():
            cfg0 = set_path(cfg0, k, v)
    od = outdir(cfg0["name"])
    lv = cfg0["levels"]
    cs = combos(lv)
    seeds = cfg0["seeds"]
    jobs = [(job_cfg(cfg0, c, s), s, cfg0["bad_err_s"]) for c in cs for s in seeds]
    res = pmap(_job, jobs)
    ns = len(seeds)
    rows = []
    for i, c in enumerate(cs):
        ana = analytic(cfg0, c, cfg0["n_mc"])
        rows.append(summarize(cfg0, c, res[i * ns:(i + 1) * ns], ana))
    rows.sort(key=lambda r: (-r["n_pass"], -r["L3"], r["bad"]))

    # Fig.3 乾淨通道：t_C 解析度
    c1 = load_config(RESULTS.parent / "config" / "T1_fig3.yaml", T=600.0)
    f3 = {}
    for res_s in (1e-6, 0.015625):
        rr = pmap(_fig3_job, [(set_path(c1, "clock_pc.tc_resolution", res_s), s) for s in range(5)])
        f3[res_s] = dict(sigma100=float(np.mean([r["sigma100"] for r in rr])),
                         step_down=float(np.mean([r["step_down"] for r in rr])))

    # 各因素水準的「最好能做到」：在其他因素任選下，最高的 p_in 與通過項目數
    marg = {}
    for f in ("src", "pull", "conv", "pc"):
        for lvl in sorted({r[f] for r in rows}):
            sub = [r for r in rows if r[f] == lvl]
            best = max(sub, key=lambda r: (r["n_pass"], r["L3"]))
            marg[(f, lvl)] = dict(best=best["name"], n_pass=best["n_pass"], p_in=best["p_in"],
                                  any_full=any(r["n_pass"] == 5 for r in sub))
    write_json(od / "summary.json", dict(rows=rows, fig3_resolution={str(k): v for k, v in f3.items()},
                                         marginal={f"{k[0]}={k[1]}": v for k, v in marg.items()}))

    # ---- 圖 1：每個組合的斜率分布（解析）----
    P = cfg0["paper"]
    lo, hi = P["slope_ppm"]
    show = rows[:14] + [r for r in rows if r["name"] in (
        "xt|matched|exact|ntp", "xt|nocap|exact|ntp", "xt|matched|exact|qpc", "hfrc|matched|exact|ntp", "lfrc|matched|exact|ntp",
        "hfadj|matched|exact|ntp")]
    seen, uniq = set(), []
    for r in show:
        if r["name"] not in seen:
            seen.add(r["name"])
            uniq.append(r)
    fig, ax = F.plt.subplots(figsize=(13, 0.42 * len(uniq) + 1.6))
    for k, r in enumerate(uniq[::-1]):
        col = F.ORANGE if r["n_pass"] == 5 else (F.BLUE if r["n_pass"] >= 3 else "0.55")
        ax.plot([r["a_p5"], r["a_p95"]], [k, k], "-", color=col, lw=5, solid_capstyle="butt")
        ax.plot(r["a_med"], k, "o", color="k", ms=4)
    ax.axvspan(lo, hi, color=F.ORANGE, alpha=0.18, zorder=0)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xscale("symlog", linthresh=300)
    ax.set_yticks(range(len(uniq)))
    ax.set_yticklabels([f"{r['name']}  (通過 {r['n_pass']}/5)" for r in uniq[::-1]], fontsize=9.5)
    ax.set_xlabel("δT 斜率 (ppm)，橫條 = 5–95 %，點 = 中位數；|x| > 300 為對數刻度")
    ax.set_title(f"各組合的 δT 斜率（橘色帶 = 論文 {lo:.0f}–{hi:.0f} ppm；橘條 = 5 項檢查全過）")
    F.save(fig, od / "slope_by_combo.png")

    # ---- 圖 2：t_P = 晶振時，有效負載電容 → δT 斜率（解析，連續掃描）----
    cls = np.arange(2.0, 10.01, 0.25)
    fig, ax = F.plt.subplots(figsize=(10, 5))
    need = {}
    for conv, pc, col, ls in (("exact", "ntp", F.BLUE, "-"), ("t30.51us", "ntp", F.ORANGE, "-"),
                              ("exact", "qpc", F.BLUE, "--")):
        q = []
        for cl in cls:
            c2 = deep_merge(cfg0, dict(levels=dict(pull=dict(_x=float(cl)))))
            a = analytic(c2, dict(src="xt", pull="_x", conv=conv, pc=pc), 4000)
            q.append(np.percentile(a, [5, 50, 95]))
        q = np.array(q)
        lab = f"{CONV_LABEL[conv]}，{PC_LABEL[pc]}"
        ax.plot(cls, q[:, 1], ls, color=col, lw=2, label=lab)
        if ls == "-":
            ax.fill_between(cls, q[:, 0], q[:, 2], color=col, alpha=0.15)
        need[lab] = float(np.interp((lo + hi) / 2, q[:, 1], cls)) if q[:, 1].min() <= (lo + hi) / 2 <= q[:, 1].max() else None
    ax.axhspan(lo, hi, color="k", alpha=0.12, label=f"論文 {lo:.0f}–{hi:.0f} ppm")
    ax.axhline(0, color="k", lw=0.8)
    for k in ("nocap", "matched", "ext7", "ext15"):
        x = lv["pull"][k]
        ax.axvline(x, color="0.4", ls=":", lw=1)
        ax.text(x, ax.get_ylim()[0] + 15, PULL_LABEL[k].replace("（", "\n（"), fontsize=9, ha="center", va="bottom",
                color="0.25", bbox=dict(fc="w", ec="none", alpha=0.8))
    ax.set_xlabel("晶振的有效負載電容 (pF)（ABS04W 規格 = 4 pF）")
    ax.set_ylabel("δT 斜率 (ppm)，線 = 中位數，色帶 = 5–95 %")
    ax.set_title("負載電容偏大 → 晶振變慢 → δT 往上飄")
    ax.legend(fontsize=10, loc="upper left")
    F.save(fig, od / "load_cap_curve.png")
    write_json(od / "load_cap_needed.json", need)
    write_readme(od, cfg0, rows, f3, marg)


def write_readme(od, cfg0, rows, f3, marg):
    P = cfg0["paper"]
    lo, hi = P["slope_ppm"]
    ck = lambda r: "".join("✓" if r["checks"][k] else "✗" for k in ("slope", "no_neg", "interval", "resid", "sharktooth"))  # noqa: E731
    top = "\n".join(
        f"| {k + 1} | {SRC_LABEL[r['src']]} | {PULL_LABEL[r['pull']]} | {CONV_LABEL[r['conv']]} | {PC_LABEL[r['pc']]} | "
        f"{r['a_med']:+.0f}（{r['a_p5']:+.0f} ～ {r['a_p95']:+.0f}） | {r['p_in']:.0%} | {r['neg']:.0%} | {r['interval']:.2f} | "
        f"{r['P50']:.1f} / {r['P90']:.1f} | {r['fail_nodes']:.0%} | {r['seg6']:.2f} | {ck(r)} |"
        for k, r in enumerate(rows[:15]))
    mg = "\n".join(f"| {f} | {lvl} | {v['n_pass']}/5 | {v['p_in']:.0%} | {'有' if v['any_full'] else '沒有'} | `{v['best']}` |"
                   for (f, lvl), v in marg.items())
    full = [r for r in rows if r["n_pass"] == 5]
    write_text(od / "README.md", f"""# T11　哪些誤差組合做得出論文的 δT（向上飄移）

**目的**：T10 證明「只有晶振 ±20 ppm + PC ±50 ppm」做不出論文 +{lo:.0f}～+{hi:.0f} ppm 的斜率。這裡把找得到的候選誤差全部放進來、
做所有組合，看哪些組合同時對上論文的多項觀察。

**時鐘誤差是固定值**：每個節點的時鐘誤差（ρ）在一次 1 小時擷取內是常數；隨機的只有「抽到哪一顆晶振、哪一塊板子、哪一台 PC」。
（例外是刻意設成會變的因素：NTP 調速。溫度：ABS04W 的溫度係數約 −0.034 ppm/°C²，麵包板上 23→33 °C 只差 < 2 ppm，所以沒有列為因素。）

## 1. 候選因素

| 因素 | 水準 | 根據 |
|---|---|---|
| t_P 的時鐘源 | 32.768 kHz 晶振 / HFRC 鎖晶振（HFADJ，{cfg0['hfadj_ppm']:+.0f} ppm，規格書 §12.1.5：HFXTADJ = 1464）/ HFRC（RC，和 ADC 同一顆）/ LFRC（RC，±3 % 假設） | 論文 §II-A：晶振是時間戳的時鐘源；使用者問是否其實是 RC |
| 晶振有效負載電容 | 不接電容 ≈2.2 / 相符 4 / 外接 7 pF ≈5.7 / ≈8 / 外接 15 pF ≈9.7 pF | ABS04W：規格負載 4 pF、C0 1.5–2.0 pF、C1 6.5 fF、靈敏度 −90～−125 ppm/pF；Apollo3 每腳內建 3.4 pF、允許外加 ≤ 7 pF；SparkFun 論壇實測 Apollo3 換負載電容後誤差 +12 / +69 / +347 ppm |
| 韌體 tick → 時間換算 | 正確 / × 30.51 µs（−248 ppm）/ × 30.5 µs（−576 ppm） | 4 B 時間戳若以 µs 存，要做這個換算 |
| PC 時鐘（t_C） | NTP 校正 / 同左但 15.6 ms 解析度 / perf_counter ±50 ppm / NTP 調速 / 很差 ±300 ppm | pc-ble-driver-py 只支援 Python < 3.11；Windows 上 Python < 3.13 的 time.time() 解析度約 15.6 ms |

負載電容造成的頻率偏移：Δf/f = (C1/2)·[1/(C0 + CL_有效) − 1/(C0 + 4 pF)]，每塊板子的 C0、C1、雜散電容各自抽樣。
其他固定條件：T4 擬合的 BLE 延遲參數、4 節點 / 1 dongle、244 B、3600 s × {len(cfg0['seeds'])} seeds。ADC 計時器誤差 {cfg0['adc_err_ppm'][0]:.0f}–{cfg0['adc_err_ppm'][1]:.0f} ppm（T0 由封包間隔反推）。

## 2. 五項檢查（對照論文）

1. **斜率**：解析抽樣 {cfg0['n_mc']} 次，斜率中位數落在 {lo:.0f}–{hi:.0f} ppm。另外列出「單一節點落在範圍內的機率 p」；論文三張圖都落在範圍內，若三張來自不同節點，機率約 p³
2. **沒有往下的節點**：δT 往下的比例 ≤ 1 %（論文沒有出現 SharkTooth 失效）
3. **封包間隔**（以 t_P 計）在 {P['interval_ms'][0]}–{P['interval_ms'][1]} ms
4. **殘差分布**（Fig.6 時間窗，對下緣線）：P50 = {P['resid_ms']['P50']} ± 3 ms、P90 = {P['resid_ms']['P90']} ± 4 ms
5. **SharkTooth（論文字面版）能用**：壞輸出 > 5 % 的節點 ≤ 10 %，第 6 段節點對 mean|e| ≤ 1 ms（論文 C4 = {P['seg6_ms']} ms）

## 3. 結果：前 15 名組合

| # | t_P 時鐘源 | 負載電容 | 換算 | PC 時鐘 | 斜率 ppm 中位（5–95 %） | 落在論文範圍 | 往下 | 封包間隔 ms | 殘差 P50 / P90 ms | 壞輸出 > 5 % 的節點 | 第 6 段 ms | 檢查 1–5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{top}

五項全過的組合：{len(full)} 個（共 {len(rows)} 個）。

![各組合斜率](slope_by_combo.png)

![負載電容 → 斜率](load_cap_curve.png)

## 4. 各因素水準：最好能做到幾項

| 因素 | 水準 | 最多通過 | 該組合落在論文範圍 | 有沒有五項全過的組合 | 最佳組合 |
|---|---|---|---|---|---|
{mg}

## 5. t_C 解析度（Fig.3 的乾淨通道，T1 設定，5 seeds × 600 s）

| t_C 解析度 | σ100 (ms) | 每包下降 (ms) |
|---|---|---|
| 1 µs | {f3[1e-6]['sigma100']:.2f} | {f3[1e-6]['step_down']:.2f} |
| 15.625 ms（Windows time.time()，Python < 3.13） | {f3[0.015625]['sigma100']:.2f} | {f3[0.015625]['step_down']:.2f} |
| 論文 Fig.3 | {P['sigma100_ms'][0]}–{P['sigma100_ms'][1]} | −10.90 |

## 6. 結論

1. **RC 振盪器當 t_P 時鐘源：排除。** HFRC 會讓 δT 斜率約 +7000～+9000 ppm，而且封包間隔剛好 100.00 ms（和 ADC 同一顆振盪器）；
   LFRC 則是 ±數萬 ppm、正負都有。論文是 +235～+245 ppm、封包間隔 100.6–100.9 ms。
   但**ADC 取樣計時器很可能是 RC**：封包間隔慢 0.6–0.9 %，而且每張圖不一樣，符合每顆晶片不同的 RC 誤差（規格書 §12.1.5：HFRC 典型變異 ±2 %；§18.2.3：ADC 重複觸發只能用 CTIMER 3，CTIMER 可選 HFRC 分頻）。
2. **只有兩類組合五項全過**（PC 時鐘用 NTP 校正或 perf_counter 都可以）：
   - **(a) 韌體把 tick 換成時間時用了 30.51 µs**（負載電容剛好相符）：斜率中位數 +248 ppm；
   - **(b) 晶振的負載電容比規格大約 4 pF**（4 pF 晶振實際接到約 8 pF，例如每腳外接約 12 pF）：斜率中位數 +232 ppm。
     如果照抄 SparkFun Artemis 的參考設計（每腳 15 pF，那是給 12.5 pF 晶振用的），斜率約 +282 ppm，稍微偏大但在同一個量級。
   兩者都是「所有節點共有的固定誤差」，所以每個節點都往上、SharkTooth 不會失效——正好是論文看到的。
3. **單靠 PC 時鐘做不出來**：PC 時鐘很差（±300 ppm）時，斜率剛好在論文範圍的機率只有約 3 %，而且一半的節點往下。
4. **論文的 t_C 不是 15.6 ms 解析度**：Windows 上 Python < 3.13 的 time.time() 會讓 Fig.6 的殘差 P50 / P90 變成約 24 / 41 ms
   （論文 18 / 32 ms），Fig.3 的 σ100 變成約 16 ms（論文 14.2–14.8 ms）。所以作者用的應該是 perf_counter 或 Linux。
5. **PC 時鐘在擷取中被 NTP 調速會讓 SharkTooth 失效**（壞輸出 > 5 % 的節點達 75 %）：調速期間 δT 的斜率暫時改變，觸發大量樞紐點。
   論文沒有這種失效 → 擷取期間 PC 時鐘沒有被調速（或用的是不受 NTP 影響的 perf_counter）。這個調速模型是假設的。
6. **溫度、老化**：ABS04W 約 −0.034 ppm/°C²，麵包板上 < 2 ppm；老化每年約 ±3 ppm。都可以忽略。

**限制**：(a)、(b) 都只是「做得出論文現象」，不代表就是原因。負載電容的 C0、C1 用的是規格書典型值，板與板差異是假設；
HFADJ 的目標值已由 Apollo3 規格書 §12.1.5 確認（1464）；LFRC 精度規格書沒給、NTP 調速的大小沒查到原始數字。(b) 需要每腳外接約 12 pF，超過 Apollo3 規格書允許的 7 pF（Table 1147），但 SparkFun 自己的參考設計就用 15 pF，有照抄的前例。負載電容模型用 SparkFun 論壇的三組實測（12.5 pF 晶振接 15 pF / 7.5 pF / 不接：+12 / +69 / +347 ppm）檢驗，模型給 +10～13 / +86～95 / +339～343 ppm，量級與方向對。ABS04W 的「K」字尾我推測是 4 pF（同家 ABS05W-…-K-2-T 是 4 pF），沒有直接查到。

## 7. 開發板實測時怎麼分辨

| 檢查 | 方法 | (a) 換算常數 | (b) 負載電容 | PC 時鐘 |
|---|---|---|---|---|
| 讀原始 tick | 韌體直接送 32.768 kHz 計數值，自己在 PC 端換算 | 斜率變回約 0 | 不變 | 不變 |
| 量晶振頻率 | CLKOUT 接頻率計（論壇做法），或換負載電容 | 正確 | 慢約 240 ppm | 正確 |
| 換一台 PC | 同一塊板子接另一台電腦 | 不變 | 不變 | 改變 |
| 看斜率的分散 | 多塊板子同時量 | 約 ±20 ppm | 約 ±30–50 ppm（C1 差異放大） | 同一台 PC 上所有板子一起變 |
""")

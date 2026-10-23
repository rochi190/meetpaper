"""T12：溫度造成的漂移——晶振暖機 + 環境溫度起伏、PC 時鐘溫度、ADC 計時器（RC）溫度；
另外加上從論文資料找到的 t_P 解析度（整數毫秒）。

和論文相符的設定（T4 擬合參數 + T11 組合 (a)）只改這些，比較四種情境：
  off      = 不加溫度（目前的設定）；
  bench    = 論文的驗證環境（板子放桌上、只有自熱）；
  body     = 實際應用（貼在皮膚上，升到 31–34 °C）；
  bench_ms = bench + t_P 是整數毫秒。
同一個房間：室溫與空調起伏每次擷取抽一次，所有節點和 PC 共用；自熱、暖機快慢每個節點各自抽。
"""
import json

import numpy as np

from .. import figures as F
from ..algorithms import sharktooth
from ..config import deep_merge, draw, load_config, set_path
from ..evaluate import lower_line_lp
from ..runner import RESULTS, algo_params, node_series, outdir, pmap, write_json, write_text
from ..simulate import simulate
from .t10 import lower_slope
from .t11 import _fig3_job

SCEN_ORDER = ["off", "bench", "body", "bench_ms"]
SCEN_SHORT = {"off": "不加溫度", "bench": "桌上", "body": "貼身", "bench_ms": "桌上 + t_P 毫秒"}
SCEN_COLOR = {"off": "#2a78d6", "bench": "#eb6834", "body": "#1baf7a", "bench_ms": "#eda100"}  # 參考色盤 1–4 格
INK, INK2 = "#0b0b0b", "#52514e"
T_MARKS_MIN = [1, 5, 10, 30, 60]


def job_cfg(cfg0, scen, seed):
    """情境 → 這次擷取的設定。房間共用的量在這裡抽；每個節點各自的量留成分布，交給 make_node 抽。"""
    if scen == "off":
        return cfg0
    sc = cfg0["scenarios"][scen]
    if sc.get("base"):
        cfg = job_cfg(cfg0, sc["base"], seed)
        cfg["clock_node"]["tick_hz"] = sc["tick_hz"]
        return cfg
    rng = np.random.default_rng([seed, 12])
    rm = cfg0["room"]
    room = dict(amb_amp_C=float(rm["amb_amp_C"]), amb_period_s=float(draw(rm["amb_period_s"], rng)),
                amb_phase=float(rng.uniform(0, 2 * np.pi)))
    T_room = float(draw(rm["T_C"], rng))
    node_t = dict(enabled=True, T_start=T_room, **cfg0["crystal"], **sc["node"], **room)
    pc_t = dict(enabled=True, T_start=0.0, k_ppm_per_C2=0.0, **cfg0["pc"], **room)
    cfg = deep_merge(cfg0, dict(clock_adc=dict(temp_coeff_ppm_per_C=cfg0["adc_temp_coeff_ppm_per_C"])))
    cfg["clock_node"]["temp"] = node_t          # 整個取代：合併會留下預設的 T_end，蓋掉 dT_C
    cfg["clock_pc"]["temp"] = pc_t
    return cfg


def interval_drift_ppm(tp, k, t0=-np.inf, t1=np.inf):
    """封包間隔在 [t0, t1] 內的相對變化（ppm）：t_P 對封包編號做二次擬合。"""
    m = (tp >= t0) & (tp <= t1)
    if m.sum() < 50:
        return np.nan
    kk = k[m] - k[m][0]
    c2, c1, _ = np.polyfit(kk, tp[m], 2)
    return 2 * c2 * kk[-1] / c1 * 1e6


def paper_evidence(cfg0):
    """從 T0 數位化的論文資料量：(1) 封包間隔在圖內的變化（區塊自助法 95 % CI）；(2) t_P 是不是整數毫秒。"""
    out = {}
    for nm, fn in cfg0["paper"]["digitized"].items():
        a = np.genfromtxt(RESULTS.parent / "data" / "digitized" / f"{fn}.csv", delimiter=",", names=True)
        x = np.sort(a["x"])
        d = np.diff(x)
        k = np.r_[0, np.cumsum(np.maximum(1, np.round(d / np.median(d))))]   # 漏掉的點補上編號
        X = np.c_[np.ones_like(k), k, k ** 2]
        beta = np.linalg.lstsq(X, x, rcond=None)[0]
        r = x - X @ beta
        rng = np.random.default_rng(0)
        L, boot = 50, []
        for _ in range(500):
            idx = np.concatenate([np.arange(s0, s0 + L) for s0 in rng.integers(0, x.size - L, x.size // L + 1)])[:x.size]
            boot.append(interval_drift_ppm(X @ beta + r[idx], k))
        dms = d[(d > 0.09) & (d < 0.11)] * 1e3
        out[nm] = dict(drift=interval_drift_ppm(x, k), ci=np.percentile(boot, [2.5, 97.5]).tolist(),
                       interval_ms=float(dms.mean()),
                       ms_lattice=float(np.mean(np.abs(dms - np.round(dms)) < 0.05)),
                       frac_low=float(np.mean(np.round(dms) == np.floor(dms.mean()))),
                       frac_low_expected=float(1 - (dms.mean() - np.floor(dms.mean()))),
                       frac_beyond=float(np.mean(np.abs(np.round(dms) - dms.mean()) > 1.0)))
    return out


def epoch_mean(t, e, T):
    """每 1 s epoch 的平均（論文的 RSE 也是 1 s 平均）。"""
    ok = np.isfinite(e) & (t >= 0) & (t < T)
    i = t[ok].astype(int)
    n = np.bincount(i, minlength=int(T))
    s = np.bincount(i, weights=e[ok], minlength=int(T))
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _job(args):
    cfg, seed, bad_err, forgets, windows = args
    sim = simulate(cfg, seed)
    T = cfg["T"]
    conv = cfg["clock_node"]["scale_err_ppm"] * 1e-6
    tg = np.arange(0.0, T + 1.0, 10.0)
    r_pc = sim["pc_clock"].rho_at(tg)
    ep = np.arange(int(T)) + 0.5
    nodes, E = [], {f: [] for f in forgets}
    for k in range(cfg["n_nodes"]):
        meta = sim["nodes"][k]["meta"]
        p = node_series(sim, k)
        tp, tc, seq = p["t_P"], p["t_C"], p["pkt_seq"]
        dt = tc - tp
        # 真正的 δT 斜率隨時間（由時鐘直接算）
        r_n = meta["clock"].rho_at(tg)
        s = ((1 + r_pc) / ((1 + r_n) * (1 + conv)) - 1) * 1e6
        a, _ = lower_slope(tp, dt)
        early, late = tp <= 600, tp >= T - 600
        w = (tp >= 5) & (tp <= 145)
        a6, b6 = lower_line_lp(tp[w], dt[w])
        res = (dt[w] - (a6 * tp[w] + b6)) * 1e3
        row = dict(slope_curve=s - s[0], slope=a * 1e6,
                   node_ch=-(r_n[-1] - r_n[0]) * 1e6, pc_ch=(r_pc[-1] - r_pc[0]) * 1e6,   # 60 分鐘後各自貢獻的斜率變化
                   slope_obs_change=(lower_slope(tp[late], dt[late])[0] - lower_slope(tp[early], dt[early])[0]) * 1e6,
                   interval_ms=(tp[-1] - tp[0]) / (seq[-1] - seq[0]) * 1e3,
                   P50=float(np.percentile(res, 50)), P90=float(np.percentile(res, 90)),
                   adc_tc=meta["adc_tc_ppm"],
                   drift={nm: interval_drift_ppm(tp, seq, *win) for nm, win in windows.items()})
        tt = p["t_true_last_sample"]
        for f in forgets:
            r = sharktooth.run(tp, tc, **dict(algo_params(cfg), forget=f))
            e = np.where(r["valid"], r["ts"] - p["t_pc_true"], np.nan)
            if f == 1.0:
                steady = r["valid"] & (tt > T / 3)
                cst = np.nanmedian(e[steady]) if steady.any() else np.nan
                row["bad"] = float(np.mean(~r["valid"] | ~(np.abs(e - cst) <= bad_err)))
            E[f].append(epoch_mean(tt, e, T))
        nodes.append(row)
    seg = {}
    for f in forgets:
        per_pair = []
        for i in range(len(E[f])):
            for j in range(i + 1, len(E[f])):
                d = np.abs(E[f][i] - E[f][j]) * 1e3
                per_pair.append([float(np.nanmean(d[(ep >= 600 * s) & (ep < 600 * (s + 1))])) for s in range(6)])
        seg[f] = np.nanmean(np.array(per_pair), axis=0)          # 各對平均，第 1–6 段
    return dict(nodes=nodes, seg=seg, t_curve=tg)


def summarize(cfg0, rr, ev):
    nodes = [n for r in rr for n in r["nodes"]]
    tg = rr[0]["t_curve"]
    C = np.array([n["slope_curve"] for n in nodes])
    pick = lambda arr: dict(med=float(np.nanmedian(arr)), p5=float(np.nanpercentile(arr, 5)),
                            p95=float(np.nanpercentile(arr, 95)))
    s = dict(n_nodes=len(nodes))
    s["slope_change"] = {m: pick(np.array([np.interp(m * 60.0, tg, c) for c in C])) for m in T_MARKS_MIN}
    s["slope_obs_change"] = pick(np.array([n["slope_obs_change"] for n in nodes]))
    s["node_ch_abs"] = float(np.median(np.abs([n["node_ch"] for n in nodes])))
    s["pc_ch_abs"] = float(np.median(np.abs([n["pc_ch"] for n in nodes])))
    slopes = np.array([n["slope"] for n in nodes])
    lo, hi = cfg0["paper"]["slope_ppm"]
    s["slope"] = pick(slopes)
    s["slope_in"] = float(np.mean((slopes >= lo) & (slopes <= hi)))
    s["neg"] = float(np.mean(slopes < 0))
    s["interval_ms"] = float(np.median([n["interval_ms"] for n in nodes]))
    s["P50"] = float(np.median([n["P50"] for n in nodes]))
    s["P90"] = float(np.median([n["P90"] for n in nodes]))
    s["fail_nodes"] = float(np.mean([n["bad"] > 0.05 for n in nodes]))
    s["drift"] = {}
    for nm in cfg0["paper"]["windows_s"]:
        v = np.abs(np.array([n["drift"][nm] for n in nodes]))
        s["drift"][nm] = dict(abs_med=float(np.nanmedian(v)), abs_p90=float(np.nanpercentile(v, 90)),
                              frac_ge_paper=float(np.nanmean(v >= abs(ev[nm]["drift"]))))
    s["seg"] = {str(f): np.nanmean(np.array([r["seg"][f] for r in rr]), axis=0).tolist() for f in cfg0["forget_values"]}
    return s


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    if cfg0.get("use_T4_best"):
        bp = json.loads((RESULTS / "T4_fig6" / "summary.json").read_text())["best_params"]
        for k, v in bp.items():
            cfg0 = set_path(cfg0, k, v)
    od = outdir(cfg0["name"])
    ev = paper_evidence(cfg0)
    seeds = cfg0["seeds"]
    windows = {k: tuple(v) for k, v in cfg0["paper"]["windows_s"].items()}
    jobs = [(job_cfg(cfg0, sc, s), s, cfg0["bad_err_s"], cfg0["forget_values"], windows)
            for sc in SCEN_ORDER for s in seeds]
    res = pmap(_job, jobs)
    ns = len(seeds)
    raw = {sc: res[i * ns:(i + 1) * ns] for i, sc in enumerate(SCEN_ORDER)}
    summ = {sc: summarize(cfg0, raw[sc], ev) for sc in SCEN_ORDER}

    # Fig.3 乾淨通道：溫度、t_P 解析度會不會改變 σ100（T1 設定）
    c1 = load_config(RESULTS.parent / "config" / "T1_fig3.yaml", T=3600.0)
    c1 = deep_merge(c1, {k: cfg0[k] for k in ("room", "crystal", "adc_temp_coeff_ppm_per_C", "pc", "scenarios")})
    for sc in ("off", "bench", "bench_ms"):
        f3 = pmap(_fig3_job, [(job_cfg(c1, sc, s), s) for s in seeds[:5]])
        summ[sc]["fig3_sigma100"] = float(np.mean([r["sigma100"] for r in f3]))

    write_json(od / "summary.json", dict(paper_evidence=ev, scenarios=summ))
    figures(od, cfg0, raw, summ, ev)
    write_readme(od, cfg0, summ, ev)


def figures(od, cfg0, raw, summ, ev):
    plt = F.plt
    # 1. δT 斜率隨時間的變化（相對開始時），每個情境一格
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, sc in zip(axs, ["bench", "body"]):
        tg = raw[sc][0]["t_curve"] / 60
        C = np.array([n["slope_curve"] for r in raw[sc] for n in r["nodes"]])
        for c in C[:40]:
            ax.plot(tg, c, color=SCEN_COLOR[sc], lw=0.6, alpha=0.35)
        ax.plot(tg, np.median(C, axis=0), color=SCEN_COLOR[sc], lw=2.0, label="中位數")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(cfg0["scenarios"][sc]["label"], color=INK)
        ax.set_xlabel("時間 (min)")
        ax.legend(loc="lower right", frameon=False)
    axs[0].set_ylabel("δT 斜率的變化 (ppm，相對開始時)")
    fig.suptitle("溫度讓 δT 斜率慢慢改變，直到穩態（細線：各節點；不加溫度時恆為 0）", color=INK)
    fig.tight_layout()
    fig.savefig(od / "slope_vs_time.png")
    plt.close(fig)

    # 2. 封包間隔在約 2 分鐘內的變化：模擬（各節點）vs 論文（t_P 解析度不影響，所以不畫 bench_ms）
    names = {"fig6": "Fig.6 窗（開始後 5–145 s）", "fig3": "Fig.3 窗（約 48 分鐘）", "fig4": "Fig.4 窗（約 56 分鐘）"}
    show = ["off", "bench", "body"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, nm in zip(axs, names):
        for i, sc in enumerate(show):
            v = np.array([n["drift"][nm] for r in raw[sc] for n in r["nodes"]])
            ax.scatter(i + rng.uniform(-0.18, 0.18, v.size), v, s=10, color=SCEN_COLOR[sc], alpha=0.6, linewidths=0)
        ax.axhline(ev[nm]["drift"], color=INK, lw=1.5, ls="--")
        ax.text(2.85, ev[nm]["drift"] + 8, "論文", ha="right", va="bottom", color=INK, fontsize=10)
        ax.set_xticks(range(3), [SCEN_SHORT[sc] for sc in show])
        ax.set_xlim(-0.5, 2.9)
        ax.set_ylim(-400, 400)
        ax.set_title(names[nm], color=INK)
    axs[0].set_ylabel("封包間隔變化 (ppm；貼身的開頭超出範圍)")
    fig.suptitle("ADC 計時器（RC）隨溫度變：封包間隔在約 2 分鐘內的變化", color=INK)
    fig.tight_layout()
    fig.savefig(od / "interval_drift.png")
    plt.close(fig)

    # 3. SharkTooth 節點對誤差（各段），兩種遺忘因子各一格
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, f in zip(axs, cfg0["forget_values"]):
        for sc in SCEN_ORDER:
            ax.plot(range(1, 7), summ[sc]["seg"][str(f)], "-o", color=SCEN_COLOR[sc], lw=2, ms=8, label=SCEN_SHORT[sc])
        ax.set_title("累積 OLS（論文）" if f == 1.0 else f"遺忘因子 λ = {f}", color=INK)
        ax.set_xlabel("段（每段 10 分鐘）")
        ax.legend(frameon=False)
    axs[0].set_ylabel("節點對 mean|誤差| (ms，真值，1 s 平均)")
    fig.tight_layout()
    fig.savefig(od / "pair_error_segments.png")
    plt.close(fig)


def write_readme(od, cfg0, summ, ev):
    lab = {"off": "不加溫度（目前的設定）", "bench": "桌上（論文驗證）", "body": "貼身穿戴", "bench_ms": "桌上 + t_P 整數毫秒"}
    S = summ
    o, b, y, bm = S["off"], S["bench"], S["body"], S["bench_ms"]
    lam = str(cfg0["forget_values"][1])

    def sc_row(sc):
        r = S[sc]
        ch = " / ".join(f"{r['slope_change'][m]['med']:+.2f}" for m in T_MARKS_MIN)
        ob = r["slope_obs_change"]
        return (f"| {lab[sc]} | {ch} | {r['slope_change'][60]['p5']:+.1f} ～ {r['slope_change'][60]['p95']:+.1f} | "
                f"{r['node_ch_abs']:.2f} / {r['pc_ch_abs']:.2f} | {ob['med']:+.1f}（{ob['p5']:+.1f} ～ {ob['p95']:+.1f}） |")

    def check_row(sc):
        r = S[sc]
        f3 = f"{r['fig3_sigma100']:.2f}" if "fig3_sigma100" in r else "—"
        return (f"| {lab[sc]} | {r['slope']['med']:+.0f}（{r['slope']['p5']:+.0f} ～ {r['slope']['p95']:+.0f}） | "
                f"{r['slope_in']:.0%} | {r['neg']:.0%} | {r['interval_ms']:.2f} | {r['P50']:.1f} / {r['P90']:.1f} | "
                f"{r['fail_nodes']:.0%} | {f3} |")

    def drift_row(sc):
        r = S[sc]["drift"]
        return f"| {lab[sc]} | " + " | ".join(
            f"{r[nm]['abs_med']:.0f}（P90 {r[nm]['abs_p90']:.0f}；{r[nm]['frac_ge_paper']:.0%}）" for nm in ("fig6", "fig3", "fig4")) + " |"

    def seg_row(sc):
        return f"| {lab[sc]} | " + " | ".join(" / ".join(f"{v:.2f}" for v in S[sc]["seg"][str(f)]) for f in cfg0["forget_values"]) + " |"

    def ev_row(nm, title):
        e = ev[nm]
        lat = e["ms_lattice"] > 0.9                     # 不在整數 ms 格點上時，後兩欄沒有意義
        low = f"{e['frac_low']:.1%}（純量化預期 {e['frac_low_expected']:.1%}）" if lat else "—"
        beyond = f"{e['frac_beyond']:.1%}" if lat else "—"
        return (f"| {title} | {e['drift']:+.0f}（{e['ci'][0]:+.0f} ～ {e['ci'][1]:+.0f}） | {e['interval_ms']:.3f} | "
                f"{e['ms_lattice']:.0%} | {low} | {beyond} |")

    seg_mean = lambda r, f="1.0": float(np.mean(r["seg"][f][1:]))       # 第 2–6 段平均
    pa = cfg0["paper"]
    text = f"""# T12　溫度造成的漂移（晶振暖機、PC 時鐘、ADC 計時器）與 t_P 解析度

**設定**（`config/T12_temperature.yaml`）：和論文相符的設定（T4 擬合的延遲參數 + T11 組合 (a)），只改溫度（和 t_P 解析度）。
4 節點 × {len(cfg0['seeds'])} seeds × 3600 s。
同一個房間的室溫、空調起伏所有節點和 PC 共用；自熱、暖機快慢每個節點各自抽。

## 加入的物理因素與數值出處

| 因素 | 模型 | 數值 | 出處 |
|---|---|---|---|
| 節點晶振溫度（A2） | ρ = k (T − T0)² | k −0.04～−0.02 ppm/°C²、T0 20～30 °C | ABS04W 規格書 |
| 暖機（A3） | T(t) = T_end − (T_end − T_start)e^(−t/τ) | 桌上：自熱 +0.5～2 °C；貼身：升到 31～34 °C；τ 3～10 min | 自熱是假設（板子約 10–20 mW）；貼身裝置約 5–10 分鐘達穩態 |
| 環境溫度起伏 | ± A sin(2πt/P) | ±0.5 °C、週期 20～40 min | 假設（一般空調溫控） |
| PC 時鐘溫度（A6） | ρ = k1 (T − T_start) | k1 ±0.6 ppm/°C；CPU 負載升溫 1～5 °C、τ 5～20 min | Murdoch 2006：AT-cut 室溫附近約 0.5 ppm/°C，負載升溫 1–1.5 °C（Tor）／5 °C（CPUBurn），正負都有 |
| ADC 計時器溫度（B6） | 週期誤差 + tc (T − T_start) | tc ±300 ppm/°C | 48 MHz RC 全溫度範圍約 ±4 %（Microchip OSC48M）；Apollo3 規格書只說 HFRC 典型 ±2 % |
| t_P 解析度（A7） | 讀數無條件捨去到 1 ms | 1 ms（原本 1/32768 s） | 論文資料，見下表 |

## 論文資料裡的證據（由 T0 數位化的點直接量）

| 圖 | 封包間隔在圖內的變化 ppm（95 % CI） | 平均間隔 ms | 相鄰差落在整數 ms | 「較短那一格」比例 | 差超過 ±1 ms 的比例 |
|---|---|---|---|---|---|
{ev_row('fig6', 'Fig.6（開始後 5–145 s）')}
{ev_row('fig3', 'Fig.3（約 48 分鐘）')}
{ev_row('fig4', 'Fig.4（約 56 分鐘）')}

- **封包間隔在幾分鐘內變了約 100 ppm**：晶振量的 t_P 只會差 < 1 ppm，所以這是取樣用的 RC 計時器在變，量級和「RC 隨溫度變」一致。
  Fig.6 是擷取剛開始：前 50 s 平，之後上升再趨平。
- **t_P 是整數毫秒**：Fig.3、Fig.4 的相鄰 t_P 差 100 % 是整數 ms。
  - 這不是數位化造成的：PDF 座標的最小單位換算是 0.70 ms，不會剛好落在 1.00 ms 的格點上。
  - Fig.4「100 ms 那一格」的比例和「只有量化」的預期一模一樣。
  - Fig.3 還有額外約 ±1 ms 的跳動（有 102 ms 的差），代表某些封包的 t_P 晚了 1 ms。
  - Fig.6 的點座標精度較差，看不清楚。
  - 韌體送出的是毫秒數，所以一定有一個 tick → ms 的換算（T11 的候選 (a) 就發生在這一步）。

## 1. δT 斜率隨時間的變化（相對開始時，ppm）

- 「節點 / PC」：60 分鐘後兩者各自造成的斜率變化（|ppm|，中位數）。
- 「從資料量到的」：最後 10 分鐘與最初 10 分鐘的下緣斜率差，也就是只看 δT 資料能量到的變化（中位數，括號 5–95 %）。

| 情境 | 1 / 5 / 10 / 30 / 60 min（中位數） | 60 min 時 5–95 % | 節點 / PC | 從資料量到的 |
|---|---|---|---|---|
{chr(10).join(sc_row(sc) for sc in ('off', 'bench', 'body'))}

![斜率隨時間](slope_vs_time.png)

## 2. 和論文的比對（同 T11 的檢查）

| 情境 | 斜率 ppm 中位（5–95 %） | 落在 {pa['slope_ppm'][0]:.0f}–{pa['slope_ppm'][1]:.0f} | 往下 | 封包間隔 ms | 殘差 P50 / P90 ms | SharkTooth 失效節點 | Fig.3 σ100 ms |
|---|---|---|---|---|---|---|---|
{chr(10).join(check_row(sc) for sc in SCEN_ORDER)}
| **論文** | +235 ～ +245 | — | 0 % | {pa['interval_ms'][0]}–{pa['interval_ms'][1]} | {pa['resid_ms']['P50']} / {pa['resid_ms']['P90']} | 0 % | 14.23–14.75 |

## 3. 封包間隔在約 2 分鐘內的變化（|ppm|，各節點中位數；括號是 P90、以及 ≥ 論文值的節點比例）

| 情境 | Fig.6 窗（論文 {ev['fig6']['drift']:+.0f}） | Fig.3 窗（論文 {ev['fig3']['drift']:+.0f}） | Fig.4 窗（論文 {ev['fig4']['drift']:+.0f}） |
|---|---|---|---|
{chr(10).join(drift_row(sc) for sc in ('off', 'bench', 'body'))}

![封包間隔變化](interval_drift.png)

## 4. SharkTooth 節點對誤差（真值，1 s 平均，各對平均，第 1–6 段，ms）

| 情境 | 累積 OLS（論文） | λ = {lam} |
|---|---|---|
{chr(10).join(seg_row(sc) for sc in SCEN_ORDER)}

![節點對誤差](pair_error_segments.png)

## 結論

1. **貼身時看得到「斜率越來越大，直到穩態」**：
   - 中位數 10 分鐘 {y['slope_change'][10]['med']:+.1f}、30 分鐘 {y['slope_change'][30]['med']:+.1f}、60 分鐘 {y['slope_change'][60]['med']:+.1f} ppm。
   - 主要來自節點晶振（60 分鐘時 {y['node_ch_abs']:.1f} ppm，PC {y['pc_ch_abs']:.1f} ppm）：皮膚溫度離晶振的轉折溫度遠，拋物線變陡。
   - 每個節點不一樣（5–95 %：{y['slope_change'][60]['p5']:+.1f} ～ {y['slope_change'][60]['p95']:+.1f}），因為轉折溫度、係數、PC 的正負號都不同。
2. **桌上（論文的環境）幾乎沒有暖機曲線**：
   - 中位數 {b['slope_change'][60]['med']:+.2f} ppm，5–95 % {b['slope_change'][60]['p5']:+.1f} ～ {b['slope_change'][60]['p95']:+.1f}。
   - 節點晶振在轉折溫度附近幾乎不動（{b['node_ch_abs']:.2f} ppm），變化主要是 PC 升溫（{b['pc_ch_abs']:.2f} ppm）和空調起伏，正負都有。
3. **兩種都比 +240 ppm 小兩個數量級。** 從 δT 資料本身量不太出來：不加溫度時，量到的變化就有 {o['slope_obs_change']['p5']:+.1f} ～ {o['slope_obs_change']['p95']:+.1f} ppm 的量測雜訊。
4. **加了溫度後仍然和論文相符**：
   - 斜率中位數 {o['slope']['med']:+.0f} → {b['slope']['med']:+.0f} ppm（桌上），差別在抽樣誤差內。
   - 殘差 P50 / P90、封包間隔、Fig.3 σ100、SharkTooth 失效節點都一樣（見第 2 節）。
   - 所以 T4 的擬合參數不用重調，T1–T11 的結論不變。
5. **ADC 計時器（RC）的溫度效應和論文資料的量級一致，但後段偏小**：
   - 桌上情境 Fig.6 窗：模擬中位數 {b['drift']['fig6']['abs_med']:.0f} ppm（P90 {b['drift']['fig6']['abs_p90']:.0f}），{b['drift']['fig6']['frac_ge_paper']:.0%} 的節點 ≥ 論文的 {abs(ev['fig6']['drift']):.0f} ppm。論文在偏高處，但在範圍內。
   - Fig.4 窗的 {abs(ev['fig4']['drift']):.0f} ppm 很普通（{b['drift']['fig4']['frac_ge_paper']:.0%} 的節點 ≥）。
   - Fig.3 窗的 {abs(ev['fig3']['drift']):.0f} ppm 只有 {b['drift']['fig3']['frac_ge_paper']:.0%} 的節點達到。論文後段 30 秒內就有數十 ppm 的起伏，比「溫度慢慢變」大。
     RC 振盪器本身還有和溫度無關的短期頻率起伏，這裡沒有建模，因為它不影響 δT 和同步結果。
   - 不加溫度時是 0。這個效應不影響 δT 斜率，只改變封包相位掃過 CI 的速度。
6. **t_P 整數毫秒幾乎沒有影響**：
   - 和論文的比對一樣（第 2 節）。
   - SharkTooth 節點對誤差第 2–6 段平均 {seg_mean(b):.2f} → {seg_mean(bm):.2f} ms。
   - 原因：封包間隔 100.x ms 不是整數毫秒，每包被捨去的量在 0–1 ms 之間輪流變化，1 s（10 包）平均後幾乎剛好是 0.5 ms。
     每個節點都一樣，所以兩兩相減就抵消了。
   - 反過來說，如果封包間隔剛好是整數毫秒（例如 100.000 ms），捨去量每包都一樣、不會抵消，就會留下最多 1 ms 的固定偏差。
7. **SharkTooth 和溫度**：
   - 桌上的節點對誤差和不加溫度時一樣（第 2–6 段平均 {seg_mean(b):.2f} vs {seg_mean(o):.2f} ms）。
   - 貼身時後段變大（第 5–6 段 {y['seg']['1.0'][4]:.2f} / {y['seg']['1.0'][5]:.2f} ms，不加溫度 {o['seg']['1.0'][4]:.2f} / {o['seg']['1.0'][5]:.2f}）：各節點升溫曲線不同，累積 OLS 把前期的彎曲留在斜率裡（和 T5 一樣）。
   - 遺忘因子 λ = {lam} 降到 {y['seg'][lam][4]:.2f} / {y['seg'][lam][5]:.2f} ms。

**限制**：
- 自熱的大小、空調起伏、RC 溫度係數的大小與正負號都是假設，或從其他晶片推的；PC 的溫度係數用的是 Murdoch 量的舊 PC。
- t_P 整數毫秒是從 Fig.3 / 4 看出來的，是「無條件捨去」還是「四捨五入」看不出來；這裡用捨去。
- 開發板上可以直接量：用 CLKOUT 量晶振、記錄封包間隔與板子溫度，就能把這些假設換成實測值。
"""
    write_text(od / "README.md", text)

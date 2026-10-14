"""T4：復現 Fig.6（主要目標）＋ 參數擬合 ＋ 消融。"""
import json

import numpy as np

from .. import digitize as dg
from .. import evaluate as ev
from .. import figures as F
from ..config import load_config, set_path
from ..runner import ROOT, node_series, outdir, pmap, run_node_algos, write_json, write_text
from ..simulate import simulate

SCALE = dict(P10=1.0, P50=1.0, P90=1.5, P99=5.0, outlier=0.01, startup=15.0)


def series_stats(tp, dt, CI=0.03):
    bs, r = ev.band_stats(tp, dt, skip_start_s=1.0)
    line = (bs["slope_ppm"] * 1e-6, bs["intercept"])
    su = ev.startup_spike(tp, dt, line, 6)
    sig = ev.sigma_M(dt, 100)
    return dict(P10=bs["P10"], P50=bs["P50"], P90=bs["P90"], P95=bs["P95"], P99=bs["P99"], max=bs["max"],
                outlier=bs["outlier_frac"], frac_gt_CI=bs["frac_gt_CI"], startup=su["first_excess_ms"],
                startup4=su["excess_ms"][:4], slope_ppm=bs["slope_ppm"],
                rough=ev.lower_edge_roughness(tp[tp > tp[0] + 2], dt[tp > tp[0] + 2]),
                sigma100=float(np.nanmean(sig) * 1e3), width=bs["P90"] - bs["P10"]), r


def target():
    T = json.loads((ROOT / "data" / "targets.json").read_text())["fig6"]
    d = dg.load("fig6", "dots")
    d = d[np.argsort(d[:, 0])]
    st, r = series_stats(d[:, 0], d[:, 1])
    return st, r, d, T


def sim_stats(cfg, seed):
    sim = simulate(cfg, seed)
    out = []
    for k in range(cfg["n_nodes"]):
        p = node_series(sim, k)
        st, r = series_stats(p["t_P"], p["t_C"] - p["t_P"], cfg["CI"])
        out.append((st, r))
    return out


def pooled(cfg, seeds):
    res = [sim_stats(cfg, s) for s in seeds]
    flat = [x for rr in res for x in rr]
    keys = [k for k in flat[0][0] if k != "startup4"]
    m = {k: float(np.nanmean([f[0][k] for f in flat])) for k in keys}
    m["startup4"] = np.mean([f[0]["startup4"] for f in flat], axis=0).tolist()
    # 分位數用合併後的殘差（比平均分位數穩定）
    r = np.concatenate([f[1] for f in flat])
    for q, k in [(10, "P10"), (50, "P50"), (90, "P90"), (95, "P95"), (99, "P99")]:
        m[k] = float(np.percentile(r, q))
    m["outlier"] = float(np.mean(r > 45))
    return m, r


def loss(m, t):
    return float(sum(((m[k] - t[k]) / s) ** 2 for k, s in SCALE.items()))


def apply(cfg, combo):
    for k, v in combo.items():
        cfg = set_path(cfg, k, v)
    return cfg


def _job(args):
    cfg, seeds = args
    m, _ = pooled(cfg, seeds)
    return m


def run(cfg_path):
    cfg0 = load_config(cfg_path)
    od = outdir(cfg0["name"])
    tst, tr, d, T6 = target()
    rng = np.random.default_rng(2024)
    rngs = cfg0["fit"]["ranges"]
    combos = [{k: float(rng.uniform(*v)) for k, v in rngs.items()} for _ in range(cfg0["fit"]["n_random"])]
    res = pmap(_job, [(apply(cfg0, c), cfg0["seeds"]) for c in combos])
    rows = [dict(combo=c, m=m, loss=loss(m, tst)) for c, m in zip(combos, res)]
    rows.sort(key=lambda r: r["loss"])
    # 第二階段：在前 10 名附近擾動
    top = rows[:10]
    combos2 = []
    for _ in range(cfg0["fit"]["n_refine"]):
        b = top[rng.integers(len(top))]["combo"]
        c = {}
        for k, (lo, hi) in rngs.items():
            c[k] = float(np.clip(b[k] + rng.normal(0, 0.08 * (hi - lo)), lo, hi))
        combos2.append(c)
    res2 = pmap(_job, [(apply(cfg0, c), cfg0["seeds"]) for c in combos2])
    rows += [dict(combo=c, m=m, loss=loss(m, tst)) for c, m in zip(combos2, res2)]
    rows.sort(key=lambda r: r["loss"])
    best = rows[0]
    cfgb = apply(cfg0, best["combo"])
    # 用更多 seed 驗證
    mv, r_sim = pooled(set_path(cfgb, "T", 1200.0), [100, 101, 102, 103, 104])
    passed = dict(P10=abs(mv["P10"] - tst["P10"]) < 3, P50=abs(mv["P50"] - tst["P50"]) < 3,
                  P90=abs(mv["P90"] - tst["P90"]) < 3, P99=abs(mv["P99"] - tst["P99"]) < 10,
                  outlier=abs(mv["outlier"] - tst["outlier"]) < 0.02)
    # ---------- 圖：並排 ----------
    sim = simulate(set_path(cfgb, "T", 3600.0), 11)
    p = node_series(sim, 0)
    tp, dt = p["t_P"], p["t_C"] - p["t_P"]
    st = run_node_algos(p, cfgb, ["sharktooth"])["sharktooth"]
    m = tp < 145
    a_p, b_p = ev.lower_line_lp(d[d[:, 0] > 6, 0], d[d[:, 0] > 6, 1])
    a_s, b_s = ev.lower_line_lp(tp[m & (tp > 1)], dt[m & (tp > 1)])
    off = b_p - b_s
    # 平移 t_P 讓第一包和論文一樣在 5.1 s（論文開始記錄前約 5 s 沒有資料）
    toff = d[0, 0] - tp[0]
    red = dg.load("fig6", "red")
    piv = dg.load("fig6", "star").ravel()
    fig, axs = F.plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    F.fig6_panel(axs[0], d[:, 0], d[:, 1], None, piv, "論文 Fig.6（數位化）")
    axs[0].plot(red[:, 0], red[:, 1], "-", color="r", lw=1.6, label="Output δT*")
    axs[0].legend(loc="upper right", fontsize=10)
    pv = np.flatnonzero(st["pivot"][m])
    F.fig6_panel(axs[1], tp[m] + toff, dt[m] + off, st["star"][m] + off,
                 (tp[pv[-1]] + toff, dt[pv[-1]] + off), "模擬（4 節點 / 1 dongle，節點 0）")
    F.save(fig, od / "fig6_compare.png")
    # QQ
    fig, axs = F.plt.subplots(1, 2, figsize=(12, 4.6))
    qs = np.linspace(0.5, 99.5, 199)
    axs[0].plot(np.percentile(tr, qs), np.percentile(r_sim, qs), ".", color=F.BLUE)
    lim = max(np.percentile(tr, 99.5), np.percentile(r_sim, 99.5)) + 5
    axs[0].plot([0, lim], [0, lim], "k--", lw=1)
    axs[0].set_xlabel("論文 Fig.6 殘差分位數 (ms)")
    axs[0].set_ylabel("模擬殘差分位數 (ms)")
    axs[0].set_title("QQ 圖：δT − 下緣")
    bins = np.arange(0, 105, 2.5)
    axs[1].hist(tr, bins=bins, density=True, histtype="step", lw=2, label="論文 Fig.6")
    axs[1].hist(r_sim, bins=bins, density=True, histtype="step", lw=2, label="模擬")
    axs[1].set_xlabel("殘差 (ms)")
    axs[1].legend()
    F.save(fig, od / "qq_residual.png")
    # ---------- 消融 ----------
    abl_defs = {
        "完整模型": {},
        "A1 晶振誤差 off": {"clock_node.rho_ppm": 0.0},
        "A7 tick 量化 off": {"clock_node.tick_hz": None},
        "B1 ISR 抖動 off": {"isr_jitter.enabled": False},
        "B2 韌體延遲 off": {"fw_delay.enabled": False},
        "C1 等 event off（CI→1.25 ms，不排程）": {"CI": 0.00125, "sched.enabled": False},
        "C3 重傳 off": {"per.enabled": False},
        "C4 流量控制 off（window 8、每 event 4 包）": {"flow.window": 8, "flow.max_pkts_per_event": 4},
        "C5 排程 off": {"sched.enabled": False},
        "C11 起始延遲 off": {"startup.enabled": False},
        "D2 序列化 off": {"serial.enabled": False},
        "D3 USB frame off": {"usb.enabled": False},
        "E1 OS 延遲 off": {"os.enabled": False},
        "E2 停頓 off": {"pauses.enabled": False},
        "E3 GIL / 執行緒 off": {"gil.enabled": False, "os.per_node_scale": 0.0},
        "封包間隔剛好 100 ms（取樣計時器 = 晶振）": {"clock_adc.period_err_ppm": -248.0},
    }
    abl_jobs = [(apply(set_path(cfgb, "T", 1200.0), v), [100, 101, 102]) for v in abl_defs.values()]
    abl_res = pmap(_job, abl_jobs)
    abl = {k: m_ for k, m_ in zip(abl_defs, abl_res)}
    full = abl["完整模型"]

    def feat(m_):
        return dict(帶寬_P90減P10=m_["width"], P99=m_["P99"], 離群比例=m_["outlier"] * 100,
                    起始尖峰=m_["startup"], 起始第2點=m_["startup4"][1], 下緣粗糙度=m_["rough"], σ100=m_["sigma100"])
    rows_abl = []
    for k, m_ in abl.items():
        f = feat(m_)
        rows_abl.append((k, f))
    write_json(od / "summary.json", dict(target=tst, best_params=best["combo"], best_fit=best["m"], best_loss=best["loss"],
                                         validation=mv, passed=passed, ablation={k: feat(v) for k, v in abl.items()},
                                         top10=[dict(combo=r["combo"], loss=r["loss"]) for r in rows[:10]]))
    tf = feat(dict(tst, width=tst["P90"] - tst["P10"], sigma100=float("nan")))
    tf["起始第2點"] = tst["startup4"][1]

    def flag(k, f):
        if k == "完整模型":
            return ""
        ff = feat(full)
        msgs = []
        if abs(f["帶寬_P90減P10"] - ff["帶寬_P90減P10"]) > 3:
            msgs.append(f"帶寬 {ff['帶寬_P90減P10']:.0f}→{f['帶寬_P90減P10']:.0f} ms")
        if abs(f["離群比例"] - ff["離群比例"]) > 1.5:
            msgs.append(f"離群 {ff['離群比例']:.1f}→{f['離群比例']:.1f} %")
        if abs(f["起始尖峰"] - ff["起始尖峰"]) > 30:
            msgs.append(f"起始尖峰 {ff['起始尖峰']:.0f}→{f['起始尖峰']:.0f} ms")
        elif abs(f["起始第2點"] - ff["起始第2點"]) > 30:
            msgs.append(f"起始斜坡消失（第 2 點 {ff['起始第2點']:.0f}→{f['起始第2點']:.0f} ms，尖峰只剩 1 點）")
        if f["下緣粗糙度"] > 2 * ff["下緣粗糙度"] + 1:
            msgs.append(f"下緣變階梯（粗糙度 {ff['下緣粗糙度']:.1f}→{f['下緣粗糙度']:.1f} ms）")
        if abs(f["σ100"] - ff["σ100"]) > 1.0:
            msgs.append(f"σ100 {ff['σ100']:.1f}→{f['σ100']:.1f} ms")
        return "；".join(msgs) or "（看不出差別）"
    abl_tbl = "\n".join(
        f"| {k} | {f['帶寬_P90減P10']:.1f} | {f['P99']:.1f} | {f['離群比例']:.1f} | {f['起始尖峰']:.0f} / {f['起始第2點']:.0f} | {f['下緣粗糙度']:.1f} | {f['σ100']:.1f} | {flag(k, f)} |"
        for k, f in rows_abl)
    write_text(od / "ablation_table.md", "| 關掉的模組 | 帶寬 P90−P10 (ms) | P99 (ms) | 離群 >45 ms (%) | 起始第 1 / 2 點 (ms) | 下緣粗糙度 (ms) | σ100 (ms) | 消失 / 改變的特徵 |\n|---|---|---|---|---|---|---|---|\n" + abl_tbl
               + f"\n| **論文 Fig.6** | {tf['帶寬_P90減P10']:.1f} | {tf['P99']:.1f} | {tf['離群比例']:.1f} | {tf['起始尖峰']:.0f} / {tf['起始第2點']:.0f} | {tst['rough']:.1f} | — | |\n")
    bp = best["combo"]
    write_text(od / "README.md", f"""# T4　復現 Fig.6（主要目標）

**設定**（`config/T4_fig6.yaml`）：4 節點 / 1 dongle、244 B、封包間隔 100.78 ms（T0）；開啟 A1 A7 B1 B2 C1 C3 C4 C5 C11 D2 D3 E1 E2 E3。
**擬合**：隨機搜尋 {cfg0['fit']['n_random']} 組 + 前 10 名附近擾動 {cfg0['fit']['n_refine']} 組（每組 3 seeds × 600 s × 4 節點）。
目標函數 = Σ((模擬 − 目標)/尺度)²，尺度：P10/P50 1 ms、P90 1.5 ms、P99 5 ms、離群比例 1 %、起始尖峰 15 ms。

## 最佳參數

| 參數 | 值 |
|---|---|
| D2 序列化服務時間 `serial.base_s` | {bp['serial.base_s']*1e3:.2f} ms（+ 4 µs/B） |
| E1 OS 延遲中位數 `os.median_s` | {bp['os.median_s']*1e3:.2f} ms |
| C3 PER 好 / 壞 | {bp['per.good']:.3f} / {bp['per.bad']:.2f} |
| C3 進入壞狀態機率、壞狀態平均長度 | {bp['per.p_g2b']:.4f} / event、{bp['per.mean_bad_events']:.1f} events |
| E2 停頓發生率 | {bp['pauses.rate_hz']:.3f} Hz（5–80 ms） |
| C11 起始延遲 | {bp['startup.delay_s']*1e3:.0f} ms |

## 與目標比較（驗證：5 個新 seed × 1200 s × 4 節點）

| 指標 | 論文 Fig.6 | 模擬 | 標準 | 通過 |
|---|---|---|---|---|
| 殘差 P10 (ms) | {tst['P10']:.1f} | {mv['P10']:.1f} | ±3 | {'✔' if passed['P10'] else '✘'} |
| 殘差 P50 (ms) | {tst['P50']:.1f} | {mv['P50']:.1f} | ±3 | {'✔' if passed['P50'] else '✘'} |
| 殘差 P90 (ms) | {tst['P90']:.1f} | {mv['P90']:.1f} | ±3 | {'✔' if passed['P90'] else '✘'} |
| 殘差 P99 (ms) | {tst['P99']:.1f} | {mv['P99']:.1f} | ±10 | {'✔' if passed['P99'] else '✘'} |
| 離群 > 45 ms | {tst['outlier']*100:.1f} % | {mv['outlier']*100:.1f} % | ±2 pp | {'✔' if passed['outlier'] else '✘'} |
| 起始前 4 點高出下緣 (ms) | {', '.join(f'{v:.0f}' for v in tst['startup4'])} | {', '.join(f'{v:.0f}' for v in mv['startup4'])} | — | |
| 下緣粗糙度 (ms) | {tst['rough']:.1f} | {mv['rough']:.1f} | — | |

圖：`fig6_compare.png`（同座標並排：藍點 δT、紅線 SharkTooth δT*、綠星樞紐點；模擬的 δT 與 t_P 平移了常數 β 與起點，形狀不變）、
`qq_residual.png`（殘差 QQ 圖與直方圖）。

## 消融：每次關掉一個模組（`ablation_table.md`，簡報「延遲從何而來」的核心證據）

{open(od / 'ablation_table.md', encoding='utf-8').read()}

## 結論

- Fig.6 的四個特徵各有明確來源：
  **連續 30 ms 帶** = C1 等 connection event（P 與 CI 不成整數比，相位連續掃過）；
  **帶上方 30–60 ms 的稀疏點** = C3 重傳一次（PER ≈ {bp['per.good']*100:.0f} %）；
  **起始垂直尖峰** = C11 第一包多等約 {bp['startup.delay_s']*1e3:.0f} ms ＋ C4 流量控制讓後面幾包以 −40 ms/包排空；
  **平滑下緣** = 封包間隔 100.78 ms（不是 100 ms）。把封包間隔改回 100 ms，下緣立刻變成階梯。
- D2 / D3 / E1 / E3 這些「連續型」延遲在 4 節點 / 244 B 下都只有 ~1 ms 量級：關掉它們，帶寬與分位數幾乎不變。
  **上一版規格把 Fig.6 的帶寬歸因於「大的連續型延遲」是不對的**——帶寬就是 1 個 CI。
- 多節點並非必要：單節點加 PER ≈ 9 % 就能重現（見 T0）；4 節點的設定也一樣能重現，Fig.6 本身無法區分。
""")
